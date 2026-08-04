package ai.liquid.lfmdemo.tts

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.os.SystemClock
import android.util.Log
import com.k2fsa.sherpa.onnx.GenerationConfig
import com.k2fsa.sherpa.onnx.OfflineTts
import com.k2fsa.sherpa.onnx.OfflineTtsConfig
import com.k2fsa.sherpa.onnx.OfflineTtsKokoroModelConfig
import com.k2fsa.sherpa.onnx.OfflineTtsMatchaModelConfig
import com.k2fsa.sherpa.onnx.OfflineTtsModelConfig
import com.k2fsa.sherpa.onnx.OfflineTtsSupertonicModelConfig
import com.k2fsa.sherpa.onnx.OfflineTtsVitsModelConfig
import com.k2fsa.sherpa.onnx.OfflineTtsZipVoiceModelConfig
import java.io.File
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

/**
 * Offline neural TTS backend for the demo app.
 *
 * Calls are serialized through one synthesis worker because sherpa-onnx's
 * [OfflineTts] object is not safe for overlapping `generate` calls. PCM is fed
 * to a separate writer and playback completion is tracked on a third worker,
 * so the next sentence can synthesize while the previous sentence is playing.
 * This preserves streaming latency without reordering speech or reporting a
 * turn complete before the speaker drains.
 *
 * Every generated PCM chunk passes [PcmGate] before it reaches AudioTrack.
 * This is required for the bundled Kokoro INT8 diagnostic candidate, which can
 * return non-finite PCM for very short Mandarin input.
 */
class SherpaTtsEngine(
    private val bundleRoot: File,
    private val listener: Listener,
) {
    interface Listener {
        fun onUtteranceStart(utteranceId: String)
        fun onUtteranceDone(utteranceId: String)
        fun onUtteranceError(utteranceId: String, reason: String)
        fun onUtteranceMetrics(utteranceId: String, metrics: Map<String, Any>)
    }

    companion object {
        private const val TAG = "SherpaTts"
        private const val TARGET_RMS_DBFS = -20.0
        private const val CUSHION_MS = 140
        private const val WRITE_QUANTUM_MS = 20
        private const val ZERO_WRITE_BACKOFF_MS = 2L
        private const val ZERO_WRITE_START_FALLBACK_RETRIES = 4
        private const val MAX_CONSECUTIVE_ZERO_WRITES = 1_000
        private const val PLAYBACK_GRACE_MS = 3_000L
        private const val WORKER_POLL_MS = 100L
        private const val PCM_QUEUE_CAPACITY = 24
        private const val TASK_QUEUE_CAPACITY = 64
        const val MIN_CHUNK_CHARS = 2
    }

    private data class SpeechTask(
        val text: String,
        val utteranceId: String,
        val generation: Long,
        val candidate: Candidate,
    )

    private data class PcmChunk(
        val generation: Long,
        val samples: FloatArray,
    )

    private data class PlaybackTarget(
        val generation: Long,
        val utteranceId: String,
        val targetSamples: Long,
        val session: PlaybackSession,
    )

    private data class ShutdownState(
        val engine: OfflineTts?,
        val synthesis: Thread?,
        val completion: Thread?,
    )

    private class PlaybackSession(
        val generation: Long,
        val sampleRate: Int,
        val track: AudioTrack,
    ) {
        val pcmQueue = LinkedBlockingQueue<PcmChunk>(PCM_QUEUE_CAPACITY)
        val enqueuedSamples = AtomicLong(0)
        val playbackStarted = AtomicBoolean(false)
        val zeroWriteRetries = AtomicLong(0)
        @Volatile var writer: Thread? = null
        @Volatile var failureReason: String? = null
    }

    private val stateLock = Any()
    private val playbackLock = Any()
    private val speechQueue = LinkedBlockingQueue<SpeechTask>(TASK_QUEUE_CAPACITY)
    private val completionQueue = LinkedBlockingQueue<PlaybackTarget>(TASK_QUEUE_CAPACITY)
    private val closed = AtomicBoolean(true)
    private val generation = AtomicLong(0)

    @Volatile private var tts: OfflineTts? = null
    @Volatile private var candidate: Candidate? = null
    @Volatile private var playbackSession: PlaybackSession? = null
    @Volatile private var synthesisWorker: Thread? = null
    @Volatile private var completionWorker: Thread? = null

    val loadedCandidateId: String? get() = candidate?.id
    val sampleRate: Int get() = tts?.sampleRate() ?: 0

    fun canReuseModel(c: Candidate): Boolean = synchronized(stateLock) {
        !closed.get() && tts != null && candidate?.sharesModelWith(c) == true
    }

    fun selectCandidate(c: Candidate) {
        synchronized(stateLock) {
            check(!closed.get() && tts != null && candidate?.sharesModelWith(c) == true) {
                "Candidate ${c.id} does not match the resident TTS graph"
            }
            candidate = c
        }
        Log.i(TAG, "reused resident graph for ${c.id} (${c.generationLanguage})")
    }

    /** Loads exactly one candidate and starts the ordered worker pipeline. */
    fun load(c: Candidate) {
        shutdown()
        val missing = c.hasAllFiles(bundleRoot)
        check(missing.isEmpty()) {
            "Voice \"${c.label}\" is missing ${missing.size} file(s): ${missing.first()}"
        }

        val model = OfflineTtsModelConfig(
            numThreads = c.numThreads,
            debug = false,
            provider = "cpu",
        )
        when (c.engine) {
            "vits" -> model.vits = OfflineTtsVitsModelConfig(
                model = c.path(bundleRoot, "model"),
                lexicon = c.path(bundleRoot, "lexicon"),
                tokens = c.path(bundleRoot, "tokens"),
                dataDir = c.path(bundleRoot, "data_dir"),
                dictDir = c.path(bundleRoot, "dict_dir"),
            )
            "matcha" -> model.matcha = OfflineTtsMatchaModelConfig(
                acousticModel = c.path(bundleRoot, "acoustic_model"),
                vocoder = c.path(bundleRoot, "vocoder"),
                lexicon = c.path(bundleRoot, "lexicon"),
                tokens = c.path(bundleRoot, "tokens"),
                dataDir = c.path(bundleRoot, "data_dir"),
                dictDir = c.path(bundleRoot, "dict_dir"),
            )
            "kokoro" -> model.kokoro = OfflineTtsKokoroModelConfig(
                model = c.path(bundleRoot, "model"),
                voices = c.path(bundleRoot, "voices"),
                tokens = c.path(bundleRoot, "tokens"),
                dataDir = c.path(bundleRoot, "data_dir"),
                dictDir = c.path(bundleRoot, "dict_dir"),
                lexicon = c.files["lexicon"]
                    ?.split(",")
                    ?.joinToString(",") { File(bundleRoot, it.trim()).absolutePath }
                    ?: "",
            )
            "supertonic" -> model.supertonic = OfflineTtsSupertonicModelConfig(
                durationPredictor = c.path(bundleRoot, "duration_predictor"),
                textEncoder = c.path(bundleRoot, "text_encoder"),
                vectorEstimator = c.path(bundleRoot, "vector_estimator"),
                vocoder = c.path(bundleRoot, "vocoder"),
                ttsJson = c.path(bundleRoot, "tts_json"),
                unicodeIndexer = c.path(bundleRoot, "unicode_indexer"),
                voiceStyle = c.path(bundleRoot, "voice_style"),
            )
            "zipvoice" -> model.zipvoice = OfflineTtsZipVoiceModelConfig(
                tokens = c.path(bundleRoot, "tokens"),
                encoder = c.path(bundleRoot, "encoder"),
                decoder = c.path(bundleRoot, "decoder"),
                vocoder = c.path(bundleRoot, "vocoder"),
                dataDir = c.path(bundleRoot, "data_dir"),
                lexicon = c.path(bundleRoot, "lexicon"),
            )
            else -> throw IllegalArgumentException("Unknown TTS engine type '${c.engine}'")
        }

        val loaded = OfflineTts(null, OfflineTtsConfig(model = model))
        synchronized(stateLock) {
            tts = loaded
            candidate = c
            closed.set(false)
            startWorkersLocked(loaded)
        }
        Log.i(TAG, "loaded ${c.id} (${c.engine}, ${c.numThreads} threads)")
    }

    /**
     * Queue one text chunk. `flush=true` begins a new playback generation and
     * cancels every task, PCM buffer, and completion from the previous turn.
     */
    fun speak(text: String, utteranceId: String, flush: Boolean) {
        val selectedCandidate = candidate
        val ready = tts != null && selectedCandidate != null && !closed.get()
        if (!ready) {
            listener.onUtteranceError(utteranceId, "no_voice_loaded")
            return
        }
        if (flush) stop()
        val task = SpeechTask(text, utteranceId, generation.get(), selectedCandidate!!)
        if (!speechQueue.offer(task)) {
            listener.onUtteranceError(utteranceId, "synthesis_queue_full")
        }
    }

    private fun startWorkersLocked(engine: OfflineTts) {
        synthesisWorker = Thread(
            { synthesisLoop(engine) },
            "sherpa-tts-synthesis",
        ).apply { isDaemon = true; start() }
        completionWorker = Thread(
            { completionLoop() },
            "sherpa-tts-completion",
        ).apply { isDaemon = true; start() }
    }

    private fun synthesisLoop(engine: OfflineTts) {
        while (!closed.get()) {
            val task = try {
                speechQueue.poll(WORKER_POLL_MS, TimeUnit.MILLISECONDS)
            } catch (_: InterruptedException) {
                if (closed.get()) return
                continue
            } ?: continue
            if (task.generation != generation.get()) continue
            synthesize(engine, task.candidate, task)
        }
    }

    private fun synthesize(engine: OfflineTts, c: Candidate, task: SpeechTask) {
        val startedAt = SystemClock.elapsedRealtime()
        var firstChunkMs = -1L
        var dropped = 0
        var totalSamples = 0
        var startReported = false
        try {
            val session = ensureTrack(engine.sampleRate(), task.generation) ?: return
            val reference = c.referenceWav?.let { WavIo.readMono(File(bundleRoot, it)) }
            val config = GenerationConfig(
                silenceScale = 1.0f,
                speed = c.speed,
                sid = c.sid,
                referenceAudio = reference?.first,
                referenceSampleRate = reference?.second ?: 0,
                referenceText = c.referenceText ?: "",
                numSteps = c.numSteps,
                extra = c.extra,
            )
            var gain = 1f
            val audio = engine.generateWithConfigAndCallback(task.text, config) { chunk ->
                if (!isActivePlaybackSession(session, task.generation)) {
                    return@generateWithConfigAndCallback 0
                }
                if (firstChunkMs < 0) {
                    firstChunkMs = SystemClock.elapsedRealtime() - startedAt
                }
                val gate = PcmGate.evaluate(chunk)
                if (!gate.passed) {
                    dropped++
                    Log.w(TAG, "dropped chunk for ${task.utteranceId}: ${gate.failures}")
                    return@generateWithConfigAndCallback 1
                }
                if (gain == 1f) gain = makeupGain(gate.rms, gate.peak)
                // The native callback owns `chunk`; playback must retain its
                // own buffer. Clamp after makeup gain so a later, louder
                // callback cannot turn the first chunk's gain into clipping.
                val output = FloatArray(chunk.size) { index ->
                    (chunk[index] * gain).coerceIn(-0.95f, 0.95f)
                }
                if (!enqueuePcm(session, PcmChunk(task.generation, output))) {
                    return@generateWithConfigAndCallback 0
                }
                if (!startReported) {
                    startReported = true
                    listener.onUtteranceStart(task.utteranceId)
                }
                totalSamples += output.size
                1
            }
            if (task.generation != generation.get() || closed.get()) return

            val elapsed = SystemClock.elapsedRealtime() - startedAt
            val seconds = audio.samples.size.toDouble() / audio.sampleRate
            listener.onUtteranceMetrics(
                task.utteranceId,
                mapOf(
                    "engine_id" to c.id,
                    "engine_family" to c.engine,
                    // Keep the engine's playback epoch distinct from the
                    // WebView turn generation injected by NativeAudioBridge.
                    "engine_generation" to task.generation,
                    // CandidateRegistry guarantees an explicit GenerationConfig
                    // language cannot disagree with the routed catalog language.
                    "generation_language" to c.generationLanguage,
                    "threads" to c.numThreads,
                    "first_chunk_ms" to firstChunkMs,
                    "synthesis_ms" to elapsed,
                    "audio_seconds" to seconds,
                    "rtf" to if (seconds > 0) elapsed / 1000.0 / seconds else -1.0,
                    "dropped_chunks" to dropped,
                    "applied_gain_db" to (20.0 * Math.log10(gain.toDouble())),
                    "sample_rate_hz" to audio.sampleRate,
                    "serialized_synthesis" to true,
                    "pending_synthesis_tasks" to speechQueue.size,
                    "audio_track_started" to session.playbackStarted.get(),
                    "audio_track_buffer_frames" to session.track.bufferSizeInFrames,
                    "audio_track_write_quantum_ms" to WRITE_QUANTUM_MS,
                    "audio_track_zero_write_retries" to session.zeroWriteRetries.get(),
                ),
            )
            if (totalSamples == 0) {
                listener.onUtteranceError(task.utteranceId, "all_chunks_rejected_invalid_pcm")
                return
            }
            completionQueue.put(
                PlaybackTarget(
                    generation = task.generation,
                    utteranceId = task.utteranceId,
                    targetSamples = session.enqueuedSamples.get(),
                    session = session,
                ),
            )
        } catch (_: InterruptedException) {
            if (task.generation == generation.get() && !closed.get()) {
                listener.onUtteranceError(task.utteranceId, "synthesis_interrupted")
            }
            Thread.currentThread().interrupt()
        } catch (error: Throwable) {
            Log.e(TAG, "synthesis failed", error)
            if (task.generation == generation.get() && !closed.get()) {
                listener.onUtteranceError(
                    task.utteranceId,
                    "${error::class.java.simpleName}: ${error.message}",
                )
            }
        }
    }

    private fun completionLoop() {
        while (!closed.get()) {
            val target = try {
                completionQueue.poll(WORKER_POLL_MS, TimeUnit.MILLISECONDS)
            } catch (_: InterruptedException) {
                if (closed.get()) return
                continue
            } ?: continue
            if (target.generation != generation.get()) continue
            val failure = waitForPlayout(target)
            if (failure == null) {
                if (target.generation == generation.get() && !closed.get()) {
                    listener.onUtteranceDone(target.utteranceId)
                }
            } else if (target.generation == generation.get() && !closed.get()) {
                listener.onUtteranceError(target.utteranceId, failure)
            }
        }
    }

    private fun makeupGain(rms: Double, peak: Float): Float {
        if (rms <= 0.0) return 1f
        val target = Math.pow(10.0, TARGET_RMS_DBFS / 20.0)
        var gain = target / rms
        if (peak > 0f && peak * gain > 0.95) gain = 0.95 / peak
        return gain.toFloat()
    }

    private fun isCurrentPlaybackSession(
        session: PlaybackSession,
        taskGeneration: Long,
    ): Boolean = !closed.get() &&
        taskGeneration == generation.get() &&
        playbackSession === session

    private fun isActivePlaybackSession(
        session: PlaybackSession,
        taskGeneration: Long,
    ): Boolean = isCurrentPlaybackSession(session, taskGeneration) &&
        session.failureReason == null

    private fun enqueuePcm(session: PlaybackSession, chunk: PcmChunk): Boolean {
        while (isCurrentPlaybackSession(session, chunk.generation)) {
            session.failureReason?.let { throw IllegalStateException(it) }
            val accepted = session.pcmQueue.offer(
                chunk,
                WORKER_POLL_MS,
                TimeUnit.MILLISECONDS,
            )
            if (accepted) {
                session.enqueuedSamples.addAndGet(chunk.samples.size.toLong())
                return true
            }
        }
        return false
    }

    private fun ensureTrack(rate: Int, taskGeneration: Long): PlaybackSession? {
        synchronized(playbackLock) {
            if (taskGeneration != generation.get() || closed.get()) return null
            playbackSession?.let { active ->
                if (active.generation == taskGeneration) return active
            }
            val minBytes = AudioTrack.getMinBufferSize(
                rate,
                AudioFormat.CHANNEL_OUT_MONO,
                AudioFormat.ENCODING_PCM_FLOAT,
            )
            val created = AudioTrack.Builder()
                .setAudioAttributes(
                    AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_MEDIA)
                        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                        .build(),
                )
                .setAudioFormat(
                    AudioFormat.Builder()
                        .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                        .setSampleRate(rate)
                        .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                        .build(),
                )
                .setBufferSizeInBytes(maxOf(minBytes * 4, rate * 4))
                .setTransferMode(AudioTrack.MODE_STREAM)
                .build()
            check(created.state == AudioTrack.STATE_INITIALIZED) {
                "AudioTrack did not initialize"
            }
            val session = PlaybackSession(taskGeneration, rate, created)
            playbackSession = session
            startWriterLocked(session)
            return session
        }
    }

    private fun startWriterLocked(session: PlaybackSession) {
        val writer = Thread({
            var written = 0L
            var consecutiveZeroWrites = 0
            val startupSamples = maxOf(
                1L,
                session.sampleRate.toLong() * CUSHION_MS / 1000L,
            )
            val writeQuantumSamples = maxOf(
                1,
                session.sampleRate * WRITE_QUANTUM_MS / 1000,
            )
            try {
                while (isActivePlaybackSession(session, session.generation)) {
                    val next = session.pcmQueue.poll(50, TimeUnit.MILLISECONDS) ?: continue
                    if (next.generation != session.generation) continue
                    var offset = 0
                    while (offset < next.samples.size &&
                        isActivePlaybackSession(session, session.generation)
                    ) {
                        val startupRemaining = (startupSamples - written)
                            .coerceAtLeast(1L)
                            .coerceAtMost(Int.MAX_VALUE.toLong())
                            .toInt()
                        val requestSize = minOf(
                            next.samples.size - offset,
                            writeQuantumSamples,
                            if (session.playbackStarted.get()) {
                                writeQuantumSamples
                            } else {
                                startupRemaining
                            },
                        )
                        val count = session.track.write(
                            next.samples,
                            offset,
                            requestSize,
                            AudioTrack.WRITE_BLOCKING,
                        )
                        if (!isCurrentPlaybackSession(session, session.generation)) break
                        if (count < 0) {
                            throw IllegalStateException("AudioTrack.write returned $count")
                        }
                        if (count == 0) {
                            session.zeroWriteRetries.incrementAndGet()
                            consecutiveZeroWrites += 1
                            if (!session.playbackStarted.get() &&
                                consecutiveZeroWrites >= ZERO_WRITE_START_FALLBACK_RETRIES
                            ) {
                                // Some vendor AudioTrack implementations refuse
                                // further pre-roll writes with a zero return. A
                                // started empty stream can underrun once, but it
                                // recovers; treating zero as fatal cannot.
                                startPlayback(session)
                            }
                            if (consecutiveZeroWrites >= MAX_CONSECUTIVE_ZERO_WRITES) {
                                throw IllegalStateException(
                                    "AudioTrack.write made no progress after " +
                                        "$consecutiveZeroWrites retries",
                                )
                            }
                            Thread.sleep(ZERO_WRITE_BACKOFF_MS)
                            continue
                        }
                        consecutiveZeroWrites = 0
                        offset += count
                        written += count
                        if (!session.playbackStarted.get() && written >= startupSamples) {
                            startPlayback(session)
                        }
                    }
                }
            } catch (_: InterruptedException) {
                Thread.currentThread().interrupt()
            } catch (error: Throwable) {
                if (isCurrentPlaybackSession(session, session.generation)) {
                    session.failureReason = "audio_track_write_failure: ${error.message}"
                    Log.e(TAG, "AudioTrack writer failed", error)
                }
            }
        }, "sherpa-tts-writer-g${session.generation}")
        session.writer = writer
        writer.apply { isDaemon = true; start() }
    }

    private fun startPlayback(session: PlaybackSession) {
        synchronized(session) {
            if (session.playbackStarted.get()) return
            if (session.track.playState != AudioTrack.PLAYSTATE_PLAYING) {
                session.track.play()
            }
            session.playbackStarted.set(true)
        }
    }

    private fun waitForPlayout(target: PlaybackTarget): String? {
        val active = target.session
        if (!isCurrentPlaybackSession(active, target.generation)) {
            return "playback_session_replaced"
        }
        active.failureReason?.let { return it }
        runCatching { startPlayback(active) }
            .getOrElse { return "audio_track_play_failure: ${it.message}" }

        val current = playbackHead(active.track)
            ?: return "audio_track_head_unavailable"
        val remaining = (target.targetSamples - current).coerceAtLeast(0)
        val deadline = SystemClock.elapsedRealtime() +
            remaining * 1000L / active.sampleRate + PLAYBACK_GRACE_MS
        while (
            isCurrentPlaybackSession(active, target.generation) &&
            SystemClock.elapsedRealtime() < deadline
        ) {
            active.failureReason?.let { return it }
            val head = playbackHead(active.track)
                ?: return "audio_track_head_unavailable"
            if (head >= target.targetSamples) return null
            try {
                Thread.sleep(10)
            } catch (_: InterruptedException) {
                Thread.currentThread().interrupt()
                return "playback_wait_interrupted"
            }
        }
        return if (isCurrentPlaybackSession(active, target.generation)) {
            "playback_drain_timeout"
        } else {
            "playback_session_replaced"
        }
    }

    private fun playbackHead(t: AudioTrack): Long? = runCatching {
        t.playbackHeadPosition.toLong() and 0xffffffffL
    }.getOrNull()

    /** Barge-in: invalidate active work and synchronously detach playback. */
    fun stop() {
        generation.incrementAndGet()
        speechQueue.clear()
        completionQueue.clear()
        val oldSession: PlaybackSession?
        synchronized(playbackLock) {
            oldSession = playbackSession
            playbackSession = null
            oldSession?.pcmQueue?.clear()
            oldSession?.writer?.interrupt()
            oldSession?.track?.let {
                runCatching { it.pause() }
                runCatching { it.flush() }
                runCatching { it.stop() }
                runCatching { it.release() }
            }
        }
    }

    private fun beginShutdown(): ShutdownState? = synchronized(stateLock) {
        if (tts == null && candidate == null && closed.get()) {
            null
        } else {
            closed.set(true)
            stop()
            val state = ShutdownState(tts, synthesisWorker, completionWorker)
            synthesisWorker = null
            completionWorker = null
            tts = null
            candidate = null
            state
        }
    }

    /** Release the model after all workers have been told to stop. */
    fun shutdown() {
        val state = beginShutdown() ?: return
        state.synthesis?.interrupt()
        state.completion?.interrupt()
        runCatching { state.synthesis?.join(2_000) }
        runCatching { state.completion?.join(500) }
        if (state.synthesis?.isAlive == true && state.engine != null) {
            Log.w(TAG, "synthesis worker is still exiting; deferring model release")
            Thread({
                runCatching { state.synthesis.join() }
                runCatching { state.engine.release() }
            }, "sherpa-tts-release").apply { isDaemon = true; start() }
        } else {
            runCatching { state.engine?.release() }
        }
    }

    /**
     * Serialized model replacement path. It never allocates the next graph
     * until the previous synthesis worker has exited and its native graph has
     * actually been released, avoiding a transient dual-model memory spike.
     */
    fun shutdownAndAwaitRelease() {
        val state = beginShutdown() ?: return
        state.synthesis?.interrupt()
        state.completion?.interrupt()
        runCatching { state.synthesis?.join() }
        runCatching { state.completion?.join() }
        runCatching { state.engine?.release() }
    }
}
