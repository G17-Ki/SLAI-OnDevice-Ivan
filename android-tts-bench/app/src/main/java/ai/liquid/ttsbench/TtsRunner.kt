package ai.liquid.ttsbench

import android.app.ActivityManager
import android.content.Context
import android.os.Debug
import android.os.SystemClock
import com.k2fsa.sherpa.onnx.GenerationConfig
import com.k2fsa.sherpa.onnx.OfflineTts
import com.k2fsa.sherpa.onnx.OfflineTtsConfig
import com.k2fsa.sherpa.onnx.OfflineTtsKokoroModelConfig
import com.k2fsa.sherpa.onnx.OfflineTtsMatchaModelConfig
import com.k2fsa.sherpa.onnx.OfflineTtsModelConfig
import com.k2fsa.sherpa.onnx.OfflineTtsSupertonicModelConfig
import com.k2fsa.sherpa.onnx.OfflineTtsVitsModelConfig
import com.k2fsa.sherpa.onnx.OfflineTtsZipVoiceModelConfig
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import kotlin.concurrent.thread

/**
 * Loads one candidate and measures it on the device.
 *
 * One model is resident at a time, matching the host harness and the phone's
 * real constraint - the demo already holds LFM, Qwen and SenseVoice, so a bench
 * that loaded several TTS models at once would measure a situation that cannot
 * ship.
 *
 * The numbers recorded here are the device-side counterparts of the host
 * metrics, so a run can be compared directly with
 * reports/android_phone/streaming_response/tts_research_20260726/.
 */
class TtsRunner(private val context: Context, private val bundleRoot: File) {

    private var tts: OfflineTts? = null
    private var loaded: Candidate? = null
    var lastLoadMs: Long = -1
        private set
    var lastLoadPssDeltaKb: Long = -1
        private set

    fun currentCandidate(): Candidate? = loaded

    fun release() {
        tts?.release()
        tts = null
        loaded = null
    }

    // ActivityManager.getProcessMemoryInfo is rate-limited for ordinary apps on
    // modern Android and returns a cached value, which made every candidate
    // report an identical 68 MiB. Debug.getMemoryInfo reads this process's own
    // memory and is not throttled.
    private val memInfo = Debug.MemoryInfo()

    private fun pssKb(): Long = synchronized(memInfo) {
        Debug.getMemoryInfo(memInfo)
        memInfo.totalPss.toLong()
    }

    // /proc/self/statm field 2 is resident pages: the same quantity the host
    // harness caps at 2.5 GiB, so device and host memory numbers mean the same
    // thing. Debug.MemoryInfo exposes PSS but not RSS.
    private val pageSizeKb: Long = 4L

    private fun rssKb(): Long = try {
        val fields = File("/proc/self/statm").readText().trim().split(" ")
        fields.getOrNull(1)?.toLongOrNull()?.times(pageSizeKb) ?: -1L
    } catch (_: Exception) {
        -1L
    }

    fun load(c: Candidate) {
        release()
        val missing = c.hasAllFiles(bundleRoot)
        if (missing.isNotEmpty()) {
            throw IllegalStateException("missing files for ${c.id}: ${missing.joinToString(", ")}")
        }
        val before = pssKb()
        val t0 = SystemClock.elapsedRealtime()

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
            else -> throw IllegalArgumentException("unknown engine '${c.engine}' for ${c.id}")
        }

        // assetManager is null: every path above is an absolute filesystem path
        // under the pushed bundle, not an APK asset.
        tts = OfflineTts(null, OfflineTtsConfig(model = model))
        lastLoadMs = SystemClock.elapsedRealtime() - t0
        lastLoadPssDeltaKb = if (before >= 0) pssKb() - before else -1
        loaded = c
    }

    data class RunResult(
        val candidateId: String,
        val text: String,
        val sampleRate: Int,
        val loadMs: Long,
        val firstChunkMs: Long,
        val totalSynthMs: Long,
        val audioSeconds: Double,
        val rtf: Double,
        val chunkCount: Int,
        val chunkMs: List<Long>,
        val firstAudibleMs: Long,
        val underruns: Int,
        val droppedChunks: Int,
        val peakPssKb: Long,
        val peakRssKb: Long,
        val gate: PcmGate.Result,
        val chunkGateFailures: List<String>,
        val appliedGainDb: Float,
    ) {
        fun toJson(): JSONObject = JSONObject().apply {
            put("candidate_id", candidateId)
            put("text", text)
            put("sample_rate_hz", sampleRate)
            put("load_ms", loadMs)
            put("first_chunk_ms", firstChunkMs)
            put("total_synthesis_ms", totalSynthMs)
            put("audio_duration_s", audioSeconds)
            put("rtf", rtf)
            put("chunk_count", chunkCount)
            put("chunk_ms", JSONArray(chunkMs))
            put("first_audible_ms", firstAudibleMs)
            put("audiotrack_underruns", underruns)
            put("dropped_chunks_failed_gate", droppedChunks)
            put("peak_total_pss_kb", peakPssKb)
            put("peak_total_rss_kb", peakRssKb)
            put("applied_gain_db", appliedGainDb.toDouble())
            put("gate_passed", gate.passed)
            put("gate_failures", JSONArray(gate.failures))
            put("gate_warnings", JSONArray(gate.warnings))
            put("chunk_gate_failures", JSONArray(chunkGateFailures))
            put("peak_abs", gate.peak.toDouble())
            put("rms", gate.rms)
            put("dc_mean", gate.dc)
            put("nonfinite_count", gate.nonFinite)
        }
    }

    /**
     * Synthesize and (optionally) play, measuring both the synthesizer and the
     * audio path.
     *
     * @param equalizeLoudness applies per-candidate makeup gain so an A/B
     *   comparison is not decided by output level; the candidates measured on
     *   the host differ by more than 20 dB.
     */
    fun speak(
        text: String,
        play: Boolean,
        cushionMs: Int,
        equalizeLoudness: Boolean,
        saveWavTo: File? = null,
        onChunk: (Int, Long, PcmGate.Result) -> Unit = { _, _, _ -> },
    ): RunResult {
        val c = loaded ?: throw IllegalStateException("no candidate loaded")
        val engine = tts ?: throw IllegalStateException("no candidate loaded")

        val gen = GenerationConfig(
            silenceScale = 1.0f,
            speed = c.speed,
            sid = c.sid,
            referenceAudio = c.referenceWav?.let { WavIo.readMono(File(bundleRoot, it)).first },
            referenceSampleRate = c.referenceWav?.let { WavIo.readMono(File(bundleRoot, it)).second } ?: 0,
            referenceText = c.referenceText ?: "",
            numSteps = c.numSteps,
            extra = c.extra,
        )

        var peakPss = pssKb()
        var peakRss = rssKb()
        val stopSampler = java.util.concurrent.atomic.AtomicBoolean(false)
        val sampler = thread(start = true, isDaemon = true) {
            while (!stopSampler.get()) {
                val p = pssKb()
                if (p > peakPss) peakPss = p
                val r = rssKb()
                if (r > peakRss) peakRss = r
                Thread.sleep(50)
            }
        }

        val player = if (play) StreamPlayer(engine.sampleRate(), cushionMs) else null
        // Loudness equalization needs the level before playback starts, so the
        // first chunk is measured and its gain reused for the rest. Judging gain
        // from the first chunk is imperfect but far better than none.
        var gainDb = 0f
        var equalizedPlayer: StreamPlayer? = null

        val chunkTimes = ArrayList<Long>()
        val chunkFailures = ArrayList<String>()
        val start = SystemClock.elapsedRealtime()
        var index = 0

        var totalMs = 0L
        var generationCompleted = false
        val audio = try {
            val generated = engine.generateWithConfigAndCallback(text, gen) { chunk ->
                val at = SystemClock.elapsedRealtime() - start
                chunkTimes.add(at)
                val gate = PcmGate.evaluate(chunk)
                if (!gate.passed) chunkFailures += "chunk$index:${gate.failures.joinToString("|")}"
                if (play) {
                    if (equalizedPlayer == null) {
                        if (equalizeLoudness) {
                            gainDb = StreamPlayer.makeupGainDb(gate.rms, gate.peak)
                        }
                        equalizedPlayer = StreamPlayer(engine.sampleRate(), cushionMs, gainDb).also { it.start() }
                    }
                    equalizedPlayer?.write(chunk, gate)
                }
                onChunk(index, at, gate)
                index++
                1
            }
            totalMs = SystemClock.elapsedRealtime() - start
            equalizedPlayer?.drain()
            generationCompleted = true
            generated
        } finally {
            stopSampler.set(true)
            sampler.join(1000)
            if (!generationCompleted) equalizedPlayer?.stop()
        }

        val samples = audio.samples
        val sr = audio.sampleRate
        val gate = PcmGate.evaluate(samples)
        val seconds = samples.size.toDouble() / sr

        // A waveform that failed a hard gate is never written out as if it were
        // audio, matching the host harness.
        if (saveWavTo != null && gate.passed) {
            runCatching { WavIo.writePcm16(saveWavTo, samples, sr) }
        }

        val result = RunResult(
            candidateId = c.id,
            text = text,
            sampleRate = sr,
            loadMs = lastLoadMs,
            firstChunkMs = chunkTimes.firstOrNull() ?: totalMs,
            totalSynthMs = totalMs,
            audioSeconds = seconds,
            rtf = if (seconds > 0) totalMs / 1000.0 / seconds else -1.0,
            chunkCount = chunkTimes.size,
            chunkMs = chunkTimes,
            firstAudibleMs = equalizedPlayer?.firstAudibleMs ?: -1,
            underruns = equalizedPlayer?.underrunCount ?: -1,
            droppedChunks = equalizedPlayer?.droppedChunks ?: 0,
            peakPssKb = peakPss,
            peakRssKb = peakRss,
            gate = gate,
            chunkGateFailures = chunkFailures,
            appliedGainDb = gainDb,
        )
        equalizedPlayer?.stop()
        player?.stop()
        return result
    }
}
