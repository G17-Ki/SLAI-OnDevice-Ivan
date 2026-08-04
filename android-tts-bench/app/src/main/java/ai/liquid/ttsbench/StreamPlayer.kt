package ai.liquid.ttsbench

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.os.SystemClock

/**
 * Clause-streamed playback with the same startup cushion the Android demo uses.
 *
 * Chunks are written as the synthesizer produces them rather than after the
 * whole utterance, which is the behaviour being evaluated. Playback starts once
 * [cushionMs] of audio is buffered, and the player reports the real
 * first-audible time from the AudioTrack playback head rather than from the
 * write timestamp - a write only proves the buffer was accepted, not that the
 * speaker moved.
 *
 * Underruns are detected from [AudioTrack.getUnderrunCount], which is the
 * device's own count, not a simulation.
 */
class StreamPlayer(
    private val sampleRate: Int,
    private val cushionMs: Int = 140,
    private val gainDb: Float = 0f,
) {
    private var track: AudioTrack? = null
    private var started = false
    private var queuedFrames = 0
    private var writeStartMs = 0L
    private var headWatcher: Thread? = null
    private val stopWatcher = java.util.concurrent.atomic.AtomicBoolean(false)
    private val queue = java.util.concurrent.LinkedBlockingQueue<FloatArray>()
    private var writer: Thread? = null
    @Volatile private var inputClosed = false
    @Volatile private var writtenFrames = 0

    var firstAudibleMs: Long = -1
        private set
    var underrunCount: Int = 0
        private set
    var droppedChunks: Int = 0
        private set

    private val gain: Float = if (gainDb == 0f) 1f else Math.pow(10.0, gainDb / 20.0).toFloat()

    fun start() {
        val minBytes = AudioTrack.getMinBufferSize(
            sampleRate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_FLOAT,
        )
        // Four times the minimum: enough to hold a clause without forcing the
        // synthesizer to block, small enough that an underrun still shows up.
        val bufferBytes = maxOf(minBytes * 4, sampleRate * 4)
        track = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build()
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                    .setSampleRate(sampleRate)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .build()
            )
            .setBufferSizeInBytes(bufferBytes)
            .setTransferMode(AudioTrack.MODE_STREAM)
            .build()
        writeStartMs = SystemClock.elapsedRealtime()
        queuedFrames = 0
        writtenFrames = 0
        started = false
        inputClosed = false
        firstAudibleMs = -1
        underrunCount = 0
        droppedChunks = 0
        queue.clear()
        startWriter(track!!)
    }

    /**
     * Hand a chunk to the player. Returns false if the PCM gate rejected it.
     *
     * The chunk is queued, not written inline. `AudioTrack.write` is blocking,
     * so writing from the synthesis callback made synthesis wait for real-time
     * playback: a 5.2 s utterance that synthesizes in 125 ms reported 3300 ms,
     * and the whole sweep ran at roughly playback speed. That is also not what
     * the shipped Android demo does - it queues each PCM delta immediately and
     * lets playback drain independently - so blocking here measured a design
     * nobody proposed.
     *
     * With an unbounded queue and a separate writer thread, synthesis runs at
     * full speed, RTF stays comparable with the no-playback sweep, and an
     * underrun means what it should: the synthesizer failed to stay ahead.
     */
    fun write(chunk: FloatArray, gate: PcmGate.Result): Boolean {
        track ?: return false
        if (!gate.passed) {
            // Never hand invalid PCM to the audio path.
            droppedChunks++
            return false
        }
        val out = if (gain == 1f) chunk else FloatArray(chunk.size) { chunk[it] * gain }
        queue.put(out)
        queuedFrames += out.size
        return true
    }

    /** Drains the queue into AudioTrack, starting playback once the cushion is met. */
    private fun startWriter(t: AudioTrack) {
        writer = Thread {
            var written = 0
            while (true) {
                val next = queue.poll(50, java.util.concurrent.TimeUnit.MILLISECONDS)
                    ?: if (inputClosed) break else continue
                t.write(next, 0, next.size, AudioTrack.WRITE_BLOCKING)
                written += next.size
                if (!started && written >= sampleRate * cushionMs / 1000) {
                    t.play()
                    started = true
                    startHeadWatcher(t)
                }
            }
            // Short final chunk that never reached the cushion still has to play.
            if (!started && written > 0) {
                t.play()
                started = true
                startHeadWatcher(t)
            }
            writtenFrames = written
        }.apply { isDaemon = true; start() }
    }

    /**
     * Watch the playback head from its own thread.
     *
     * `AudioTrack.write` is blocking, so checking the head immediately after a
     * write reports whenever that write happened to return - which for a long
     * first chunk was over a second late. The first-audible time has to be
     * sampled independently of the writer to mean anything.
     */
    private fun startHeadWatcher(t: AudioTrack) {
        stopWatcher.set(false)
        headWatcher = Thread {
            while (!stopWatcher.get()) {
                if (t.playbackHeadPosition > 0) {
                    firstAudibleMs = SystemClock.elapsedRealtime() - writeStartMs
                    return@Thread
                }
                try {
                    Thread.sleep(1)
                } catch (_: InterruptedException) {
                    return@Thread
                }
            }
        }.apply { isDaemon = true; priority = Thread.MAX_PRIORITY; start() }
    }

    /** Blocks until every queued frame has actually been played out. */
    fun drain() {
        val t = track ?: return
        inputClosed = true
        writer?.join(120_000)
        // Bounded by the audio itself plus a margin, so a track that stops
        // short cannot cost a fixed 60 s per utterance.
        val expectedMs = queuedFrames * 1000L / sampleRate
        val deadline = SystemClock.elapsedRealtime() + expectedMs + 5_000
        while (t.playbackHeadPosition < queuedFrames && SystemClock.elapsedRealtime() < deadline) {
            Thread.sleep(10)
        }
        stopWatcher.set(true)
        underrunCount = t.underrunCount
    }

    fun stop() {
        stopWatcher.set(true)
        inputClosed = true
        writer?.join(2_000)
        writer = null
        headWatcher = null
        track?.let {
            underrunCount = it.underrunCount
            runCatching { it.pause(); it.flush(); it.stop() }
            it.release()
        }
        track = null
        started = false
    }

    companion object {
        /** Gain that brings a measured RMS to the target, capped so it cannot clip. */
        fun makeupGainDb(measuredRms: Double, measuredPeak: Float, targetRmsDbfs: Double = -20.0): Float {
            if (measuredRms <= 0.0) return 0f
            val target = Math.pow(10.0, targetRmsDbfs / 20.0)
            var g = target / measuredRms
            if (measuredPeak > 0f && measuredPeak * g > 0.95) g = 0.95 / measuredPeak
            return (20.0 * Math.log10(g)).toFloat()
        }
    }
}
