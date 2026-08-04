package ai.liquid.ttsbench

import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Real-time safety subset of the host waveform gates.
 *
 * This is not decoration. One of the candidates in the bundle (Kokoro INT8)
 * returns an entirely non-finite buffer for short Mandarin input, and handing
 * that to AudioTrack produces full-scale noise through the phone speaker. Every
 * chunk is checked before it reaches the audio path, and a chunk that fails is
 * dropped rather than played.
 *
 * Shared thresholds match scripts/tts_research_20260726/gates.py. The host
 * complete-waveform pass additionally checks long internal silence and repeated
 * windows; those checks require whole-utterance context and are not performed
 * here.
 */
object PcmGate {

    private const val CLIP_LEVEL = 0.999f
    private const val MIN_RMS = 0.005
    private const val WARN_ABS_DC = 0.01
    private const val MAX_ABS_DC = 0.10
    private const val MAX_CLIP_FRACTION = 0.001

    data class Result(
        val passed: Boolean,
        val failures: List<String>,
        val warnings: List<String>,
        val samples: Int,
        val nonFinite: Int,
        val peak: Float,
        val rms: Double,
        val dc: Double,
        val clipCount: Int,
    ) {
        fun short(): String =
            if (passed) "ok peak=%.3f rms=%.4f".format(peak, rms)
            else "REJECTED ${failures.joinToString(",")}"
    }

    fun evaluate(samples: FloatArray): Result {
        if (samples.isEmpty()) {
            return Result(false, listOf("empty"), emptyList(), 0, 0, 0f, 0.0, 0.0, 0)
        }
        var nonFinite = 0
        var peak = 0f
        var sum = 0.0
        var sumSq = 0.0
        var clip = 0
        var finiteCount = 0
        for (v in samples) {
            if (v.isNaN() || v.isInfinite()) {
                nonFinite++
                continue
            }
            finiteCount++
            val a = abs(v)
            if (a > peak) peak = a
            if (a >= CLIP_LEVEL) clip++
            sum += v
            sumSq += v.toDouble() * v.toDouble()
        }
        if (finiteCount == 0) {
            return Result(
                false, listOf("all_samples_non_finite"), emptyList(),
                samples.size, nonFinite, 0f, 0.0, 0.0, 0,
            )
        }
        val rms = sqrt(sumSq / finiteCount)
        val dc = sum / finiteCount
        val clipFraction = clip.toDouble() / finiteCount

        val failures = mutableListOf<String>()
        val warnings = mutableListOf<String>()
        if (nonFinite > 0) failures += "non_finite_pcm:$nonFinite"
        if (rms < MIN_RMS) failures += "near_silent_rms:%.6f".format(rms)
        if (clipFraction > MAX_CLIP_FRACTION) failures += "clipping:%.6f".format(clipFraction)
        if (abs(dc) > MAX_ABS_DC) failures += "severe_dc_offset:%.6f".format(dc)
        else if (abs(dc) > WARN_ABS_DC) warnings += "dc_offset_needs_blocker:%.6f".format(dc)
        if (peak > 0.99f) warnings += "peak_near_full_scale:%.4f".format(peak)

        return Result(
            failures.isEmpty(), failures, warnings,
            samples.size, nonFinite, peak, rms, dc, clip,
        )
    }
}
