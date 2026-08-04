package ai.liquid.ttsbench

import android.content.Intent
import android.os.Build
import android.os.PowerManager
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * Unattended sweep driven by intent extras.
 *
 * The interactive UI cannot carry an overnight run: nobody is there to tap, and
 * driving spinners with `adb shell input tap` is too fragile to trust for a
 * hundred measurements. This mode is launched with `am start ... --es mode
 * autorun`, runs to completion on its own, and writes a marker file the host
 * polls, so the host never has to guess whether the device is still working.
 *
 * Every result is flushed to disk as it is produced rather than at the end, so
 * a crash, a reboot, or a battery cut-off still leaves everything measured up
 * to that point.
 */
object AutoRun {

    data class Options(
        val enabled: Boolean,
        val repeats: Int,
        val play: Boolean,
        val saveWav: Boolean,
        val threads: List<Int>,
        val candidateFilter: String?,
        val phraseFilter: String?,
        val tag: String,
        val equalize: Boolean,
    )

    fun parse(intent: Intent?): Options {
        val mode = intent?.getStringExtra("mode")
        val threadsRaw = intent?.getStringExtra("threads")
        return Options(
            enabled = mode == "autorun",
            repeats = intent?.getIntExtra("repeats", 3) ?: 3,
            play = intent?.getBooleanExtra("play", false) ?: false,
            saveWav = intent?.getBooleanExtra("save_wav", false) ?: false,
            threads = threadsRaw?.split(",")?.mapNotNull { it.trim().toIntOrNull() }
                ?.ifEmpty { null } ?: listOf(0),   // 0 = whatever candidates.json says
            candidateFilter = intent?.getStringExtra("candidates")?.takeIf { it != "all" },
            phraseFilter = intent?.getStringExtra("phrases")?.takeIf { it != "all" },
            tag = intent?.getStringExtra("tag") ?: "autorun",
            equalize = intent?.getBooleanExtra("equalize", false) ?: false,
        )
    }

    /**
     * @param progress called with a human-readable line after every measurement
     *   so the UI (and `adb logcat`) shows the run advancing.
     */
    fun execute(
        options: Options,
        candidates: List<Candidate>,
        bundleRoot: File,
        resultsDir: File,
        runner: TtsRunner,
        powerManager: PowerManager?,
        progress: (String) -> Unit,
    ) {
        resultsDir.mkdirs()
        val stamp = System.currentTimeMillis()
        val out = File(resultsDir, "autorun_${options.tag}_$stamp.jsonl")
        val marker = File(resultsDir, "autorun_${options.tag}_$stamp.done")
        val statusFile = File(resultsDir, "autorun_status.json")

        // A long sweep must not be suspended halfway through by the screen
        // turning off; without this the host would see a run that simply stops.
        val wake = powerManager?.newWakeLock(
            PowerManager.PARTIAL_WAKE_LOCK, "ttsbench:autorun"
        )?.apply { setReferenceCounted(false); acquire(6 * 60 * 60 * 1000L) }

        val selected = candidates.filter { c ->
            options.candidateFilter == null ||
                options.candidateFilter.split(",").map { it.trim() }.contains(c.id)
        }

        var done = 0
        var failed = 0
        val started = System.currentTimeMillis()
        val plannedUnits = selected.sumOf { c ->
            val n = CandidateRegistry.phrases(bundleRoot, c.language)
                .count { options.phraseFilter == null ||
                    options.phraseFilter.split(",").map { p -> p.trim() }.contains(it.first) }
            n * options.threads.size
        }

        fun writeStatus(state: String, current: String) {
            runCatching {
                statusFile.writeText(
                    JSONObject().apply {
                        put("state", state)
                        put("tag", options.tag)
                        put("output", out.absolutePath)
                        put("completed_units", done)
                        put("failed_units", failed)
                        put("planned_units", plannedUnits)
                        put("current", current)
                        put("elapsed_ms", System.currentTimeMillis() - started)
                        put("device", Build.MODEL)
                    }.toString(2)
                )
            }
        }

        writeStatus("running", "starting")
        out.appendText(
            JSONObject().apply {
                put("record", "header")
                put("schema_version", 1)
                put("tag", options.tag)
                put("repeats", options.repeats)
                put("play", options.play)
                put("thread_sweep", JSONArray(options.threads))
                put("planned_units", plannedUnits)
                put("device", JSONObject().apply {
                    put("model", Build.MODEL)
                    put("device", Build.DEVICE)
                    put("android_release", Build.VERSION.RELEASE)
                    put("sdk_int", Build.VERSION.SDK_INT)
                    put("abi", Build.SUPPORTED_ABIS.firstOrNull() ?: "")
                    put("hardware", Build.HARDWARE)
                    put("soc_model", Build.SOC_MODEL)
                })
            }.toString() + "\n"
        )

        try {
            for (candidate in selected) {
                for (threads in options.threads) {
                    val effective = if (threads > 0) candidate.copy(numThreads = threads) else candidate
                    val label = "${effective.id}@${effective.numThreads}t"
                    try {
                        writeStatus("running", "loading $label")
                        runner.load(effective)
                    } catch (e: Throwable) {
                        failed++
                        out.appendText(
                            JSONObject().apply {
                                put("record", "load_failure")
                                put("candidate_id", effective.id)
                                put("threads", effective.numThreads)
                                put("error", "${e::class.java.simpleName}: ${e.message}")
                            }.toString() + "\n"
                        )
                        progress("$label LOAD FAILED: ${e.message}")
                        continue
                    }

                    val phraseList = CandidateRegistry.phrases(bundleRoot, effective.language)
                        .filter { options.phraseFilter == null ||
                            options.phraseFilter.split(",").map { p -> p.trim() }.contains(it.first) }

                    for ((phraseId, text) in phraseList) {
                        for (repeat in 0 until options.repeats) {
                            writeStatus("running", "$label $phraseId #${repeat + 1}")
                            try {
                                val wav = if (options.saveWav && repeat == 0) {
                                    File(resultsDir, "wav/${effective.id}__${phraseId}.wav")
                                } else null
                                val r = runner.speak(
                                    text = text,
                                    play = options.play,
                                    cushionMs = 140,
                                    equalizeLoudness = options.equalize,
                                    saveWavTo = wav,
                                )
                                val json = r.toJson().apply {
                                    put("record", "run")
                                    put("phrase_id", phraseId)
                                    put("repeat_index", repeat)
                                    put("threads", effective.numThreads)
                                    put("language", effective.language)
                                    put("engine", effective.engine)
                                    put("num_steps", effective.numSteps)
                                    if (wav != null && wav.exists()) put("wav_file", wav.name)
                                }
                                // Flushed per measurement: a crash or a dead
                                // battery must not cost the whole night.
                                out.appendText(json.toString() + "\n")
                            } catch (e: Throwable) {
                                failed++
                                out.appendText(
                                    JSONObject().apply {
                                        put("record", "run_failure")
                                        put("candidate_id", effective.id)
                                        put("phrase_id", phraseId)
                                        put("repeat_index", repeat)
                                        put("threads", effective.numThreads)
                                        put("error", "${e::class.java.simpleName}: ${e.message}")
                                    }.toString() + "\n"
                                )
                                progress("$label $phraseId FAILED: ${e.message}")
                            }
                        }
                        done++
                        progress("$label $phraseId done ($done/$plannedUnits)")
                    }
                    runner.release()
                }
            }
        } finally {
            runner.release()
            writeStatus("complete", "finished")
            marker.writeText(
                JSONObject().apply {
                    put("completed_units", done)
                    put("failed_units", failed)
                    put("planned_units", plannedUnits)
                    put("elapsed_ms", System.currentTimeMillis() - started)
                    put("output", out.absolutePath)
                }.toString(2)
            )
            wake?.let { runCatching { it.release() } }
        }
    }
}
