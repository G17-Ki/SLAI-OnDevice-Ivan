package ai.liquid.ttsbench

import android.os.Build
import android.os.Bundle
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import androidx.appcompat.app.AppCompatActivity
import ai.liquid.ttsbench.databinding.ActivityMainBinding
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Locale
import java.util.TimeZone
import kotlin.concurrent.thread

/**
 * On-device TTS candidate bench.
 *
 * Deliberately a separate app from the LFM demo: the demo's English
 * Expressive-LFM route is validated and must not be perturbed while ten
 * candidates are being compared. Nothing here touches it.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var ui: ActivityMainBinding
    private lateinit var bundleRoot: File
    private var candidates: List<Candidate> = emptyList()
    private var phrases: List<Pair<String, String>> = emptyList()
    private var runner: TtsRunner? = null
    private val results = ArrayList<JSONObject>()
    private var busy = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        ui = ActivityMainBinding.inflate(layoutInflater)
        setContentView(ui.root)

        bundleRoot = File(getExternalFilesDir(null), "tts-eval")
        runner = TtsRunner(this, bundleRoot)

        val (list, error) = CandidateRegistry.load(bundleRoot)
        candidates = list
        if (error != null || candidates.isEmpty()) {
            ui.statusBanner.text = error ?: "No candidates in candidates.json"
            ui.output.text = "Expected bundle at:\n${bundleRoot.absolutePath}\n\n" +
                "Push it with tools/push_tts_bundle.sh, then reopen this app."
            setControlsEnabled(false)
            return
        }

        ui.statusBanner.text = "${candidates.size} candidates · ${Build.MODEL} · ${Build.SUPPORTED_ABIS.firstOrNull()}"
        val labels = candidates.map { c ->
            buildString {
                append(c.language.uppercase()).append("  ").append(c.label)
                if (c.known != null) append("  [!]")
            }
        }
        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, labels).apply {
            setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        }
        ui.candidateA.adapter = adapter
        ui.candidateB.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, labels).apply {
            setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        }
        if (candidates.size > 1) ui.candidateB.setSelection(1)

        ui.candidateA.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(p: AdapterView<*>?, v: View?, pos: Int, id: Long) {
                val c = candidates[pos]
                ui.candidateANotes.text = buildString {
                    append(c.engine).append(" · ").append(c.numThreads).append(" threads")
                    if (c.numSteps > 0) append(" · ").append(c.numSteps).append(" steps")
                    if (c.known != null) append("\n⚠ ").append(c.known)
                    if (c.notes.isNotEmpty()) append("\n").append(c.notes)
                }
                refreshPhrases(c.language)
            }
            override fun onNothingSelected(p: AdapterView<*>?) {}
        }

        val auto = AutoRun.parse(intent)
        if (auto.enabled) {
            startAutoRun(auto)
            return
        }

        ui.speakA.setOnClickListener { runOne(selected(ui.candidateA), play = true) }
        ui.speakB.setOnClickListener { runOne(selected(ui.candidateB), play = true) }
        ui.abTest.setOnClickListener { runAb() }
        ui.runAll.setOnClickListener { runAll() }
        ui.export.setOnClickListener { export() }
    }

    private fun selected(spinner: android.widget.Spinner): Candidate =
        candidates[spinner.selectedItemPosition.coerceIn(candidates.indices)]

    private fun refreshPhrases(language: String) {
        phrases = CandidateRegistry.phrases(bundleRoot, language)
        ui.phrase.adapter = ArrayAdapter(
            this, android.R.layout.simple_spinner_item,
            phrases.map { "${it.first}: ${it.second.take(28)}" },
        ).apply { setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item) }
    }

    private fun currentText(c: Candidate): String {
        val custom = ui.customText.text.toString().trim()
        if (custom.isNotEmpty()) return custom
        val list = CandidateRegistry.phrases(bundleRoot, c.language)
        val idx = ui.phrase.selectedItemPosition
        return list.getOrNull(idx)?.second ?: list.firstOrNull()?.second ?: "测试"
    }

    private fun setControlsEnabled(enabled: Boolean) {
        ui.speakA.isEnabled = enabled
        ui.speakB.isEnabled = enabled
        ui.abTest.isEnabled = enabled
        ui.runAll.isEnabled = enabled
        ui.export.isEnabled = enabled
    }

    private fun append(line: String) = runOnUiThread {
        ui.output.text = buildString {
            append(line).append("\n\n").append(ui.output.text)
        }
    }

    /** Runs off the UI thread; only one synthesis at a time, one model resident. */
    private fun runOne(c: Candidate, play: Boolean, thenRun: (() -> Unit)? = null) {
        if (busy) { append("busy — ignoring"); return }
        busy = true
        setControlsEnabled(false)
        val text = currentText(c)
        thread {
            try {
                val r = runner!!
                if (r.currentCandidate()?.id != c.id) {
                    runOnUiThread { ui.statusBanner.text = "loading ${c.label}…" }
                    r.load(c)
                }
                val wavOut = if (ui.saveWav.isChecked) wavPathFor(c, text) else null
                val res = r.speak(
                    text = text,
                    play = play,
                    cushionMs = 140,
                    equalizeLoudness = ui.equalizeLoudness.isChecked,
                    saveWavTo = wavOut,
                )
                results += res.toJson()
                append(format(c, res) + if (wavOut != null && wavOut.exists()) "\n  wav ${wavOut.name}" else "")
            } catch (e: Throwable) {
                append("${c.id}: FAILED\n${e::class.java.simpleName}: ${e.message}")
            } finally {
                busy = false
                runOnUiThread {
                    setControlsEnabled(true)
                    ui.statusBanner.text = "${candidates.size} candidates · ${Build.MODEL}"
                }
                thenRun?.invoke()
            }
        }
    }

    private fun runAb() {
        val a = selected(ui.candidateA)
        val b = selected(ui.candidateB)
        runOne(a, play = true) {
            Thread.sleep(400)
            runOne(b, play = true)
        }
    }

    /**
     * Every candidate, one after another, without playback: a latency and
     * stability sweep rather than a listening test.
     */
    private fun runAll() {
        if (busy) return
        busy = true
        setControlsEnabled(false)
        thread {
            for (c in candidates) {
                try {
                    val r = runner!!
                    r.load(c)
                    val text = currentText(c)
                    val res = r.speak(
                        text = text, play = false, cushionMs = 140,
                        equalizeLoudness = false,
                        saveWavTo = if (ui.saveWav.isChecked) wavPathFor(c, text) else null,
                    )
                    results += res.toJson()
                    append(format(c, res))
                } catch (e: Throwable) {
                    append("${c.id}: FAILED — ${e.message}")
                }
            }
            runner?.release()
            busy = false
            runOnUiThread { setControlsEnabled(true) }
        }
    }

    private fun format(c: Candidate, r: TtsRunner.RunResult): String = buildString {
        append("${c.label}  [${c.id}]\n")
        append("  load %d ms · first chunk %d ms · total %d ms\n".format(r.loadMs, r.firstChunkMs, r.totalSynthMs))
        append("  audio %.2f s · RTF %.3f · %d chunks @ %d Hz\n".format(r.audioSeconds, r.rtf, r.chunkCount, r.sampleRate))
        if (r.firstAudibleMs >= 0) {
            append("  first audible %d ms · underruns %d\n".format(r.firstAudibleMs, r.underruns))
        }
        append("  peak PSS %.0f MiB · gain %+.1f dB\n".format(r.peakPssKb / 1024.0, r.appliedGainDb))
        append("  gate ${r.gate.short()}")
        if (r.droppedChunks > 0) append("\n  ⚠ ${r.droppedChunks} chunk(s) dropped before playback")
        if (r.chunkGateFailures.isNotEmpty()) append("\n  ⚠ ${r.chunkGateFailures.joinToString("; ")}")
        if (r.gate.warnings.isNotEmpty()) append("\n  · ${r.gate.warnings.joinToString("; ")}")
    }

    /**
     * Unattended sweep. Keeps the screen on so the run is not throttled or
     * suspended halfway, and reports progress through both the UI and logcat so
     * the host can watch without pulling files.
     */
    private fun startAutoRun(options: AutoRun.Options) {
        window.addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setControlsEnabled(false)
        ui.statusBanner.text = "AUTORUN ${options.tag} — do not touch"
        busy = true
        thread {
            val pm = getSystemService(POWER_SERVICE) as? android.os.PowerManager
            try {
                AutoRun.execute(
                    options = options,
                    candidates = candidates,
                    bundleRoot = bundleRoot,
                    resultsDir = File(getExternalFilesDir(null), "results"),
                    runner = runner!!,
                    powerManager = pm,
                ) { line ->
                    android.util.Log.i("ttsbench", line)
                    append(line)
                }
                append("AUTORUN COMPLETE")
            } catch (e: Throwable) {
                android.util.Log.e("ttsbench", "autorun failed", e)
                append("AUTORUN FAILED: ${e::class.java.simpleName}: ${e.message}")
            } finally {
                busy = false
                runOnUiThread { ui.statusBanner.text = "autorun finished" }
            }
        }
    }

    /** Stable per-(candidate, text) filename so repeats overwrite rather than pile up. */
    private fun wavPathFor(c: Candidate, text: String): File {
        val key = text.hashCode().toUInt().toString(16)
        return File(File(getExternalFilesDir(null), "results/wav"), "${c.id}__$key.wav")
    }

    private fun export() {
        val dir = File(getExternalFilesDir(null), "results")
        dir.mkdirs()
        val stamp = SimpleDateFormat("yyyyMMdd'T'HHmmss'Z'", Locale.US).apply {
            timeZone = TimeZone.getTimeZone("UTC")
        }.format(System.currentTimeMillis())
        val payload = JSONObject().apply {
            put("schema_version", 1)
            put("device", JSONObject().apply {
                put("model", Build.MODEL)
                put("device", Build.DEVICE)
                put("android_release", Build.VERSION.RELEASE)
                put("sdk_int", Build.VERSION.SDK_INT)
                put("abi", Build.SUPPORTED_ABIS.firstOrNull() ?: "")
                put("hardware", Build.HARDWARE)
            })
            put("bundle_root", bundleRoot.absolutePath)
            put("startup_cushion_ms", 140)
            put("runs", JSONArray(results))
        }
        val out = File(dir, "tts_bench_$stamp.json")
        out.writeText(payload.toString(2))
        append("exported ${results.size} run(s)\n  ${out.absolutePath}")
    }
}
