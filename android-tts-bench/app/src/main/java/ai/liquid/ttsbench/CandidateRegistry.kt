package ai.liquid.ttsbench

import org.json.JSONObject
import java.io.File

/**
 * Candidates are described by `candidates.json` inside the pushed model bundle,
 * not compiled in.
 *
 * That is deliberate: adding, removing, or re-tuning a candidate (a different
 * step count, a different speaker id, a new voice) should not require
 * rebuilding and reinstalling the APK while someone is sitting with the phone
 * doing listening comparisons.
 */
data class Candidate(
    val id: String,
    val label: String,
    val language: String,
    val engine: String,
    val numThreads: Int,
    val sid: Int,
    val speed: Float,
    val numSteps: Int,
    val extra: Map<String, String>,
    val files: Map<String, String>,
    val referenceWav: String?,
    val referenceText: String?,
    val notes: String,
    val known: String?,
) {
    fun path(root: File, key: String): String {
        val rel = files[key] ?: return ""
        return File(root, rel).absolutePath
    }

    /**
     * Missing files, if any.
     *
     * Values may hold several comma-separated paths - Kokoro takes one lexicon
     * per language that way - so each value is split before checking. Treating
     * the whole comma-joined string as one filename made both Kokoro candidates
     * fail to load with "missing files" while the paths were in fact present.
     */
    fun hasAllFiles(root: File): List<String> =
        files.values
            .flatMap { it.split(",") }
            .map { it.trim() }
            .filter { it.isNotEmpty() && !File(root, it).exists() }
}

object CandidateRegistry {

    fun load(bundleRoot: File): Pair<List<Candidate>, String?> {
        val f = File(bundleRoot, "candidates.json")
        if (!f.exists()) {
            return emptyList<Candidate>() to
                "candidates.json not found under ${bundleRoot.absolutePath}. " +
                "Push the model bundle first (see tools/push_tts_bundle.sh)."
        }
        return try {
            val root = JSONObject(f.readText())
            val arr = root.getJSONArray("candidates")
            val out = ArrayList<Candidate>(arr.length())
            for (i in 0 until arr.length()) {
                val o = arr.getJSONObject(i)
                val files = HashMap<String, String>()
                val fo = o.optJSONObject("files")
                fo?.keys()?.forEach { k -> files[k] = fo.getString(k) }
                val extra = HashMap<String, String>()
                val eo = o.optJSONObject("extra")
                eo?.keys()?.forEach { k -> extra[k] = eo.getString(k) }
                out += Candidate(
                    id = o.getString("id"),
                    label = o.optString("label", o.getString("id")),
                    language = o.optString("language", "zh"),
                    engine = o.getString("engine"),
                    numThreads = o.optInt("num_threads", 2),
                    sid = o.optInt("sid", 0),
                    speed = o.optDouble("speed", 1.0).toFloat(),
                    numSteps = o.optInt("num_steps", 0),
                    extra = extra,
                    files = files,
                    referenceWav = o.optString("reference_wav", "").ifEmpty { null },
                    referenceText = o.optString("reference_text", "").ifEmpty { null },
                    notes = o.optString("notes", ""),
                    known = o.optString("known_issue", "").ifEmpty { null },
                )
            }
            out to null
        } catch (e: Exception) {
            emptyList<Candidate>() to "candidates.json is not readable: ${e.message}"
        }
    }

    fun phrases(bundleRoot: File, language: String): List<Pair<String, String>> {
        val f = File(bundleRoot, "phrases.json")
        if (!f.exists()) return emptyList()
        return runCatching {
            val root = JSONObject(f.readText())
            val arr = root.optJSONArray(language) ?: return emptyList()
            (0 until arr.length()).map {
                val o = arr.getJSONObject(it)
                o.getString("id") to o.getString("text")
            }
        }.getOrDefault(emptyList())
    }
}
