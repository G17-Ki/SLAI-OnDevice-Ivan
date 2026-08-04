package ai.liquid.lfmdemo.tts

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
    val autoPriority: Int,
    val extra: Map<String, String>,
    val files: Map<String, String>,
    val referenceWav: String?,
    val referenceText: String?,
    val notes: String,
    val known: String?,
) {
    /** Language that the per-request GenerationConfig will actually carry. */
    val generationLanguage: String get() = extra["lang"] ?: language

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

    /** True when two registry entries can share one resident OfflineTts graph. */
    fun sharesModelWith(other: Candidate): Boolean =
        engine == other.engine &&
            numThreads == other.numThreads &&
            files == other.files
}

object CandidateRegistry {

    private val canonicalLanguageCode = Regex("^[a-z]{2,3}$")
    private val supertonicRequiredFiles = setOf(
        "duration_predictor",
        "text_encoder",
        "vector_estimator",
        "vocoder",
        "tts_json",
        "unicode_indexer",
        "voice_style",
    )

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
            require(arr.length() > 0) { "TTS candidates array is empty" }
            val out = ArrayList<Candidate>(arr.length())
            val ids = HashSet<String>()
            for (i in 0 until arr.length()) {
                val o = arr.getJSONObject(i)
                val files = HashMap<String, String>()
                val fo = o.optJSONObject("files")
                fo?.keys()?.forEach { k -> files[k] = fo.getString(k) }
                val extra = HashMap<String, String>()
                val eo = o.optJSONObject("extra")
                eo?.keys()?.forEach { k -> extra[k] = eo.getString(k) }
                val id = o.getString("id")
                val language = o.getString("language")
                val engine = o.getString("engine")
                require(id.isNotBlank() && id == id.trim()) {
                    "TTS candidate id must be non-empty and canonical"
                }
                require(ids.add(id)) { "Duplicate TTS candidate id: $id" }
                require(canonicalLanguageCode.matches(language)) {
                    "TTS candidate $id has non-canonical language code: $language"
                }
                require(files.isNotEmpty()) { "TTS candidate $id has no model files" }
                require(files.all { (key, value) -> key.isNotBlank() && value.isNotBlank() }) {
                    "TTS candidate $id has an empty model-file key or path"
                }
                require(extra["lang"] == null || extra["lang"] == language) {
                    "TTS candidate $id language '$language' does not match " +
                        "extra.lang '${extra["lang"]}'"
                }
                if (engine == "supertonic") {
                    val missing = supertonicRequiredFiles.filter { files[it].isNullOrBlank() }
                    require(missing.isEmpty()) {
                        "Supertonic candidate $id is missing required files: " +
                            missing.joinToString()
                    }
                    require(o.optInt("num_steps", 0) > 0) {
                        "Supertonic candidate $id must use a positive num_steps"
                    }
                    require(extra["lang"] != null) {
                        "Supertonic candidate $id must declare extra.lang"
                    }
                }
                out += Candidate(
                    id = id,
                    label = o.optString("label", id),
                    language = language,
                    engine = engine,
                    numThreads = o.optInt("num_threads", 2),
                    sid = o.optInt("sid", 0),
                    speed = o.optDouble("speed", 1.0).toFloat(),
                    numSteps = o.optInt("num_steps", 0),
                    autoPriority = o.optInt(
                        "auto_priority",
                        when (o.getString("id")) {
                            "matcha_zh_baker",
                            "supertonic3_en_s5",
                            "supertonic3_s5",
                            "supertonic3_ja_s5",
                            "supertonic3_ko_s5",
                            // ES and FR reuse the identical Supertonic graph set,
                            // so sharesModelWith() lets them switch languages with
                            // no second model load. See stage_device_bundle.py.
                            "supertonic3_es_s5",
                            "supertonic3_fr_s5" -> 10
                            else -> 0
                        },
                    ),
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
