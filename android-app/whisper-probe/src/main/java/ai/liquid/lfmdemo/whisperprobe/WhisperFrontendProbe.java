package ai.liquid.lfmdemo.whisperprobe;

import android.content.Context;
import android.content.res.AssetManager;
import android.os.SystemClock;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/** Runs the profile-pinned bundled-WAV frontend comparator before QNN execution. */
final class WhisperFrontendProbe {
    private static final String V1_ASSET_ROOT = "whisper_frontend/v1/";
    private static final String V1_GATE_ID = "whisper_qai_hf_bundled_wav_frontend_v1";
    private static final String V1_MANIFEST_SHA256 =
            "aea54e5c751f9a2b489c709e5ca149db7c39cbdabea3ed4675483c00e236693f";
    private static final String V2_ASSET_ROOT = "whisper_frontend/v2/";
    private static final String V2_GATE_ID =
            "whisper_qai_hf_bundled_wav_frontend_multilang_v2";
    private static final String V2_MANIFEST_SHA256 =
            "acb5c46f3bbbe3c11620bd662242157b30695500c1a0e8b6e7510acdc3157736";

    private static final int FEATURE_ELEMENTS = 80 * 3000;
    private static final int FEATURE_BYTES = FEATURE_ELEMENTS * 2;
    private static final int MAX_UINT16_DELTA = 1;
    private static final int MAX_DIFFERING_VALUES = 240;
    private static final double MAX_DIFFERING_FRACTION = 0.001;

    private WhisperFrontendProbe() {}

    static Result run(Context context) throws IOException, JSONException {
        return run(
                context,
                BuildConfig.WHISPER_SEMANTIC_LANGUAGE,
                BuildConfig.WHISPER_SEMANTIC_DECODE_ENABLED
        );
    }

    static Result run(
            Context context,
            String selectedLanguage,
            boolean semanticDecodeEnabled
    ) throws IOException, JSONException {
        if (!BuildConfig.WHISPER_FRONTEND_FIXTURES_EMBEDDED) {
            throw new IllegalStateException(
                    "This APK was built without WHISPER_FRONTEND_ASSET_DIR"
            );
        }
        if (ByteOrder.nativeOrder() != ByteOrder.LITTLE_ENDIAN) {
            throw new IllegalStateException(
                    "frontend output contract requires little-endian arm64"
            );
        }

        FrontendProfile profile = activeProfile();
        String normalizedLanguage = normalizeLanguage(selectedLanguage);
        if (!profile.fixtures.containsKey(normalizedLanguage)) {
            throw new IllegalArgumentException(
                    "no bundled frontend fixture for language: " + selectedLanguage
            );
        }
        Set<String> executionLanguages = executionLanguages(
                profile, normalizedLanguage, semanticDecodeEnabled
        );

        AssetManager assets = context.getAssets();
        String manifestAsset = profile.assetRoot + "manifest.json";
        byte[] manifestBytes = readAsset(assets, manifestAsset);
        requireIdentity(manifestBytes, new AssetIdentity(
                manifestAsset, manifestBytes.length, profile.manifestSha256
        ));
        JSONObject manifest = new JSONObject(
                new String(manifestBytes, StandardCharsets.UTF_8)
        );
        verifyManifest(context, manifest, profile);

        byte[] hann = readChecked(assets, profile.hann);
        byte[] mel = readChecked(assets, profile.mel);
        Map<String, FixtureAssets> authenticatedFixtures = new LinkedHashMap<>();
        for (Map.Entry<String, Fixture> entry : profile.fixtures.entrySet()) {
            Fixture fixture = entry.getValue();
            authenticatedFixtures.put(
                    entry.getKey(),
                    new FixtureAssets(
                            readChecked(assets, fixture.wav),
                            readChecked(assets, fixture.golden)
                    )
            );
        }

        long suiteStart = SystemClock.elapsedRealtimeNanos();
        Map<String, FixtureResult> fixtureResults = new LinkedHashMap<>();
        boolean boundedGatePassed = true;
        for (String language : executionLanguages) {
            FixtureResult result = runFixture(
                    profile.fixtures.get(language),
                    authenticatedFixtures.get(language),
                    hann,
                    mel
            );
            fixtureResults.put(language, result);
            boundedGatePassed &= result.boundedGatePassed;
        }
        long suiteEnd = SystemClock.elapsedRealtimeNanos();

        JSONObject fixtureReports = new JSONObject();
        JSONObject fixtureTimings = new JSONObject();
        double aggregateComputeMilliseconds = 0.0;
        for (Map.Entry<String, FixtureResult> entry : fixtureResults.entrySet()) {
            fixtureReports.put(entry.getKey(), entry.getValue().report);
            fixtureTimings.put(entry.getKey(), entry.getValue().computeMilliseconds);
            aggregateComputeMilliseconds += entry.getValue().computeMilliseconds;
        }
        fixtureTimings.put("aggregate", roundMilliseconds(aggregateComputeMilliseconds));

        JSONObject report = new JSONObject();
        report.put("schema_version", 1);
        report.put("gate_id", profile.gateId);
        report.put("manifest_sha256", profile.manifestSha256);
        report.put("frontend_profile", profile.id);
        report.put("available_fixture_languages", jsonArray(profile.fixtures.keySet()));
        report.put("executed_fixture_languages", jsonArray(executionLanguages));
        report.put(
                "fixture_execution_mode",
                executionLanguages.size() == profile.fixtures.size()
                        ? "all_profile_fixtures"
                        : "english_plus_semantic_language"
        );
        report.put("authenticated_assets", profile.fixtures.size() * 2 + 2);
        report.put("implementation", "jni_persistent_kissfft_400_real");
        report.put("feature_dtype", "uint16");
        report.put("feature_shape", new JSONArray().put(1).put(80).put(3000));
        report.put("feature_layout", "mel_major_frame_minor");
        report.put("fixtures", fixtureReports);
        report.put("frontend_compute_ms", fixtureTimings);
        report.put("frontend_verification_total_ms", millis(suiteEnd - suiteStart));
        report.put("verification_policy", new JSONObject()
                .put("max_uint16_delta", MAX_UINT16_DELTA)
                .put("max_differing_values", MAX_DIFFERING_VALUES)
                .put("max_differing_fraction", MAX_DIFFERING_FRACTION)
                .put("target_acceptance_promoted", false));
        report.put("bounded_gate_passed", boundedGatePassed);
        report.put("real_audio_frontend", false);
        report.put("real_audio_transcription", false);
        report.put("microphone_input", false);
        report.put("status", boundedGatePassed
                ? "bundled_wav_frontend_bound_pass"
                : "bundled_wav_frontend_failed_closed");
        report.put(
                "claim_boundary",
                "This verifies only bundled WAV preprocessing on the executing ABI. "
                        + "It does not enable microphone input or claim a semantic transcription."
        );
        File nativeLibrary = new File(
                context.getApplicationInfo().nativeLibraryDir,
                "libwhisper_frontend.so"
        );
        if (!nativeLibrary.isFile()) {
            throw new IOException("missing packaged frontend JNI library: " + nativeLibrary);
        }
        report.put("native_library_sha256", sha256(nativeLibrary));

        Map<String, byte[]> outputs = new LinkedHashMap<>();
        FixtureResult english = fixtureResults.get("en");
        if (english == null) {
            throw new IllegalStateException("English frontend fixture must always execute");
        }
        outputs.put("en", boundedGatePassed ? english.output : null);
        for (Map.Entry<String, FixtureResult> entry : fixtureResults.entrySet()) {
            if (!"en".equals(entry.getKey())) {
                outputs.put(
                        entry.getKey(),
                        boundedGatePassed ? entry.getValue().output : null
                );
            }
        }
        return new Result(report, outputs, boundedGatePassed);
    }

    private static Set<String> executionLanguages(
            FrontendProfile profile,
            String selectedLanguage,
            boolean semanticDecodeEnabled
    ) {
        LinkedHashSet<String> languages = new LinkedHashSet<>();
        // V1 behavior is immutable: all historical EN/JA fixtures still run.
        if ("v1".equals(profile.id) || !semanticDecodeEnabled) {
            languages.addAll(profile.fixtures.keySet());
        } else {
            // V2 semantic candidates avoid three unnecessary frontend computes.
            languages.add("en");
            languages.add(selectedLanguage);
        }
        return languages;
    }

    private static FixtureResult runFixture(
            Fixture fixture,
            FixtureAssets assets,
            byte[] hann,
            byte[] mel
    ) throws IOException, JSONException {
        long computeStart = SystemClock.elapsedRealtimeNanos();
        byte[] actual = WhisperFrontendNative.compute(
                assets.wav, fixture.pcmFrames, hann, mel
        );
        long computeEnd = SystemClock.elapsedRealtimeNanos();
        if (actual == null || actual.length != FEATURE_BYTES) {
            throw new IOException(
                    fixture.language + " native frontend emitted "
                            + (actual == null ? "null" : actual.length + " bytes")
            );
        }
        long differing = 0;
        int maxDelta = 0;
        for (int offset = 0; offset < FEATURE_BYTES; offset += 2) {
            int actualValue = (actual[offset] & 0xff)
                    | ((actual[offset + 1] & 0xff) << 8);
            int goldenValue = (assets.golden[offset] & 0xff)
                    | ((assets.golden[offset + 1] & 0xff) << 8);
            int delta = Math.abs(actualValue - goldenValue);
            if (delta != 0) {
                differing++;
                maxDelta = Math.max(maxDelta, delta);
            }
        }
        double differingFraction = differing / (double) FEATURE_ELEMENTS;
        boolean boundedGatePassed = maxDelta <= MAX_UINT16_DELTA
                && differing <= MAX_DIFFERING_VALUES
                && differingFraction <= MAX_DIFFERING_FRACTION;
        String actualSha256 = sha256(actual);
        JSONObject report = new JSONObject();
        report.put("wav_sha256", fixture.wav.sha256);
        report.put("wav_pcm_frames", fixture.pcmFrames);
        report.put("golden_tensor_sha256", fixture.golden.sha256);
        report.put("produced_tensor_sha256", actualSha256);
        report.put("exact_sha256_match", fixture.golden.sha256.equals(actualSha256));
        report.put("differing_uint16_values", differing);
        report.put("differing_fraction", differingFraction);
        report.put("max_uint16_delta", maxDelta);
        report.put("frontend_compute_ms", millis(computeEnd - computeStart));
        report.put("bounded_gate_passed", boundedGatePassed);
        return new FixtureResult(
                report,
                actual,
                boundedGatePassed,
                millis(computeEnd - computeStart)
        );
    }

    private static void verifyManifest(
            Context context,
            JSONObject manifest,
            FrontendProfile profile
    ) throws JSONException {
        if (manifest.getInt("schema_version") != 1
                || !profile.gateId.equals(manifest.getString("gate_id"))) {
            throw new IllegalStateException("unexpected frontend manifest identity");
        }
        if ("v2".equals(profile.id)) {
            if (!profile.id.equals(manifest.getString("frontend_profile"))
                    || !jsonArrayEquals(
                            manifest.getJSONArray("fixture_languages"),
                            profile.fixtures.keySet()
                    )) {
                throw new IllegalStateException("frontend-v2 fixture profile mismatch");
            }
        }
        JSONObject target = manifest.getJSONObject("target");
        if (!context.getPackageName().equals(target.getString("package"))
                || !"arm64-v8a".equals(target.getString("abi"))) {
            throw new IllegalStateException("frontend manifest target mismatch");
        }
        JSONObject capabilities = manifest.getJSONObject("capabilities");
        if (!capabilities.getBoolean("bundled_wav_frontend_verification")
                || capabilities.getBoolean("microphone_input")
                || capabilities.getBoolean("real_audio_transcription")) {
            throw new IllegalStateException("frontend manifest crosses claim boundary");
        }
        JSONObject policy = manifest.getJSONObject("verification_policy");
        if (!policy.getBoolean("strict_asset_sha256")
                || !policy.getBoolean("report_exact_output_sha256")
                || policy.getInt("max_uint16_delta") != MAX_UINT16_DELTA
                || Double.compare(
                        policy.getDouble("max_differing_fraction"),
                        MAX_DIFFERING_FRACTION
                ) != 0
                || policy.getBoolean("target_acceptance_promoted")) {
            throw new IllegalStateException("frontend verification policy mismatch");
        }
        if (policy.has("max_differing_uint16_values")
                && policy.getInt("max_differing_uint16_values") != MAX_DIFFERING_VALUES) {
            throw new IllegalStateException("frontend differing-value policy mismatch");
        }
        if ("v2".equals(profile.id)
                && !policy.has("max_differing_uint16_values")) {
            throw new IllegalStateException("frontend-v2 differing-value policy is missing");
        }

        Map<String, JSONObject> records = new LinkedHashMap<>();
        JSONArray manifestAssets = manifest.getJSONArray("assets");
        for (int index = 0; index < manifestAssets.length(); index++) {
            JSONObject record = manifestAssets.getJSONObject(index);
            String path = record.getString("path");
            if (records.put(path, record) != null) {
                throw new IllegalStateException("duplicate frontend asset record: " + path);
            }
        }
        verifyManifestAsset(records, profile.hann);
        verifyManifestAsset(records, profile.mel);
        for (Fixture fixture : profile.fixtures.values()) {
            verifyManifestAsset(records, fixture.wav);
            verifyManifestAsset(records, fixture.golden);
        }
        if (!records.isEmpty()) {
            throw new IllegalStateException("frontend manifest contains unexpected assets");
        }
    }

    private static void verifyManifestAsset(
            Map<String, JSONObject> records,
            AssetIdentity identity
    ) throws JSONException {
        JSONObject record = records.remove(identity.path);
        if (record == null
                || record.getLong("bytes") != identity.bytes
                || !identity.sha256.equals(record.getString("sha256"))) {
            throw new IllegalStateException(
                    "frontend manifest asset mismatch: " + identity.path
            );
        }
    }

    private static FrontendProfile activeProfile() {
        String id = BuildConfig.WHISPER_FRONTEND_PROFILE;
        if ("v1".equals(id)) {
            requireBuildContract(id, V1_ASSET_ROOT, V1_GATE_ID, V1_MANIFEST_SHA256);
            LinkedHashMap<String, Fixture> fixtures = new LinkedHashMap<>();
            addFixture(fixtures, fixtureEn(V1_ASSET_ROOT));
            addFixture(fixtures, fixtureJa(V1_ASSET_ROOT));
            return profile(id, V1_ASSET_ROOT, V1_GATE_ID, V1_MANIFEST_SHA256, fixtures);
        }
        if ("v2".equals(id)) {
            requireBuildContract(id, V2_ASSET_ROOT, V2_GATE_ID, V2_MANIFEST_SHA256);
            LinkedHashMap<String, Fixture> fixtures = new LinkedHashMap<>();
            addFixture(fixtures, fixtureEn(V2_ASSET_ROOT));
            addFixture(fixtures, fixtureZh(V2_ASSET_ROOT));
            addFixture(fixtures, fixtureRu(V2_ASSET_ROOT));
            addFixture(fixtures, fixtureKo(V2_ASSET_ROOT));
            addFixture(fixtures, fixtureJa(V2_ASSET_ROOT));
            return profile(id, V2_ASSET_ROOT, V2_GATE_ID, V2_MANIFEST_SHA256, fixtures);
        }
        throw new IllegalStateException("unsupported frontend profile: " + id);
    }

    private static FrontendProfile profile(
            String id,
            String assetRoot,
            String gateId,
            String manifestSha256,
            LinkedHashMap<String, Fixture> fixtures
    ) {
        return new FrontendProfile(
                id,
                assetRoot,
                gateId,
                manifestSha256,
                new AssetIdentity(
                        assetRoot + "tables/hann.f32le.bin",
                        1600,
                        "c692d19f0b2bd745f0dfc967261c3e3d21a2473e485a677d219e9b4a73e0218e"
                ),
                new AssetIdentity(
                        assetRoot + "tables/mel_filters.f32le.bin",
                        64320,
                        "d18b48f33345d958ea71098bb990b5e32cebc78d9a2cf65ca0d7302f34ebe8a9"
                ),
                fixtures
        );
    }

    private static void requireBuildContract(
            String id,
            String assetRoot,
            String gateId,
            String manifestSha256
    ) {
        String configuredRoot = BuildConfig.WHISPER_FRONTEND_ASSET_ROOT + "/";
        if (!assetRoot.equals(configuredRoot)
                || !gateId.equals(BuildConfig.WHISPER_FRONTEND_GATE_ID)
                || !manifestSha256.equals(
                        BuildConfig.WHISPER_FRONTEND_MANIFEST_SHA256
                )) {
            throw new IllegalStateException(
                    "BuildConfig frontend contract mismatch for profile " + id
            );
        }
    }

    private static Fixture fixtureEn(String root) {
        return fixture(
                root,
                "en",
                105682,
                "b5fa81a74e608624b52fb3e4cf895d6982a455022597a0da0ad3dc52866f8b69",
                52819,
                "29c478d40bb98b7782280dcfbc98662cee514df7e62c180c5ba849db24a2f6a1"
        );
    }

    private static Fixture fixtureZh(String root) {
        return fixture(
                root,
                "zh",
                83478,
                "bfc02160e9b86c487dc122a7e501e71d411ada2d7a5b94ad59b1bf0f8e149c63",
                41717,
                "6d18da5e1ec42bbd8161881a598e835c365138377b3d42364ff360e4849d4263"
        );
    }

    private static Fixture fixtureRu(String root) {
        return fixture(
                root,
                "ru",
                48674,
                "3545b8a20e3f320174a4e4dee57ff77268f30f79526eef91c86c778e3c180683",
                24315,
                "0e0c25a24157edabe3c1d193e60e8b59886aa9b2ba9b1581fb5c90c2eaf3bbc4"
        );
    }

    private static Fixture fixtureKo(String root) {
        return fixture(
                root,
                "ko",
                66908,
                "0ce8353dc9faa844cc8214c613f182a127338b71b139955ccb903ca806a7b3de",
                33432,
                "808b38d3c4198c15566116562eb7ad61fbafc01912a747ae439047fb9d7eb1c5"
        );
    }

    private static Fixture fixtureJa(String root) {
        return fixture(
                root,
                "ja",
                79464,
                "0da6ee3b9514a96597bf9225a1bcdf7cae86608e095e157490899a443948b761",
                39710,
                "8cf37ba9a60f5cd91453dae2d721e9fd7fec69919582223bf02a3f76f8421a80"
        );
    }

    private static Fixture fixture(
            String root,
            String language,
            int wavBytes,
            String wavSha256,
            int pcmFrames,
            String goldenSha256
    ) {
        return new Fixture(
                language,
                new AssetIdentity(
                        root + "fixtures/question_" + language + ".wav",
                        wavBytes,
                        wavSha256
                ),
                pcmFrames,
                new AssetIdentity(
                        root + "goldens/" + language + ".features.u16le.bin",
                        FEATURE_BYTES,
                        goldenSha256
                )
        );
    }

    private static void addFixture(
            LinkedHashMap<String, Fixture> fixtures,
            Fixture fixture
    ) {
        if (fixtures.put(fixture.language, fixture) != null) {
            throw new AssertionError("duplicate frontend fixture: " + fixture.language);
        }
    }

    private static String normalizeLanguage(String language) {
        if (language == null) {
            throw new IllegalArgumentException("frontend language is required");
        }
        return language.toLowerCase(Locale.ROOT);
    }

    private static JSONArray jsonArray(Iterable<String> values) {
        JSONArray result = new JSONArray();
        for (String value : values) {
            result.put(value);
        }
        return result;
    }

    private static boolean jsonArrayEquals(JSONArray actual, Iterable<String> expected)
            throws JSONException {
        int index = 0;
        for (String value : expected) {
            if (index >= actual.length() || !value.equals(actual.getString(index))) {
                return false;
            }
            index++;
        }
        return index == actual.length();
    }

    private static byte[] readChecked(AssetManager assets, AssetIdentity identity)
            throws IOException {
        byte[] value = readAsset(assets, identity.path);
        requireIdentity(value, identity);
        return value;
    }

    private static void requireIdentity(byte[] value, AssetIdentity identity)
            throws IOException {
        if (value.length != identity.bytes) {
            throw new IOException("frontend asset byte count mismatch: " + identity.path);
        }
        String actual = sha256(value);
        if (!identity.sha256.equals(actual)) {
            throw new IOException("frontend asset SHA-256 mismatch: " + identity.path);
        }
    }

    private static byte[] readAsset(AssetManager assets, String path) throws IOException {
        try (InputStream input = assets.open(path, AssetManager.ACCESS_STREAMING);
             ByteArrayOutputStream output = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[1024 * 1024];
            int count;
            while ((count = input.read(buffer)) != -1) {
                output.write(buffer, 0, count);
            }
            return output.toByteArray();
        }
    }

    private static String sha256(byte[] value) {
        MessageDigest digest = digest();
        digest.update(value);
        return hex(digest.digest());
    }

    private static String sha256(File path) throws IOException {
        MessageDigest digest = digest();
        try (FileInputStream input = new FileInputStream(path)) {
            byte[] buffer = new byte[1024 * 1024];
            int count;
            while ((count = input.read(buffer)) != -1) {
                digest.update(buffer, 0, count);
            }
        }
        return hex(digest.digest());
    }

    private static MessageDigest digest() {
        try {
            return MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException error) {
            throw new AssertionError(error);
        }
    }

    private static String hex(byte[] value) {
        StringBuilder encoded = new StringBuilder(value.length * 2);
        for (byte element : value) {
            encoded.append(String.format(Locale.ROOT, "%02x", element & 0xff));
        }
        return encoded.toString();
    }

    private static double millis(long nanoseconds) {
        return Math.round((nanoseconds / 1_000_000.0) * 1000.0) / 1000.0;
    }

    private static double roundMilliseconds(double milliseconds) {
        return Math.round(milliseconds * 1000.0) / 1000.0;
    }

    static final class Result {
        final JSONObject report;
        final Map<String, byte[]> featuresByLanguage;
        final byte[] englishFeatures;
        final byte[] japaneseFeatures;
        final boolean boundedGatePassed;

        Result(
                JSONObject report,
                Map<String, byte[]> featuresByLanguage,
                boolean boundedGatePassed
        ) {
            this.report = report;
            this.featuresByLanguage = Collections.unmodifiableMap(
                    new LinkedHashMap<>(featuresByLanguage)
            );
            this.englishFeatures = this.featuresByLanguage.get("en");
            this.japaneseFeatures = this.featuresByLanguage.get("ja");
            this.boundedGatePassed = boundedGatePassed;
        }

        byte[] featuresForLanguage(String languageCode) {
            String normalized = normalizeLanguage(languageCode);
            if (!featuresByLanguage.containsKey(normalized)) {
                throw new IllegalArgumentException(
                        "frontend fixture was not executed for language: " + languageCode
                );
            }
            return featuresByLanguage.get(normalized);
        }
    }

    private static final class FrontendProfile {
        final String id;
        final String assetRoot;
        final String gateId;
        final String manifestSha256;
        final AssetIdentity hann;
        final AssetIdentity mel;
        final Map<String, Fixture> fixtures;

        FrontendProfile(
                String id,
                String assetRoot,
                String gateId,
                String manifestSha256,
                AssetIdentity hann,
                AssetIdentity mel,
                LinkedHashMap<String, Fixture> fixtures
        ) {
            this.id = id;
            this.assetRoot = assetRoot;
            this.gateId = gateId;
            this.manifestSha256 = manifestSha256;
            this.hann = hann;
            this.mel = mel;
            this.fixtures = Collections.unmodifiableMap(
                    new LinkedHashMap<>(fixtures)
            );
        }
    }

    private static final class AssetIdentity {
        final String path;
        final int bytes;
        final String sha256;

        AssetIdentity(String path, int bytes, String sha256) {
            this.path = path;
            this.bytes = bytes;
            this.sha256 = sha256;
        }
    }

    private static final class Fixture {
        final String language;
        final AssetIdentity wav;
        final int pcmFrames;
        final AssetIdentity golden;

        Fixture(String language, AssetIdentity wav, int pcmFrames, AssetIdentity golden) {
            this.language = language;
            this.wav = wav;
            this.pcmFrames = pcmFrames;
            this.golden = golden;
        }
    }

    private static final class FixtureAssets {
        final byte[] wav;
        final byte[] golden;

        FixtureAssets(byte[] wav, byte[] golden) {
            this.wav = wav;
            this.golden = golden;
        }
    }

    private static final class FixtureResult {
        final JSONObject report;
        final byte[] output;
        final boolean boundedGatePassed;
        final double computeMilliseconds;

        FixtureResult(
                JSONObject report,
                byte[] output,
                boolean boundedGatePassed,
                double computeMilliseconds
        ) {
            this.report = report;
            this.output = output;
            this.boundedGatePassed = boundedGatePassed;
            this.computeMilliseconds = computeMilliseconds;
        }
    }
}
