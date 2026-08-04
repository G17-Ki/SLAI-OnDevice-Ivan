package ai.liquid.lfmdemo.supertonicchaindiagprobe;

import android.content.Context;
import android.content.res.AssetManager;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

/** Installs and verifies the immutable four-stage probe payload in app storage. */
public final class SupertonicChainDiagnosticPayload {
    public static final String MANIFEST_ASSET =
            "supertonic_chain_diagnostic_probe/payload_manifest.json";
    public static final String EXPECTED_GATE_ID =
            "supertonic3_full_chain_stage_diagnostic_raw_onnx_ort_qnn_sm8850_probe_v2";

    private final File root;
    private final File nativeLibraryDir;
    private final JSONObject manifest;

    private SupertonicChainDiagnosticPayload(
            File root,
            File nativeLibraryDir,
            JSONObject manifest
    ) {
        this.root = root;
        this.nativeLibraryDir = nativeLibraryDir;
        this.manifest = manifest;
    }

    public static SupertonicChainDiagnosticPayload installAndVerify(Context context)
            throws IOException, JSONException {
        if (!BuildConfig.SUPERTONIC_CHAIN_DIAGNOSTIC_PAYLOAD_EMBEDDED) {
            throw new IllegalStateException(
                    "APK was built without SUPERTONIC_CHAIN_PROBE_PAYLOAD_DIR"
            );
        }
        byte[] manifestBytes = readAsset(context.getAssets(), MANIFEST_ASSET);
        JSONObject manifest = new JSONObject(
                new String(manifestBytes, StandardCharsets.UTF_8)
        );
        verifyManifestIdentity(context, manifest);

        File root = new File(
                context.getFilesDir(), "supertonic_chain_diagnostic_probe/payload-v2"
        );
        if (!root.isDirectory() && !root.mkdirs()) {
            throw new IOException("cannot create app-private Supertonic chain diagnostic root");
        }
        writeIfChanged(
                checkedDestination(root, "payload_manifest.json"), manifestBytes
        );

        JSONArray assets = manifest.getJSONArray("assets");
        for (int index = 0; index < assets.length(); index++) {
            JSONObject record = assets.getJSONObject(index);
            String assetPath = record.getString("path");
            String prefix = "supertonic_chain_diagnostic_probe/";
            if (!assetPath.startsWith(prefix)) {
                throw new IOException(
                        "asset is outside supertonic_chain_diagnostic_probe/: " + assetPath
                );
            }
            File destination = checkedDestination(
                    root, assetPath.substring(prefix.length())
            );
            installAsset(
                    context.getAssets(),
                    assetPath,
                    destination,
                    record.getLong("bytes"),
                    record.getString("sha256")
            );
        }

        File nativeLibraryDir = new File(
                context.getApplicationInfo().nativeLibraryDir
        );
        JSONArray libraries = manifest.getJSONArray("native_libraries");
        for (int index = 0; index < libraries.length(); index++) {
            JSONObject record = libraries.getJSONObject(index);
            String packagedPath = record.getString("path");
            String prefix = "arm64-v8a/";
            if (!packagedPath.startsWith(prefix)) {
                throw new IOException(
                        "unexpected native-library path: " + packagedPath
                );
            }
            verifyFile(
                    new File(
                            nativeLibraryDir,
                            packagedPath.substring(prefix.length())
                    ),
                    record.getLong("bytes"),
                    record.getString("sha256")
            );
        }

        SupertonicChainDiagnosticPayload payload = new SupertonicChainDiagnosticPayload(
                root, nativeLibraryDir, manifest
        );
        payload.requireCoreFiles();
        return payload;
    }

    private static void verifyManifestIdentity(
            Context context,
            JSONObject manifest
    ) throws JSONException {
        if (manifest.getInt("schema_version") != 1
                || !EXPECTED_GATE_ID.equals(manifest.getString("gate_id"))) {
            throw new IllegalStateException("unexpected Supertonic chain diagnostic identity");
        }
        if (!context.getPackageName().equals(
                manifest.getJSONObject("target").getString("package")
        )) {
            throw new IllegalStateException("Supertonic chain diagnostic package mismatch");
        }
        JSONObject runtime = manifest.getJSONObject("runtime");
        if (!"1.25.0".equals(runtime.getString("onnx_runtime"))
                || !"2.42.0.251225135753_193295".equals(
                        runtime.getString("qairt")
                )
                || !runtime.getBoolean("disable_cpu_ep_fallback")
                || runtime.getInt("inter_op_threads") != 1
                || runtime.getInt("intra_op_threads") != 1) {
            throw new IllegalStateException("Supertonic chain diagnostic runtime mismatch");
        }
        JSONObject policy = manifest.getJSONObject("verification_policy");
        if (!policy.getBoolean("require_all_four_sessions")
                || !policy.getBoolean("require_chained_outputs")
                || !policy.getBoolean("require_finite_tensors")
                || !policy.getBoolean("require_exact_shapes")
                || !policy.getBoolean("require_zero_masked_padding")
                || !policy.getBoolean("require_non_silent_waveform")
                || !policy.getBoolean("host_reference_comparisons_recorded")
                || policy.getBoolean("host_reference_tolerances_calibrated_on_phone")
                || policy.getBoolean("host_reference_comparisons_promoted")
                || policy.getBoolean("audio_quality_promoted")) {
            throw new IllegalStateException(
                    "Supertonic chain diagnostic verification policy crosses claim boundary"
            );
        }
        JSONObject capabilities = manifest.getJSONObject("capabilities");
        if (!capabilities.getBoolean("duration_execution")
                || !capabilities.getBoolean("text_encoder_execution")
                || !capabilities.getBoolean("four_step_vector_execution")
                || !capabilities.getBoolean("vocoder_execution")
                || !capabilities.getBoolean("chained_pcm_tensor")
                || capabilities.getBoolean("audio_file_output")
                || capabilities.getBoolean("audio_playback")
                || capabilities.getBoolean("audio_quality_acceptance")
                || capabilities.getBoolean("production_integration")) {
            throw new IllegalStateException(
                    "Supertonic chain diagnostic capabilities cross claim boundary"
            );
        }
        JSONObject diagnostic = manifest.getJSONObject("diagnostic_policy");
        if (!diagnostic.getBoolean("completed_stages_recorded")
                || !diagnostic.getBoolean("cpu_ep_fallback_remains_disabled")
                || !diagnostic.getBoolean("failure_phase_recorded")
                || !diagnostic.getBoolean("failure_stage_recorded")
                || diagnostic.getBoolean("promotion_allowed")) {
            throw new IllegalStateException(
                    "Supertonic chain diagnostic policy mismatch"
            );
        }
        JSONObject source = manifest.getJSONObject("source_payload_manifest");
        if (source.getLong("bytes") != 19_378L
                || !"735e61b070c7ad199a67bc18829036f4575551b5533a49e9ead5b36392f0819f"
                        .equals(source.getString("sha256"))) {
            throw new IllegalStateException(
                    "Supertonic chain diagnostic source provenance mismatch"
            );
        }
    }

    private void requireCoreFiles() throws IOException {
        for (String name : new String[] {
                "models/duration_predictor.int8.onnx",
                "models/text_encoder.int8.onnx",
                "models/vector_estimator.int8.onnx",
                "models/vocoder.int8.onnx",
                "fixtures/text_ids.i64le",
                "fixtures/text_mask.f32le",
                "fixtures/style_dp.f32le",
                "fixtures/style_ttl.f32le",
                "fixtures/latent_mask.f32le",
                "fixtures/latent_initial.f32le",
                "goldens/duration.f32le",
                "goldens/text_embedding.f32le",
                "goldens/latent_step_0.f32le",
                "goldens/latent_step_1.f32le",
                "goldens/latent_step_2.f32le",
                "goldens/latent_step_3.f32le",
                "goldens/waveform_padded.f32le"
        }) {
            if (!file(name).isFile()) {
                throw new IOException("missing installed Supertonic chain diagnostic file: " + name);
            }
        }
        for (String name : new String[] {
                "libQnnSystem.so",
                "libQnnHtp.so",
                "libQnnHtpPrepare.so",
                "libQnnHtpV81Stub.so",
                "libQnnHtpV81Skel.so"
        }) {
            if (!nativeLibrary(name).isFile()) {
                throw new IOException("missing installed QNN library: " + name);
            }
        }
    }

    public File file(String relativePath) {
        return new File(root, relativePath);
    }

    public File nativeLibrary(String name) {
        return new File(nativeLibraryDir, name);
    }

    public File root() {
        return root;
    }

    public File nativeLibraryDir() {
        return nativeLibraryDir;
    }

    public JSONObject manifest() {
        return manifest;
    }

    private static byte[] readAsset(AssetManager assets, String path)
            throws IOException {
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

    private static void installAsset(
            AssetManager assets,
            String assetPath,
            File destination,
            long expectedBytes,
            String expectedSha256
    ) throws IOException {
        if (destination.isFile()) {
            try {
                verifyFile(destination, expectedBytes, expectedSha256);
                return;
            } catch (IOException ignored) {
                // Replace only this package's manifest-owned private file.
            }
        }
        File parent = destination.getParentFile();
        if (parent == null || (!parent.isDirectory() && !parent.mkdirs())) {
            throw new IOException("cannot create payload directory for " + destination);
        }
        File temporary = new File(parent, destination.getName() + ".partial");
        if (temporary.exists() && !temporary.delete()) {
            throw new IOException("cannot clear stale partial payload: " + temporary);
        }
        try (InputStream input = assets.open(assetPath, AssetManager.ACCESS_STREAMING);
             FileOutputStream output = new FileOutputStream(temporary)) {
            byte[] buffer = new byte[1024 * 1024];
            int count;
            while ((count = input.read(buffer)) != -1) {
                output.write(buffer, 0, count);
            }
            output.getFD().sync();
        }
        verifyFile(temporary, expectedBytes, expectedSha256);
        atomicMove(temporary, destination);
    }

    private static void writeIfChanged(File destination, byte[] payload)
            throws IOException {
        if (destination.isFile()) {
            byte[] existing = Files.readAllBytes(destination.toPath());
            if (MessageDigest.isEqual(existing, payload)) {
                return;
            }
        }
        File parent = destination.getParentFile();
        if (parent == null || (!parent.isDirectory() && !parent.mkdirs())) {
            throw new IOException("cannot create manifest destination directory");
        }
        File temporary = new File(parent, destination.getName() + ".partial");
        if (temporary.exists() && !temporary.delete()) {
            throw new IOException("cannot clear stale manifest partial");
        }
        try (FileOutputStream output = new FileOutputStream(temporary)) {
            output.write(payload);
            output.getFD().sync();
        }
        atomicMove(temporary, destination);
    }

    private static void atomicMove(File source, File destination)
            throws IOException {
        try {
            Files.move(
                    source.toPath(),
                    destination.toPath(),
                    StandardCopyOption.ATOMIC_MOVE,
                    StandardCopyOption.REPLACE_EXISTING
            );
        } catch (AtomicMoveNotSupportedException error) {
            throw new IOException("atomic payload install is unsupported", error);
        }
    }

    private static File checkedDestination(File root, String relativePath)
            throws IOException {
        File destination = new File(root, relativePath);
        String rootPath = root.getCanonicalPath() + File.separator;
        if (!destination.getCanonicalPath().startsWith(rootPath)) {
            throw new IOException("payload path traversal rejected: " + relativePath);
        }
        return destination;
    }

    private static void verifyFile(
            File file,
            long expectedBytes,
            String expectedSha256
    ) throws IOException {
        if (!file.isFile() || file.length() != expectedBytes) {
            throw new IOException("file size/type mismatch: " + file);
        }
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            try (InputStream input = Files.newInputStream(file.toPath())) {
                byte[] buffer = new byte[1024 * 1024];
                int count;
                while ((count = input.read(buffer)) != -1) {
                    digest.update(buffer, 0, count);
                }
            }
            StringBuilder actual = new StringBuilder();
            for (byte value : digest.digest()) {
                actual.append(String.format("%02x", value & 0xff));
            }
            if (!expectedSha256.equals(actual.toString())) {
                throw new IOException("file SHA-256 mismatch: " + file);
            }
        } catch (NoSuchAlgorithmException error) {
            throw new IOException("SHA-256 unavailable", error);
        }
    }
}
