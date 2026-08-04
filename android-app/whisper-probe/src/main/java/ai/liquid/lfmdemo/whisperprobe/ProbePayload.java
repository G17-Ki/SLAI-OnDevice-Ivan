package ai.liquid.lfmdemo.whisperprobe;

import android.content.Context;
import android.content.res.AssetManager;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

/** Installs and verifies the immutable APK payload into app-private storage. */
public final class ProbePayload {
    public static final String MANIFEST_ASSET = "whisper_probe/payload_manifest.json";
    public static final String EXPECTED_GATE_ID =
            "nubia_whisper_small_ort_qnn_epcontext_sm8850_probe_v1";

    private final File root;
    private final File nativeLibraryDir;
    private final JSONObject manifest;

    private ProbePayload(File root, File nativeLibraryDir, JSONObject manifest) {
        this.root = root;
        this.nativeLibraryDir = nativeLibraryDir;
        this.manifest = manifest;
    }

    public static ProbePayload installAndVerify(Context context)
            throws IOException, JSONException {
        if (!BuildConfig.WHISPER_PAYLOAD_EMBEDDED) {
            throw new IllegalStateException(
                    "This APK was built without WHISPER_PROBE_PAYLOAD_DIR"
            );
        }
        byte[] manifestBytes = readAsset(context.getAssets(), MANIFEST_ASSET);
        JSONObject manifest = new JSONObject(new String(manifestBytes, StandardCharsets.UTF_8));
        verifyManifestIdentity(context, manifest);

        File root = new File(context.getFilesDir(), "whisper_probe/payload-v1");
        if (!root.isDirectory() && !root.mkdirs()) {
            throw new IOException("cannot create app-private probe root: " + root);
        }
        File manifestCopy = checkedDestination(root, "payload_manifest.json");
        writeIfChanged(manifestCopy, manifestBytes);

        JSONArray assets = manifest.getJSONArray("assets");
        for (int index = 0; index < assets.length(); index++) {
            JSONObject record = assets.getJSONObject(index);
            String assetPath = record.getString("path");
            if (!assetPath.startsWith("whisper_probe/")) {
                throw new IOException("asset is outside whisper_probe/: " + assetPath);
            }
            String installedPath = assetPath.substring("whisper_probe/".length());
            File destination = checkedDestination(root, installedPath);
            installAsset(
                    context.getAssets(),
                    assetPath,
                    destination,
                    record.getLong("bytes"),
                    record.getString("sha256")
            );
        }

        File nativeLibraryDir = new File(context.getApplicationInfo().nativeLibraryDir);
        JSONArray nativeLibraries = manifest.getJSONArray("native_libraries");
        for (int index = 0; index < nativeLibraries.length(); index++) {
            JSONObject record = nativeLibraries.getJSONObject(index);
            String packagedPath = record.getString("path");
            String prefix = "arm64-v8a/";
            if (!packagedPath.startsWith(prefix)) {
                throw new IOException("unexpected native-library path: " + packagedPath);
            }
            File library = new File(nativeLibraryDir, packagedPath.substring(prefix.length()));
            verifyFile(library, record.getLong("bytes"), record.getString("sha256"));
        }

        ProbePayload payload = new ProbePayload(root, nativeLibraryDir, manifest);
        payload.requireCoreFiles();
        return payload;
    }

    private static void verifyManifestIdentity(Context context, JSONObject manifest)
            throws JSONException {
        if (manifest.getInt("schema_version") != 1) {
            throw new IllegalStateException("unsupported probe payload schema");
        }
        if (!EXPECTED_GATE_ID.equals(manifest.getString("gate_id"))) {
            throw new IllegalStateException("unexpected probe gate_id");
        }
        if (!context.getPackageName().equals(
                manifest.getJSONObject("target").getString("package"))) {
            throw new IllegalStateException("probe payload package mismatch");
        }
        JSONObject capabilities = manifest.getJSONObject("capabilities");
        if (!capabilities.getBoolean("session_load")
                || !capabilities.getBoolean("contract_enumeration")
                || !capabilities.getBoolean("four_step_decode")) {
            throw new IllegalStateException("probe payload lacks compatibility capabilities");
        }
        if (capabilities.getBoolean("real_audio_frontend")
                || capabilities.getBoolean("real_audio_transcription")
                || capabilities.getBoolean("microphone_input")) {
            throw new IllegalStateException("scaffold manifest must fail closed for real audio");
        }
        JSONObject runtime = manifest.getJSONObject("runtime");
        if (!"1.25.0".equals(runtime.getString("onnx_runtime"))
                || !"2.42.0.251225135753_193295".equals(runtime.getString("qairt"))
                || !runtime.getBoolean("disable_cpu_ep_fallback")) {
            throw new IllegalStateException("probe runtime identity mismatch");
        }
    }

    private void requireCoreFiles() throws IOException {
        for (String name : new String[] {
                "encoder.onnx",
                "encoder_qairt_context.bin",
                "decoder.onnx",
                "decoder_qairt_context.bin",
                "metadata.json"
        }) {
            if (!modelFile(name).isFile()) {
                throw new IOException("missing installed model file: " + name);
            }
        }
        if (!voiceAiFile("vocab.bin").isFile() || !voiceAiFile("config.json").isFile()) {
            throw new IOException("missing installed VoiceAI vocabulary/config provenance");
        }
        if (!nativeLibrary("libQnnHtp.so").isFile()
                || !nativeLibrary("libQnnHtpV81Skel.so").isFile()) {
            throw new IOException("missing installed QNN HTP runtime");
        }
    }

    public File modelFile(String name) {
        return new File(new File(root, "models"), name);
    }

    public File voiceAiFile(String name) {
        return new File(new File(root, "voice_ai"), name);
    }

    public File nativeLibrary(String name) {
        return new File(nativeLibraryDir, name);
    }

    public File optionalFeatureFixture() {
        File fixture = modelFile("input_features.u16le");
        return fixture.isFile() ? fixture : null;
    }

    public JSONObject manifest() {
        return manifest;
    }

    public File root() {
        return root;
    }

    public File nativeLibraryDir() {
        return nativeLibraryDir;
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
                // Replace only this app-private, manifest-owned file.
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
        Files.move(
                temporary.toPath(),
                destination.toPath(),
                java.nio.file.StandardCopyOption.REPLACE_EXISTING,
                java.nio.file.StandardCopyOption.ATOMIC_MOVE
        );
        verifyFile(destination, expectedBytes, expectedSha256);
    }

    private static File checkedDestination(File root, String relative) throws IOException {
        File destination = new File(root, relative);
        String rootPath = root.getCanonicalPath() + File.separator;
        String destinationPath = destination.getCanonicalPath();
        if (!destinationPath.startsWith(rootPath)) {
            throw new IOException("payload path traversal: " + relative);
        }
        return destination;
    }

    private static void writeIfChanged(File path, byte[] bytes) throws IOException {
        if (path.isFile() && path.length() == bytes.length) {
            byte[] existing = Files.readAllBytes(path.toPath());
            if (java.util.Arrays.equals(existing, bytes)) {
                return;
            }
        }
        File parent = path.getParentFile();
        if (parent == null || (!parent.isDirectory() && !parent.mkdirs())) {
            throw new IOException("cannot create parent for " + path);
        }
        try (FileOutputStream output = new FileOutputStream(path)) {
            output.write(bytes);
            output.getFD().sync();
        }
    }

    private static void verifyFile(File path, long expectedBytes, String expectedSha256)
            throws IOException {
        if (!path.isFile()) {
            throw new IOException("missing payload file: " + path);
        }
        if (path.length() != expectedBytes) {
            throw new IOException("payload size mismatch: " + path);
        }
        String actual = sha256(path);
        if (!expectedSha256.equals(actual)) {
            throw new IOException("payload SHA-256 mismatch: " + path);
        }
    }

    private static String sha256(File path) throws IOException {
        MessageDigest digest;
        try {
            digest = MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException error) {
            throw new AssertionError(error);
        }
        try (FileInputStream input = new FileInputStream(path)) {
            byte[] buffer = new byte[1024 * 1024];
            int count;
            while ((count = input.read(buffer)) != -1) {
                digest.update(buffer, 0, count);
            }
        }
        StringBuilder encoded = new StringBuilder(64);
        for (byte value : digest.digest()) {
            encoded.append(String.format(java.util.Locale.ROOT, "%02x", value & 0xff));
        }
        return encoded.toString();
    }
}
