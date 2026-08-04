package ai.liquid.lfmdemo.supertonicprobe;

import android.os.Process;
import android.os.SystemClock;
import android.system.ErrnoException;
import android.system.Os;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtProvider;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.Map;

/** Executes only the deterministic Supertonic duration graph through QNN HTP. */
public final class SupertonicQnnProbe {
    private static final long[] TEXT_IDS_SHAPE = {1, 32};
    private static final long[] STYLE_DP_SHAPE = {1, 8, 16};
    private static final long[] TEXT_MASK_SHAPE = {1, 1, 32};

    private SupertonicQnnProbe() {}

    public static JSONObject run(SupertonicPayload payload)
            throws IOException, JSONException, OrtException, ErrnoException {
        long probeStart = SystemClock.elapsedRealtimeNanos();
        if (ByteOrder.nativeOrder() != ByteOrder.LITTLE_ENDIAN) {
            throw new IllegalStateException("probe requires little-endian arm64");
        }

        String nativePath = payload.nativeLibraryDir().getAbsolutePath();
        prependEnvironment("ADSP_LIBRARY_PATH", nativePath, ";");
        prependEnvironment("LD_LIBRARY_PATH", nativePath, ":");
        System.load(payload.nativeLibrary("libQnnSystem.so").getAbsolutePath());
        System.load(payload.nativeLibrary("libQnnHtpV81Stub.so").getAbsolutePath());
        System.load(payload.nativeLibrary("libQnnHtp.so").getAbsolutePath());

        OrtEnvironment environment = OrtEnvironment.getEnvironment(
                "supertonic-duration-ort-qnn-probe"
        );
        String ortVersion = environment.getVersion();
        if (!"1.25.0".equals(ortVersion)) {
            throw new IllegalStateException("unexpected ORT version: " + ortVersion);
        }
        EnumSet<OrtProvider> providers = OrtEnvironment.getAvailableProviders();
        if (!providers.contains(OrtProvider.QNN)) {
            throw new IllegalStateException("QNNExecutionProvider is unavailable");
        }

        File model = payload.file("models/duration_predictor.int8.onnx");
        long loadStart = SystemClock.elapsedRealtimeNanos();
        try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
            options.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL);
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
            options.setInterOpNumThreads(1);
            options.setIntraOpNumThreads(1);
            options.setSymbolicDimensionValue("batch_size", 1);
            options.setSymbolicDimensionValue("text_length", 32);
            options.addConfigEntry("session.disable_cpu_ep_fallback", "1");
            Map<String, String> qnn = new HashMap<>();
            qnn.put("backend_path", payload.nativeLibrary("libQnnHtp.so").getAbsolutePath());
            qnn.put("htp_performance_mode", "burst");
            qnn.put("enable_htp_fp16_precision", "1");
            options.addQnn(qnn);

            try (OrtSession session = environment.createSession(model.getAbsolutePath(), options)) {
                long loaded = SystemClock.elapsedRealtimeNanos();
                JSONObject contract = describeContract(session);

                ByteBuffer textBytes = directBytes(
                        payload.file("fixtures/text_ids.i64le")
                );
                ByteBuffer styleBytes = directBytes(
                        payload.file("fixtures/style_dp.f32le")
                );
                ByteBuffer maskBytes = directBytes(
                        payload.file("fixtures/text_mask.f32le")
                );
                ByteBuffer goldenBytes = directBytes(
                        payload.file("fixtures/duration.f32le")
                );
                LongBuffer textValues = textBytes.asLongBuffer();
                FloatBuffer styleValues = styleBytes.asFloatBuffer();
                FloatBuffer maskValues = maskBytes.asFloatBuffer();
                float expectedDuration = goldenBytes.asFloatBuffer().get(0);

                try (OnnxTensor textIds = OnnxTensor.createTensor(
                             environment, textValues, TEXT_IDS_SHAPE
                     );
                     OnnxTensor styleDp = OnnxTensor.createTensor(
                             environment, styleValues, STYLE_DP_SHAPE
                     );
                     OnnxTensor textMask = OnnxTensor.createTensor(
                             environment, maskValues, TEXT_MASK_SHAPE
                     )) {
                    Map<String, OnnxTensor> inputs = new HashMap<>();
                    inputs.put("text_ids", textIds);
                    inputs.put("style_dp", styleDp);
                    inputs.put("text_mask", textMask);
                    long executeStart = SystemClock.elapsedRealtimeNanos();
                    try (OrtSession.Result result = session.run(inputs)) {
                        long executed = SystemClock.elapsedRealtimeNanos();
                        OnnxValue outputValue = result.get("duration").orElseThrow(
                                () -> new IllegalStateException("duration output is absent")
                        );
                        if (!(outputValue instanceof OnnxTensor)) {
                            throw new IllegalStateException("duration output is not a tensor");
                        }
                        OnnxTensor durationTensor = (OnnxTensor) outputValue;
                        FloatBuffer durationValues = durationTensor.getFloatBuffer();
                        if (durationValues.remaining() != 1) {
                            throw new IllegalStateException(
                                    "duration output element count is not one"
                            );
                        }
                        float actualDuration = durationValues.get(durationValues.position());
                        float absoluteError = Math.abs(actualDuration - expectedDuration);
                        float tolerance = 0.1f;
                        boolean finite = Float.isFinite(actualDuration);
                        boolean numericalPass = finite && absoluteError <= tolerance;
                        if (!numericalPass) {
                            throw new IllegalStateException(
                                    "duration numerical gate failed: actual="
                                            + actualDuration + ", expected=" + expectedDuration
                            );
                        }

                        JSONObject report = new JSONObject();
                        report.put("schema_version", 1);
                        report.put("gate_id", SupertonicPayload.EXPECTED_GATE_ID);
                        report.put("package", "ai.liquid.lfmdemo.supertonicprobe");
                        report.put("uid", Process.myUid());
                        report.put("ort_version", ortVersion);
                        JSONArray providerNames = new JSONArray();
                        for (OrtProvider provider : providers) {
                            providerNames.put(provider.getName());
                        }
                        report.put("available_providers", providerNames);
                        report.put("qnn_provider_requested", true);
                        report.put("cpu_ep_fallback_disabled", true);
                        report.put("backend_path", payload.nativeLibrary(
                                "libQnnHtp.so"
                        ).getAbsolutePath());
                        report.put("model_storage", payload.root().getAbsolutePath());
                        report.put(
                                "runtime_storage",
                                payload.nativeLibraryDir().getAbsolutePath()
                        );
                        report.put("host_transport_required", false);
                        report.put("component", "duration_predictor");
                        report.put("source_format", "raw_onnx");
                        report.put("symbolic_dimension_overrides", new JSONObject()
                                .put("batch_size", 1)
                                .put("text_length", 32));
                        report.put("session_created", true);
                        report.put("contract", contract);
                        report.put("execution", new JSONObject()
                                .put("output_name", "duration")
                                .put("expected_duration", expectedDuration)
                                .put("actual_duration", actualDuration)
                                .put("absolute_error", absoluteError)
                                .put("absolute_tolerance", tolerance)
                                .put("finite", finite)
                                .put("numerical_sanity_passed", numericalPass));
                        report.put("timing_ms", new JSONObject()
                                .put("session_load", millis(loaded - loadStart))
                                .put("execute", millis(executed - executeStart))
                                .put("total", millis(executed - probeStart)));
                        report.put("full_tts", false);
                        report.put("audio_output", false);
                        report.put("production_integration", false);
                        report.put("accepted_packages_mutated", false);
                        report.put("status", "duration_component_probe_pass");
                        report.put(
                                "claim_boundary",
                                "Raw-ONNX duration-component compatibility only. No full TTS, "
                                        + "audio quality, end-to-end latency, or production claim."
                        );
                        return report;
                    }
                }
            }
        }
    }

    private static JSONObject describeContract(OrtSession session)
            throws JSONException, OrtException {
        JSONObject result = new JSONObject();
        result.put("inputs", describeNodes(session.getInputInfo()));
        result.put("outputs", describeNodes(session.getOutputInfo()));
        if (!session.getInputNames().equals(
                new java.util.HashSet<>(Arrays.asList("text_ids", "style_dp", "text_mask")))) {
            throw new IllegalStateException("duration input-name contract mismatch");
        }
        if (!session.getOutputNames().equals(
                new java.util.HashSet<>(Arrays.asList("duration")))) {
            throw new IllegalStateException("duration output-name contract mismatch");
        }
        return result;
    }

    private static JSONArray describeNodes(Map<String, NodeInfo> nodes) throws JSONException {
        JSONArray result = new JSONArray();
        for (Map.Entry<String, NodeInfo> entry : nodes.entrySet()) {
            if (!(entry.getValue().getInfo() instanceof TensorInfo)) {
                throw new IllegalStateException("non-tensor ONNX contract: " + entry.getKey());
            }
            TensorInfo info = (TensorInfo) entry.getValue().getInfo();
            JSONArray shape = new JSONArray();
            for (long dimension : info.getShape()) {
                shape.put(dimension);
            }
            JSONArray dimensionNames = new JSONArray();
            for (String name : info.getDimensionNames()) {
                dimensionNames.put(name == null ? JSONObject.NULL : name);
            }
            result.put(new JSONObject()
                    .put("name", entry.getKey())
                    .put("dtype", info.type.toString())
                    .put("shape", shape)
                    .put("dimension_names", dimensionNames));
        }
        return result;
    }

    private static ByteBuffer directBytes(File file) throws IOException {
        byte[] bytes = Files.readAllBytes(file.toPath());
        ByteBuffer result = ByteBuffer.allocateDirect(bytes.length).order(
                ByteOrder.LITTLE_ENDIAN
        );
        result.put(bytes);
        result.flip();
        return result;
    }

    private static void prependEnvironment(String key, String value, String separator)
            throws ErrnoException {
        String previous = System.getenv(key);
        String combined = value;
        if (previous != null && !previous.trim().isEmpty() && !previous.startsWith(value)) {
            combined = value + separator + previous;
        }
        Os.setenv(key, combined, true);
    }

    private static double millis(long nanoseconds) {
        return Math.round((nanoseconds / 1_000_000.0) * 1000.0) / 1000.0;
    }
}
