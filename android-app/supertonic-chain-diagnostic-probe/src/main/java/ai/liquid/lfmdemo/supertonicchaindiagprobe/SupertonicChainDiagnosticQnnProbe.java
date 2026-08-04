package ai.liquid.lfmdemo.supertonicchaindiagprobe;

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
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

/** Executes the deterministic four-component Supertonic neural chain on QNN. */
public final class SupertonicChainDiagnosticQnnProbe {
    private static final int TEXT_LENGTH = 32;
    private static final int LATENT_LENGTH = 32;
    private static final int LATENT_CHANNELS = 144;
    private static final int TEXT_EMBEDDING_CHANNELS = 256;
    private static final int VECTOR_STEPS = 4;
    private static final int SAMPLE_RATE = 44_100;
    private static final int SAMPLES_PER_LATENT = 3_072;
    private static final int WAVEFORM_SAMPLES = 98_304;
    private static final int EXPECTED_REAL_LATENT_LENGTH = 31;

    private static final long[] TEXT_IDS_SHAPE = {1, TEXT_LENGTH};
    private static final long[] TEXT_MASK_SHAPE = {1, 1, TEXT_LENGTH};
    private static final long[] STYLE_DP_SHAPE = {1, 8, 16};
    private static final long[] STYLE_TTL_SHAPE = {1, 50, 256};
    private static final long[] TEXT_EMBEDDING_SHAPE = {
            1, TEXT_EMBEDDING_CHANNELS, TEXT_LENGTH
    };
    private static final long[] LATENT_SHAPE = {
            1, LATENT_CHANNELS, LATENT_LENGTH
    };
    private static final long[] LATENT_MASK_SHAPE = {1, 1, LATENT_LENGTH};
    private static final long[] SCALAR_SHAPE = {1};
    private static final long[] WAVEFORM_SHAPE = {1, WAVEFORM_SAMPLES};

    public static final int DIAGNOSTIC_SCHEMA_VERSION = 2;
    public static final String DIAGNOSTIC_MARKER =
            "SUPERTONIC_CHAIN_STAGE_DIAGNOSTIC_V2";

    private SupertonicChainDiagnosticQnnProbe() {}

    public static JSONObject run(
            SupertonicChainDiagnosticPayload payload,
            Progress progress
    )
            throws IOException, JSONException, OrtException, ErrnoException {
        long probeStart = SystemClock.elapsedRealtimeNanos();
        progress.enterStage("runtime_initialization", "native_library_load");
        if (ByteOrder.nativeOrder() != ByteOrder.LITTLE_ENDIAN) {
            throw new IllegalStateException("probe requires little-endian arm64");
        }
        String nativePath = payload.nativeLibraryDir().getAbsolutePath();
        prependEnvironment("ADSP_LIBRARY_PATH", nativePath, ";");
        prependEnvironment("LD_LIBRARY_PATH", nativePath, ":");
        System.load(payload.nativeLibrary("libQnnSystem.so").getAbsolutePath());
        System.load(payload.nativeLibrary("libQnnHtpV81Stub.so").getAbsolutePath());
        System.load(payload.nativeLibrary("libQnnHtp.so").getAbsolutePath());

        progress.enterStage("runtime_initialization", "ort_environment_create");
        OrtEnvironment environment = OrtEnvironment.getEnvironment(
                "supertonic-full-chain-stage-diagnostic-ort-qnn-probe"
        );
        progress.enterStage("runtime_initialization", "provider_validation");
        String ortVersion = environment.getVersion();
        if (!"1.25.0".equals(ortVersion)) {
            throw new IllegalStateException("unexpected ORT version: " + ortVersion);
        }
        EnumSet<OrtProvider> providers = OrtEnvironment.getAvailableProviders();
        if (!providers.contains(OrtProvider.QNN)) {
            throw new IllegalStateException("QNNExecutionProvider is unavailable");
        }

        JSONArray providerNames = new JSONArray();
        for (OrtProvider provider : providers) {
            providerNames.put(provider.getName());
        }
        progress.enterStage("runtime_initialization", "fixture_load");
        long[] textIds = readLongs(
                payload.file("fixtures/text_ids.i64le"), TEXT_LENGTH
        );
        float[] textMask = readFloats(
                payload.file("fixtures/text_mask.f32le"), TEXT_LENGTH
        );
        float[] styleDp = readFloats(
                payload.file("fixtures/style_dp.f32le"), 8 * 16
        );
        float[] styleTtl = readFloats(
                payload.file("fixtures/style_ttl.f32le"), 50 * 256
        );
        float[] latentMask = readFloats(
                payload.file("fixtures/latent_mask.f32le"), LATENT_LENGTH
        );
        float[] latentInitial = readFloats(
                payload.file("fixtures/latent_initial.f32le"),
                LATENT_CHANNELS * LATENT_LENGTH
        );
        float[] durationGolden = readFloats(
                payload.file("goldens/duration.f32le"), 1
        );
        float[] textEmbeddingGolden = readFloats(
                payload.file("goldens/text_embedding.f32le"),
                TEXT_EMBEDDING_CHANNELS * TEXT_LENGTH
        );
        float[][] latentGoldens = new float[VECTOR_STEPS][];
        for (int step = 0; step < VECTOR_STEPS; step++) {
            latentGoldens[step] = readFloats(
                    payload.file("goldens/latent_step_" + step + ".f32le"),
                    LATENT_CHANNELS * LATENT_LENGTH
            );
        }
        float[] waveformGolden = readFloats(
                payload.file("goldens/waveform_padded.f32le"), WAVEFORM_SAMPLES
        );
        progress.completeStage(
                "runtime_initialization",
                new JSONObject()
                        .put("component", "runtime_initialization")
                        .put("native_libraries_loaded", true)
                        .put("ort_version", ortVersion)
                        .put("available_providers", providerNames)
                        .put("qnn_provider_available", true)
                        .put("fixture_files_loaded", true)
        );

        StageResult duration = runDuration(
                environment,
                payload,
                progress,
                textIds,
                styleDp,
                textMask,
                durationGolden
        );
        int wavLength = Math.max(1, (int) (duration.output[0] * SAMPLE_RATE));
        int realLatentLength =
                (wavLength + SAMPLES_PER_LATENT - 1) / SAMPLES_PER_LATENT;
        if (realLatentLength != EXPECTED_REAL_LATENT_LENGTH) {
            throw new IllegalStateException(
                    "duration changed fixed latent contract: " + realLatentLength
            );
        }
        requireMask(latentMask, realLatentLength);
        progress.completeStage("duration_predictor", duration.report);

        StageResult textEncoder = runTextEncoder(
                environment,
                payload,
                progress,
                textIds,
                styleTtl,
                textMask,
                textEmbeddingGolden
        );
        progress.completeStage("text_encoder", textEncoder.report);
        VectorResult vector = runVectorEstimator(
                environment,
                payload,
                progress,
                latentInitial,
                textEncoder.output,
                styleTtl,
                latentMask,
                textMask,
                latentGoldens
        );
        progress.completeStage("vector_estimator", vector.report);
        StageResult vocoder = runVocoder(
                environment,
                payload,
                progress,
                vector.output,
                waveformGolden
        );
        progress.completeStage("vocoder", vocoder.report);

        progress.enterStage("report_assembly", "output_validation");
        JSONObject waveformStats = tensorStats(vocoder.output);
        double rms = waveformStats.getDouble("rms");
        if (!(rms > 1.0e-5)) {
            throw new IllegalStateException("vocoder waveform is silent");
        }
        long executeNanos = duration.executeNanos
                + textEncoder.executeNanos
                + vector.executeNanos
                + vocoder.executeNanos;
        long loadNanos = duration.loadNanos
                + textEncoder.loadNanos
                + vector.loadNanos
                + vocoder.loadNanos;

        JSONObject report = new JSONObject();
        report.put("schema_version", 1);
        report.put("gate_id", SupertonicChainDiagnosticPayload.EXPECTED_GATE_ID);
        report.put("package", "ai.liquid.lfmdemo.supertonicchaindiagprobe");
        report.put("uid", Process.myUid());
        report.put("ort_version", ortVersion);
        report.put("available_providers", providerNames);
        report.put("qnn_provider_requested", true);
        report.put("cpu_ep_fallback_disabled", true);
        report.put(
                "backend_path",
                payload.nativeLibrary("libQnnHtp.so").getAbsolutePath()
        );
        report.put("model_storage", payload.root().getAbsolutePath());
        report.put(
                "runtime_storage", payload.nativeLibraryDir().getAbsolutePath()
        );
        report.put("host_transport_required", false);
        report.put("source_format", "raw_onnx");
        report.put("symbolic_dimension_overrides", new JSONObject()
                .put("batch_size", 1)
                .put("text_length", TEXT_LENGTH)
                .put("latent_length", LATENT_LENGTH));
        report.put("sessions_created", new JSONObject()
                .put("duration_predictor", true)
                .put("text_encoder", true)
                .put("vector_estimator", true)
                .put("vocoder", true));
        report.put("chain", new JSONObject()
                .put("duration_predictor", duration.report)
                .put("text_encoder", textEncoder.report)
                .put("vector_estimator", vector.report)
                .put("vocoder", vocoder.report)
                .put("vector_steps", VECTOR_STEPS)
                .put("sample_rate", SAMPLE_RATE)
                .put("wav_length_from_device_duration", wavLength)
                .put("fixed_latent_length", LATENT_LENGTH)
                .put("real_latent_length", realLatentLength)
                .put("waveform_samples", vocoder.output.length)
                .put("waveform_rms", rms)
                .put("finite_non_silent_pcm_tensor", true));
        report.put("timing_ms", new JSONObject()
                .put("duration_session_load", millis(duration.loadNanos))
                .put("duration_execute", millis(duration.executeNanos))
                .put("text_encoder_session_load", millis(textEncoder.loadNanos))
                .put("text_encoder_execute", millis(textEncoder.executeNanos))
                .put("vector_session_load", millis(vector.loadNanos))
                .put("vector_execute_four_steps", millis(vector.executeNanos))
                .put("vocoder_session_load", millis(vocoder.loadNanos))
                .put("vocoder_execute", millis(vocoder.executeNanos))
                .put("session_load_total", millis(loadNanos))
                .put("neural_execute_total", millis(executeNanos))
                .put("total", millis(SystemClock.elapsedRealtimeNanos() - probeStart)));
        report.put("full_neural_chain", true);
        report.put("chained_pcm_tensor", true);
        report.put("full_tts", false);
        report.put("audio_file_output", false);
        report.put("audio_playback", false);
        report.put("audio_quality_accepted", false);
        report.put("host_reference_comparisons_promoted", false);
        report.put("production_integration", false);
        report.put("accepted_packages_mutated", false);
        report.put("status", "four_component_chain_stage_diagnostic_pass");
        report.put(
                "claim_boundary",
                "All four raw ONNX components executed as one deterministic chain with "
                        + "finite shape-safe tensors and non-silent PCM. Host reference "
                        + "comparisons are diagnostic only; no audio quality, language "
                        + "identity, playback, promoted latency, unplugged durability, or "
                        + "production integration is claimed."
        );
        return report;
    }

    private static StageResult runDuration(
            OrtEnvironment environment,
            SupertonicChainDiagnosticPayload payload,
            Progress progress,
            long[] textIds,
            float[] styleDp,
            float[] textMask,
            float[] golden
    ) throws OrtException, IOException, JSONException {
        Map<String, TensorExpectation> inputs = new LinkedHashMap<>();
        inputs.put("text_ids", expectation("INT64", TEXT_IDS_SHAPE));
        inputs.put("style_dp", expectation("FLOAT", STYLE_DP_SHAPE));
        inputs.put("text_mask", expectation("FLOAT", TEXT_MASK_SHAPE));
        Map<String, TensorExpectation> outputs = new LinkedHashMap<>();
        outputs.put("duration", expectation("FLOAT", SCALAR_SHAPE));
        long loadStart = SystemClock.elapsedRealtimeNanos();
        progress.enterStage("duration_predictor", "session_create");
        try (OrtSession.SessionOptions options = sessionOptions(payload, true, false);
             OrtSession session = environment.createSession(
                     payload.file("models/duration_predictor.int8.onnx")
                             .getAbsolutePath(),
                     options
             )) {
            long loaded = SystemClock.elapsedRealtimeNanos();
            progress.markSessionCreated("duration_predictor");
            progress.enterStage("duration_predictor", "contract_validation");
            JSONObject contract = requireContract(session, inputs, outputs);
            progress.enterStage("duration_predictor", "execute");
            try (OnnxTensor textTensor = OnnxTensor.createTensor(
                         environment, directLongs(textIds), TEXT_IDS_SHAPE
                 );
                 OnnxTensor styleTensor = OnnxTensor.createTensor(
                         environment, directFloats(styleDp), STYLE_DP_SHAPE
                 );
                 OnnxTensor maskTensor = OnnxTensor.createTensor(
                         environment, directFloats(textMask), TEXT_MASK_SHAPE
                 )) {
                Map<String, OnnxTensor> values = new HashMap<>();
                values.put("text_ids", textTensor);
                values.put("style_dp", styleTensor);
                values.put("text_mask", maskTensor);
                long executeStart = SystemClock.elapsedRealtimeNanos();
                float[] actual;
                try (OrtSession.Result result = session.run(values)) {
                    progress.enterStage("duration_predictor", "output_validation");
                    actual = floatOutput(result, "duration", 1);
                }
                long executed = SystemClock.elapsedRealtimeNanos();
                requireFinite(actual, "duration output");
                double absoluteError = Math.abs(actual[0] - golden[0]);
                boolean withinDiagnosticTolerance = absoluteError <= 0.1;
                JSONObject stage = new JSONObject()
                        .put("component", "duration_predictor")
                        .put("session_created", true)
                        .put("contract", contract)
                        .put("output", tensorStats(actual))
                        .put("reference_comparison", comparison(actual, golden))
                        .put("diagnostic_absolute_tolerance", 0.1)
                        .put(
                                "reference_within_diagnostic_tolerance",
                                withinDiagnosticTolerance
                        )
                        .put("finite_shape_gate_passed", true);
                return new StageResult(
                        actual,
                        stage,
                        loaded - loadStart,
                        executed - executeStart
                );
            }
        }
    }

    private static StageResult runTextEncoder(
            OrtEnvironment environment,
            SupertonicChainDiagnosticPayload payload,
            Progress progress,
            long[] textIds,
            float[] styleTtl,
            float[] textMask,
            float[] golden
    ) throws OrtException, IOException, JSONException {
        Map<String, TensorExpectation> inputs = new LinkedHashMap<>();
        inputs.put("text_ids", expectation("INT64", TEXT_IDS_SHAPE));
        inputs.put("style_ttl", expectation("FLOAT", STYLE_TTL_SHAPE));
        inputs.put("text_mask", expectation("FLOAT", TEXT_MASK_SHAPE));
        Map<String, TensorExpectation> outputs = new LinkedHashMap<>();
        outputs.put("text_emb", expectation("FLOAT", TEXT_EMBEDDING_SHAPE));
        long loadStart = SystemClock.elapsedRealtimeNanos();
        progress.enterStage("text_encoder", "session_create");
        try (OrtSession.SessionOptions options = sessionOptions(payload, true, false);
             OrtSession session = environment.createSession(
                     payload.file("models/text_encoder.int8.onnx").getAbsolutePath(),
                     options
             )) {
            long loaded = SystemClock.elapsedRealtimeNanos();
            progress.markSessionCreated("text_encoder");
            progress.enterStage("text_encoder", "contract_validation");
            JSONObject contract = requireContract(session, inputs, outputs);
            progress.enterStage("text_encoder", "execute");
            try (OnnxTensor textTensor = OnnxTensor.createTensor(
                         environment, directLongs(textIds), TEXT_IDS_SHAPE
                 );
                 OnnxTensor styleTensor = OnnxTensor.createTensor(
                         environment, directFloats(styleTtl), STYLE_TTL_SHAPE
                 );
                 OnnxTensor maskTensor = OnnxTensor.createTensor(
                         environment, directFloats(textMask), TEXT_MASK_SHAPE
                 )) {
                Map<String, OnnxTensor> values = new HashMap<>();
                values.put("text_ids", textTensor);
                values.put("style_ttl", styleTensor);
                values.put("text_mask", maskTensor);
                long executeStart = SystemClock.elapsedRealtimeNanos();
                float[] actual;
                try (OrtSession.Result result = session.run(values)) {
                    progress.enterStage("text_encoder", "output_validation");
                    actual = floatOutput(
                            result,
                            "text_emb",
                            TEXT_EMBEDDING_CHANNELS * TEXT_LENGTH
                    );
                }
                long executed = SystemClock.elapsedRealtimeNanos();
                requireFinite(actual, "text encoder output");
                JSONObject stage = new JSONObject()
                        .put("component", "text_encoder")
                        .put("session_created", true)
                        .put("contract", contract)
                        .put("output", tensorStats(actual))
                        .put("reference_comparison", comparison(actual, golden))
                        .put("finite_shape_gate_passed", true);
                return new StageResult(
                        actual,
                        stage,
                        loaded - loadStart,
                        executed - executeStart
                );
            }
        }
    }

    private static VectorResult runVectorEstimator(
            OrtEnvironment environment,
            SupertonicChainDiagnosticPayload payload,
            Progress progress,
            float[] latentInitial,
            float[] textEmbedding,
            float[] styleTtl,
            float[] latentMask,
            float[] textMask,
            float[][] goldens
    ) throws OrtException, IOException, JSONException {
        Map<String, TensorExpectation> inputs = new LinkedHashMap<>();
        inputs.put("noisy_latent", expectation("FLOAT", LATENT_SHAPE));
        inputs.put("text_emb", expectation("FLOAT", TEXT_EMBEDDING_SHAPE));
        inputs.put("style_ttl", expectation("FLOAT", STYLE_TTL_SHAPE));
        inputs.put("latent_mask", expectation("FLOAT", LATENT_MASK_SHAPE));
        inputs.put("text_mask", expectation("FLOAT", TEXT_MASK_SHAPE));
        inputs.put("current_step", expectation("FLOAT", SCALAR_SHAPE));
        inputs.put("total_step", expectation("FLOAT", SCALAR_SHAPE));
        Map<String, TensorExpectation> outputs = new LinkedHashMap<>();
        outputs.put("denoised_latent", expectation("FLOAT", LATENT_SHAPE));
        long loadStart = SystemClock.elapsedRealtimeNanos();
        progress.enterStage("vector_estimator", "session_create");
        try (OrtSession.SessionOptions options = sessionOptions(payload, true, true);
             OrtSession session = environment.createSession(
                     payload.file("models/vector_estimator.int8.onnx")
                             .getAbsolutePath(),
                     options
             )) {
            long loaded = SystemClock.elapsedRealtimeNanos();
            progress.markSessionCreated("vector_estimator");
            progress.enterStage("vector_estimator", "contract_validation");
            JSONObject contract = requireContract(session, inputs, outputs);
            float[] latent = Arrays.copyOf(latentInitial, latentInitial.length);
            JSONArray stepReports = new JSONArray();
            long executeNanos = 0;
            for (int step = 0; step < VECTOR_STEPS; step++) {
                progress.enterVectorStep("execute", step);
                try (OnnxTensor latentTensor = OnnxTensor.createTensor(
                             environment, directFloats(latent), LATENT_SHAPE
                     );
                     OnnxTensor textTensor = OnnxTensor.createTensor(
                             environment,
                             directFloats(textEmbedding),
                             TEXT_EMBEDDING_SHAPE
                     );
                     OnnxTensor styleTensor = OnnxTensor.createTensor(
                             environment, directFloats(styleTtl), STYLE_TTL_SHAPE
                     );
                     OnnxTensor latentMaskTensor = OnnxTensor.createTensor(
                             environment,
                             directFloats(latentMask),
                             LATENT_MASK_SHAPE
                     );
                     OnnxTensor textMaskTensor = OnnxTensor.createTensor(
                             environment, directFloats(textMask), TEXT_MASK_SHAPE
                     );
                     OnnxTensor currentTensor = OnnxTensor.createTensor(
                             environment,
                             directFloats(new float[] {step}),
                             SCALAR_SHAPE
                     );
                     OnnxTensor totalTensor = OnnxTensor.createTensor(
                             environment,
                             directFloats(new float[] {VECTOR_STEPS}),
                             SCALAR_SHAPE
                     )) {
                    Map<String, OnnxTensor> values = new HashMap<>();
                    values.put("noisy_latent", latentTensor);
                    values.put("text_emb", textTensor);
                    values.put("style_ttl", styleTensor);
                    values.put("latent_mask", latentMaskTensor);
                    values.put("text_mask", textMaskTensor);
                    values.put("current_step", currentTensor);
                    values.put("total_step", totalTensor);
                    long executeStart = SystemClock.elapsedRealtimeNanos();
                    float[] raw;
                    try (OrtSession.Result result = session.run(values)) {
                        progress.enterVectorStep("output_validation", step);
                        raw = floatOutput(
                                result,
                                "denoised_latent",
                                LATENT_CHANNELS * LATENT_LENGTH
                        );
                    }
                    long executed = SystemClock.elapsedRealtimeNanos();
                    executeNanos += executed - executeStart;
                    requireFinite(raw, "vector step " + step);
                    applyLatentMask(raw, latentMask);
                    requireZeroPadding(raw, EXPECTED_REAL_LATENT_LENGTH);
                    JSONObject stepReport = new JSONObject()
                            .put("step", step)
                            .put("total_steps", VECTOR_STEPS)
                            .put("execute_ms", millis(executed - executeStart))
                            .put("output", tensorStats(raw))
                            .put(
                                    "reference_comparison",
                                    comparison(raw, goldens[step])
                            )
                            .put("finite_shape_gate_passed", true)
                            .put("masked_padding_zero", true);
                    stepReports.put(stepReport);
                    progress.completeVectorStep(step, stepReport);
                    latent = raw;
                }
            }
            JSONObject stage = new JSONObject()
                    .put("component", "vector_estimator")
                    .put("session_created", true)
                    .put("contract", contract)
                    .put("steps", stepReports)
                    .put("step_count", VECTOR_STEPS)
                    .put("finite_shape_gate_passed", true)
                    .put("masked_padding_zero", true);
            return new VectorResult(
                    latent,
                    stage,
                    loaded - loadStart,
                    executeNanos
            );
        }
    }

    private static StageResult runVocoder(
            OrtEnvironment environment,
            SupertonicChainDiagnosticPayload payload,
            Progress progress,
            float[] latent,
            float[] golden
    ) throws OrtException, IOException, JSONException {
        Map<String, TensorExpectation> inputs = new LinkedHashMap<>();
        inputs.put("latent", expectation("FLOAT", LATENT_SHAPE));
        Map<String, TensorExpectation> outputs = new LinkedHashMap<>();
        outputs.put("wav_tts", expectation("FLOAT", WAVEFORM_SHAPE));
        long loadStart = SystemClock.elapsedRealtimeNanos();
        progress.enterStage("vocoder", "session_create");
        try (OrtSession.SessionOptions options = sessionOptions(payload, false, true);
             OrtSession session = environment.createSession(
                     payload.file("models/vocoder.int8.onnx").getAbsolutePath(),
                     options
             )) {
            long loaded = SystemClock.elapsedRealtimeNanos();
            progress.markSessionCreated("vocoder");
            progress.enterStage("vocoder", "contract_validation");
            JSONObject contract = requireContract(session, inputs, outputs);
            progress.enterStage("vocoder", "execute");
            try (OnnxTensor latentTensor = OnnxTensor.createTensor(
                    environment, directFloats(latent), LATENT_SHAPE
            )) {
                Map<String, OnnxTensor> values = new HashMap<>();
                values.put("latent", latentTensor);
                long executeStart = SystemClock.elapsedRealtimeNanos();
                float[] actual;
                try (OrtSession.Result result = session.run(values)) {
                    progress.enterStage("vocoder", "output_validation");
                    actual = floatOutput(result, "wav_tts", WAVEFORM_SAMPLES);
                }
                long executed = SystemClock.elapsedRealtimeNanos();
                requireFinite(actual, "vocoder output");
                JSONObject stats = tensorStats(actual);
                boolean nonSilent = stats.getDouble("rms") > 1.0e-5;
                if (!nonSilent) {
                    throw new IllegalStateException("vocoder output is silent");
                }
                JSONObject stage = new JSONObject()
                        .put("component", "vocoder")
                        .put("session_created", true)
                        .put("contract", contract)
                        .put("output", stats)
                        .put("reference_comparison", comparison(actual, golden))
                        .put("finite_shape_gate_passed", true)
                        .put("non_silent", true)
                        .put("audio_quality_accepted", false);
                return new StageResult(
                        actual,
                        stage,
                        loaded - loadStart,
                        executed - executeStart
                );
            }
        }
    }

    private static OrtSession.SessionOptions sessionOptions(
            SupertonicChainDiagnosticPayload payload,
            boolean textDimension,
            boolean latentDimension
    ) throws OrtException {
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        options.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL);
        options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        options.setInterOpNumThreads(1);
        options.setIntraOpNumThreads(1);
        options.setSymbolicDimensionValue("batch_size", 1);
        if (textDimension) {
            options.setSymbolicDimensionValue("text_length", TEXT_LENGTH);
        }
        if (latentDimension) {
            options.setSymbolicDimensionValue("latent_length", LATENT_LENGTH);
        }
        options.addConfigEntry("session.disable_cpu_ep_fallback", "1");
        Map<String, String> qnn = new HashMap<>();
        qnn.put(
                "backend_path",
                payload.nativeLibrary("libQnnHtp.so").getAbsolutePath()
        );
        qnn.put("htp_performance_mode", "burst");
        qnn.put("enable_htp_fp16_precision", "1");
        options.addQnn(qnn);
        return options;
    }

    private static JSONObject requireContract(
            OrtSession session,
            Map<String, TensorExpectation> expectedInputs,
            Map<String, TensorExpectation> expectedOutputs
    ) throws OrtException, JSONException {
        if (!session.getInputNames().equals(expectedInputs.keySet())) {
            throw new IllegalStateException("ONNX input-name contract mismatch");
        }
        if (!session.getOutputNames().equals(expectedOutputs.keySet())) {
            throw new IllegalStateException("ONNX output-name contract mismatch");
        }
        requireNodes(session.getInputInfo(), expectedInputs);
        requireNodes(session.getOutputInfo(), expectedOutputs);
        return new JSONObject()
                .put("inputs", describeNodes(session.getInputInfo()))
                .put("outputs", describeNodes(session.getOutputInfo()));
    }

    private static void requireNodes(
            Map<String, NodeInfo> nodes,
            Map<String, TensorExpectation> expected
    ) {
        for (Map.Entry<String, TensorExpectation> entry : expected.entrySet()) {
            NodeInfo node = nodes.get(entry.getKey());
            if (node == null || !(node.getInfo() instanceof TensorInfo)) {
                throw new IllegalStateException("missing tensor node: " + entry.getKey());
            }
            TensorInfo info = (TensorInfo) node.getInfo();
            TensorExpectation contract = entry.getValue();
            if (!contract.dtype.equals(info.type.toString())
                    || !Arrays.equals(contract.shape, info.getShape())) {
                throw new IllegalStateException(
                        "tensor contract mismatch: " + entry.getKey()
                );
            }
        }
    }

    private static JSONArray describeNodes(Map<String, NodeInfo> nodes)
            throws JSONException {
        JSONArray result = new JSONArray();
        for (Map.Entry<String, NodeInfo> entry : nodes.entrySet()) {
            if (!(entry.getValue().getInfo() instanceof TensorInfo)) {
                throw new IllegalStateException(
                        "non-tensor ONNX contract: " + entry.getKey()
                );
            }
            TensorInfo info = (TensorInfo) entry.getValue().getInfo();
            JSONArray shape = new JSONArray();
            for (long dimension : info.getShape()) {
                shape.put(dimension);
            }
            result.put(new JSONObject()
                    .put("name", entry.getKey())
                    .put("dtype", info.type.toString())
                    .put("shape", shape));
        }
        return result;
    }

    private static TensorExpectation expectation(String dtype, long[] shape) {
        return new TensorExpectation(dtype, shape);
    }

    private static float[] floatOutput(
            OrtSession.Result result,
            String name,
            int expectedCount
    ) throws OrtException {
        OnnxValue value = result.get(name).orElseThrow(
                () -> new IllegalStateException("missing output: " + name)
        );
        if (!(value instanceof OnnxTensor)) {
            throw new IllegalStateException("output is not a tensor: " + name);
        }
        FloatBuffer buffer = ((OnnxTensor) value).getFloatBuffer().duplicate();
        if (buffer.remaining() != expectedCount) {
            throw new IllegalStateException("output element count mismatch: " + name);
        }
        float[] output = new float[expectedCount];
        buffer.get(output);
        return output;
    }

    private static long[] readLongs(File file, int expectedCount)
            throws IOException {
        byte[] bytes = Files.readAllBytes(file.toPath());
        if (bytes.length != expectedCount * Long.BYTES) {
            throw new IOException("int64 fixture byte count mismatch: " + file);
        }
        LongBuffer buffer = ByteBuffer.wrap(bytes)
                .order(ByteOrder.LITTLE_ENDIAN)
                .asLongBuffer();
        long[] output = new long[expectedCount];
        buffer.get(output);
        return output;
    }

    private static float[] readFloats(File file, int expectedCount)
            throws IOException {
        byte[] bytes = Files.readAllBytes(file.toPath());
        if (bytes.length != expectedCount * Float.BYTES) {
            throw new IOException("float fixture byte count mismatch: " + file);
        }
        FloatBuffer buffer = ByteBuffer.wrap(bytes)
                .order(ByteOrder.LITTLE_ENDIAN)
                .asFloatBuffer();
        float[] output = new float[expectedCount];
        buffer.get(output);
        return output;
    }

    private static LongBuffer directLongs(long[] values) {
        ByteBuffer bytes = ByteBuffer.allocateDirect(values.length * Long.BYTES)
                .order(ByteOrder.nativeOrder());
        LongBuffer buffer = bytes.asLongBuffer();
        buffer.put(values);
        buffer.flip();
        return buffer;
    }

    private static FloatBuffer directFloats(float[] values) {
        ByteBuffer bytes = ByteBuffer.allocateDirect(values.length * Float.BYTES)
                .order(ByteOrder.nativeOrder());
        FloatBuffer buffer = bytes.asFloatBuffer();
        buffer.put(values);
        buffer.flip();
        return buffer;
    }

    private static void requireMask(float[] mask, int realLength) {
        if (mask.length != LATENT_LENGTH) {
            throw new IllegalStateException("latent mask length mismatch");
        }
        for (int index = 0; index < mask.length; index++) {
            float expected = index < realLength ? 1.0f : 0.0f;
            if (mask[index] != expected) {
                throw new IllegalStateException("latent mask content mismatch");
            }
        }
    }

    private static void applyLatentMask(float[] latent, float[] mask) {
        for (int channel = 0; channel < LATENT_CHANNELS; channel++) {
            for (int position = 0; position < LATENT_LENGTH; position++) {
                int index = channel * LATENT_LENGTH + position;
                latent[index] *= mask[position];
            }
        }
    }

    private static void requireZeroPadding(float[] latent, int realLength) {
        for (int channel = 0; channel < LATENT_CHANNELS; channel++) {
            for (int position = realLength; position < LATENT_LENGTH; position++) {
                if (latent[channel * LATENT_LENGTH + position] != 0.0f) {
                    throw new IllegalStateException("masked latent padding is non-zero");
                }
            }
        }
    }

    private static void requireFinite(float[] values, String label) {
        for (float value : values) {
            if (!Float.isFinite(value)) {
                throw new IllegalStateException(label + " contains NaN or infinity");
            }
        }
    }

    private static JSONObject tensorStats(float[] values) throws JSONException {
        requireFinite(values, "tensor");
        double sum = 0.0;
        double sumSquares = 0.0;
        float minimum = Float.POSITIVE_INFINITY;
        float maximum = Float.NEGATIVE_INFINITY;
        float maximumAbsolute = 0.0f;
        int nonzero = 0;
        int clipped = 0;
        for (float value : values) {
            sum += value;
            sumSquares += (double) value * value;
            minimum = Math.min(minimum, value);
            maximum = Math.max(maximum, value);
            maximumAbsolute = Math.max(maximumAbsolute, Math.abs(value));
            if (value != 0.0f) {
                nonzero++;
            }
            if (Math.abs(value) > 1.0f) {
                clipped++;
            }
        }
        return new JSONObject()
                .put("elements", values.length)
                .put("finite", true)
                .put("min", minimum)
                .put("max", maximum)
                .put("mean", sum / values.length)
                .put("rms", Math.sqrt(sumSquares / values.length))
                .put("max_abs", maximumAbsolute)
                .put("nonzero_values", nonzero)
                .put("values_over_unit_range", clipped)
                .put("sha256_float32le", sha256Float32(values));
    }

    private static JSONObject comparison(
            float[] actual,
            float[] reference
    ) throws JSONException {
        if (actual.length != reference.length) {
            throw new IllegalStateException("reference element count mismatch");
        }
        requireFinite(actual, "actual tensor");
        requireFinite(reference, "reference tensor");
        double absoluteSum = 0.0;
        double squaredSum = 0.0;
        double actualSquared = 0.0;
        double referenceSquared = 0.0;
        double dot = 0.0;
        double maximum = 0.0;
        for (int index = 0; index < actual.length; index++) {
            double error = (double) actual[index] - reference[index];
            double absolute = Math.abs(error);
            absoluteSum += absolute;
            squaredSum += error * error;
            maximum = Math.max(maximum, absolute);
            actualSquared += (double) actual[index] * actual[index];
            referenceSquared += (double) reference[index] * reference[index];
            dot += (double) actual[index] * reference[index];
        }
        double denominator = Math.sqrt(actualSquared * referenceSquared);
        return new JSONObject()
                .put("diagnostic_only", true)
                .put("promoted", false)
                .put("elements", actual.length)
                .put("max_abs_error", maximum)
                .put("mean_abs_error", absoluteSum / actual.length)
                .put("rmse", Math.sqrt(squaredSum / actual.length))
                .put("cosine_similarity", denominator == 0.0 ? 1.0 : dot / denominator)
                .put("actual_sha256", sha256Float32(actual))
                .put("reference_sha256", sha256Float32(reference));
    }

    private static String sha256Float32(float[] values) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            ByteBuffer bytes = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
            for (float value : values) {
                bytes.clear();
                bytes.putFloat(value);
                digest.update(bytes.array());
            }
            StringBuilder result = new StringBuilder();
            for (byte value : digest.digest()) {
                result.append(String.format("%02x", value & 0xff));
            }
            return result.toString();
        } catch (NoSuchAlgorithmException error) {
            throw new IllegalStateException("SHA-256 unavailable", error);
        }
    }

    private static void prependEnvironment(
            String key,
            String value,
            String separator
    ) throws ErrnoException {
        String previous = System.getenv(key);
        String combined = value;
        if (previous != null
                && !previous.trim().isEmpty()
                && !previous.startsWith(value)) {
            combined = value + separator + previous;
        }
        Os.setenv(key, combined, true);
    }

    private static double millis(long nanoseconds) {
        return Math.round((nanoseconds / 1_000_000.0) * 1000.0) / 1000.0;
    }

    /** Mutable, single-run evidence ledger used to attribute fail-closed reports. */
    public static final class Progress {
        private final JSONArray completedStages = new JSONArray();
        private final JSONObject completedStageReports = new JSONObject();
        private final JSONObject sessionsCreated = new JSONObject();
        private final JSONArray completedVectorStepReports = new JSONArray();
        private String currentStage = "not_started";
        private String currentPhase = "not_started";
        private Integer currentVectorStep;
        private int completedVectorSteps;

        public Progress() {
            try {
                sessionsCreated.put("duration_predictor", false);
                sessionsCreated.put("text_encoder", false);
                sessionsCreated.put("vector_estimator", false);
                sessionsCreated.put("vocoder", false);
            } catch (JSONException error) {
                throw new IllegalStateException("cannot initialize diagnostic ledger", error);
            }
        }

        public synchronized void enterStage(String stage, String phase) {
            requireLabel(stage, "stage");
            requireLabel(phase, "phase");
            currentStage = stage;
            currentPhase = phase;
            currentVectorStep = null;
        }

        public synchronized void enterVectorStep(String phase, int step) {
            if (step < 0 || step >= VECTOR_STEPS) {
                throw new IllegalArgumentException("invalid vector step: " + step);
            }
            enterStage("vector_estimator", phase);
            currentVectorStep = step;
        }

        public synchronized void markSessionCreated(String stage)
                throws JSONException {
            if (!sessionsCreated.has(stage)) {
                throw new IllegalArgumentException("unexpected session stage: " + stage);
            }
            sessionsCreated.put(stage, true);
        }

        public synchronized void completeVectorStep(int step, JSONObject report)
                throws JSONException {
            if (!"vector_estimator".equals(currentStage)
                    || step != completedVectorSteps
                    || currentVectorStep == null
                    || currentVectorStep != step) {
                throw new IllegalStateException("vector-step progress is out of order");
            }
            completedVectorStepReports.put(copy(report));
            completedVectorSteps++;
            currentPhase = "step_completed";
        }

        public synchronized void completeStage(String stage, JSONObject report)
                throws JSONException {
            if (!stage.equals(currentStage)) {
                throw new IllegalStateException(
                        "completed stage differs from active stage: " + stage
                );
            }
            for (int index = 0; index < completedStages.length(); index++) {
                if (stage.equals(completedStages.getString(index))) {
                    throw new IllegalStateException("stage completed twice: " + stage);
                }
            }
            completedStages.put(stage);
            completedStageReports.put(stage, copy(report));
            currentPhase = "completed";
            currentVectorStep = null;
        }

        public synchronized void appendSuccessEvidence(JSONObject report)
                throws JSONException {
            appendCommonEvidence(report);
            report.put("diagnostic_outcome", "all_stages_completed");
        }

        public synchronized void appendFailureEvidence(JSONObject report)
                throws JSONException {
            appendCommonEvidence(report);
            report.put("diagnostic_outcome", "failure_localized");
            report.put("failure_stage", currentStage);
            report.put("failure_phase", currentPhase);
            if (currentVectorStep != null) {
                report.put("failure_vector_step", currentVectorStep);
            }
        }

        private void appendCommonEvidence(JSONObject report) throws JSONException {
            report.put("diagnostic_schema_version", DIAGNOSTIC_SCHEMA_VERSION);
            report.put("diagnostic_marker", DIAGNOSTIC_MARKER);
            report.put("completed_stages", copy(completedStages));
            report.put("completed_stage_count", completedStages.length());
            report.put("completed_stage_reports", copy(completedStageReports));
            report.put("sessions_created", copy(sessionsCreated));
            report.put("completed_vector_steps", completedVectorSteps);
            report.put(
                    "completed_vector_step_reports",
                    copy(completedVectorStepReports)
            );
        }

        private static JSONObject copy(JSONObject value) throws JSONException {
            return new JSONObject(value.toString());
        }

        private static JSONArray copy(JSONArray value) throws JSONException {
            return new JSONArray(value.toString());
        }

        private static void requireLabel(String value, String label) {
            if (value == null || value.isEmpty()) {
                throw new IllegalArgumentException("empty diagnostic " + label);
            }
        }
    }

    private static final class TensorExpectation {
        final String dtype;
        final long[] shape;

        TensorExpectation(String dtype, long[] shape) {
            this.dtype = dtype;
            this.shape = Arrays.copyOf(shape, shape.length);
        }
    }

    private static class StageResult {
        final float[] output;
        final JSONObject report;
        final long loadNanos;
        final long executeNanos;

        StageResult(
                float[] output,
                JSONObject report,
                long loadNanos,
                long executeNanos
        ) {
            this.output = output;
            this.report = report;
            this.loadNanos = loadNanos;
            this.executeNanos = executeNanos;
        }
    }

    private static final class VectorResult extends StageResult {
        VectorResult(
                float[] output,
                JSONObject report,
                long loadNanos,
                long executeNanos
        ) {
            super(output, report, loadNanos, executeNanos);
        }
    }
}
