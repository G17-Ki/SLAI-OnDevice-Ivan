package ai.liquid.lfmdemo.whisperprobe;

import android.os.Process;
import android.os.SystemClock;
import android.system.ErrnoException;
import android.system.Os;

import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtProvider;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.UnsignedTensorFactory;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.nio.ShortBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/** Loads both external EPContext sessions and executes a fixed compatibility probe. */
public final class WhisperQnnProbe {
    private static final long[] FEATURE_SHAPE = {1, 80, 3000};
    private static final long[] ATTENTION_MASK_SHAPE = {1, 1, 1, 200};
    private static final long[] INPUT_ID_SHAPE = {1, 1};
    private static final long[] POSITION_ID_SHAPE = {1};
    private static final long[] SELF_K_SHAPE = {12, 1, 64, 199};
    private static final long[] SELF_V_SHAPE = {12, 1, 199, 64};
    private static final int FEATURE_ELEMENTS = 80 * 3000;
    private static final int FEATURE_ZERO_POINT = 32072;
    private static final int CACHE_ZERO_POINT = 128;
    private static final int INITIAL_TOKEN_ID = 50258;
    private static final int LOGIT_COUNT = 51865;
    private static final int DECODER_STEPS = 4;

    private WhisperQnnProbe() {}

    public static JSONObject run(ProbePayload payload)
            throws IOException, JSONException, OrtException, ErrnoException {
        return run(
                payload,
                null,
                null,
                WhisperDecodePolicy.Language.ENGLISH,
                false
        );
    }

    public static JSONObject run(ProbePayload payload, byte[] bundledWavFeatures)
            throws IOException, JSONException, OrtException, ErrnoException {
        return run(
                payload,
                bundledWavFeatures,
                bundledWavFeatures,
                WhisperDecodePolicy.Language.ENGLISH,
                false
        );
    }

    public static JSONObject run(
            ProbePayload payload,
            byte[] bundledWavFeatures,
            boolean semanticDecodeEnabled
    ) throws IOException, JSONException, OrtException, ErrnoException {
        return run(
                payload,
                bundledWavFeatures,
                bundledWavFeatures,
                WhisperDecodePolicy.Language.ENGLISH,
                semanticDecodeEnabled
        );
    }

    public static JSONObject run(
            ProbePayload payload,
            byte[] bundledEnglishFeatures,
            byte[] semanticFeatures,
            WhisperDecodePolicy.Language semanticLanguage,
            boolean semanticDecodeEnabled
    ) throws IOException, JSONException, OrtException, ErrnoException {
        long probeStart = SystemClock.elapsedRealtimeNanos();
        if (ByteOrder.nativeOrder() != ByteOrder.LITTLE_ENDIAN) {
            throw new IllegalStateException("release UINT16 payload requires little-endian arm64");
        }
        if (semanticLanguage == null) {
            throw new IllegalArgumentException("semantic language is required");
        }

        String nativePath = payload.nativeLibraryDir().getAbsolutePath();
        prependEnvironment("ADSP_LIBRARY_PATH", nativePath, ";");
        prependEnvironment("LD_LIBRARY_PATH", nativePath, ":");

        // Load only host ELFs here; the v81 skeleton is a Hexagon ELF located
        // by FastRPC through ADSP_LIBRARY_PATH.
        System.load(payload.nativeLibrary("libQnnSystem.so").getAbsolutePath());
        System.load(payload.nativeLibrary("libQnnHtpV81Stub.so").getAbsolutePath());
        System.load(payload.nativeLibrary("libQnnHtp.so").getAbsolutePath());

        OrtEnvironment environment = OrtEnvironment.getEnvironment("whisper-ort-qnn-probe");
        String ortVersion = environment.getVersion();
        if (!"1.25.0".equals(ortVersion)) {
            throw new IllegalStateException("unexpected ORT version: " + ortVersion);
        }
        EnumSet<OrtProvider> providers = OrtEnvironment.getAvailableProviders();
        if (!providers.contains(OrtProvider.QNN)) {
            throw new IllegalStateException("pinned ORT AAR does not expose QNNExecutionProvider");
        }

        JSONObject metadata = new JSONObject(
                new String(
                        Files.readAllBytes(payload.modelFile("metadata.json").toPath()),
                        StandardCharsets.UTF_8
                )
        );
        long encoderLoadStart = SystemClock.elapsedRealtimeNanos();
        try (OrtSession encoder = createQnnSession(
                environment,
                payload.modelFile("encoder.onnx"),
                payload.nativeLibrary("libQnnHtp.so")
        )) {
            long encoderLoaded = SystemClock.elapsedRealtimeNanos();
            try (OrtSession decoder = createQnnSession(
                    environment,
                    payload.modelFile("decoder.onnx"),
                    payload.nativeLibrary("libQnnHtp.so")
            )) {
                long decoderLoaded = SystemClock.elapsedRealtimeNanos();
                JSONObject encoderContract = ContractVerifier.verify(
                        encoder, metadata, "encoder.onnx"
                );
                JSONObject decoderContract = ContractVerifier.verify(
                        decoder, metadata, "decoder.onnx"
                );
                SyntheticResult synthetic = runSynthetic(
                        environment, encoder, decoder, payload, bundledEnglishFeatures
                );
                JSONObject semantic;
                if (!semanticDecodeEnabled) {
                    semantic = semanticDisabled();
                } else if (semanticFeatures == null) {
                    semantic = semanticMissingBundledFrontend();
                } else {
                    semantic = runSemantic(
                            environment,
                            encoder,
                            decoder,
                            payload,
                            semanticFeatures,
                            semanticLanguage
                    );
                }
                long completed = SystemClock.elapsedRealtimeNanos();

                JSONArray providerNames = new JSONArray();
                for (OrtProvider provider : providers) {
                    providerNames.put(provider.getName());
                }
                JSONObject report = new JSONObject();
                report.put("schema_version", 1);
                report.put("gate_id", ProbePayload.EXPECTED_GATE_ID);
                report.put("package", "ai.liquid.lfmdemo.whisperprobe");
                report.put("uid", Process.myUid());
                report.put("ort_version", ortVersion);
                report.put("available_providers", providerNames);
                report.put("qnn_provider_requested", true);
                report.put("cpu_ep_fallback_disabled", true);
                report.put("backend_path", payload.nativeLibrary("libQnnHtp.so").getAbsolutePath());
                report.put("model_storage", payload.root().getAbsolutePath());
                report.put("runtime_storage", payload.nativeLibraryDir().getAbsolutePath());
                report.put("host_transport_required", false);
                report.put("real_audio_transcription", false);
                report.put("encoder_session_created", true);
                report.put("decoder_session_created", true);
                report.put("encoder_contract", encoderContract);
                report.put("decoder_contract", decoderContract);
                report.put("synthetic_decode", synthetic.toJson());
                report.put("semantic_decode", semantic);
                report.put("timing_ms", new JSONObject()
                        .put("encoder_session_load", millis(encoderLoaded - encoderLoadStart))
                        .put("decoder_session_load", millis(decoderLoaded - encoderLoaded))
                        .put("encoder_execute", synthetic.encoderExecuteMilliseconds)
                        .put("decoder_execute", synthetic.decoderExecuteMilliseconds)
                        .put("decoder_loop", synthetic.decoderLoopMilliseconds)
                        .put("decoder_step", new JSONArray(synthetic.decoderStepMilliseconds))
                        .put("execute", millis(completed - decoderLoaded))
                        .put("total", millis(completed - probeStart)));
                report.put("status", "compatibility_probe_pass");
                report.put(
                        "claim_boundary",
                        "Session/run success is probe-package feasibility evidence only; it is not "
                                + "final ai.liquid.lfmdemo.runtime integration or real-audio acceptance."
                );
                return report;
            }
        }
    }

    private static OrtSession createQnnSession(
            OrtEnvironment environment,
            File model,
            File backend
    ) throws OrtException {
        try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
            options.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL);
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
            options.setInterOpNumThreads(1);
            options.setIntraOpNumThreads(1);
            options.addConfigEntry("session.disable_cpu_ep_fallback", "1");
            Map<String, String> qnn = new HashMap<>();
            qnn.put("backend_path", backend.getAbsolutePath());
            qnn.put("htp_performance_mode", "burst");
            options.addQnn(qnn);
            return environment.createSession(model.getAbsolutePath(), options);
        }
    }

    private static SyntheticResult runSynthetic(
            OrtEnvironment environment,
            OrtSession encoder,
            OrtSession decoder,
            ProbePayload payload,
            byte[] bundledWavFeatures
    ) throws IOException, OrtException {
        String featureSource;
        OnnxTensor features;
        if (bundledWavFeatures != null) {
            features = loadUInt16Bytes(environment, bundledWavFeatures, FEATURE_SHAPE);
            featureSource = "bundled_question_en_native_frontend";
        } else if (payload.optionalFeatureFixture() != null) {
            File featureFixture = payload.optionalFeatureFixture();
            features = loadUInt16Fixture(environment, featureFixture, FEATURE_SHAPE);
            featureSource = "prequantized_fixture";
        } else {
            features = UnsignedTensorFactory.createUInt16Filled(
                    environment, FEATURE_SHAPE, FEATURE_ZERO_POINT
            );
            featureSource = "synthetic_logical_zero";
        }

        Map<String, OnnxTensor> encoderInputs = new HashMap<>();
        encoderInputs.put("input_features", features);
        try (features) {
            long encoderExecuteStart = SystemClock.elapsedRealtimeNanos();
            try (OrtSession.Result encoderResult = encoder.run(encoderInputs)) {
                long encoderExecuteEnd = SystemClock.elapsedRealtimeNanos();
            List<OnnxTensor> initialSelfCaches = createInitialSelfCaches(environment);
            OrtSession.Result previousDecoderResult = null;
            List<Integer> tokens = new ArrayList<>(DECODER_STEPS);
            List<Integer> unmaskedIndices = new ArrayList<>(DECODER_STEPS);
            List<Double> decoderStepMilliseconds = new ArrayList<>(DECODER_STEPS);
            int inputToken = INITIAL_TOKEN_ID;
            long decoderExecuteStart = SystemClock.elapsedRealtimeNanos();
            try {
                for (int step = 0; step < DECODER_STEPS; step++) {
                    Map<String, OnnxTensor> decoderInputs = new HashMap<>();
                    decoderInputs.put("input_ids", createInt32(environment, inputToken, INPUT_ID_SHAPE));
                    decoderInputs.put("position_ids", createInt32(environment, step, POSITION_ID_SHAPE));
                    int unmaskedIndex = 199 - step;
                    decoderInputs.put(
                            "attention_mask",
                            createAttentionMask(environment, unmaskedIndex)
                    );
                    unmaskedIndices.add(unmaskedIndex);

                    for (int layer = 0; layer < 12; layer++) {
                        decoderInputs.put(
                                "k_cache_cross_" + layer,
                                resultTensor(encoderResult, "k_cache_cross_" + layer)
                        );
                        decoderInputs.put(
                                "v_cache_cross_" + layer,
                                resultTensor(encoderResult, "v_cache_cross_" + layer)
                        );
                        if (previousDecoderResult == null) {
                            decoderInputs.put(
                                    "k_cache_self_" + layer + "_in",
                                    initialSelfCaches.get(layer * 2)
                            );
                            decoderInputs.put(
                                    "v_cache_self_" + layer + "_in",
                                    initialSelfCaches.get(layer * 2 + 1)
                            );
                        } else {
                            decoderInputs.put(
                                    "k_cache_self_" + layer + "_in",
                                    resultTensor(previousDecoderResult, "k_cache_self_" + layer + "_out")
                            );
                            decoderInputs.put(
                                    "v_cache_self_" + layer + "_in",
                                    resultTensor(previousDecoderResult, "v_cache_self_" + layer + "_out")
                            );
                        }
                    }

                    OrtSession.Result current;
                    try {
                        long stepStart = SystemClock.elapsedRealtimeNanos();
                        current = decoder.run(decoderInputs);
                        decoderStepMilliseconds.add(millis(
                                SystemClock.elapsedRealtimeNanos() - stepStart
                        ));
                    } finally {
                        decoderInputs.get("input_ids").close();
                        decoderInputs.get("position_ids").close();
                        decoderInputs.get("attention_mask").close();
                    }
                    int token = argmaxUInt16(resultTensor(current, "logits"));
                    tokens.add(token);
                    inputToken = token;

                    if (previousDecoderResult != null) {
                        previousDecoderResult.close();
                    } else {
                        closeTensors(initialSelfCaches);
                    }
                    previousDecoderResult = current;
                }
            } finally {
                closeTensors(initialSelfCaches);
                if (previousDecoderResult != null) {
                    previousDecoderResult.close();
                }
            }
            long decoderExecuteEnd = SystemClock.elapsedRealtimeNanos();
            VocabDecoder vocab = VocabDecoder.load(payload.voiceAiFile("vocab.bin"));
            return new SyntheticResult(
                    featureSource,
                    tokens,
                    unmaskedIndices,
                    vocab.decode(tokens),
                    millis(encoderExecuteEnd - encoderExecuteStart),
                    sumMilliseconds(decoderStepMilliseconds),
                    millis(decoderExecuteEnd - decoderExecuteStart),
                    decoderStepMilliseconds
            );
            }
        }
    }

    private static JSONObject runSemantic(
            OrtEnvironment environment,
            OrtSession encoder,
            OrtSession decoder,
            ProbePayload payload,
            byte[] bundledWavFeatures,
            WhisperDecodePolicy.Language language
    ) throws IOException, OrtException, JSONException {
        int[] prompt = WhisperDecodePolicy.forcedPrompt(language);
        int maximumGeneratedTokens = WhisperDecodePolicy.maximumGeneratedTokens(prompt.length);
        OnnxTensor features = loadUInt16Bytes(environment, bundledWavFeatures, FEATURE_SHAPE);
        Map<String, OnnxTensor> encoderInputs = new HashMap<>();
        encoderInputs.put("input_features", features);
        try (features) {
            long encoderExecuteStart = SystemClock.elapsedRealtimeNanos();
            try (OrtSession.Result encoderResult = encoder.run(encoderInputs)) {
                long encoderExecuteEnd = SystemClock.elapsedRealtimeNanos();
                List<OnnxTensor> initialSelfCaches = createInitialSelfCaches(environment);
                OrtSession.Result previousDecoderResult = null;
                List<Integer> generatedTokens = new ArrayList<>(maximumGeneratedTokens);
                List<Integer> inputTokens = new ArrayList<>(
                        WhisperDecodePolicy.MAX_DECODER_CALLS
                );
                List<Integer> firstUnmaskedIndices = new ArrayList<>(
                        WhisperDecodePolicy.MAX_DECODER_CALLS
                );
                List<Double> decoderStepMilliseconds = new ArrayList<>(
                        WhisperDecodePolicy.MAX_DECODER_CALLS
                );
                int decoderCalls = 0;
                boolean eotReached = false;
                String terminationReason = "maximum_generated_tokens";
                long decoderExecuteStart = SystemClock.elapsedRealtimeNanos();
                try {
                    for (int step = 0;
                         step < WhisperDecodePolicy.MAX_DECODER_CALLS;
                         step++) {
                        int inputToken = WhisperDecodePolicy.inputTokenForStep(
                                step,
                                prompt,
                                generatedTokens
                        );
                        inputTokens.add(inputToken);
                        int firstUnmasked = WhisperDecodePolicy.firstUnmaskedIndex(step);
                        firstUnmaskedIndices.add(firstUnmasked);

                        Map<String, OnnxTensor> decoderInputs = new HashMap<>();
                        decoderInputs.put(
                                "input_ids",
                                createInt32(environment, inputToken, INPUT_ID_SHAPE)
                        );
                        decoderInputs.put(
                                "position_ids",
                                createInt32(environment, step, POSITION_ID_SHAPE)
                        );
                        decoderInputs.put(
                                "attention_mask",
                                createCumulativeAttentionMask(environment, step)
                        );
                        for (int layer = 0; layer < 12; layer++) {
                            decoderInputs.put(
                                    "k_cache_cross_" + layer,
                                    resultTensor(encoderResult, "k_cache_cross_" + layer)
                            );
                            decoderInputs.put(
                                    "v_cache_cross_" + layer,
                                    resultTensor(encoderResult, "v_cache_cross_" + layer)
                            );
                            if (previousDecoderResult == null) {
                                decoderInputs.put(
                                        "k_cache_self_" + layer + "_in",
                                        initialSelfCaches.get(layer * 2)
                                );
                                decoderInputs.put(
                                        "v_cache_self_" + layer + "_in",
                                        initialSelfCaches.get(layer * 2 + 1)
                                );
                            } else {
                                decoderInputs.put(
                                        "k_cache_self_" + layer + "_in",
                                        resultTensor(
                                                previousDecoderResult,
                                                "k_cache_self_" + layer + "_out"
                                        )
                                );
                                decoderInputs.put(
                                        "v_cache_self_" + layer + "_in",
                                        resultTensor(
                                                previousDecoderResult,
                                                "v_cache_self_" + layer + "_out"
                                        )
                                );
                            }
                        }

                        OrtSession.Result current;
                        try {
                            long stepStart = SystemClock.elapsedRealtimeNanos();
                            current = decoder.run(decoderInputs);
                            decoderStepMilliseconds.add(millis(
                                    SystemClock.elapsedRealtimeNanos() - stepStart
                            ));
                        } finally {
                            decoderInputs.get("input_ids").close();
                            decoderInputs.get("position_ids").close();
                            decoderInputs.get("attention_mask").close();
                        }
                        decoderCalls++;

                        if (WhisperDecodePolicy.shouldSelectGeneratedToken(
                                step,
                                prompt.length
                        )) {
                            int token = argmaxAllowedUInt16(resultTensor(current, "logits"));
                            if (!WhisperDecodePolicy.isAllowedGeneratedToken(token)) {
                                current.close();
                                throw new IllegalStateException(
                                        "semantic policy selected forbidden token: " + token
                                );
                            }
                            generatedTokens.add(token);
                            if (WhisperDecodePolicy.isEndOfTranscript(token)) {
                                eotReached = true;
                                terminationReason = "end_of_transcript";
                            }
                        }

                        if (previousDecoderResult != null) {
                            previousDecoderResult.close();
                        } else {
                            closeTensors(initialSelfCaches);
                        }
                        previousDecoderResult = current;
                        if (eotReached) {
                            break;
                        }
                    }
                } finally {
                    closeTensors(initialSelfCaches);
                    if (previousDecoderResult != null) {
                        previousDecoderResult.close();
                    }
                }
                long decoderExecuteEnd = SystemClock.elapsedRealtimeNanos();

                if (generatedTokens.size() > maximumGeneratedTokens) {
                    throw new IllegalStateException("semantic decoder exceeded token budget");
                }
                int expectedDecoderCalls = WhisperDecodePolicy.expectedDecoderCalls(
                        prompt.length,
                        generatedTokens.size()
                );
                if (decoderCalls != expectedDecoderCalls) {
                    throw new IllegalStateException(
                            "semantic decoder call/token invariant failed: calls="
                                    + decoderCalls
                                    + " expected="
                                    + expectedDecoderCalls
                    );
                }
                if (!eotReached && generatedTokens.size() != maximumGeneratedTokens) {
                    throw new IllegalStateException(
                            "semantic decoder stopped before EOT or maximum token budget"
                    );
                }

                VocabDecoder vocab = VocabDecoder.load(payload.voiceAiFile("vocab.bin"));
                JSONObject value = new JSONObject();
                value.put("enabled", true);
                value.put(
                        "status",
                        eotReached ? "candidate_decode_eot" : "candidate_decode_max_tokens"
                );
                value.put(
                        "feature_source",
                        "bundled_question_" + language.code() + "_native_frontend"
                );
                value.put("language_code", language.code());
                value.put("language_token_id", language.tokenId());
                value.put("prompt_token_ids", new JSONArray(toIntegerList(prompt)));
                value.put("decoder_calls", decoderCalls);
                value.put("decoder_calls_expected", expectedDecoderCalls);
                value.put("generated_token_count", generatedTokens.size());
                value.put("generated_token_ids", new JSONArray(generatedTokens));
                value.put("generated_token_ids_include_eot", true);
                value.put("eot_reached", eotReached);
                value.put("termination_reason", terminationReason);
                value.put("maximum_generated_tokens", maximumGeneratedTokens);
                value.put("candidate_transcript", vocab.decode(generatedTokens));
                value.put("candidate_transcript_skips_eot", true);
                value.put("input_token_ids", new JSONArray(inputTokens));
                value.put("attention_mask_mode", "cumulative_suffix_uint16");
                value.put(
                        "attention_first_unmasked_indices",
                        new JSONArray(firstUnmaskedIndices)
                );
                value.put("allowed_generated_token_min", 0);
                value.put(
                        "allowed_generated_token_max",
                        WhisperDecodePolicy.EOT_TOKEN_ID
                );
                value.put(
                        "suppressed_generated_token_min",
                        WhisperDecodePolicy.FIRST_SUPPRESSED_TOKEN_ID
                );
                value.put(
                        "suppressed_generated_token_max",
                        WhisperDecodePolicy.LAST_SUPPRESSED_TOKEN_ID
                );
                value.put("encoder_execute_ms", millis(
                        encoderExecuteEnd - encoderExecuteStart
                ));
                value.put("decoder_execute_ms", sumMilliseconds(decoderStepMilliseconds));
                value.put("decoder_loop_ms", millis(
                        decoderExecuteEnd - decoderExecuteStart
                ));
                value.put("decoder_step_ms", new JSONArray(decoderStepMilliseconds));
                value.put("semantic_transcript_claimed", false);
                value.put("real_audio_transcription", false);
                return value;
            }
        }
    }

    private static JSONObject semanticDisabled() throws JSONException {
        return new JSONObject()
                .put("enabled", false)
                .put("status", "disabled_by_build")
                .put("semantic_transcript_claimed", false)
                .put("real_audio_transcription", false);
    }

    private static JSONObject semanticMissingBundledFrontend() throws JSONException {
        return new JSONObject()
                .put("enabled", true)
                .put("status", "failed_closed_missing_bundled_frontend")
                .put("semantic_transcript_claimed", false)
                .put("real_audio_transcription", false);
    }

    private static List<Integer> toIntegerList(int[] values) {
        List<Integer> result = new ArrayList<>(values.length);
        for (int value : values) {
            result.add(value);
        }
        return result;
    }

    private static OnnxTensor loadUInt16Bytes(
            OrtEnvironment environment,
            byte[] encoded,
            long[] shape
    ) throws IOException, OrtException {
        int expectedBytes = FEATURE_ELEMENTS * 2;
        if (encoded.length != expectedBytes) {
            throw new IOException(
                    "native frontend tensor must be exactly " + expectedBytes + " bytes"
            );
        }
        ByteBuffer bytes = ByteBuffer
                .allocateDirect(expectedBytes)
                .order(ByteOrder.LITTLE_ENDIAN);
        bytes.put(encoded);
        bytes.flip();
        return UnsignedTensorFactory.createUInt16(environment, bytes.asShortBuffer(), shape);
    }

    private static OnnxTensor loadUInt16Fixture(
            OrtEnvironment environment,
            File fixture,
            long[] shape
    ) throws IOException, OrtException {
        int expectedBytes = FEATURE_ELEMENTS * 2;
        if (fixture.length() != expectedBytes) {
            throw new IOException("prequantized feature fixture must be exactly " + expectedBytes + " bytes");
        }
        ByteBuffer bytes = ByteBuffer.allocateDirect(expectedBytes).order(ByteOrder.LITTLE_ENDIAN);
        try (FileInputStream input = new FileInputStream(fixture);
             FileChannel channel = input.getChannel()) {
            while (bytes.hasRemaining()) {
                if (channel.read(bytes) < 0) {
                    throw new IOException("truncated prequantized feature fixture");
                }
            }
        }
        bytes.flip();
        ShortBuffer values = bytes.asShortBuffer();
        return UnsignedTensorFactory.createUInt16(environment, values, shape);
    }

    private static OnnxTensor createAttentionMask(
            OrtEnvironment environment,
            int unmaskedIndex
    ) throws OrtException {
        ShortBuffer mask = ByteBuffer
                .allocateDirect(200 * 2)
                .order(ByteOrder.nativeOrder())
                .asShortBuffer();
        while (mask.hasRemaining()) {
            mask.put((short) 0);
        }
        mask.put(unmaskedIndex, (short) 0xffff);
        mask.position(0);
        return UnsignedTensorFactory.createUInt16(environment, mask, ATTENTION_MASK_SHAPE);
    }

    private static OnnxTensor createCumulativeAttentionMask(
            OrtEnvironment environment,
            int decoderStep
    ) throws OrtException {
        short[] encoded = WhisperDecodePolicy.cumulativeAttentionMask(decoderStep);
        ShortBuffer mask = ByteBuffer
                .allocateDirect(encoded.length * 2)
                .order(ByteOrder.nativeOrder())
                .asShortBuffer();
        mask.put(encoded);
        mask.flip();
        return UnsignedTensorFactory.createUInt16(environment, mask, ATTENTION_MASK_SHAPE);
    }

    private static OnnxTensor createInt32(
            OrtEnvironment environment,
            int value,
            long[] shape
    ) throws OrtException {
        IntBuffer values = ByteBuffer
                .allocateDirect(4)
                .order(ByteOrder.nativeOrder())
                .asIntBuffer();
        values.put(value);
        values.flip();
        return OnnxTensor.createTensor(environment, values, shape);
    }

    private static List<OnnxTensor> createInitialSelfCaches(OrtEnvironment environment)
            throws OrtException {
        List<OnnxTensor> tensors = new ArrayList<>(24);
        try {
            for (int layer = 0; layer < 12; layer++) {
                tensors.add(createUInt8Filled(environment, SELF_K_SHAPE, CACHE_ZERO_POINT));
                tensors.add(createUInt8Filled(environment, SELF_V_SHAPE, CACHE_ZERO_POINT));
            }
            return tensors;
        } catch (OrtException | RuntimeException | Error error) {
            closeTensors(tensors);
            throw error;
        }
    }

    private static OnnxTensor createUInt8Filled(
            OrtEnvironment environment,
            long[] shape,
            int value
    ) throws OrtException {
        int elements = Math.toIntExact(elementCount(shape));
        ByteBuffer bytes = ByteBuffer.allocateDirect(elements).order(ByteOrder.nativeOrder());
        byte encoded = (byte) value;
        while (bytes.hasRemaining()) {
            bytes.put(encoded);
        }
        bytes.flip();
        return OnnxTensor.createTensor(environment, bytes, shape, OnnxJavaType.UINT8);
    }

    private static OnnxTensor resultTensor(OrtSession.Result result, String name) {
        OnnxValue value = result.get(name).orElseThrow(
                () -> new IllegalStateException("missing ORT output: " + name)
        );
        if (!(value instanceof OnnxTensor)) {
            throw new IllegalStateException("ORT output is not a tensor: " + name);
        }
        return (OnnxTensor) value;
    }

    private static int argmaxUInt16(OnnxTensor logits) {
        ByteBuffer bytes = logits.getByteBuffer().order(ByteOrder.nativeOrder());
        ShortBuffer values = bytes.asShortBuffer();
        if (values.remaining() != LOGIT_COUNT) {
            throw new IllegalStateException(
                    "logits has " + values.remaining() + " values; expected " + LOGIT_COUNT
            );
        }
        int bestIndex = 0;
        int bestValue = -1;
        for (int index = 0; index < LOGIT_COUNT; index++) {
            int value = values.get(index) & 0xffff;
            if (value > bestValue) {
                bestValue = value;
                bestIndex = index;
            }
        }
        return bestIndex;
    }

    private static int argmaxAllowedUInt16(OnnxTensor logits) {
        ByteBuffer bytes = logits.getByteBuffer().order(ByteOrder.nativeOrder());
        return WhisperDecodePolicy.selectAllowedToken(bytes.asShortBuffer());
    }

    private static long elementCount(long[] shape) {
        long count = 1;
        for (long dimension : shape) {
            count = Math.multiplyExact(count, dimension);
        }
        return count;
    }

    private static void closeTensors(List<OnnxTensor> tensors) {
        for (OnnxTensor tensor : tensors) {
            if (tensor != null && !tensor.isClosed()) {
                tensor.close();
            }
        }
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

    private static double sumMilliseconds(List<Double> values) {
        double total = 0.0;
        for (double value : values) {
            total += value;
        }
        return Math.round(total * 1000.0) / 1000.0;
    }

    private static final class SyntheticResult {
        private final String featureSource;
        private final List<Integer> tokens;
        private final List<Integer> unmaskedIndices;
        private final String decodedBaseText;
        private final double encoderExecuteMilliseconds;
        private final double decoderExecuteMilliseconds;
        private final double decoderLoopMilliseconds;
        private final List<Double> decoderStepMilliseconds;

        SyntheticResult(
                String featureSource,
                List<Integer> tokens,
                List<Integer> unmaskedIndices,
                String decodedBaseText,
                double encoderExecuteMilliseconds,
                double decoderExecuteMilliseconds,
                double decoderLoopMilliseconds,
                List<Double> decoderStepMilliseconds
        ) {
            this.featureSource = featureSource;
            this.tokens = tokens;
            this.unmaskedIndices = unmaskedIndices;
            this.decodedBaseText = decodedBaseText;
            this.encoderExecuteMilliseconds = encoderExecuteMilliseconds;
            this.decoderExecuteMilliseconds = decoderExecuteMilliseconds;
            this.decoderLoopMilliseconds = decoderLoopMilliseconds;
            this.decoderStepMilliseconds = decoderStepMilliseconds;
        }

        JSONObject toJson() throws JSONException {
            JSONObject value = new JSONObject();
            value.put("feature_source", featureSource);
            value.put("feature_dtype", "uint16");
            value.put("feature_zero_point", FEATURE_ZERO_POINT);
            value.put("encoder_execute_passed", true);
            value.put("decoder_execute_passed", true);
            value.put("decoder_steps", tokens.size());
            value.put("token_ids", new JSONArray(tokens));
            value.put("first_token_id", tokens.isEmpty() ? JSONObject.NULL : tokens.get(0));
            value.put("attention_unmasked_indices", new JSONArray(unmaskedIndices));
            value.put("decoded_base_text", decodedBaseText);
            value.put("encoder_execute_ms", encoderExecuteMilliseconds);
            value.put("decoder_execute_ms", decoderExecuteMilliseconds);
            value.put("decoder_loop_ms", decoderLoopMilliseconds);
            value.put("decoder_step_ms", new JSONArray(decoderStepMilliseconds));
            value.put("semantic_transcript_claimed", false);
            return value;
        }
    }
}
