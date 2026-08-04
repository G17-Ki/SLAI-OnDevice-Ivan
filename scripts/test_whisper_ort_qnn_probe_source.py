#!/usr/bin/env python3
"""Static fail-closed gates for the isolated Whisper ORT-QNN probe source."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "android-app/whisper-probe"


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def main() -> None:
    settings = read("android-app/settings.gradle")
    gradle = read("android-app/whisper-probe/build.gradle")
    manifest = read("android-app/whisper-probe/src/main/AndroidManifest.xml")
    unsigned = read(
        "android-app/whisper-probe/src/main/java/ai/onnxruntime/UnsignedTensorFactory.java"
    )
    runner = read(
        "android-app/whisper-probe/src/main/java/ai/liquid/lfmdemo/whisperprobe/"
        "WhisperQnnProbe.java"
    )
    payload = read(
        "android-app/whisper-probe/src/main/java/ai/liquid/lfmdemo/whisperprobe/"
        "ProbePayload.java"
    )
    activity = read(
        "android-app/whisper-probe/src/main/java/ai/liquid/lfmdemo/whisperprobe/"
        "ProbeActivity.java"
    )
    vocab = read(
        "android-app/whisper-probe/src/main/java/ai/liquid/lfmdemo/whisperprobe/"
        "VocabDecoder.java"
    )
    decode_policy = read(
        "android-app/whisper-probe/src/main/java/ai/liquid/lfmdemo/whisperprobe/"
        "WhisperDecodePolicy.java"
    )
    stager = read("scripts/stage_whisper_ort_qnn_probe.py")
    frontend_stager = read("scripts/stage_whisper_android_frontend_assets.py")
    frontend = read(
        "android-app/whisper-probe/src/main/java/ai/liquid/lfmdemo/whisperprobe/"
        "WhisperFrontendProbe.java"
    )
    frontend_core = read(
        "android-app/whisper-probe/src/main/cpp/whisper_frontend_core.c"
    )
    frontend_jni = read("android-app/whisper-probe/src/main/cpp/whisper_frontend_jni.c")

    assert 'include(":whisper-probe")' in settings
    assert 'applicationId "ai.liquid.lfmdemo.whisperprobe"' in gradle
    assert "397277c3e50326f5972409ca42ab5b4e69172b218557923630d68e55b11d6198" in gradle
    assert "implementation files(ortAar)" in gradle
    assert "WHISPER_PROBE_PAYLOAD_DIR" in gradle
    assert 'keepDebugSymbols += ["**/*.so"]' in gradle
    assert "WHISPER_FRONTEND_ASSET_DIR" in gradle
    assert 'System.getenv("WHISPER_FRONTEND_PROFILE") ?: "v1"' in gradle
    assert '["en", "ja"]' in gradle
    assert '["en", "zh", "ru", "ko", "ja"]' in gradle
    assert 'buildConfigField "String", "WHISPER_FRONTEND_PROFILE"' in gradle
    assert 'System.getenv("WHISPER_SEMANTIC_DECODE_ENABLED") ?: "false"' in gradle
    assert 'buildConfigField "boolean", "WHISPER_SEMANTIC_DECODE_ENABLED"' in gradle
    assert 'System.getenv("WHISPER_SEMANTIC_LANGUAGE") ?: "en"' in gradle
    assert 'buildConfigField "String", "WHISPER_SEMANTIC_LANGUAGE"' in gradle
    assert "-fno-fast-math" in gradle and "-ffp-contract=off" in gradle

    assert "android.permission.INTERNET" not in manifest
    assert "android.permission.RECORD_AUDIO" not in manifest
    assert 'android:name="libcdsprpc.so"' in manifest
    assert "REAL_AUDIO_ENABLED" in manifest and 'android:value="false"' in manifest

    assert (
        'getDeclaredMethod(\n                    "createTensorFromBuffer"' in unsigned
    )
    assert "ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16.value" in unsigned
    assert "OnnxJavaType.UNKNOWN" in unsigned
    assert "createTensor(environment, values, shape)" not in unsigned

    assert 'prependEnvironment("ADSP_LIBRARY_PATH", nativePath, ";")' in runner
    assert runner.index('prependEnvironment("ADSP_LIBRARY_PATH"') < runner.index(
        'System.load(payload.nativeLibrary("libQnnHtp.so")'
    )
    assert 'options.addConfigEntry("session.disable_cpu_ep_fallback", "1")' in runner
    assert "options.addQnn(qnn)" in runner
    assert 'qnn.put("backend_path", backend.getAbsolutePath())' in runner
    assert "FEATURE_ZERO_POINT = 32072" in runner
    assert "CACHE_ZERO_POINT = 128" in runner
    assert "int unmaskedIndex = 199 - step" in runner
    assert "DECODER_STEPS = 4" in runner
    assert "argmaxUInt16" in runner
    assert "runSemantic(" in runner
    assert "createCumulativeAttentionMask" in runner
    assert "argmaxAllowedUInt16" in runner
    assert 'value.put("semantic_transcript_claimed", false)' in runner
    assert 'value.put("real_audio_transcription", false)' in runner
    assert '"candidate_decode_eot" : "candidate_decode_max_tokens"' in runner
    assert 'value.put("attention_mask_mode", "cumulative_suffix_uint16")' in runner
    assert 'value.put("decoder_calls_expected", expectedDecoderCalls)' in runner
    assert 'value.put("generated_token_ids_include_eot", true)' in runner
    assert 'value.put("candidate_transcript_skips_eot", true)' in runner
    assert "new JSONArray(firstUnmaskedIndices)" in runner
    assert '"bundled_question_" + language.code() + "_native_frontend"' in runner
    assert 'semantic_transcript_claimed", false' in runner
    assert 'featureSource = "bundled_question_en_native_frontend"' in runner
    assert 'put("encoder_execute_ms"' in runner
    assert 'put("decoder_execute_ms"' in runner
    assert 'put("decoder_loop_ms"' in runner
    assert 'put("decoder_step_ms"' in runner

    assert 'capabilities.getBoolean("real_audio_frontend")' in payload
    assert 'capabilities.getBoolean("real_audio_transcription")' in payload
    assert "scaffold manifest must fail closed for real audio" in payload

    assert 'EXTRA_RUN_PROBE = "run_probe"' in activity
    assert 'EXTRA_RUN_ID = "run_id"' in activity
    assert (
        "state == null && getIntent().getBooleanExtra(EXTRA_RUN_PROBE, false)"
        in activity
    )
    assert "getIntent().removeExtra(EXTRA_RUN_PROBE)" in activity
    assert "getIntent().removeExtra(EXTRA_RUN_ID)" in activity
    assert "runButton.post(() -> runProbe(runButton, run))" in activity
    assert "BuildConfig.WHISPER_SEMANTIC_DECODE_ENABLED" in activity
    assert "BuildConfig.WHISPER_SEMANTIC_LANGUAGE" in activity
    assert "frontend.featuresForLanguage(semanticLanguage.code())" in activity
    assert "isValidRunId(automationRunId)" in activity
    persist = activity.split("private File persistReport", 1)[1]
    assert 'new File(reportDir, "latest.json.partial")' in persist
    assert "output.getFD().sync()" in persist
    assert "StandardCopyOption.ATOMIC_MOVE" in persist
    assert 'report.put("run_id", run.runId)' in persist
    assert 'report.put("report_complete", true)' in persist
    assert 'report.put("completion_marker", REPORT_COMPLETION_MARKER)' in persist
    assert 'report.put("started_at_utc"' in persist
    assert 'report.put("completed_at_utc"' in persist
    assert (
        'report.put("elapsed_wall_ms", completedAtUnixMs - run.startedAtUnixMs)'
        in persist
    )
    assert 'report.put(\n                "elapsed_monotonic_ms"' in persist
    assert 'report.put("report_path", reportFile.getAbsolutePath())' in persist
    assert persist.index('report.put("report_path"') < persist.index(
        "report.toString(2)"
    )

    assert "RAW_NUL_TOKEN_ID = 188" in vocab
    assert "EMPTY_TOKEN_ID = 50256" in vocab
    assert "FIRST_SPECIAL_TOKEN_ID = 50257" in vocab
    assert "joined.write(0)" in vocab
    assert "CodingErrorAction.REPORT" in vocab
    assert "decodeUtf8Strict(joined.toByteArray())" in vocab
    assert "codePoint == 0xfffd" in vocab
    assert "Character.isISOControl(codePoint)" in vocab

    assert "MAX_DECODER_CALLS = DECODER_CONTEXT_LENGTH - 1" in decode_policy
    assert 'ENGLISH("en", 50259)' in decode_policy
    assert 'CHINESE("zh", 50260)' in decode_policy
    assert 'RUSSIAN("ru", 50263)' in decode_policy
    assert 'KOREAN("ko", 50264)' in decode_policy
    assert 'JAPANESE("ja", 50266)' in decode_policy
    assert "TRANSCRIBE_TOKEN_ID = 50359" in decode_policy
    assert "NO_TIMESTAMPS_TOKEN_ID = 50363" in decode_policy
    assert "return DECODER_CONTEXT_LENGTH - promptLength" in decode_policy
    assert "return promptLength - 1 + generatedTokenCount" in decode_policy
    assert "cumulativeAttentionMask" in decode_policy
    assert "tokenId <= EOT_TOKEN_ID" in decode_policy

    assert '"real_audio_frontend": False' in stager
    assert '"real_audio_transcription": False' in stager
    assert '"attention_unmask_order": [199, 198, 197, 196]' in stager
    assert "require_external_output(output)" in stager
    assert "adb" not in stager.lower()

    assert (
        "8cf37ba9a60f5cd91453dae2d721e9fd7fec69919582223bf02a3f76f8421a80" in frontend
    )
    assert (
        "6d18da5e1ec42bbd8161881a598e835c365138377b3d42364ff360e4849d4263" in frontend
    )
    assert (
        "808b38d3c4198c15566116562eb7ad61fbafc01912a747ae439047fb9d7eb1c5" in frontend
    )
    assert (
        "0e0c25a24157edabe3c1d193e60e8b59886aa9b2ba9b1581fb5c90c2eaf3bbc4" in frontend
    )
    assert "final Map<String, byte[]> featuresByLanguage" in frontend
    assert "final byte[] japaneseFeatures" in frontend
    assert "featuresByLanguage.containsKey(normalized)" in frontend
    assert 'languages.add("en")' in frontend
    assert "languages.add(selectedLanguage)" in frontend
    assert 'report.put("produced_tensor_sha256"' in frontend
    assert 'report.put("differing_uint16_values"' in frontend
    assert 'report.put("max_uint16_delta"' in frontend
    assert "boundedGatePassed ? english.output : null" in frontend
    assert "torch_reflect_index" in frontend_core
    assert "nearbyint((double)normalized" in frontend_core
    assert "pthread_once(&g_frontend_once" in frontend_jni
    assert "05ec7e5f" not in frontend + frontend_stager + frontend_core + frontend_jni
    assert "adb" not in frontend_stager.lower()

    java_sources = "\n".join(
        path.read_text(encoding="utf-8") for path in MODULE.rglob("*.java")
    )
    assert "package ai.liquid.lfmdemo.runtime" not in java_sources
    assert "RuntimeService.class" not in java_sources
    assert "ai.liquid.lfmdemo.MainActivity" not in java_sources
    print("whisper ORT-QNN probe source gates passed")


if __name__ == "__main__":
    main()
