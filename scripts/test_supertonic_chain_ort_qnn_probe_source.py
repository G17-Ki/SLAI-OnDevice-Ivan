#!/usr/bin/env python3
"""Static claim-boundary gates for the isolated Supertonic chain probe."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "android-app/supertonic-chain-probe"


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def main() -> None:
    settings = read("android-app/settings.gradle")
    gradle = read("android-app/supertonic-chain-probe/build.gradle")
    manifest = read("android-app/supertonic-chain-probe/src/main/AndroidManifest.xml")
    payload = read(
        "android-app/supertonic-chain-probe/src/main/java/ai/liquid/lfmdemo/"
        "supertonicchainprobe/SupertonicChainPayload.java"
    )
    runner = read(
        "android-app/supertonic-chain-probe/src/main/java/ai/liquid/lfmdemo/"
        "supertonicchainprobe/SupertonicChainQnnProbe.java"
    )
    activity = read(
        "android-app/supertonic-chain-probe/src/main/java/ai/liquid/lfmdemo/"
        "supertonicchainprobe/ProbeActivity.java"
    )

    assert 'include(":supertonic-chain-probe")' in settings
    assert 'expectedPackage = "ai.liquid.lfmdemo.supertonicchainprobe"' in gradle
    assert "SUPERTONIC_CHAIN_PROBE_PAYLOAD_DIR" in gradle
    assert "applicationId expectedPackage" in gradle
    assert "implementation files(ortAar)" in gradle
    assert "397277c3e50326f5972409ca42ab5b4e69172b218557923630d68e55b11d6198" in gradle
    assert 'keepDebugSymbols += ["**/*.so"]' in gradle
    assert "host_reference_comparisons_promoted != false" in gradle
    assert "require_non_silent_waveform != true" in gradle

    assert "android.permission.INTERNET" not in manifest
    assert "android.permission.RECORD_AUDIO" not in manifest
    assert "READ_EXTERNAL_STORAGE" not in manifest
    assert 'android:usesCleartextTraffic="false"' in manifest
    assert 'android:name="libcdsprpc.so"' in manifest
    assert 'android:required="false"' in manifest

    assert "require_all_four_sessions" in payload
    assert "require_chained_outputs" in payload
    assert "host_reference_comparisons_recorded" in payload
    assert "host_reference_comparisons_promoted" in payload
    assert "audio_quality_promoted" in payload
    assert "StandardCopyOption.ATOMIC_MOVE" in payload
    assert "output.getFD().sync()" in payload

    assert 'prependEnvironment("ADSP_LIBRARY_PATH", nativePath, ";")' in runner
    assert runner.index('prependEnvironment("ADSP_LIBRARY_PATH"') < runner.index(
        'System.load(payload.nativeLibrary("libQnnHtp.so")'
    )
    assert 'options.addConfigEntry("session.disable_cpu_ep_fallback", "1")' in runner
    assert 'qnn.put(\n                "backend_path"' in runner
    assert 'qnn.put("htp_performance_mode", "burst")' in runner
    assert 'qnn.put("enable_htp_fp16_precision", "1")' in runner
    assert "VECTOR_STEPS = 4" in runner
    assert "for (int step = 0; step < VECTOR_STEPS; step++)" in runner
    assert "applyLatentMask(raw, latentMask)" in runner
    assert "requireZeroPadding(raw, EXPECTED_REAL_LATENT_LENGTH)" in runner
    assert 'values.put("text_emb", textTensor)' in runner
    assert 'values.put("latent", latentTensor)' in runner
    assert '.put("diagnostic_only", true)' in runner
    assert '.put("promoted", false)' in runner
    assert "boolean promoted" not in runner
    assert "duration numerical gate failed" not in runner
    assert 'report.put("full_tts", false)' in runner
    assert 'report.put("audio_quality_accepted", false)' in runner
    assert 'report.put("production_integration", false)' in runner

    assert 'EXTRA_RUN_ID = "run_id"' in activity
    assert "isValidRunId(runId)" in activity
    persist = activity.split("private File persistReport", 1)[1]
    assert 'new File(reportDir, "latest.json.partial")' in persist
    assert "output.getFD().sync()" in persist
    assert "StandardCopyOption.ATOMIC_MOVE" in persist
    assert 'report.put("run_id", run.runId)' in persist
    assert 'report.put("report_complete", true)' in persist
    assert 'report.put("completion_marker", REPORT_COMPLETION_MARKER)' in persist
    assert 'report.put("started_at_utc"' in persist
    assert 'report.put("completed_at_utc"' in persist
    assert 'report.put("elapsed_wall_ms"' in persist
    assert '"elapsed_monotonic_ms"' in persist

    sources = "\n".join(
        path.read_text(encoding="utf-8") for path in MODULE.rglob("*.java")
    )
    assert "package ai.liquid.lfmdemo.runtime" not in sources
    assert "ai.liquid.lfmdemo.MainActivity" not in sources
    assert "RuntimeService" not in sources
    assert "adb" not in sources.lower()
    print("Supertonic chain ORT-QNN source gates passed")


if __name__ == "__main__":
    main()
