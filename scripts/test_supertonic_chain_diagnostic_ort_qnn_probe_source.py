#!/usr/bin/env python3
"""Static fail-closed gates for the Supertonic stage-diagnostic probe."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "android-app/supertonic-chain-diagnostic-probe"
PACKAGE = "ai.liquid.lfmdemo.supertonicchaindiagprobe"


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def method(source: str, start: str, end: str) -> str:
    return source.split(start, 1)[1].split(end, 1)[0]


def require_order(source: str, *needles: str) -> None:
    offsets = [source.index(needle) for needle in needles]
    assert offsets == sorted(offsets), (needles, offsets)


def main() -> None:
    settings = read("android-app/settings.gradle")
    gradle = read("android-app/supertonic-chain-diagnostic-probe/build.gradle")
    manifest = read(
        "android-app/supertonic-chain-diagnostic-probe/src/main/AndroidManifest.xml"
    )
    payload = read(
        "android-app/supertonic-chain-diagnostic-probe/src/main/java/ai/liquid/"
        "lfmdemo/supertonicchaindiagprobe/SupertonicChainDiagnosticPayload.java"
    )
    runner = read(
        "android-app/supertonic-chain-diagnostic-probe/src/main/java/ai/liquid/"
        "lfmdemo/supertonicchaindiagprobe/SupertonicChainDiagnosticQnnProbe.java"
    )
    activity = read(
        "android-app/supertonic-chain-diagnostic-probe/src/main/java/ai/liquid/"
        "lfmdemo/supertonicchaindiagprobe/ProbeActivity.java"
    )

    assert 'include(":supertonic-chain-diagnostic-probe")' in settings
    assert f'expectedPackage = "{PACKAGE}"' in gradle
    assert "SUPERTONIC_CHAIN_DIAGNOSTIC_PROBE_PAYLOAD_DIR" in gradle
    assert "0.2.0-stage-diagnostic-raw-onnx-qnn-probe" in gradle
    assert "cpu_ep_fallback_remains_disabled: true" in gradle
    assert "failure_stage_recorded: true" in gradle
    assert "failure_phase_recorded: true" in gradle
    assert "promotion_allowed: false" in gradle
    assert "735e61b070c7ad199a67bc18829036f4575551b5533a49e9ead5b36392f0819f" in gradle

    assert "android.permission.INTERNET" not in manifest
    assert "android.permission.RECORD_AUDIO" not in manifest
    assert "READ_EXTERNAL_STORAGE" not in manifest
    assert 'android:usesCleartextTraffic="false"' in manifest
    assert 'android:name="libcdsprpc.so"' in manifest
    assert 'android:required="false"' in manifest

    assert "payload-v2" in payload
    assert "diagnostic_policy" in payload
    assert "source_payload_manifest" in payload
    assert "StandardCopyOption.ATOMIC_MOVE" in payload
    assert "output.getFD().sync()" in payload

    assert 'options.addConfigEntry("session.disable_cpu_ep_fallback", "1")' in runner
    assert 'qnn.put("htp_performance_mode", "burst")' in runner
    assert 'qnn.put("enable_htp_fp16_precision", "1")' in runner
    assert 'DIAGNOSTIC_MARKER =\n            "SUPERTONIC_CHAIN_STAGE_DIAGNOSTIC_V2"' in runner
    assert 'progress.enterStage("runtime_initialization", "native_library_load")' in runner
    assert 'progress.enterStage("runtime_initialization", "ort_environment_create")' in runner
    assert 'progress.enterStage("runtime_initialization", "provider_validation")' in runner
    assert 'progress.enterStage("runtime_initialization", "fixture_load")' in runner
    assert 'progress.completeStage(\n                "runtime_initialization"' in runner

    stages = {
        "duration_predictor": method(runner, "private static StageResult runDuration(",
                                     "private static StageResult runTextEncoder("),
        "text_encoder": method(runner, "private static StageResult runTextEncoder(",
                                "private static VectorResult runVectorEstimator("),
        "vector_estimator": method(runner, "private static VectorResult runVectorEstimator(",
                                    "private static StageResult runVocoder("),
        "vocoder": method(runner, "private static StageResult runVocoder(",
                           "private static OrtSession.SessionOptions sessionOptions("),
    }
    for stage, source in stages.items():
        require_order(
            source,
            f'progress.enterStage("{stage}", "session_create")',
            "environment.createSession(",
            f'progress.markSessionCreated("{stage}")',
            f'progress.enterStage("{stage}", "contract_validation")',
            "requireContract(session, inputs, outputs)",
        )
    for stage in ("duration_predictor", "text_encoder", "vocoder"):
        source = stages[stage]
        require_order(
            source,
            f'progress.enterStage("{stage}", "execute")',
            "session.run(values)",
            f'progress.enterStage("{stage}", "output_validation")',
            "floatOutput(",
        )
    vector = stages["vector_estimator"]
    require_order(
        vector,
        'progress.enterVectorStep("execute", step)',
        "session.run(values)",
        'progress.enterVectorStep("output_validation", step)',
        "floatOutput(",
        "progress.completeVectorStep(step, stepReport)",
    )
    for stage in (
        "duration_predictor",
        "text_encoder",
        "vector_estimator",
        "vocoder",
    ):
        assert f'progress.completeStage("{stage}"' in runner

    assert 'progress.enterStage("payload_install", "install_and_hash")' in activity
    require_order(
        activity,
        'progress.enterStage("payload_install", "install_and_hash")',
        "SupertonicChainDiagnosticPayload.installAndVerify(this)",
        'progress.completeStage(\n                        "payload_install"',
    )
    assert 'failure.put("status", "failed_closed_stage_diagnostic")' in activity
    assert 'failure.put("cpu_ep_fallback_disabled", true)' in activity
    require_order(
        activity,
        "progress.appendFailureEvidence(failure)",
        "persistReport(failure, run)",
    )
    assert 'report.put("failure_stage", currentStage)' in runner
    assert 'report.put("failure_phase", currentPhase)' in runner
    assert 'report.put("failure_vector_step", currentVectorStep)' in runner
    assert 'report.put("completed_stages", copy(completedStages))' in runner
    assert 'report.put("completed_stage_reports", copy(completedStageReports))' in runner
    assert 'report.put("sessions_created", copy(sessionsCreated))' in runner
    assert '"completed_vector_step_reports"' in runner

    persist = activity.split("private File persistReport", 1)[1]
    assert 'new File(reportDir, "latest.json.partial")' in persist
    assert "output.getFD().sync()" in persist
    assert "StandardCopyOption.ATOMIC_MOVE" in persist
    assert 'report.put("run_id", run.runId)' in persist
    assert 'report.put("report_complete", true)' in persist
    assert 'report.put("completion_marker", REPORT_COMPLETION_MARKER)' in persist

    assert 'report.put("full_tts", false)' in runner
    assert 'report.put("audio_quality_accepted", false)' in runner
    assert 'report.put("production_integration", false)' in runner
    assert 'failure.put("full_tts", false)' in activity
    assert 'failure.put("audio_quality_accepted", false)' in activity
    assert 'failure.put("production_integration", false)' in activity

    sources = "\n".join(
        path.read_text(encoding="utf-8") for path in MODULE.rglob("*.java")
    )
    assert "package ai.liquid.lfmdemo.supertonicchainprobe" not in sources
    assert "ai.liquid.lfmdemo.MainActivity" not in sources
    assert "RuntimeService" not in sources
    assert "adb" not in sources.lower()
    print("Supertonic chain diagnostic source gates passed")


if __name__ == "__main__":
    main()
