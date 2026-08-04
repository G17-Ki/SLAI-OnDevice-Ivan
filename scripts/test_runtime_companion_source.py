#!/usr/bin/env python3
"""Lightweight lifecycle gates for the app-owned runtime companion."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVICE = ROOT / (
    "android-app/runtime-companion/src/main/java/ai/liquid/lfmdemo/runtime/"
    "RuntimeService.java"
)
MANIFEST = ROOT / "android-app/runtime-companion/src/main/AndroidManifest.xml"
MAIN_ACTIVITY = ROOT / (
    "android-app/app/src/main/java/ai/liquid/lfmdemo/MainActivity.java"
)
BOOTSTRAP_ACTIVITY = ROOT / (
    "android-app/runtime-companion/src/main/java/ai/liquid/lfmdemo/runtime/"
    "RuntimeBootstrapActivity.java"
)
BOOTSTRAP_PROVIDER = ROOT / (
    "android-app/runtime-companion/src/main/java/ai/liquid/lfmdemo/runtime/"
    "RuntimeBootstrapProvider.java"
)
ASR_SERVER = ROOT / (
    "android-app/runtime-companion/src/main/java/ai/liquid/lfmdemo/runtime/"
    "AsrHttpServer.java"
)
INSTALLER = ROOT / "scripts/install_app_owned_runtime.sh"


def main() -> None:
    service = SERVICE.read_text(encoding="utf-8")
    manifest = MANIFEST.read_text(encoding="utf-8")
    main_activity = MAIN_ACTIVITY.read_text(encoding="utf-8")
    bootstrap_activity = BOOTSTRAP_ACTIVITY.read_text(encoding="utf-8")
    bootstrap_provider = BOOTSTRAP_PROVIDER.read_text(encoding="utf-8")
    asr_server = ASR_SERVER.read_text(encoding="utf-8")
    installer = INSTALLER.read_text(encoding="utf-8")

    assert '"qwen2.5-1.5b-instruct-q4_k_m.gguf"' in service
    assert '"Qwen2.5-1.5B-Instruct-Q4_K_M"' in service
    assert 'command.add("--reasoning");' not in service
    assert 'command.add("--reasoning-format");' not in service
    assert 'android.permission.WAKE_LOCK' in manifest
    assert "PowerManager.PARTIAL_WAKE_LOCK" in service
    assert 'ai.liquid.lfmdemo.runtime:active-client' in service
    assert "activeClientWakeLock.setReferenceCounted(false)" in service
    assert "acquireActiveClientWakeLock();\n        startRuntimes();" in service
    assert "public boolean onUnbind(Intent intent)" in service
    assert "releaseActiveClientWakeLock();\n        return true;" in service
    assert "public void onRebind(Intent intent)" in service
    assert (
        "public void onRebind(Intent intent) {\n"
        "        super.onRebind(intent);\n"
        "        acquireActiveClientWakeLock();\n"
        "        startRuntimes();\n"
        "    }"
    ) in service
    assert service.count("releaseActiveClientWakeLock();") >= 2
    assert "START_STICKY" in service
    start_runtimes = service.split("private synchronized void startRuntimes()", 1)[1].split(
        "private synchronized void startTextModel()", 1
    )[0]
    assert "AsrHttpServer candidate = null;" in start_runtimes
    assert "candidate = new AsrHttpServer(" in start_runtimes
    assert "candidate.start();" in start_runtimes
    assert "asrServer = candidate;" in start_runtimes
    assert "asrServer = new AsrHttpServer(" not in start_runtimes
    assert start_runtimes.index("candidate.start();") < start_runtimes.index(
        "asrServer = candidate;"
    )
    asr_start_failure = start_runtimes.split("catch (IOException error)", 1)[1]
    assert "if (candidate != null) candidate.close();" in asr_start_failure
    assert "asrServer = null;" in asr_start_failure
    assert asr_start_failure.index("candidate.close();") < asr_start_failure.index(
        "asrServer = null;"
    )

    on_create = main_activity.split("protected void onCreate", 1)[1].split(
        "protected void onStart", 1
    )[0]
    on_start = main_activity.split("protected void onStart", 1)[1].split(
        "protected void onStop", 1
    )[0]
    on_stop = main_activity.split("protected void onStop", 1)[1].split(
        "private Intent runtimeCompanionIntent", 1
    )[0]
    assert "startRuntimeCompanion();" not in on_create
    assert 'recordRuntimeEvent("lifecycle_on_create"' in on_create
    assert "cancelRuntimeAttachCallbacks();" in on_start
    assert "runtimeAttachEpoch += 1L;" in on_start
    assert "runtimeBindingDesired = true;" in on_start
    assert "runtimeBootstrapAttempted = false;" in on_start
    assert "runtimeConnectionRecoveries = 0;" in on_start
    assert '"lifecycle_on_start"' in on_start
    assert on_start.index("cancelRuntimeAttachCallbacks();") < on_start.index(
        "runtimeAttachEpoch += 1L;"
    )
    assert on_start.index("runtimeAttachEpoch += 1L;") < on_start.index(
        "scheduleRuntimeBootstrapFallback();"
    )
    assert "startRuntimeCompanion();" in on_start
    assert "bindRuntimeCompanion();" in on_start
    assert "scheduleRuntimeBootstrapFallback();" in on_start
    assert 'recordRuntimeEvent("lifecycle_on_stop_entry"' in on_stop
    assert "runtimeBindingDesired = false;" in on_stop
    assert "cancelRuntimeAttachCallbacks();" in on_stop
    assert 'unbindRuntimeCompanion("on_stop");' in on_stop
    assert 'recordRuntimeEvent("lifecycle_on_stop_complete"' in on_stop
    assert "super.onStop();" in on_stop
    assert "runtimeBindingRegistered || !runtimeBindingDesired" in main_activity
    assert "boolean wasRegistered = runtimeBindingRegistered;" in main_activity
    assert "runtimeBindingRegistered = false;\n        runtimeConnected = false;" in main_activity
    assert 'recordRuntimeEvent("lifecycle_on_destroy_entry"' in main_activity
    assert 'unbindRuntimeCompanion("on_destroy");' in main_activity
    assert 'recordRuntimeEvent("lifecycle_on_destroy_complete"' in main_activity
    assert "Context.BIND_AUTO_CREATE | Context.BIND_IMPORTANT" in main_activity
    assert "public boolean onShowFileChooser(" in main_activity
    assert "params.createIntent()" in main_activity
    assert "WebChromeClient.FileChooserParams.parseResult(resultCode, data)" in main_activity
    assert "fileChooserCallback.onReceiveValue(null);" in main_activity
    for event in (
        "direct_start_attempt",
        "direct_start_result",
        "direct_start_exception",
        "bind_attempt",
        "bind_result",
        "bind_exception",
        "bootstrap_schedule_decision",
    ):
        assert f'"{event}"' in main_activity
    for callback_event in (
        "connection_on_service_connected",
        "connection_on_service_disconnected",
        "connection_on_binding_died",
        "connection_on_null_binding",
    ):
        assert f'"{callback_event}"' in main_activity
    disconnected = main_activity.split(
        "public void onServiceDisconnected", 1
    )[1].split("public void onBindingDied", 1)[0]
    assert 'recoverRuntimeConnection("service_disconnected");' in disconnected
    recovery = main_activity.split(
        "private void recoverRuntimeConnection", 1
    )[1].split("private void startRuntimeBootstrapProvider", 1)[0]
    assert 'unbindRuntimeCompanion(reason);' in recovery
    assert "MAX_RUNTIME_CONNECTION_RECOVERIES = 1" in main_activity
    assert "runtimeConnectionRecoveries >= MAX_RUNTIME_CONNECTION_RECOVERIES" in recovery
    assert "runtimeConnectionRecoveries += 1;" in recovery
    assert "runtimeBootstrapAttempted = false;" in recovery
    assert "startRuntimeCompanion();" in recovery
    assert "bindRuntimeCompanion();" in recovery
    assert "scheduleRuntimeBootstrapFallback();" in recovery
    assert 'recordRuntimeEvent("connection_recovery_bounded"' in recovery
    assert "RuntimeBootstrapActivity" in main_activity
    assert "RUNTIME_BOOTSTRAP_DELAY_MS = 1200L" in main_activity
    assert "runtimeHandler.postDelayed(" in main_activity
    assert "Runnable callback = () -> runRuntimeBootstrapFallback(attachEpoch);" in main_activity
    assert "runtimeHandler.removeCallbacks(runtimeBootstrapFallback);" in main_activity
    assert "Runnable callback = () -> runRuntimePostBootstrapRetry(attachEpoch);" in main_activity
    assert "runtimeHandler.postDelayed(callback, 900L);" in main_activity
    assert "runtimeHandler.removeCallbacks(runtimePostBootstrapRetry);" in main_activity
    assert "RUNTIME_PROVIDER_PROOF_DELAY_MS = 5000L" in main_activity
    assert "Runnable callback = () -> runRuntimeActivityBootstrapFallback(attachEpoch);" in main_activity
    activity_fallback = main_activity.split(
        "private void runRuntimeActivityBootstrapFallback", 1
    )[1].split("private final ServiceConnection", 1)[0]
    assert "!isRuntimeAttachEpochActive(attachEpoch) || runtimeConnected" in activity_fallback
    bootstrap_fallback = main_activity.split(
        "private void runRuntimeBootstrapFallback", 1
    )[1].split("private void runRuntimePostBootstrapRetry", 1)[0]
    assert "!isRuntimeAttachEpochActive(attachEpoch)" in bootstrap_fallback
    post_bootstrap_retry = main_activity.split(
        "private void runRuntimePostBootstrapRetry", 1
    )[1].split("private void runRuntimeActivityBootstrapFallback", 1)[0]
    assert "if (!isRuntimeAttachEpochActive(attachEpoch))" in post_bootstrap_retry
    assert post_bootstrap_retry.index("isRuntimeAttachEpochActive") < post_bootstrap_retry.index(
        'unbindRuntimeCompanion("post_bootstrap_retry");'
    )
    assert "runtimeHandler.removeCallbacks(runtimeActivityBootstrapFallback);" in main_activity
    assert "RUNTIME_PROVIDER_PROOF_DELAY_MS\n        );" in main_activity
    assert "runtimeStartFailed = true;" in main_activity
    assert "(!runtimeConnected || runtimeStartFailed)" in main_activity
    assert "runtimeBootstrapAttempted = true;" in main_activity
    assert '"bootstrap_runnable_entry"' in main_activity
    assert '"bootstrap_runnable_skipped"' in main_activity
    assert '"bootstrap_runnable_dispatch"' in main_activity
    assert "startRuntimeBootstrapProvider(attachEpoch);" in main_activity
    assert 'Uri.parse("content://ai.liquid.lfmdemo.runtime.bootstrap")' in main_activity
    assert 'RUNTIME_BOOTSTRAP_METHOD = "start_runtime"' in main_activity
    assert "getContentResolver().call(" in main_activity
    assert '}, "runtime-bootstrap-provider");' in main_activity
    assert "handleRuntimeBootstrapProviderResult(" in main_activity
    provider_method = main_activity.split(
        "private void startRuntimeBootstrapProvider(long attachEpoch)", 1
    )[1].split("private void handleRuntimeBootstrapProviderResult", 1)[0]
    assert "scheduleRuntimeActivityBootstrapFallback(attachEpoch)" in provider_method
    assert "RUNTIME_PROVIDER_PROOF_DELAY_MS" in provider_method
    assert provider_method.index("scheduleRuntimeActivityBootstrapFallback") < provider_method.index(
        "getContentResolver().call("
    )
    assert "attachEpoch,\n                    deliveredResult" in provider_method
    provider_result = main_activity.split(
        "private void handleRuntimeBootstrapProviderResult", 1
    )[1].split("private String runtimeProviderSummary", 1)[0]
    assert "if (!isRuntimeAttachEpochActive(attachEpoch))" in provider_result
    assert 'recordRuntimeEvent("provider_result_stale_epoch"' in provider_result
    assert provider_result.index("provider_result_stale_epoch") < provider_result.index(
        "startRuntimeBootstrapActivity(attachEpoch);"
    )
    assert "runtimeConnected && !runtimeStartFailed" in provider_result
    assert '"connection_ready_before_result"' in provider_result
    assert 'result.getBoolean("start_requested", false)' in main_activity
    assert 'beginRuntimeBootstrapAttempt("provider");' in main_activity
    assert 'finishRuntimeBootstrapAttempt(' in main_activity
    assert 'RUNTIME_BOOTSTRAP_DIAGNOSTICS' in main_activity
    assert "RUNTIME_EVENT_JOURNAL_SLOTS = 48" in main_activity
    assert 'putLong("journal_sequence"' in main_activity
    assert 'putString("journal_last_event"' in main_activity
    assert 'putString("journal_slot_" + slot, entry)' in main_activity
    assert 'putBoolean("state_binding_desired"' in main_activity
    assert 'putBoolean("state_connected"' in main_activity
    assert 'putInt("state_connection_recoveries"' in main_activity
    assert 'putLong("state_attach_epoch"' in main_activity
    assert ".commit();" in main_activity
    assert "private boolean isRuntimeAttachEpochActive(long attachEpoch)" in main_activity
    assert "return runtimeBindingDesired && attachEpoch == runtimeAttachEpoch;" in main_activity
    assert "private void startRuntimeBootstrapActivity(long attachEpoch)" in main_activity
    activity_start = main_activity.split(
        "private void startRuntimeBootstrapActivity(long attachEpoch)", 1
    )[1].split("private void unbindRuntimeCompanion", 1)[0]
    assert "if (!isRuntimeAttachEpochActive(attachEpoch))" in activity_start
    assert activity_start.index("isRuntimeAttachEpochActive") < activity_start.index(
        "startActivity(bootstrap);"
    )
    assert "startRuntimeBootstrapActivity();" not in main_activity
    cancel_callbacks = main_activity.split(
        "private void cancelRuntimeAttachCallbacks()", 1
    )[1].split("private void cancelRuntimeBootstrapFallback()", 1)[0]
    assert "cancelRuntimeBootstrapFallback();" in cancel_callbacks
    assert "cancelRuntimePostBootstrapRetry();" in cancel_callbacks
    assert "cancelRuntimeActivityBootstrapFallback();" in cancel_callbacks

    assert 'android:name=".RuntimeBootstrapProvider"' in manifest
    assert 'android:authorities="ai.liquid.lfmdemo.runtime.bootstrap"' in manifest
    assert 'android:exported="true"' in manifest
    assert 'android:grantUriPermissions="false"' in manifest
    assert (
        'android:permission="ai.liquid.lfmdemo.permission.START_RUNTIME"'
        in manifest
    )
    assert "extends ContentProvider" in bootstrap_provider
    assert "enforceCallingOrSelfPermission(" in bootstrap_provider
    assert 'ALLOWED_CALLER = "ai.liquid.lfmdemo"' in bootstrap_provider
    assert "getCallingPackage();" in bootstrap_provider
    assert 'METHOD_START_RUNTIME.equals(method)' in bootstrap_provider
    assert "new Intent(context, RuntimeService.class)" in bootstrap_provider
    assert "context.startForegroundService(service);" in bootstrap_provider
    assert 'DIAGNOSTICS = "runtime_bootstrap_diagnostics"' in bootstrap_provider
    assert 'putInt("provider_create_count"' in bootstrap_provider
    assert 'putInt("call_count"' in bootstrap_provider
    assert bootstrap_provider.count(".commit();") >= 2
    assert 'result.putBoolean("start_requested", startRequested)' in bootstrap_provider
    assert "throw new UnsupportedOperationException" in bootstrap_provider

    fixed_route_start = asr_server.index('} else if ("en".equals(language)')
    fixed_route_end = asr_server.index(
        '} else if ("ru".equals(language)) {', fixed_route_start
    )
    fixed_sensevoice_route = asr_server[fixed_route_start:fixed_route_end]
    for language in ("en", "zh", "ja", "ko"):
        assert f'"{language}".equals(language)' in fixed_sensevoice_route
    assert 'runQnn(wavFile, language)' in fixed_sensevoice_route
    assert 'runQnn(wavFile, "auto")' not in fixed_sensevoice_route
    assert '"Use auto, qnn-auto, en, zh, ja, ko, ru, es, or fr"' in asr_server
    assert 'pathOf(target).equals("/prepare")' in asr_server
    assert 'command.add("--persistent-server=true")' in asr_server
    assert 'command.add("--persistent-copies=" + spec.copies)' in asr_server
    assert 'readResidentJsonLine(' in asr_server
    assert '"persistent_ready"' in asr_server
    assert 'readResidentJsonLine(runner, RUNNER_TIMEOUT_MS, "text")' in asr_server
    assert 'if (payload == null) continue;' in asr_server, (
        "QNN stdout diagnostics must be skipped before persistent JSON frames"
    )
    assert '"sensevoice_qnn:" + language' in asr_server
    assert '"zipformer_ru_cpu"' in asr_server
    assert '"parakeet_v3_cpu"' in asr_server
    assert 'stopResidentRunnerLocked();\n\n        List<String> command' in asr_server, (
        "the previous selected ASR process must exit before a new route is allocated"
    )
    assert 'writer.write(RUNNER_QUIT_COMMAND)' in asr_server
    assert '"__SHERPA_ONNX_PERSISTENT_QUIT__"' in asr_server
    assert '"selected_route_only", true' in asr_server
    assert '"protocol", "stdin-json-v1"' in asr_server
    assert "question_ja.wav" in installer and "question_ko.wav" in installer
    assert "0da6ee3b9514a96597bf9225a1bcdf7cae86608e095e157490899a443948b761" in installer
    assert "0ce8353dc9faa844cc8214c613f182a127338b71b139955ccb903ca806a7b3de" in installer
    assert '"language_tag": "<|ja|>"' in installer
    assert '"language_tag": "<|ko|>"' in installer
    assert '"text": "フランス の 首都 は どこ で すか？"' in installer
    assert '"text": "프랑스의 수도는 어디인가요?"' in installer
    assert '("en", "zh", "ja", "ko", "ru")' in installer

    assert 'android:name=".RuntimeBootstrapActivity"' in manifest
    assert 'android:noHistory="true"' in manifest
    assert 'android:excludeFromRecents="true"' in manifest
    assert 'android:permission="ai.liquid.lfmdemo.permission.START_RUNTIME"' in manifest
    assert 'android:theme="@style/RuntimeBootstrapTheme"' in manifest
    assert "startForegroundService(service);" in bootstrap_activity
    assert "finish();\n            overridePendingTransition(0, 0);" in bootstrap_activity

    print("runtime companion source tests passed")


if __name__ == "__main__":
    main()
