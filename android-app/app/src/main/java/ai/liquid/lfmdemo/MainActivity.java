package ai.liquid.lfmdemo;

import android.Manifest;
import android.app.Activity;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.ServiceConnection;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.IBinder;
import android.os.Looper;
import android.util.Log;
import android.view.WindowManager;
import android.webkit.PermissionRequest;
import android.webkit.ValueCallback;
import android.webkit.WebChromeClient;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;

import java.util.ArrayList;
import java.util.List;

public final class MainActivity extends Activity {
    private static final String TAG = "LfmMainActivity";
    private static final int PERMISSION_REQUEST = 10;
    private static final int FILE_CHOOSER_REQUEST = 11;
    private static final String RUNTIME_PACKAGE = "ai.liquid.lfmdemo.runtime";
    private static final String RUNTIME_SERVICE =
            "ai.liquid.lfmdemo.runtime.RuntimeService";
    private static final String RUNTIME_BOOTSTRAP_ACTIVITY =
            "ai.liquid.lfmdemo.runtime.RuntimeBootstrapActivity";
    private static final Uri RUNTIME_BOOTSTRAP_URI =
            Uri.parse("content://ai.liquid.lfmdemo.runtime.bootstrap");
    private static final String RUNTIME_BOOTSTRAP_METHOD = "start_runtime";
    private static final String RUNTIME_BOOTSTRAP_DIAGNOSTICS =
            "runtime_bootstrap_diagnostics";
    private static final long RUNTIME_BOOTSTRAP_DELAY_MS = 1200L;
    private static final long RUNTIME_PROVIDER_PROOF_DELAY_MS = 5000L;
    private static final int RUNTIME_EVENT_JOURNAL_SLOTS = 48;
    private static final int MAX_RUNTIME_CONNECTION_RECOVERIES = 1;
    private WebView webView;
    private NativeAudioBridge nativeAudio;
    private ValueCallback<Uri[]> fileChooserCallback;
    private final Handler runtimeHandler = new Handler(Looper.getMainLooper());
    private boolean runtimeBindingDesired;
    private boolean runtimeBindingRegistered;
    private boolean runtimeConnected;
    private boolean runtimeStartFailed;
    private boolean runtimeBootstrapAttempted;
    private int runtimeConnectionRecoveries;
    private long runtimeAttachEpoch;
    private Runnable runtimeBootstrapFallback;
    private Runnable runtimePostBootstrapRetry;
    private Runnable runtimeActivityBootstrapFallback;

    private void runRuntimeBootstrapFallback(long attachEpoch) {
        recordRuntimeEvent("bootstrap_runnable_entry", "");
        if (!isRuntimeAttachEpochActive(attachEpoch)
                || (runtimeConnected && !runtimeStartFailed)
                || runtimeBootstrapAttempted) {
            recordRuntimeEvent(
                    "bootstrap_runnable_skipped",
                    runtimeEpochDetail(attachEpoch) + ", guard_rejected"
            );
            return;
        }
        runtimeBootstrapAttempted = true;
        recordRuntimeEvent(
                "bootstrap_runnable_dispatch",
                "provider, " + runtimeEpochDetail(attachEpoch)
        );
        startRuntimeBootstrapProvider(attachEpoch);
    }

    private void runRuntimePostBootstrapRetry(long attachEpoch) {
        recordRuntimeEvent("post_bootstrap_retry_entry", "");
        if (!isRuntimeAttachEpochActive(attachEpoch)) {
            recordRuntimeEvent(
                    "post_bootstrap_retry_skipped",
                    runtimeEpochDetail(attachEpoch) + ", attach_not_active"
            );
            return;
        }
        unbindRuntimeCompanion("post_bootstrap_retry");
        startRuntimeCompanion();
        bindRuntimeCompanion();
    }

    private void runRuntimeActivityBootstrapFallback(long attachEpoch) {
        recordRuntimeEvent("activity_fallback_runnable_entry", "");
        if (!isRuntimeAttachEpochActive(attachEpoch) || runtimeConnected) {
            recordRuntimeEvent(
                    "activity_fallback_runnable_skipped",
                    runtimeEpochDetail(attachEpoch) + ", guard_rejected"
            );
            return;
        }
        startRuntimeBootstrapActivity(attachEpoch);
    }

    private final ServiceConnection runtimeConnection = new ServiceConnection() {
        @Override
        public void onServiceConnected(ComponentName name, IBinder service) {
            runtimeConnected = runtimeBindingDesired && runtimeBindingRegistered;
            if (runtimeConnected) runtimeStartFailed = false;
            recordRuntimeEvent(
                    "connection_on_service_connected",
                    componentDetail(name) + ", binderPresent=" + (service != null)
            );
            cancelRuntimeActivityBootstrapFallback();
            if (!runtimeStartFailed) {
                cancelRuntimeBootstrapFallback();
                cancelRuntimePostBootstrapRetry();
            }
        }

        @Override
        public void onServiceDisconnected(ComponentName name) {
            runtimeConnected = false;
            recordRuntimeEvent("connection_on_service_disconnected", componentDetail(name));
            recoverRuntimeConnection("service_disconnected");
        }

        @Override
        public void onBindingDied(ComponentName name) {
            runtimeConnected = false;
            recordRuntimeEvent("connection_on_binding_died", componentDetail(name));
            recoverRuntimeConnection("binding_died");
        }

        @Override
        public void onNullBinding(ComponentName name) {
            runtimeConnected = false;
            recordRuntimeEvent("connection_on_null_binding", componentDetail(name));
            unbindRuntimeCompanion("null_binding");
            if (runtimeBindingDesired) scheduleRuntimeBootstrapFallback();
        }
    };

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        recordRuntimeEvent("lifecycle_on_create", "restored=" + (state != null));
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        LocalUiServer.ensureStarted(this);
        requestPermissionsIfNeeded();
        createWebView();
    }

    @Override
    protected void onStart() {
        super.onStart();
        cancelRuntimeAttachCallbacks();
        runtimeAttachEpoch += 1L;
        runtimeBindingDesired = true;
        runtimeBootstrapAttempted = false;
        runtimeConnectionRecoveries = 0;
        recordRuntimeEvent(
                "lifecycle_on_start",
                "runtime_attach_begin, epoch=" + runtimeAttachEpoch
        );
        startRuntimeCompanion();
        bindRuntimeCompanion();
        scheduleRuntimeBootstrapFallback();
    }

    @Override
    protected void onStop() {
        recordRuntimeEvent("lifecycle_on_stop_entry", "runtime_attach_cancel");
        runtimeBindingDesired = false;
        cancelRuntimeAttachCallbacks();
        unbindRuntimeCompanion("on_stop");
        recordRuntimeEvent("lifecycle_on_stop_complete", "callbacks_removed");
        super.onStop();
    }

    private Intent runtimeCompanionIntent() {
        Intent runtime = new Intent();
        runtime.setClassName(RUNTIME_PACKAGE, RUNTIME_SERVICE);
        runtime.setAction("ai.liquid.lfmdemo.runtime.START");
        return runtime;
    }

    private void startRuntimeCompanion() {
        Intent runtime = runtimeCompanionIntent();
        runtimeStartFailed = false;
        recordRuntimeEvent("direct_start_attempt", runtime.getComponent().flattenToShortString());
        try {
            ComponentName started;
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                started = startForegroundService(runtime);
            } else {
                started = startService(runtime);
            }
            if (started == null) {
                runtimeStartFailed = true;
                recordRuntimeEvent("direct_start_result", "component=null");
                Log.w(TAG, "Runtime companion start was rejected");
            } else {
                recordRuntimeEvent(
                        "direct_start_result",
                        "component=" + started.flattenToShortString()
                );
            }
        } catch (RuntimeException error) {
            // The Web UI keeps polling both loopback endpoints and presents a
            // route-specific readiness message when the companion APK is not
            // installed or its app-owned model bundle is incomplete.
            runtimeStartFailed = true;
            recordRuntimeEvent("direct_start_exception", throwableDetail(error));
            Log.w(TAG, "Could not start runtime companion service", error);
        }
    }

    private void bindRuntimeCompanion() {
        if (runtimeBindingRegistered || !runtimeBindingDesired) {
            recordRuntimeEvent(
                    "bind_skipped",
                    runtimeBindingRegistered ? "already_registered" : "binding_not_desired"
            );
            return;
        }
        recordRuntimeEvent("bind_attempt", runtimeCompanionIntent().getComponent().flattenToShortString());
        try {
            runtimeBindingRegistered = bindService(
                    runtimeCompanionIntent(),
                    runtimeConnection,
                    Context.BIND_AUTO_CREATE | Context.BIND_IMPORTANT
            );
            recordRuntimeEvent(
                    "bind_result",
                    "registered=" + runtimeBindingRegistered
            );
        } catch (RuntimeException error) {
            runtimeBindingRegistered = false;
            recordRuntimeEvent("bind_exception", throwableDetail(error));
            Log.w(TAG, "Could not bind runtime companion service", error);
        }
    }

    private void scheduleRuntimeBootstrapFallback() {
        cancelRuntimeBootstrapFallback();
        long attachEpoch = runtimeAttachEpoch;
        boolean shouldSchedule = runtimeBindingDesired
                && (!runtimeConnected || runtimeStartFailed)
                && !runtimeBootstrapAttempted;
        boolean posted = false;
        if (shouldSchedule) {
            Runnable callback = () -> runRuntimeBootstrapFallback(attachEpoch);
            runtimeBootstrapFallback = callback;
            posted = runtimeHandler.postDelayed(callback, RUNTIME_BOOTSTRAP_DELAY_MS);
            if (!posted && runtimeBootstrapFallback == callback) {
                runtimeBootstrapFallback = null;
            }
        }
        recordRuntimeEvent(
                "bootstrap_schedule_decision",
                "shouldSchedule=" + shouldSchedule
                        + ", posted=" + posted
                        + ", epoch=" + attachEpoch
                        + ", delayMs=" + RUNTIME_BOOTSTRAP_DELAY_MS
        );
    }

    private void recoverRuntimeConnection(String reason) {
        unbindRuntimeCompanion(reason);
        if (!runtimeBindingDesired) {
            recordRuntimeEvent("connection_recovery_skipped", reason + ": binding_not_desired");
            return;
        }
        if (runtimeConnectionRecoveries >= MAX_RUNTIME_CONNECTION_RECOVERIES) {
            recordRuntimeEvent("connection_recovery_bounded", reason);
            scheduleRuntimeBootstrapFallback();
            return;
        }

        runtimeConnectionRecoveries += 1;
        runtimeBootstrapAttempted = false;
        recordRuntimeEvent(
                "connection_recovery_begin",
                reason + ": attempt=" + runtimeConnectionRecoveries
        );
        startRuntimeCompanion();
        bindRuntimeCompanion();
        scheduleRuntimeBootstrapFallback();
    }

    private void startRuntimeBootstrapProvider(long attachEpoch) {
        if (!isRuntimeAttachEpochActive(attachEpoch)) {
            recordRuntimeEvent("provider_call_skipped", runtimeEpochDetail(attachEpoch));
            return;
        }
        beginRuntimeBootstrapAttempt("provider");
        // Also arm the Activity path before the IPC. If Nubia accepts the
        // provider acquisition but never returns from it, the background call
        // remains diagnosable as "attempting" and cannot suppress fallback.
        boolean activityFallbackPosted = scheduleRuntimeActivityBootstrapFallback(attachEpoch);
        recordRuntimeEvent(
                "provider_call_dispatch",
                "activityFallbackPosted=" + activityFallbackPosted
                        + ", proofDelayMs=" + RUNTIME_PROVIDER_PROOF_DELAY_MS
        );
        Thread providerCall = new Thread(() -> {
            Bundle result = null;
            RuntimeException failure = null;
            try {
                result = getContentResolver().call(
                        RUNTIME_BOOTSTRAP_URI,
                        RUNTIME_BOOTSTRAP_METHOD,
                        null,
                        null
                );
            } catch (RuntimeException error) {
                failure = error;
            }
            Bundle deliveredResult = result;
            RuntimeException deliveredFailure = failure;
            runtimeHandler.post(() -> handleRuntimeBootstrapProviderResult(
                    attachEpoch,
                    deliveredResult,
                    deliveredFailure
            ));
        }, "runtime-bootstrap-provider");
        providerCall.start();
    }

    private void handleRuntimeBootstrapProviderResult(
            long attachEpoch,
            Bundle result,
            RuntimeException error
    ) {
        if (!isRuntimeAttachEpochActive(attachEpoch)) {
            recordRuntimeEvent("provider_result_stale_epoch", runtimeEpochDetail(attachEpoch));
            return;
        }
        if (runtimeConnected && !runtimeStartFailed) {
            finishRuntimeBootstrapAttempt(
                    "provider",
                    "connection_ready_before_result",
                    runtimeEpochDetail(attachEpoch)
            );
            cancelRuntimePostBootstrapRetry();
            cancelRuntimeActivityBootstrapFallback();
            return;
        }
        if (error != null) {
            finishRuntimeBootstrapAttempt(
                    "provider",
                    "call_error",
                    error.getClass().getName() + ": " + String.valueOf(error.getMessage())
            );
            Log.w(TAG, "Could not call runtime bootstrap provider", error);
            cancelRuntimeActivityBootstrapFallback();
            startRuntimeBootstrapActivity(attachEpoch);
            return;
        }
        if (result == null) {
            finishRuntimeBootstrapAttempt("provider", "null_result", "");
            Log.w(TAG, "Runtime bootstrap provider returned no result");
            cancelRuntimeActivityBootstrapFallback();
            startRuntimeBootstrapActivity(attachEpoch);
            return;
        }

        boolean startRequested = result.getBoolean("start_requested", false);
        String summary = runtimeProviderSummary(result);
        finishRuntimeBootstrapAttempt(
                "provider",
                startRequested ? "start_requested" : "start_rejected",
                summary
        );
        Log.i(TAG, "Runtime provider bootstrap: " + summary);
        if (!startRequested) {
            cancelRuntimeActivityBootstrapFallback();
            startRuntimeBootstrapActivity(attachEpoch);
            return;
        }

        // A ComponentName only proves that Android accepted the request. Nubia
        // can still leave the target process absent after returning success.
        // Retry the normal important binding, then require an actual Binder
        // connection before suppressing the known Activity fallback.
        boolean retryPosted = scheduleRuntimePostBootstrapRetry(attachEpoch);
        boolean activityFallbackPosted = scheduleRuntimeActivityBootstrapFallback(attachEpoch);
        recordRuntimeEvent(
                "provider_start_followup_scheduled",
                "retryPosted=" + retryPosted
                        + ", activityFallbackPosted=" + activityFallbackPosted
        );
    }

    private String runtimeProviderSummary(Bundle result) {
        return "outcome=" + result.getString("outcome", "missing")
                + ", providerCreated=" + result.getBoolean("provider_created", false)
                + ", providerCreateCount=" + result.getInt("provider_create_count", -1)
                + ", callCount=" + result.getInt("call_count", -1)
                + ", diagnosticsPersisted="
                + result.getBoolean("diagnostics_persisted", false)
                + ", serviceComponent=" + result.getString("service_component", "")
                + ", error=" + result.getString("error_message", "");
    }

    private void beginRuntimeBootstrapAttempt(String route) {
        SharedPreferences diagnostics = getSharedPreferences(
                RUNTIME_BOOTSTRAP_DIAGNOSTICS,
                MODE_PRIVATE
        );
        String prefix = route + "_";
        int count = diagnostics.getInt(prefix + "attempt_count", 0) + 1;
        boolean persisted = diagnostics.edit()
                .putInt(prefix + "attempt_count", count)
                .putLong(prefix + "last_attempt_epoch_ms", System.currentTimeMillis())
                .putString(prefix + "last_outcome", "attempting")
                .remove(prefix + "last_detail")
                .commit();
        if (!persisted) Log.w(TAG, "Could not persist " + route + " bootstrap attempt");
        recordRuntimeEvent("bootstrap_" + route + "_attempt", "count=" + count);
    }

    private void finishRuntimeBootstrapAttempt(String route, String outcome, String detail) {
        String prefix = route + "_";
        boolean persisted = getSharedPreferences(RUNTIME_BOOTSTRAP_DIAGNOSTICS, MODE_PRIVATE)
                .edit()
                .putLong(prefix + "last_result_epoch_ms", System.currentTimeMillis())
                .putString(prefix + "last_outcome", outcome)
                .putString(prefix + "last_detail", detail)
                .commit();
        if (!persisted) Log.w(TAG, "Could not persist " + route + " bootstrap result");
        recordRuntimeEvent("bootstrap_" + route + "_result", outcome + ": " + detail);
    }

    private void startRuntimeBootstrapActivity(long attachEpoch) {
        if (!isRuntimeAttachEpochActive(attachEpoch)) {
            recordRuntimeEvent("activity_start_skipped", runtimeEpochDetail(attachEpoch));
            return;
        }
        beginRuntimeBootstrapAttempt("activity");
        Intent bootstrap = new Intent();
        bootstrap.setClassName(RUNTIME_PACKAGE, RUNTIME_BOOTSTRAP_ACTIVITY);
        bootstrap.addFlags(
                Intent.FLAG_ACTIVITY_NO_ANIMATION
                        | Intent.FLAG_ACTIVITY_NO_HISTORY
                        | Intent.FLAG_ACTIVITY_EXCLUDE_FROM_RECENTS
        );
        try {
            startActivity(bootstrap);
            overridePendingTransition(0, 0);
            boolean retryPosted = scheduleRuntimePostBootstrapRetry(attachEpoch);
            finishRuntimeBootstrapAttempt("activity", "launch_requested", "");
            recordRuntimeEvent(
                    "activity_start_followup_scheduled",
                    "retryPosted=" + retryPosted
            );
        } catch (RuntimeException error) {
            finishRuntimeBootstrapAttempt(
                    "activity",
                    "launch_error",
                    error.getClass().getName() + ": " + String.valueOf(error.getMessage())
            );
            Log.w(TAG, "Could not launch runtime bootstrap activity", error);
        }
    }

    private void unbindRuntimeCompanion(String reason) {
        boolean wasRegistered = runtimeBindingRegistered;
        runtimeBindingRegistered = false;
        runtimeConnected = false;
        if (!wasRegistered) {
            recordRuntimeEvent("unbind_skipped", reason + ": not_registered");
            return;
        }
        try {
            unbindService(runtimeConnection);
            recordRuntimeEvent("unbind_result", reason + ": requested");
        } catch (RuntimeException error) {
            recordRuntimeEvent("unbind_exception", reason + ": " + throwableDetail(error));
        }
    }

    private boolean scheduleRuntimePostBootstrapRetry(long attachEpoch) {
        cancelRuntimePostBootstrapRetry();
        if (!isRuntimeAttachEpochActive(attachEpoch)) return false;
        Runnable callback = () -> runRuntimePostBootstrapRetry(attachEpoch);
        runtimePostBootstrapRetry = callback;
        boolean posted = runtimeHandler.postDelayed(callback, 900L);
        if (!posted && runtimePostBootstrapRetry == callback) {
            runtimePostBootstrapRetry = null;
        }
        return posted;
    }

    private boolean scheduleRuntimeActivityBootstrapFallback(long attachEpoch) {
        cancelRuntimeActivityBootstrapFallback();
        if (!isRuntimeAttachEpochActive(attachEpoch)) return false;
        Runnable callback = () -> runRuntimeActivityBootstrapFallback(attachEpoch);
        runtimeActivityBootstrapFallback = callback;
        boolean posted = runtimeHandler.postDelayed(
                callback,
                RUNTIME_PROVIDER_PROOF_DELAY_MS
        );
        if (!posted && runtimeActivityBootstrapFallback == callback) {
            runtimeActivityBootstrapFallback = null;
        }
        return posted;
    }

    private void cancelRuntimeAttachCallbacks() {
        cancelRuntimeBootstrapFallback();
        cancelRuntimePostBootstrapRetry();
        cancelRuntimeActivityBootstrapFallback();
    }

    private void cancelRuntimeBootstrapFallback() {
        if (runtimeBootstrapFallback == null) return;
        runtimeHandler.removeCallbacks(runtimeBootstrapFallback);
        runtimeBootstrapFallback = null;
    }

    private void cancelRuntimePostBootstrapRetry() {
        if (runtimePostBootstrapRetry == null) return;
        runtimeHandler.removeCallbacks(runtimePostBootstrapRetry);
        runtimePostBootstrapRetry = null;
    }

    private void cancelRuntimeActivityBootstrapFallback() {
        if (runtimeActivityBootstrapFallback == null) return;
        runtimeHandler.removeCallbacks(runtimeActivityBootstrapFallback);
        runtimeActivityBootstrapFallback = null;
    }

    private boolean isRuntimeAttachEpochActive(long attachEpoch) {
        return runtimeBindingDesired && attachEpoch == runtimeAttachEpoch;
    }

    private String runtimeEpochDetail(long attachEpoch) {
        return "callbackEpoch=" + attachEpoch + ", currentEpoch=" + runtimeAttachEpoch;
    }

    private synchronized void recordRuntimeEvent(String event, String detail) {
        SharedPreferences diagnostics = getSharedPreferences(
                RUNTIME_BOOTSTRAP_DIAGNOSTICS,
                MODE_PRIVATE
        );
        long sequence = diagnostics.getLong("journal_sequence", 0L) + 1L;
        long timestamp = System.currentTimeMillis();
        int slot = (int) ((sequence - 1L) % RUNTIME_EVENT_JOURNAL_SLOTS);
        String normalizedDetail = detail == null
                ? ""
                : detail.replace('\n', ' ').replace('\r', ' ');
        if (normalizedDetail.length() > 800) {
            normalizedDetail = normalizedDetail.substring(0, 800);
        }
        String state = runtimeStateSnapshot();
        String entry = sequence + "|" + timestamp + "|" + event
                + "|" + state + "|" + normalizedDetail;
        boolean persisted = diagnostics.edit()
                .putLong("journal_sequence", sequence)
                .putString("journal_last_event", event)
                .putLong("journal_last_event_epoch_ms", timestamp)
                .putString("journal_last_detail", normalizedDetail)
                .putString("journal_last_state", state)
                .putString("journal_slot_" + slot, entry)
                .putBoolean("state_binding_desired", runtimeBindingDesired)
                .putBoolean("state_binding_registered", runtimeBindingRegistered)
                .putBoolean("state_connected", runtimeConnected)
                .putBoolean("state_start_failed", runtimeStartFailed)
                .putBoolean("state_bootstrap_attempted", runtimeBootstrapAttempted)
                .putInt("state_connection_recoveries", runtimeConnectionRecoveries)
                .putLong("state_attach_epoch", runtimeAttachEpoch)
                .commit();
        if (!persisted) Log.w(TAG, "Could not persist runtime event " + event);
        Log.i(TAG, "runtimeEvent=" + event + ", " + state + ", " + normalizedDetail);
    }

    private String runtimeStateSnapshot() {
        return "desired=" + runtimeBindingDesired
                + ", registered=" + runtimeBindingRegistered
                + ", connected=" + runtimeConnected
                + ", startFailed=" + runtimeStartFailed
                + ", bootstrapAttempted=" + runtimeBootstrapAttempted
                + ", recoveries=" + runtimeConnectionRecoveries
                + ", attachEpoch=" + runtimeAttachEpoch;
    }

    private String componentDetail(ComponentName component) {
        return "component=" + (component == null ? "null" : component.flattenToShortString());
    }

    private String throwableDetail(Throwable error) {
        return error.getClass().getName() + ": " + String.valueOf(error.getMessage());
    }

    private void createWebView() {
        webView = new WebView(this);
        WebSettings settings = webView.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setDomStorageEnabled(true);
        settings.setMediaPlaybackRequiresUserGesture(false);
        settings.setMixedContentMode(WebSettings.MIXED_CONTENT_ALWAYS_ALLOW);
        nativeAudio = new NativeAudioBridge(this, webView);
        webView.addJavascriptInterface(nativeAudio, "NativeAudio");
        webView.setWebViewClient(new WebViewClient());
        webView.setWebChromeClient(new WebChromeClient() {
            @Override
            public boolean onShowFileChooser(
                    WebView view,
                    ValueCallback<Uri[]> callback,
                    FileChooserParams params
            ) {
                if (fileChooserCallback != null) {
                    fileChooserCallback.onReceiveValue(null);
                }
                fileChooserCallback = callback;
                try {
                    startActivityForResult(params.createIntent(), FILE_CHOOSER_REQUEST);
                    return true;
                } catch (RuntimeException error) {
                    Log.w(TAG, "Could not open the audio file chooser", error);
                    fileChooserCallback = null;
                    callback.onReceiveValue(null);
                    return false;
                }
            }

            @Override
            public void onPermissionRequest(PermissionRequest request) {
                runOnUiThread(() -> {
                    if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
                        List<String> grants = new ArrayList<>();
                        for (String resource : request.getResources()) {
                            if (PermissionRequest.RESOURCE_AUDIO_CAPTURE.equals(resource)) {
                                grants.add(resource);
                            }
                        }
                        if (!grants.isEmpty()) {
                            request.grant(grants.toArray(new String[0]));
                            return;
                        }
                    }
                    request.deny();
                });
            }
        });
        setContentView(webView);
        webView.loadUrl("http://127.0.0.1:8765/");
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (requestCode == FILE_CHOOSER_REQUEST) {
            ValueCallback<Uri[]> callback = fileChooserCallback;
            fileChooserCallback = null;
            if (callback != null) {
                callback.onReceiveValue(
                        WebChromeClient.FileChooserParams.parseResult(resultCode, data)
                );
            }
            return;
        }
        super.onActivityResult(requestCode, resultCode, data);
    }

    private void requestPermissionsIfNeeded() {
        List<String> missing = new ArrayList<>();
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            missing.add(Manifest.permission.RECORD_AUDIO);
        }
        if (Build.VERSION.SDK_INT >= 33
                && checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
            missing.add(Manifest.permission.POST_NOTIFICATIONS);
        }
        if (!missing.isEmpty()) {
            requestPermissions(missing.toArray(new String[0]), PERMISSION_REQUEST);
        }
    }

    @Override
    public void onBackPressed() {
        if (webView != null && webView.canGoBack()) {
            webView.goBack();
        } else {
            super.onBackPressed();
        }
    }

    @Override
    protected void onDestroy() {
        recordRuntimeEvent("lifecycle_on_destroy_entry", "runtime_attach_cancel");
        runtimeBindingDesired = false;
        cancelRuntimeAttachCallbacks();
        unbindRuntimeCompanion("on_destroy");
        if (fileChooserCallback != null) {
            fileChooserCallback.onReceiveValue(null);
            fileChooserCallback = null;
        }
        if (nativeAudio != null) {
            nativeAudio.release();
        }
        if (webView != null) {
            webView.destroy();
        }
        recordRuntimeEvent("lifecycle_on_destroy_complete", "resources_released");
        super.onDestroy();
    }
}
