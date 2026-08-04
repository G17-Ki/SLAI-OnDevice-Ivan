package ai.liquid.lfmdemo.runtime;

import android.content.ComponentName;
import android.content.ContentProvider;
import android.content.ContentValues;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.database.Cursor;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;

/**
 * Minimal same-signature process bootstrap for devices that reject a
 * cross-package service or Activity launch immediately after reboot.
 *
 * Resolving this provider creates the runtime process under its own UID. The
 * only supported call then requests the explicit foreground service from that
 * UID. Persistent app-private counters make it possible to distinguish provider
 * creation, call delivery, and service-start acceptance after a failed run.
 */
public final class RuntimeBootstrapProvider extends ContentProvider {
    static final String AUTHORITY = "ai.liquid.lfmdemo.runtime.bootstrap";
    static final String METHOD_START_RUNTIME = "start_runtime";

    private static final String TAG = "LfmRuntimeProvider";
    private static final String START_PERMISSION =
            "ai.liquid.lfmdemo.permission.START_RUNTIME";
    private static final String ALLOWED_CALLER = "ai.liquid.lfmdemo";
    private static final String DIAGNOSTICS = "runtime_bootstrap_diagnostics";

    private int providerCreateCount;
    private long providerCreatedAtEpochMs;

    @Override
    public boolean onCreate() {
        Context context = getContext();
        if (context == null) return false;

        SharedPreferences diagnostics = diagnostics(context);
        providerCreateCount = diagnostics.getInt("provider_create_count", 0) + 1;
        providerCreatedAtEpochMs = System.currentTimeMillis();
        boolean persisted = diagnostics.edit()
                .putInt("provider_create_count", providerCreateCount)
                .putLong("provider_last_created_epoch_ms", providerCreatedAtEpochMs)
                .commit();
        Log.i(
                TAG,
                "Runtime bootstrap provider created; count=" + providerCreateCount
                        + ", diagnosticsPersisted=" + persisted
        );
        return true;
    }

    @Override
    public synchronized Bundle call(String method, String arg, Bundle extras) {
        Context context = getContext();
        if (context == null) throw new IllegalStateException("Provider context is unavailable");
        context.enforceCallingOrSelfPermission(
                START_PERMISSION,
                "Runtime bootstrap requires the same-signature permission"
        );
        String callerPackage = getCallingPackage();
        if (!ALLOWED_CALLER.equals(callerPackage)) {
            throw new SecurityException("Unexpected runtime bootstrap caller: " + callerPackage);
        }
        if (!METHOD_START_RUNTIME.equals(method)) {
            throw new IllegalArgumentException("Unsupported runtime bootstrap method: " + method);
        }

        SharedPreferences diagnostics = diagnostics(context);
        int callCount = diagnostics.getInt("call_count", 0) + 1;
        long callAtEpochMs = System.currentTimeMillis();
        boolean attemptPersisted = diagnostics.edit()
                .putInt("call_count", callCount)
                .putLong("last_call_epoch_ms", callAtEpochMs)
                .putString("last_caller_package", callerPackage)
                .putString("last_outcome", "attempting")
                .remove("last_error_class")
                .remove("last_error_message")
                .commit();

        boolean startRequested = false;
        String outcome;
        String serviceComponent = "";
        String errorClass = "";
        String errorMessage = "";
        try {
            Intent service = new Intent(context, RuntimeService.class)
                    .setAction("ai.liquid.lfmdemo.runtime.START");
            ComponentName started;
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                started = context.startForegroundService(service);
            } else {
                started = context.startService(service);
            }
            startRequested = started != null;
            outcome = startRequested ? "start_requested" : "start_rejected";
            if (started != null) serviceComponent = started.flattenToShortString();
        } catch (RuntimeException error) {
            outcome = "start_error";
            errorClass = error.getClass().getName();
            errorMessage = String.valueOf(error.getMessage());
            Log.e(TAG, "Could not start runtime service from bootstrap provider", error);
        }

        SharedPreferences.Editor resultEditor = diagnostics.edit()
                .putLong("last_result_epoch_ms", System.currentTimeMillis())
                .putString("last_outcome", outcome)
                .putBoolean("last_start_requested", startRequested)
                .putString("last_service_component", serviceComponent)
                .putString("last_error_class", errorClass)
                .putString("last_error_message", errorMessage);
        boolean resultPersisted = resultEditor.commit();

        Bundle result = new Bundle();
        result.putBoolean("provider_created", true);
        result.putInt("provider_create_count", providerCreateCount);
        result.putLong("provider_created_epoch_ms", providerCreatedAtEpochMs);
        result.putInt("call_count", callCount);
        result.putLong("call_epoch_ms", callAtEpochMs);
        result.putString("caller_package", callerPackage);
        result.putString("outcome", outcome);
        result.putBoolean("start_requested", startRequested);
        result.putString("service_component", serviceComponent);
        result.putString("error_class", errorClass);
        result.putString("error_message", errorMessage);
        result.putBoolean(
                "diagnostics_persisted",
                attemptPersisted && resultPersisted
        );
        Log.i(
                TAG,
                "Runtime bootstrap call completed; outcome=" + outcome
                        + ", callCount=" + callCount
        );
        return result;
    }

    private SharedPreferences diagnostics(Context context) {
        return context.getSharedPreferences(DIAGNOSTICS, Context.MODE_PRIVATE);
    }

    @Override
    public Cursor query(
            Uri uri,
            String[] projection,
            String selection,
            String[] selectionArgs,
            String sortOrder
    ) {
        throw new UnsupportedOperationException("Runtime bootstrap exposes no query surface");
    }

    @Override
    public String getType(Uri uri) {
        throw new UnsupportedOperationException("Runtime bootstrap exposes no MIME type");
    }

    @Override
    public Uri insert(Uri uri, ContentValues values) {
        throw new UnsupportedOperationException("Runtime bootstrap exposes no insert surface");
    }

    @Override
    public int delete(Uri uri, String selection, String[] selectionArgs) {
        throw new UnsupportedOperationException("Runtime bootstrap exposes no delete surface");
    }

    @Override
    public int update(
            Uri uri,
            ContentValues values,
            String selection,
            String[] selectionArgs
    ) {
        throw new UnsupportedOperationException("Runtime bootstrap exposes no update surface");
    }
}
