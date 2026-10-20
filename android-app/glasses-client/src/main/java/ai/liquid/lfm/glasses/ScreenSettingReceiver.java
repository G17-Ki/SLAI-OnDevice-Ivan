package ai.liquid.lfm.glasses;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.provider.Settings;
import android.util.Log;

/**
 * The Rokid firmware resets screen_off_timeout to its 5s factory default on
 * every boot, which also mutes the microphone (mic gain tracks the display
 * wake state on this device). Re-apply the 30-minute timeout after boot so
 * the glasses stay usable without a host computer.
 */
public final class ScreenSettingReceiver extends BroadcastReceiver {
    private static final String TAG = "LfmGlassesClient";
    private static final long SCREEN_OFF_TIMEOUT_MS = 1_800_000L;

    @Override
    public void onReceive(Context context, Intent intent) {
        if (intent == null || !Intent.ACTION_BOOT_COMPLETED.equals(intent.getAction())) return;
        try {
            Settings.System.putLong(
                    context.getContentResolver(),
                    Settings.System.SCREEN_OFF_TIMEOUT,
                    SCREEN_OFF_TIMEOUT_MS);
            Log.i(TAG, "screen_off_timeout re-applied after boot: " + SCREEN_OFF_TIMEOUT_MS);
        } catch (Exception error) {
            Log.w(TAG, "screen_off_timeout re-apply failed: " + error);
        }
    }
}
