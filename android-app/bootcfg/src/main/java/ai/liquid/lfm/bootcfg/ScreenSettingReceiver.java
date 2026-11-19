package ai.liquid.lfm.bootcfg;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.provider.Settings;
import android.util.Log;
public final class ScreenSettingReceiver extends BroadcastReceiver {
    private static final String TAG = "BootCfg";
    public static final String TEST_ACTION = "ai.liquid.lfm.bootcfg.REAPPLY";
    @Override
    public void onReceive(Context context, Intent intent) {
        if (intent == null) return;
        String action = intent.getAction();
        if (!Intent.ACTION_BOOT_COMPLETED.equals(action) && !TEST_ACTION.equals(action)) return;
        try {
            Settings.System.putLong(context.getContentResolver(),
                    Settings.System.SCREEN_OFF_TIMEOUT, 1_800_000L);
            Log.i(TAG, "re-applied screen_off_timeout=1800000 (action=" + action + ")");
        } catch (Exception error) {
            Log.w(TAG, "re-apply failed: " + error);
        }
    }
}
