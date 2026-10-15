package ai.liquid.lfmdemo.runtime;

import android.app.Activity;
import android.content.Intent;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;

/**
 * User-initiated bridge around Nubia's post-reboot AutoLaunch policy.
 *
 * The OEM can block a different package from directly starting or binding this
 * service until one Activity in the runtime package has been launched in the
 * current boot. The visible main app opens this translucent Activity only when
 * its normal bind has not connected. Starting the foreground service from the
 * runtime package's own Activity is accepted, after which the main app's normal
 * lifecycle binding takes over.
 */
public final class RuntimeBootstrapActivity extends Activity {
    private static final String TAG = "LfmRuntimeBootstrap";

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        Intent service = new Intent(this, RuntimeService.class)
                .setAction("ai.liquid.lfmdemo.dev.runtime.START");
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                startForegroundService(service);
            } else {
                startService(service);
            }
        } catch (RuntimeException error) {
            Log.e(TAG, "Could not bootstrap runtime service", error);
        } finally {
            finish();
            overridePendingTransition(0, 0);
        }
    }
}
