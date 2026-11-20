package ai.liquid.lfm.bootcfg;

import android.accessibilityservice.AccessibilityService;
import android.view.accessibility.AccessibilityEvent;
import android.content.Intent;
import android.util.Log;

public class BootA11y extends AccessibilityService {
    private boolean done;

    @Override
    protected void onServiceConnected() {
        super.onServiceConnected();
        if (!done) {
            done = true;
            Log.i("BootCfg", "a11y: bound, applying timeout + starting alarm chain");
            BootService.applyTimeout(this);
            BootService.scheduleNext(this);
        }
    }

    @Override
    public void onAccessibilityEvent(AccessibilityEvent event) {
    }

    @Override
    public void onInterrupt() {
    }

    @Override
    public boolean onUnbind(Intent intent) {
        return super.onUnbind(intent);
    }
}
