package ai.liquid.lfm.bootcfg;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;

public class Boot extends Activity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        BootService.scheduleNext(this);
        Log.i("BootCfg", "alarm chain started");
        finish();
    }
}
