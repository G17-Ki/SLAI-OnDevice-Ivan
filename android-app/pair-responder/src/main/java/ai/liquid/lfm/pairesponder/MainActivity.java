package ai.liquid.lfm.pairesponder;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.view.Gravity;
import android.widget.TextView;

public class MainActivity extends Activity {

    public static volatile String pendingPin;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        TextView tv = new TextView(this);
        tv.setGravity(Gravity.CENTER);
        tv.setText("Pair responder active.\n\n"
                + "Pairing requests are answered automatically.\n"
                + "Deliver the phone-side PIN with:\n"
                + "am start -n ai.liquid.lfm.pairesponder/.MainActivity --es pin <PIN>");
        setContentView(tv);
        applyPin(getIntent());
    }

    @Override
    protected void onNewIntent(Intent intent) {
        super.onNewIntent(intent);
        setIntent(intent);
        applyPin(intent);
    }

    @Override
    protected void onResume() {
        super.onResume();
        applyPin(getIntent());
    }

    private void applyPin(Intent intent) {
        if (intent == null) {
            return;
        }
        String pin = intent.getStringExtra("pin");
        if (pin != null && pin.trim().length() > 0) {
            pendingPin = pin.trim();
        }
    }
}
