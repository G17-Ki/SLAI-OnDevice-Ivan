package ai.liquid.lfmdemo.runtime;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Typeface;
import android.os.Build;
import android.os.Bundle;
import android.view.Gravity;
import android.widget.LinearLayout;
import android.widget.TextView;

public final class RuntimeActivity extends Activity {
    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        Intent service = new Intent(this, RuntimeService.class)
                .setAction("ai.liquid.lfmdemo.runtime.START");
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            startForegroundService(service);
        } else {
            startService(service);
        }

        int padding = Math.round(24 * getResources().getDisplayMetrics().density);
        LinearLayout layout = new LinearLayout(this);
        layout.setOrientation(LinearLayout.VERTICAL);
        layout.setGravity(Gravity.CENTER_VERTICAL);
        layout.setPadding(padding, padding, padding, padding);

        TextView title = new TextView(this);
        title.setText("LFM offline runtime");
        title.setTextSize(24);
        title.setTypeface(Typeface.DEFAULT, Typeface.BOLD);
        layout.addView(title);

        TextView message = new TextView(this);
        message.setText(
                "ASR and the compact response model are starting as app-owned "
                        + "phone services. You can return to LFM Audio; USB and "
                        + "the Mac are not used for inference."
        );
        message.setTextSize(17);
        message.setPadding(0, padding, 0, 0);
        layout.addView(message);
        setContentView(layout);
    }
}
