package ai.liquid.lfm.glasses;

import android.Manifest;
import android.app.Activity;
import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothDevice;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.Typeface;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.view.Gravity;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import java.util.ArrayList;
import java.util.List;

/**
 * Minimal control surface for the glasses bridge client: pick the bonded
 * phone, pick the task mode (prompt spec V1), start or stop the SPP stream.
 */
public class MainActivity extends Activity
        implements GlassesClientService.UiListener {
    private static final String TAG = "LfmGlassesMain";
    private static final int PERMISSION_REQUEST = 7;
    private static final String PREFS = "lfm.glasses";
    private static final String[] MODES = {"qa", "zh2en", "en2zh"};
    private static final String[] MODE_LABELS = {"QA 问答", "中译英", "英译中"};

    private final List<String> deviceAddresses = new ArrayList<>();
    private final List<String> deviceLabels = new ArrayList<>();
    private Spinner deviceSpinner;
    private final Button[] modeButtons = new Button[MODES.length];
    private int modeIndex = 0;
    private Button startButton;
    private Button stopButton;
    private TextView statusText;
    private TextView heardText;
    private TextView saidText;
    private boolean serviceRunning = false;
    private boolean autoStartPending = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(buildLayout());
        SharedPreferences prefs = getSharedPreferences(PREFS, MODE_PRIVATE);
        String savedMode = prefs.getString("mode", "qa");
        for (int i = 0; i < MODES.length; i++) {
            if (MODES[i].equals(savedMode)) modeIndex = i;
        }
        for (int i = 0; i < MODES.length; i++) {
            final int index = i;
            modeButtons[i].setOnClickListener(v -> onModeSelected(index));
        }
        updateModeButtons();
        updateTranscriptLabels();
        startButton.setOnClickListener(v -> startService());
        stopButton.setOnClickListener(v -> stopService());
        requestPermissionsIfNeeded();
        refreshDevices(prefs.getString("address", null));
        serviceRunning = isClientServiceRunning();
        updateControls();
        // Open the app and it connects (user-approved behavior).
        autoStartPending = true;
        maybeAutoStart();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == PERMISSION_REQUEST) {
            SharedPreferences prefs = getSharedPreferences(PREFS, MODE_PRIVATE);
            refreshDevices(prefs.getString("address", null));
            if (autoStartPending) maybeAutoStart();
        }
    }

    private void maybeAutoStart() {
        if (!autoStartPending || serviceRunning) return;
        if (deviceAddresses.isEmpty()) {
            statusText.setText("No paired phone; pair in system Bluetooth settings");
            return;
        }
        boolean btGranted = Build.VERSION.SDK_INT < 31
                || checkSelfPermission(Manifest.permission.BLUETOOTH_CONNECT)
                == PackageManager.PERMISSION_GRANTED;
        boolean micGranted = checkSelfPermission(Manifest.permission.RECORD_AUDIO)
                == PackageManager.PERMISSION_GRANTED;
        if (!btGranted || !micGranted) {
            statusText.setText("Waiting for Bluetooth / microphone permission...");
            return;
        }
        autoStartPending = false;
        startService();
    }

    @Override
    public void onHeard(String text) {
        runOnUiThread(() -> heardText.setText(prefixHeard() + text));
    }

    @Override
    public void onSaid(String text) {
        runOnUiThread(() -> saidText.setText(prefixSaid() + text));
    }

    @Override
    public void onStatus(String text) {
        runOnUiThread(() -> statusText.setText(text));
    }

    private boolean isTranslationMode() {
        String mode = MODES[modeIndex];
        return "zh2en".equals(mode) || "en2zh".equals(mode);
    }

    private String prefixHeard() {
        return isTranslationMode() ? "原文: " : "我: ";
    }

    private String prefixSaid() {
        return isTranslationMode() ? "译文: " : "助手: ";
    }

    private void onModeSelected(int index) {
        modeIndex = index;
        getSharedPreferences(PREFS, MODE_PRIVATE).edit()
                .putString("mode", MODES[index])
                .apply();
        updateModeButtons();
        updateTranscriptLabels();
        GlassesClientService service = GlassesClientService.getInstance();
        if (service != null) {
            service.setMode(MODES[index]);
        }
    }

    private void updateModeButtons() {
        for (int i = 0; i < modeButtons.length; i++) {
            boolean selected = i == modeIndex;
            modeButtons[i].setBackgroundColor(selected ? 0xFF1976D2 : 0x14000000);
            modeButtons[i].setTextColor(selected ? 0xFFFFFFFF : 0xFF888888);
            modeButtons[i].setTypeface(Typeface.DEFAULT,
                    selected ? Typeface.BOLD : Typeface.NORMAL);
        }
    }

    private void updateTranscriptLabels() {
        heardText.setText(prefixHeard() + "…");
        saidText.setText(prefixSaid() + "…");
    }

    @Override
    protected void onStart() {
        super.onStart();
        GlassesClientService.addUiListener(this);
    }

    @Override
    protected void onStop() {
        GlassesClientService.removeUiListener(this);
        super.onStop();
    }

    private LinearLayout buildLayout() {
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(40, 32, 40, 24);

        heardText = new TextView(this);
        heardText.setText("我: …");
        heardText.setTextSize(26);
        root.addView(heardText);

        saidText = new TextView(this);
        saidText.setText("助手: …");
        saidText.setTextSize(30);
        root.addView(saidText);

        View spacer = new View(this);
        LinearLayout.LayoutParams spacerParams =
                new LinearLayout.LayoutParams(0, 0, 1f);
        root.addView(spacer, spacerParams);

        statusText = new TextView(this);
        statusText.setText("Idle");
        statusText.setTextSize(14);
        root.addView(statusText);

        // Full-width stacked mode buttons: the waveguide scan-out rotates the
        // framebuffer, so wide single-column targets are far easier to hit
        // than a 3-across row of ~133px buttons.
        for (int i = 0; i < MODES.length; i++) {
            Button button = new Button(this);
            button.setText(MODE_LABELS[i]);
            button.setTextSize(16);
            modeButtons[i] = button;
            root.addView(button,
                    new LinearLayout.LayoutParams(
                            LinearLayout.LayoutParams.MATCH_PARENT, 72));
        }

        LinearLayout settings = new LinearLayout(this);
        settings.setOrientation(LinearLayout.HORIZONTAL);
        deviceSpinner = new Spinner(this);
        settings.addView(deviceSpinner,
                new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));
        startButton = new Button(this);
        startButton.setText("Start");
        stopButton = new Button(this);
        stopButton.setText("Stop");
        startButton.setTextSize(13);
        stopButton.setTextSize(13);
        settings.addView(startButton);
        settings.addView(stopButton);
        root.addView(settings);

        return root;
    }

    private void requestPermissionsIfNeeded() {
        List<String> missing = new ArrayList<>();
        if (Build.VERSION.SDK_INT >= 31
                && checkSelfPermission(Manifest.permission.BLUETOOTH_CONNECT)
                != PackageManager.PERMISSION_GRANTED) {
            missing.add(Manifest.permission.BLUETOOTH_CONNECT);
        }
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            missing.add(Manifest.permission.RECORD_AUDIO);
        }
        if (Build.VERSION.SDK_INT >= 33
                && checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS)
                != PackageManager.PERMISSION_GRANTED) {
            missing.add(Manifest.permission.POST_NOTIFICATIONS);
        }
        if (!missing.isEmpty()) {
            requestPermissions(missing.toArray(new String[0]), PERMISSION_REQUEST);
        }
    }

    private void refreshDevices(String preferred) {
        deviceAddresses.clear();
        deviceLabels.clear();
        BluetoothAdapter adapter = BluetoothAdapter.getDefaultAdapter();
        if (adapter == null) {
            deviceLabels.add("Bluetooth unavailable");
        } else if (Build.VERSION.SDK_INT >= 31
                && checkSelfPermission(Manifest.permission.BLUETOOTH_CONNECT)
                != PackageManager.PERMISSION_GRANTED) {
            deviceLabels.add("Waiting for Bluetooth permission");
        } else {
            for (BluetoothDevice device : adapter.getBondedDevices()) {
                deviceAddresses.add(device.getAddress());
                deviceLabels.add(device.getName() + "  " + device.getAddress());
            }
            if (deviceAddresses.isEmpty()) {
                deviceLabels.add("No paired devices");
            }
        }
        deviceSpinner.setAdapter(new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_item, deviceLabels));
        if (preferred != null) {
            int index = deviceAddresses.indexOf(preferred);
            if (index >= 0) deviceSpinner.setSelection(index);
        }
    }

    private void startService() {
        if (deviceAddresses.isEmpty()) {
            statusText.setText("No paired phone selected");
            return;
        }
        int position = deviceSpinner.getSelectedItemPosition();
        String address = deviceAddresses.get(position);
        String mode = MODES[modeIndex];
        getSharedPreferences(PREFS, MODE_PRIVATE).edit()
                .putString("address", address)
                .putString("mode", mode)
                .apply();
        Intent intent = new Intent(this, GlassesClientService.class);
        intent.putExtra("address", address);
        intent.putExtra("mode", mode);
        startForegroundService(intent);
        serviceRunning = true;
        statusText.setText("Connecting to " + address + " ...");
        updateControls();
    }

    private void stopService() {
        stopService(new Intent(this, GlassesClientService.class));
        serviceRunning = false;
        statusText.setText("Stopped");
        updateControls();
    }

    private boolean isClientServiceRunning() {
        android.app.ActivityManager manager =
                (android.app.ActivityManager) getSystemService(ACTIVITY_SERVICE);
        List<android.app.ActivityManager.RunningServiceInfo> services =
                manager.getRunningServices(64);
        for (android.app.ActivityManager.RunningServiceInfo info : services) {
            if (GlassesClientService.class.getName().equals(info.service.getClassName())) {
                return true;
            }
        }
        return false;
    }

    @Override
    protected void onResume() {
        super.onResume();
        serviceRunning = isClientServiceRunning();
        updateControls();
        String status = GlassesClientService.lastStatus();
        if (!status.isEmpty()) statusText.setText(status);
        String heard = GlassesClientService.lastHeard();
        if (!heard.isEmpty()) heardText.setText(prefixHeard() + heard);
        String said = GlassesClientService.lastSaid();
        if (!said.isEmpty()) saidText.setText(prefixSaid() + said);
    }

    private void updateControls() {
        startButton.setEnabled(!serviceRunning);
        stopButton.setEnabled(serviceRunning);
        if (!serviceRunning && deviceSpinner.getSelectedItemPosition() < 0) {
            return;
        }
    }
}
