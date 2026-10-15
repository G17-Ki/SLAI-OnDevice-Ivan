package ai.liquid.lfm.glasses;

import android.Manifest;
import android.app.Activity;
import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothDevice;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
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
 * phone, pick the cascade language, start or stop the SPP stream.
 */
public class MainActivity extends Activity
        implements GlassesClientService.UiListener {
    private static final String TAG = "LfmGlassesMain";
    private static final int PERMISSION_REQUEST = 7;
    private static final String PREFS = "lfm.glasses";
    private static final String[] LANGUAGES = {"zh", "en", "ja", "ko", "ru"};
    private static final String[] LANGUAGE_LABELS = {
            "Chinese (中文)", "English", "日本語", "한국어", "Русский"
    };

    private final List<String> deviceAddresses = new ArrayList<>();
    private final List<String> deviceLabels = new ArrayList<>();
    private Spinner deviceSpinner;
    private Spinner languageSpinner;
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
        int langIndex = 0;
        for (int i = 0; i < LANGUAGES.length; i++) {
            if (LANGUAGES[i].equals(prefs.getString("lang", "zh"))) langIndex = i;
        }
        languageSpinner.setSelection(langIndex);
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
        runOnUiThread(() -> heardText.setText("我: " + text));
    }

    @Override
    public void onSaid(String text) {
        runOnUiThread(() -> saidText.setText("助手: " + text));
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

        LinearLayout settings = new LinearLayout(this);
        settings.setOrientation(LinearLayout.HORIZONTAL);
        deviceSpinner = new Spinner(this);
        languageSpinner = new Spinner(this);
        languageSpinner.setAdapter(new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_item, LANGUAGE_LABELS));
        settings.addView(deviceSpinner,
                new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));
        settings.addView(languageSpinner,
                new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));
        root.addView(settings);

        LinearLayout buttons = new LinearLayout(this);
        buttons.setOrientation(LinearLayout.HORIZONTAL);
        startButton = new Button(this);
        startButton.setText("Start");
        stopButton = new Button(this);
        stopButton.setText("Stop");
        startButton.setTextSize(13);
        stopButton.setTextSize(13);
        buttons.addView(startButton);
        buttons.addView(stopButton);
        root.addView(buttons);

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
        String language = LANGUAGES[languageSpinner.getSelectedItemPosition()];
        getSharedPreferences(PREFS, MODE_PRIVATE).edit()
                .putString("address", address)
                .putString("lang", language)
                .apply();
        Intent intent = new Intent(this, GlassesClientService.class);
        intent.putExtra("address", address);
        intent.putExtra("lang", language);
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
    }

    private void updateControls() {
        startButton.setEnabled(!serviceRunning);
        stopButton.setEnabled(serviceRunning);
        if (!serviceRunning && deviceSpinner.getSelectedItemPosition() < 0) {
            return;
        }
    }
}
