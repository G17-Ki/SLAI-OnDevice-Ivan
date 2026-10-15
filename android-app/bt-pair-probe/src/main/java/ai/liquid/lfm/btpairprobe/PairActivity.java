package ai.liquid.lfm.btpairprobe;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothDevice;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.widget.TextView;

/**
 * One-shot helper: initiates createBond() for a device address passed as the
 * "address" extra, so two fresh devices can establish their first bond
 * headlessly. Shows live bonding state.
 */
public class PairActivity extends Activity {
    private TextView status;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        status = new TextView(this);
        status.setPadding(48, 48, 48, 48);
        setContentView(status);

        String address = getIntent().getStringExtra("address");
        if (address == null || address.isEmpty()) {
            status.setText("missing 'address' extra");
            return;
        }
        BluetoothAdapter adapter = BluetoothAdapter.getDefaultAdapter();
        if (adapter == null) {
            status.setText("no adapter");
            return;
        }
        if (checkSelfPermission(android.Manifest.permission.BLUETOOTH_CONNECT)
                != PackageManager.PERMISSION_GRANTED) {
            status.setText("BLUETOOTH_CONNECT missing; grant via adb then relaunch");
            return;
        }
        if (!adapter.isEnabled()) {
            status.setText("adapter off");
            return;
        }
        status.setText("address " + address
                + " · bond state "
                + adapter.getRemoteDevice(address).getBondState());

        IntentFilter filter = new IntentFilter(BluetoothDevice.ACTION_BOND_STATE_CHANGED);
        registerReceiver(receiver, filter);
        status.setText("pairing " + address + " ... accept the system dialog if it appears");
        adapter.getRemoteDevice(address).createBond();
    }

    private final BroadcastReceiver receiver = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {
            BluetoothDevice device = intent.getParcelableExtra(BluetoothDevice.EXTRA_DEVICE);
            int state = intent.getIntExtra(BluetoothDevice.EXTRA_BOND_STATE, -1);
            if (device != null) {
                status.setText("device " + device.getAddress() + " bond state " + state
                        + (state == BluetoothDevice.BOND_BONDED ? " (bonded)"
                        : state == BluetoothDevice.BOND_NONE ? " (none)" : " (bonding)"));
            }
        }
    };

    @Override
    protected void onDestroy() {
        try {
            unregisterReceiver(receiver);
        } catch (Exception ignored) {
        }
        super.onDestroy();
    }
}
