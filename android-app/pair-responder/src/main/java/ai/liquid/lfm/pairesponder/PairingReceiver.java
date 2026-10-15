package ai.liquid.lfm.pairesponder;

import android.bluetooth.BluetoothDevice;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.util.Log;

/**
 * Headless responder for ACTION_PAIRING_REQUEST.
 *
 * The Rokid glasses never render the system pairing dialog (the display
 * manager only mirrors the foreground app window), so passkey confirmation
 * can never be tapped. This receiver answers the request programmatically:
 * it waits for the phone-side code to be delivered via
 * `am start -n ai.liquid.lfm.pairesponder/.MainActivity --es pin <CODE>`
 * and confirms with setPin(). That covers both PIN entry and
 * passkey (numeric) confirmation, since both sides carry the same code.
 *
 * The pairing-request parcelable and its methods are resolved via
 * reflection because the project's android.jar stub omits
 * BluetoothPairingRequest.
 */
@SuppressWarnings({"deprecation", "unchecked"})
public class PairingReceiver extends BroadcastReceiver {

    private static final String TAG = "LfmPairResponder";
    private static final long PIN_WAIT_MS = 60000;
    private static final String EXTRA_PAIRING_REQ = "android.bluetooth.device.extra.PAIRING_REQ";

    @Override
    public void onReceive(Context context, Intent intent) {
        if (!BluetoothDevice.ACTION_PAIRING_REQUEST.equals(intent.getAction())) {
            return;
        }
        final int variant = intent.getIntExtra(BluetoothDevice.EXTRA_PAIRING_VARIANT, -1);
        final Object req = intent.getParcelableExtra(EXTRA_PAIRING_REQ);
        if (req == null) {
            Log.w(TAG, "pairing request with null payload (variant=" + variant + ")");
            return;
        }
        Log.i(TAG, "PAIRING_REQUEST variant=" + variantName(variant) + " payload=" + req.getClass().getName());
        if (variant == BluetoothDevice.PAIRING_VARIANT_PASSKEY_CONFIRMATION
                || variant == BluetoothDevice.PAIRING_VARIANT_PIN) {
            Thread t = new Thread(() -> waitAndSetPin(req), "pair-responder");
            t.start();
        } else {
            Log.i(TAG, "variant not pin/passkey; ignoring (no user confirmation possible)");
        }
    }

    private static String variantName(int v) {
        if (v == BluetoothDevice.PAIRING_VARIANT_PIN) return "PIN";
        if (v == BluetoothDevice.PAIRING_VARIANT_PASSKEY_CONFIRMATION) return "PASSKEY_CONFIRMATION";
        return String.valueOf(v);
    }

    private static void waitAndSetPin(Object req) {
        long deadline = System.currentTimeMillis() + PIN_WAIT_MS;
        while (System.currentTimeMillis() < deadline) {
            String pin = MainActivity.pendingPin;
            if (pin != null && pin.length() >= 4) {
                Log.i(TAG, "confirming with code len=" + pin.length());
                try {
                    req.getClass().getMethod("setPin", char[].class).invoke(req, pin.toCharArray());
                    Log.i(TAG, "code accepted by bluetooth stack");
                } catch (Exception e) {
                    Log.e(TAG, "setPin failed", e);
                }
                return;
            }
            try {
                Thread.sleep(300);
            } catch (InterruptedException e) {
                return;
            }
        }
        Log.w(TAG, "no code delivered in time; cancelling pairing request");
        try {
            req.getClass().getMethod("cancel").invoke(req);
        } catch (Exception ignored) {
        }
    }
}
