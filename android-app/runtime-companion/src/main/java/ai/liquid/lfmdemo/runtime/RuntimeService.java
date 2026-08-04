package ai.liquid.lfmdemo.runtime;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Intent;
import android.os.Handler;
import android.os.Binder;
import android.os.IBinder;
import android.os.Looper;
import android.os.PowerManager;
import android.util.Log;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public final class RuntimeService extends Service {
    private static final String TAG = "LfmRuntimeService";
    private static final String CHANNEL_ID = "lfm-offline-runtime";
    private static final int NOTIFICATION_ID = 61;
    private static final String TEXT_MODEL_FILE =
            "qwen2.5-1.5b-instruct-q4_k_m.gguf";
    private static final String TEXT_MODEL_ALIAS =
            "Qwen2.5-1.5B-Instruct-Q4_K_M";

    private final Handler handler = new Handler(Looper.getMainLooper());
    private final IBinder localBinder = new Binder();
    private AsrHttpServer asrServer;
    private Process textProcess;
    private PowerManager.WakeLock activeClientWakeLock;
    private boolean shuttingDown;

    @Override
    public void onCreate() {
        super.onCreate();
        createNotificationChannel();
        startForeground(NOTIFICATION_ID, notification("Starting ASR and response model…"));
        startRuntimes();
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        startRuntimes();
        return START_STICKY;
    }

    @Override
    public IBinder onBind(Intent intent) {
        acquireActiveClientWakeLock();
        startRuntimes();
        return localBinder;
    }

    @Override
    public boolean onUnbind(Intent intent) {
        releaseActiveClientWakeLock();
        return true;
    }

    @Override
    public void onRebind(Intent intent) {
        super.onRebind(intent);
        acquireActiveClientWakeLock();
        startRuntimes();
    }

    @Override
    public void onDestroy() {
        shuttingDown = true;
        handler.removeCallbacksAndMessages(null);
        releaseActiveClientWakeLock();
        if (asrServer != null) {
            asrServer.close();
            asrServer = null;
        }
        if (textProcess != null && textProcess.isAlive()) {
            textProcess.destroy();
        }
        super.onDestroy();
    }

    /**
     * Nubia's freezer can suspend this foreground service even when Android has
     * promoted its bound process to adj 0. Keep the UID runnable only while the
     * visible demo has an active binding; otherwise loopback sockets remain
     * open but ASR and Qwen requests block indefinitely in the freezer trap.
     */
    private synchronized void acquireActiveClientWakeLock() {
        if (activeClientWakeLock == null) {
            PowerManager power = getSystemService(PowerManager.class);
            activeClientWakeLock = power.newWakeLock(
                    PowerManager.PARTIAL_WAKE_LOCK,
                    "ai.liquid.lfmdemo.runtime:active-client"
            );
            activeClientWakeLock.setReferenceCounted(false);
        }
        if (!activeClientWakeLock.isHeld()) {
            activeClientWakeLock.acquire();
        }
    }

    private synchronized void releaseActiveClientWakeLock() {
        if (activeClientWakeLock != null && activeClientWakeLock.isHeld()) {
            activeClientWakeLock.release();
        }
    }

    private synchronized void startRuntimes() {
        if (shuttingDown) return;
        if (asrServer == null) {
            AsrHttpServer candidate = null;
            try {
                candidate = new AsrHttpServer(
                        new File(getApplicationInfo().nativeLibraryDir),
                        new File(getFilesDir(), "asr"),
                        new File(getCacheDir(), "asr-requests")
                );
                candidate.start();
                asrServer = candidate;
                updateNotification("ASR ready; loading response model…");
            } catch (IOException error) {
                if (candidate != null) candidate.close();
                asrServer = null;
                Log.e(TAG, "Could not start app-owned ASR server", error);
                updateNotification("ASR could not start: " + error.getMessage());
            }
        }
        startTextModel();
    }

    private synchronized void startTextModel() {
        if (shuttingDown || (textProcess != null && textProcess.isAlive())) return;

        File nativeDir = new File(getApplicationInfo().nativeLibraryDir);
        File server = new File(nativeDir, "libqwenserver.so");
        File textRoot = new File(getFilesDir(), "text");
        File model = new File(new File(textRoot, "models"), TEXT_MODEL_FILE);
        if (!server.isFile() || !model.isFile()) {
            Log.w(TAG, "Text runtime or model is not installed yet");
            updateNotification("ASR ready; response model bundle is incomplete");
            return;
        }

        File temporary = new File(textRoot, "tmp");
        File home = new File(textRoot, "home");
        if (!temporary.isDirectory() && !temporary.mkdirs()) {
            Log.e(TAG, "Could not create text temporary directory");
            return;
        }
        if (!home.isDirectory() && !home.mkdirs()) {
            Log.e(TAG, "Could not create text home directory");
            return;
        }

        List<String> command = new ArrayList<>();
        command.add(server.getAbsolutePath());
        command.add("--model");
        command.add(model.getAbsolutePath());
        command.add("--alias");
        command.add(TEXT_MODEL_ALIAS);
        command.add("--ctx-size");
        command.add("2048");
        command.add("--threads");
        command.add("4");
        command.add("--threads-batch");
        command.add("4");
        command.add("--threads-http");
        command.add("1");
        command.add("--batch-size");
        command.add("128");
        command.add("--ubatch-size");
        command.add("64");
        command.add("--parallel");
        command.add("1");
        command.add("--n-gpu-layers");
        command.add("0");
        command.add("--jinja");
        command.add("--host");
        command.add("127.0.0.1");
        command.add("--port");
        command.add("6020");
        command.add("--no-webui");
        command.add("--metrics");
        command.add("--log-colors");
        command.add("off");

        try {
            File log = new File(textRoot, "qwen-server.log");
            if (log.isFile() && log.length() > 8_000_000L && !log.delete()) {
                Log.w(TAG, "Could not rotate Qwen log");
            }
            ProcessBuilder builder = new ProcessBuilder(command);
            builder.directory(textRoot);
            builder.environment().put("LD_LIBRARY_PATH", nativeDir.getAbsolutePath());
            builder.environment().put("TMPDIR", temporary.getAbsolutePath());
            builder.environment().put("HOME", home.getAbsolutePath());
            builder.redirectErrorStream(true);
            builder.redirectOutput(ProcessBuilder.Redirect.appendTo(log));
            textProcess = builder.start();
            updateNotification("ASR ready; response model is loading on-device");

            Process launched = textProcess;
            Thread monitor = new Thread(() -> monitorTextProcess(launched), "qwen-monitor");
            monitor.setDaemon(true);
            monitor.start();
        } catch (IOException error) {
            Log.e(TAG, "Could not start app-owned text model", error);
            updateNotification("ASR ready; response model failed to start");
            scheduleTextRestart();
        }
    }

    private void monitorTextProcess(Process launched) {
        try {
            int code = launched.waitFor();
            Log.e(TAG, "Text server exited with code " + code);
            synchronized (this) {
                if (textProcess == launched) textProcess = null;
            }
            if (!shuttingDown) {
                updateNotification("ASR ready; response model stopped (code " + code + ")");
                scheduleTextRestart();
            }
        } catch (InterruptedException ignored) {
            Thread.currentThread().interrupt();
        }
    }

    private void scheduleTextRestart() {
        if (!shuttingDown) handler.postDelayed(this::startTextModel, 5000);
    }

    private void createNotificationChannel() {
        NotificationChannel channel = new NotificationChannel(
                CHANNEL_ID,
                "LFM offline ASR and response model",
                NotificationManager.IMPORTANCE_LOW
        );
        getSystemService(NotificationManager.class).createNotificationChannel(channel);
    }

    private Notification notification(String text) {
        Intent intent = new Intent(this, RuntimeActivity.class);
        PendingIntent pending = PendingIntent.getActivity(
                this,
                0,
                intent,
                PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT
        );
        return new Notification.Builder(this, CHANNEL_ID)
                .setContentTitle("LFM Offline Runtime")
                .setContentText(text)
                .setSmallIcon(android.R.drawable.ic_btn_speak_now)
                .setContentIntent(pending)
                .setOngoing(true)
                .build();
    }

    private void updateNotification(String text) {
        getSystemService(NotificationManager.class).notify(
                NOTIFICATION_ID,
                notification(text)
        );
    }
}
