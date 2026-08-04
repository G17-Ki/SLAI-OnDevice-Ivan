package ai.liquid.lfmdemo;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Intent;
import android.os.IBinder;
import android.util.Log;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public final class LfmService extends Service {
    private static final String TAG = "LfmService";
    private static final String CHANNEL_ID = "lfm-runtime";
    private static final int NOTIFICATION_ID = 25;
    private static final String[] MODEL_FILES = {
            "LFM2.5-Audio-1.5B-Q4_0.gguf",
            "mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf",
            "vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf",
            "tokenizer-LFM2.5-Audio-1.5B-Q4_0.gguf"
    };
    private static final String[] DEFAULT_RUNTIME_ARGUMENTS = {
            "--threads", "4",
            "--threads-batch", "4",
            "--poll", "0",
            "--poll-batch", "0"
    };

    private Process modelProcess;

    @Override
    public void onCreate() {
        super.onCreate();
        createNotificationChannel();
        startForeground(NOTIFICATION_ID, notification("Starting LFM2.5-Audio…"));
        LocalUiServer.ensureStarted(this);
        startModel();
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (modelProcess == null || !modelProcess.isAlive()) {
            startModel();
        }
        return START_STICKY;
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    @Override
    public void onDestroy() {
        if (modelProcess != null && modelProcess.isAlive()) {
            modelProcess.destroy();
        }
        super.onDestroy();
    }

    private synchronized void startModel() {
        if (modelProcess != null && modelProcess.isAlive()) {
            return;
        }
        File models = new File(getFilesDir(), "models");
        for (String name : MODEL_FILES) {
            if (!new File(models, name).isFile()) {
                String message = "Models missing from " + models;
                Log.e(TAG, message);
                updateNotification(message);
                return;
            }
        }

        File nativeDir = new File(getApplicationInfo().nativeLibraryDir);
        File server = new File(nativeDir, "liblfmserver.so");
        if (!server.isFile()) {
            updateNotification("Native LFM runner is missing");
            return;
        }

        List<String> command = new ArrayList<>();
        command.add(server.getAbsolutePath());
        command.add("-m");
        command.add(new File(models, MODEL_FILES[0]).getAbsolutePath());
        command.add("-mm");
        command.add(new File(models, MODEL_FILES[1]).getAbsolutePath());
        command.add("-mv");
        command.add(new File(models, MODEL_FILES[2]).getAbsolutePath());
        command.add("--tts-speaker-file");
        command.add(new File(models, MODEL_FILES[3]).getAbsolutePath());
        command.add("--log-colors");
        command.add("off");
        command.add("--perf");
        command.addAll(readRuntimeArguments());
        command.add("--host");
        command.add("127.0.0.1");
        command.add("--port");
        command.add("8080");

        try {
            File log = new File(getFilesDir(), "lfm-server.log");
            ProcessBuilder builder = new ProcessBuilder(command);
            builder.directory(models);
            builder.environment().put("LD_LIBRARY_PATH", nativeDir.getAbsolutePath());
            builder.redirectErrorStream(true);
            builder.redirectOutput(ProcessBuilder.Redirect.appendTo(log));
            modelProcess = builder.start();
            updateNotification("LFM2.5-Audio is running on-device");
            Thread monitor = new Thread(() -> {
                try {
                    int code = modelProcess.waitFor();
                    Log.e(TAG, "LFM server exited with code " + code);
                    updateNotification("LFM server stopped (code " + code + ")");
                } catch (InterruptedException ignored) {
                    Thread.currentThread().interrupt();
                }
            }, "lfm-process-monitor");
            monitor.setDaemon(true);
            monitor.start();
        } catch (IOException error) {
            Log.e(TAG, "Could not start LFM", error);
            updateNotification("Could not start LFM: " + error.getMessage());
        }
    }

    private List<String> readRuntimeArguments() {
        List<String> arguments = new ArrayList<>();
        File config = new File(getFilesDir(), "runtime.args");
        if (!config.isFile()) {
            for (String value : DEFAULT_RUNTIME_ARGUMENTS) {
                arguments.add(value);
            }
            Log.i(TAG, "Using balanced default runtime arguments: " + arguments);
            return arguments;
        }
        try (BufferedReader reader = new BufferedReader(new FileReader(config))) {
            String line;
            while ((line = reader.readLine()) != null) {
                String value = line.trim();
                if (!value.isEmpty() && !value.startsWith("#")) {
                    arguments.add(value);
                }
            }
            Log.i(TAG, "Loaded runtime arguments: " + arguments);
        } catch (IOException error) {
            Log.e(TAG, "Could not read runtime.args", error);
        }
        return arguments;
    }

    private void createNotificationChannel() {
        NotificationChannel channel = new NotificationChannel(
                CHANNEL_ID,
                "LFM on-device runtime",
                NotificationManager.IMPORTANCE_LOW
        );
        getSystemService(NotificationManager.class).createNotificationChannel(channel);
    }

    private Notification notification(String text) {
        Intent intent = new Intent(this, MainActivity.class);
        PendingIntent pending = PendingIntent.getActivity(
                this,
                0,
                intent,
                PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT
        );
        return new Notification.Builder(this, CHANNEL_ID)
                .setContentTitle("LFM Audio")
                .setContentText(text)
                .setSmallIcon(android.R.drawable.ic_btn_speak_now)
                .setContentIntent(pending)
                .setOngoing(true)
                .build();
    }

    private void updateNotification(String text) {
        getSystemService(NotificationManager.class).notify(NOTIFICATION_ID, notification(text));
    }
}
