package ai.liquid.lfm.glasses;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothDevice;
import android.bluetooth.BluetoothSocket;
import android.content.Context;
import android.content.Intent;
import android.media.AudioAttributes;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.AudioTrack;
import android.media.MediaRecorder;
import android.os.IBinder;
import android.util.Log;

import org.json.JSONObject;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.UUID;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.LinkedBlockingQueue;

/**
 * Glasses side of the LFM bridge.
 *
 * Streams the glasses microphone (16 kHz PCM16) to the phone over Bluetooth
 * SPP, where the phone runs the whole cascade (VAD, ASR, LLM, TTS). TTS audio
 * comes back as PCM16 frames and is played through the glasses speaker.
 *
 * Half-duplex: on TTS_START the microphone stream is paused and the speaker
 * plays; on TTS_STOP the speaker releases and the microphone resumes. There is
 * no echo path because the two directions never overlap.
 */
public class GlassesClientService extends Service {
    private static final String TAG = "LfmGlassesClient";
    private static final String CHANNEL_ID = "glasses-client";
    private static final int NOTIFICATION_ID = 63;
    private static final UUID SPP_UUID = UUID.fromString("00001101-0000-1000-8000-0010a98c3f83");
    private static final int SAMPLE_RATE = 16_000;
    private static final int MIC_FRAME_BYTES = 1_280;
    private static final int RECONNECT_DELAY_MS = 3_000;

    private volatile boolean running = false;
    private String deviceAddress;
    private String language = "zh";
    private String mode = "qa";
    private Thread loopThread;

    // Static instance so the activity can push UI commands (mode switch)
    // without a binding.
    private static volatile GlassesClientService instance;

    public static GlassesClientService getInstance() {
        return instance;
    }

    private BluetoothSocket socket;
    private InputStream input;
    private OutputStream output;
    private AudioRecord mic;
    private AudioTrack speaker;
    private volatile boolean micStreaming = false;
    private volatile boolean speakerActive = false;
    private LinkedBlockingQueue<byte[]> speakerQueue;
    private Thread speakerThread;
    private volatile boolean speakerDrain = false;

    // ---- UI observers (transcript display in the activity) ----

    public interface UiListener {
        void onHeard(String text);

        void onSaid(String text);

        void onStatus(String text);
    }

    private static final List<UiListener> uiListeners = new CopyOnWriteArrayList<>();

    public static void addUiListener(UiListener listener) {
        if (listener != null) uiListeners.add(listener);
    }

    public static void removeUiListener(UiListener listener) {
        uiListeners.remove(listener);
    }

    private static void fireHeard(String text) {
        lastHeard = text;
        for (UiListener listener : uiListeners) listener.onHeard(text);
    }

    private static void fireSaid(String text) {
        lastSaid = text;
        for (UiListener listener : uiListeners) listener.onSaid(text);
    }

    private static volatile String lastHeard = "";
    private static volatile String lastSaid = "";

    public static String lastHeard() {
        return lastHeard;
    }

    public static String lastSaid() {
        return lastSaid;
    }

    private static void fireStatus(String text) {
        lastStatus = text;
        for (UiListener listener : uiListeners) listener.onStatus(text);
    }

    // AssistServer periodically pauses this app, dropping the activity's
    // listener; the activity re-syncs from here on resume.
    private static volatile String lastStatus = "";

    public static String lastStatus() {
        return lastStatus;
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        instance = this;
        startForeground(NOTIFICATION_ID, buildNotification());
        if (intent != null) {
            deviceAddress = intent.getStringExtra("address");
            String modeExtra = intent.getStringExtra("mode");
            if (modeExtra != null && !modeExtra.isEmpty()) {
                setModeLocal(modeExtra);
            }
            String lang = intent.getStringExtra("lang");
            if (lang != null && !lang.isEmpty()) language = lang;
        }
        if (deviceAddress == null) {
            Log.e(TAG, "no device address; stopping");
            stopSelf();
            return START_NOT_STICKY;
        }
        if (!running) {
            running = true;
            loopThread = new Thread(this::connectionLoop, "lfm-glasses-client");
            loopThread.start();
        }
        return START_NOT_STICKY;
    }

    @Override
    public void onDestroy() {
        instance = null;
        running = false;
        closeSocket();
        stopMic();
        stopSpeaker();
        super.onDestroy();
    }

    // ---- connection loop with 3s backoff ----

    private void connectionLoop() {
        while (running) {
            try {
                BluetoothAdapter adapter = BluetoothAdapter.getDefaultAdapter();
                if (adapter == null) {
                    Log.e(TAG, "no Bluetooth adapter");
                    break;
                }
                if (!adapter.isEnabled()) {
                    Log.e(TAG, "Bluetooth is off");
                    break;
                }
                BluetoothDevice device = adapter.getRemoteDevice(deviceAddress);
                BluetoothSocket candidate;
                try {
                    candidate = device.createRfcommSocketToServiceRecord(SPP_UUID);
                } catch (Exception error) {
                    Log.w(TAG, "secure SPP create failed; trying insecure", error);
                    candidate = device.createInsecureRfcommSocketToServiceRecord(SPP_UUID);
                }
                Log.i(TAG, "connecting to " + device.getName() + " (" + deviceAddress + ")");
                candidate.connect();
                socket = candidate;
                input = socket.getInputStream();
                output = socket.getOutputStream();
                Log.i(TAG, "connected");
                sendControl(new JSONObject()
                        .put("t", "hello")
                        .put("lang", language)
                        .put("mode", mode));
                startMic();
                try {
                    readLoop();
                } finally {
                    stopMic();
                    stopSpeaker();
                }
                Log.i(TAG, "connection closed");
            } catch (Exception error) {
                Log.w(TAG, "connection attempt failed", error);
                stopMic();
                stopSpeaker();
            }
            closeSocket();
            if (!running) break;
            try {
                Thread.sleep(RECONNECT_DELAY_MS);
            } catch (InterruptedException error) {
                Thread.currentThread().interrupt();
                break;
            }
        }
        running = false;
    }

    // ---- read side ----

    private void readLoop() throws IOException {
        SppProtocol.Reader reader = new SppProtocol.Reader(input);
        while (running) {
            SppProtocol.Frame frame = reader.nextFrame();
            if (frame == null) return;
            if (frame.type == SppProtocol.AUDIO_OUT) {
                enqueueSpeaker(frame.payload);
            } else if (frame.type == SppProtocol.CONTROL) {
                handleControl(new String(frame.payload, StandardCharsets.UTF_8));
            } else {
                Log.w(TAG, "unexpected frame type " + frame.type);
            }
        }
    }

    private void handleControl(String raw) {
        try {
            JSONObject json = new JSONObject(raw);
            switch (json.optString("t")) {
                case "hello_ack":
                case "ready":
                case "listening":
                    Log.i(TAG, "bridge " + json.optString("t"));
                    if ("hello_ack".equals(json.optString("t"))) {
                        fireStatus("已连接 · " + modeLabel(mode));
                    }
                    return;
                case "mode_ack":
                    String ackMode = json.optString("mode", "");
                    Log.i(TAG, "bridge mode: " + ackMode);
                    fireStatus("模式: " + modeLabel(ackMode));
                    return;
                case "heard": {
                    String text = json.optString("text", "");
                    Log.i(TAG, "bridge heard: " + text);
                    fireHeard(text);
                    return;
                }
                case "said": {
                    String text = json.optString("text", "");
                    Log.i(TAG, "bridge said: " + text);
                    fireSaid(text);
                    return;
                }
                case "tts_start": {
                    int rate = json.optInt("rate", 24_000);
                    startSpeaker(rate);
                    return;
                }
                case "tts_stop":
                    // Let the playback thread finish the queued PCM before
                    // tearing the track down; stopping now would cut the tail.
                    speakerDrain = true;
                    LinkedBlockingQueue<byte[]> pending = speakerQueue;
                    if (pending != null && pending.isEmpty()) {
                        stopSpeaker();
                    }
                    return;
                case "error":
                    Log.w(TAG, "bridge error: " + json.optString("msg"));
                    return;
                default:
                    Log.d(TAG, "unhandled control: " + raw);
            }
        } catch (Exception error) {
            Log.w(TAG, "bad control frame: " + raw);
        }
    }

    private void sendControl(JSONObject json) {
        try {
            SppProtocol.writeFrame(output, SppProtocol.CONTROL,
                    json.toString().getBytes(StandardCharsets.UTF_8));
        } catch (IOException error) {
            Log.w(TAG, "control send failed: " + error.getMessage());
        }
    }

    // ---- mode (prompt spec V1): qa | zh2en | en2zh ----

    /**
     * Switch the session task mode from the UI. Takes effect on the next turn;
     * a fresh session picks it up through hello.
     */
    public void setMode(String newMode) {
        setModeLocal(newMode);
        Log.i(TAG, "set_mode " + mode);
        try {
            sendControl(new JSONObject().put("t", "set_mode").put("mode", mode));
        } catch (Exception error) {
            Log.w(TAG, "set_mode build failed: " + error.getMessage());
        }
    }

    private void setModeLocal(String newMode) {
        String normalized = newMode == null ? "qa" : newMode.trim().toLowerCase(Locale.ROOT);
        if (!"qa".equals(normalized) && !"zh2en".equals(normalized) && !"en2zh".equals(normalized)) {
            normalized = "qa";
        }
        mode = normalized;
        // Keep the legacy lang field consistent (used by older bridges).
        language = "en2zh".equals(normalized) ? "en" : "zh";
    }

    /** Accepts client-side (zh2en) and bridge-side (ZH2EN) spellings. */
    private static String modeLabel(String modeName) {
        String m = modeName == null ? "" : modeName.trim().toLowerCase(Locale.ROOT);
        if (m.equals("zh2en") || m.equals("zh-en") || m.equals("zh_en")) return "中译英";
        if (m.equals("en2zh") || m.equals("en-zh") || m.equals("en_zh")) return "英译中";
        return "QA 问答";
    }

    // ---- microphone (glasses -> phone) ----

    /**
     * Sequential 2-second probe of candidate capture sources; some OEM audio
     * policies deliver digital silence for specific sources to third parties.
     * Returns the source with the highest observed peak.
     */
    private int probeSource() {
        // MIC is the fallback: on this device VOICE_RECOGNITION delivers
        // digital silence to third parties, so a silent probe must not fall
        // back to it.
        int best = MediaRecorder.AudioSource.MIC;
        int bestPeak = 0;
        String[] names = {"VR", "MIC", "DEFAULT"};
        int[] sources = {
                MediaRecorder.AudioSource.VOICE_RECOGNITION,
                MediaRecorder.AudioSource.MIC,
                MediaRecorder.AudioSource.DEFAULT
        };
        for (int i = 0; i < sources.length; i++) {
            AudioRecord record = null;
            try {
                int minBuffer = AudioRecord.getMinBufferSize(
                        SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO,
                        AudioFormat.ENCODING_PCM_16BIT);
                record = new AudioRecord(sources[i], SAMPLE_RATE,
                        AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT,
                        Math.max(minBuffer, MIC_FRAME_BYTES * 4));
                if (record.getState() != AudioRecord.STATE_INITIALIZED) {
                    Log.w(TAG, "probe " + names[i] + ": not initialized");
                    continue;
                }
                record.startRecording();
                byte[] buf = new byte[MIC_FRAME_BYTES];
                int peak = 0;
                for (int f = 0; f < 50; f++) {
                    int read = record.read(buf, 0, buf.length);
                    if (read < 0) break;
                    for (int j = 0; j + 1 < read; j += 2) {
                        int s = (buf[j] & 0xFF) | (buf[j + 1] << 8);
                        if (s < 0) s += 65536;
                        int a = s > 32767 ? 65535 - s : s;
                        if (a > peak) peak = a;
                    }
                }
                Log.i(TAG, "probe " + names[i] + " peak: " + peak);
                if (peak > bestPeak) {
                    bestPeak = peak;
                    best = sources[i];
                }
            } catch (Exception error) {
                Log.w(TAG, "probe " + names[i] + " failed: " + error.getMessage());
            } finally {
                if (record != null) {
                    try {
                        record.stop();
                    } catch (Exception ignored) {
                    }
                    record.release();
                }
            }
        }
        Log.i(TAG, "probe winner: source " + best + " (peak " + bestPeak + ")");
        return best;
    }

    private void startMic() {
        if (mic != null) {
            resumeMic();
            return;
        }
        if (checkSelfPermission(android.Manifest.permission.RECORD_AUDIO)
                != android.content.pm.PackageManager.PERMISSION_GRANTED) {
            Log.e(TAG, "RECORD_AUDIO not granted; mic disabled");
            return;
        }
        int source = probeSource();
        int minBuffer = AudioRecord.getMinBufferSize(
                SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
        AudioRecord record;
        try {
            record = new AudioRecord(
                    source,
                    SAMPLE_RATE,
                    AudioFormat.CHANNEL_IN_MONO,
                    AudioFormat.ENCODING_PCM_16BIT,
                    Math.max(minBuffer, MIC_FRAME_BYTES * 4));
        } catch (Exception error) {
            Log.e(TAG, "AudioRecord init failed", error);
            return;
        }
        if (record.getState() != AudioRecord.STATE_INITIALIZED) {
            record.release();
            return;
        }
        mic = record;
        resumeMic();
        Thread micThread = new Thread(() -> {
            byte[] buffer = new byte[MIC_FRAME_BYTES];
            long frameCount = 0;
            String stopReason = "loop_end";
            try {
                while (running) {
                    if (mic == null) {
                        stopReason = "released";
                        break;
                    }
                    if (!micStreaming) {
                        try {
                            Thread.sleep(20);
                        } catch (InterruptedException error) {
                            Thread.currentThread().interrupt();
                            stopReason = "interrupted";
                            break;
                        }
                        continue;
                    }
                    int read;
                    try {
                        read = record.read(buffer, 0, buffer.length);
                    } catch (Exception error) {
                        stopReason = "read exception: " + error;
                        break;
                    }
                    if (read < 0) {
                        stopReason = "read " + read;
                        break;
                    }
                    int peak = 0;
                    for (int i = 0; i + 1 < read; i += 2) {
                        int s = (buffer[i] & 0xFF) | (buffer[i + 1] << 8);
                        if (s < 0) s += 65536;
                        int a = s > 32767 ? 65535 - s : s;
                        if (a > peak) peak = a;
                    }
                    frameCount++;
                    if (frameCount % 100 == 0) {
                        Log.i(TAG, "mic peak: " + peak + " (frame " + frameCount + ")");
                    }
                    if (read < buffer.length) {
                        if ((read & 1) != 0) read--;
                        if (read == 0) continue;
                        byte[] partial = Arrays.copyOf(buffer, read);
                        try {
                            SppProtocol.writeFrame(output, SppProtocol.AUDIO_IN, partial);
                        } catch (IOException error) {
                            stopReason = "write " + error;
                            break;
                        }
                        continue;
                    }
                    try {
                        SppProtocol.writeFrame(output, SppProtocol.AUDIO_IN, buffer);
                    } catch (IOException error) {
                        stopReason = "write " + error;
                        break;
                    }
                }
            } finally {
                Log.w(TAG, "mic thread stopped: " + stopReason + " after " + frameCount + " frames");
                micStreaming = false;
            }
        }, "lfm-glasses-mic");
        micThread.start();
    }

    private void resumeMic() {
        AudioRecord record = mic;
        if (record == null || micStreaming) return;
        if (record.getRecordingState() != AudioRecord.RECORDSTATE_RECORDING) {
            record.startRecording();
        }
        micStreaming = true;
    }

    private void pauseMic() {
        micStreaming = false;
        AudioRecord record = mic;
        if (record != null && record.getRecordingState() == AudioRecord.RECORDSTATE_RECORDING) {
            record.stop();
        }
    }

    private void stopMic() {
        pauseMic();
        AudioRecord record = mic;
        mic = null;
        if (record != null) {
            record.release();
        }
    }

    // ---- speaker (phone -> glasses) ----

    private void startSpeaker(int rate) {
        stopSpeaker();
        speakerDrain = false;
        int minBuffer = AudioTrack.getMinBufferSize(
                rate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT);
        AudioTrack track;
        try {
            track = new AudioTrack(
                    new AudioAttributes.Builder()
                            // USAGE_ASSISTANT is silently dropped on this device;
                            // the media route reaches the glasses speaker.
                            .setUsage(AudioAttributes.USAGE_MEDIA)
                            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                            .build(),
                    new AudioFormat.Builder()
                            .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                            .setSampleRate(rate)
                            .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                            .build(),
                    Math.max(minBuffer, rate),
                    AudioTrack.MODE_STREAM,
                    0);
        } catch (Exception error) {
            Log.e(TAG, "AudioTrack init failed", error);
            return;
        }
        if (track.getState() != AudioTrack.STATE_INITIALIZED) {
            track.release();
            return;
        }
        track.play();
        speaker = track;
        speakerActive = true;
        speakerQueue = new LinkedBlockingQueue<>(256);
        final AudioTrack playbackTrack = track;
        final LinkedBlockingQueue<byte[]> queue = speakerQueue;
        speakerThread = new Thread(() -> {
            while (running) {
                byte[] pcm;
                try {
                    pcm = queue.take();
                } catch (InterruptedException error) {
                    return;
                }
                writeSpeaker(playbackTrack, pcm);
                if (playbackTrack != speaker) return;
                if (speakerDrain && queue.isEmpty()) {
                    stopSpeaker();
                    return;
                }
            }
        }, "lfm-glasses-speaker");
        speakerThread.start();
        Log.i(TAG, "speaker up at " + rate + " Hz");
        pauseMic();
    }

    private void enqueueSpeaker(byte[] pcm) {
        LinkedBlockingQueue<byte[]> queue = speakerQueue;
        if (queue == null) return;
        if (!queue.offer(pcm)) {
            Log.w(TAG, "speaker queue full; dropping " + pcm.length + " bytes");
        }
    }

    private void writeSpeaker(AudioTrack track, byte[] pcm) {
        if (track == null) return;
        int written = 0;
        while (written < pcm.length && running) {
            try {
                int count = track.write(pcm, written, pcm.length - written, AudioTrack.WRITE_BLOCKING);
                if (count <= 0) {
                    try {
                        Thread.sleep(2);
                    } catch (InterruptedException error) {
                        Thread.currentThread().interrupt();
                        return;
                    }
                    continue;
                }
                written += count;
            } catch (Exception error) {
                Log.w(TAG, "speaker write failed", error);
                stopSpeaker();
                return;
            }
        }
    }

    private void stopSpeaker() {
        AudioTrack track = speaker;
        speaker = null;
        speakerActive = false;
        Thread thread = speakerThread;
        speakerThread = null;
        speakerQueue = null;
        if (thread != null && thread != Thread.currentThread()) {
            thread.interrupt();
            try {
                thread.join(500);
            } catch (InterruptedException ignored) {
                Thread.currentThread().interrupt();
            }
        }
        if (track != null) {
            try {
                track.stop();
            } catch (Exception ignored) {
            }
            track.release();
        }
        resumeMic();
    }

    // ---- socket teardown ----

    private void closeSocket() {
        BluetoothSocket candidate = socket;
        socket = null;
        input = null;
        output = null;
        if (candidate != null) {
            try {
                candidate.close();
            } catch (IOException ignored) {
            }
        }
    }

    private Notification buildNotification() {
        NotificationManager manager = (NotificationManager) getSystemService(Context.NOTIFICATION_SERVICE);
        if (manager != null) {
            manager.createNotificationChannel(
                    new NotificationChannel(CHANNEL_ID, "Glasses client",
                            NotificationManager.IMPORTANCE_LOW));
        }
        Intent open = new Intent(this, MainActivity.class);
        PendingIntent pending = PendingIntent.getActivity(
                this, 0, open, PendingIntent.FLAG_IMMUTABLE);
        return new Notification.Builder(this, CHANNEL_ID)
                .setSmallIcon(android.R.drawable.ic_btn_speak_now)
                .setContentTitle("LFM glasses client")
                .setContentText("Streaming to phone")
                .setOngoing(true)
                .setContentIntent(pending)
                .build();
    }
}
