package ai.liquid.lfmdemo.whisperprobe;

import android.app.Activity;
import android.graphics.Color;
import android.graphics.Typeface;
import android.os.Bundle;
import android.os.SystemClock;
import android.view.Gravity;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.time.Instant;
import java.util.Locale;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.regex.Pattern;

/** Minimal, coexistable launcher for the isolated ORT-QNN compatibility probe. */
public final class ProbeActivity extends Activity {
    public static final String EXTRA_RUN_PROBE = "run_probe";
    public static final String EXTRA_RUN_ID = "run_id";
    public static final String REPORT_COMPLETION_MARKER =
            "WHISPER_PROBE_REPORT_COMPLETE_V1";
    private static final Pattern RUN_ID_PATTERN = Pattern.compile(
            "[A-Za-z0-9][A-Za-z0-9._-]{0,127}"
    );

    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private Button runButton;
    private TextView status;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        int padding = Math.round(22 * getResources().getDisplayMetrics().density);

        LinearLayout content = new LinearLayout(this);
        content.setOrientation(LinearLayout.VERTICAL);
        content.setPadding(padding, padding, padding, padding);

        TextView title = new TextView(this);
        title.setText("Whisper Small · ORT-QNN probe");
        title.setTextSize(24);
        title.setTypeface(Typeface.DEFAULT, Typeface.BOLD);
        title.setTextColor(Color.rgb(16, 42, 36));
        content.addView(title);

        TextView boundary = new TextView(this);
        boundary.setText(
                "Offline compatibility only. No microphone and no network permission. "
                        + "This package never replaces the accepted LFM app/runtime."
        );
        boundary.setTextSize(16);
        boundary.setPadding(0, padding / 2, 0, padding);
        content.addView(boundary);

        runButton = new Button(this);
        runButton.setText(
                BuildConfig.WHISPER_SEMANTIC_DECODE_ENABLED
                        ? "Verify frontend/QNN and run bounded "
                                + BuildConfig.WHISPER_SEMANTIC_LANGUAGE.toUpperCase(Locale.ROOT)
                                + " semantic candidate"
                        : "Verify "
                                + BuildConfig.WHISPER_FRONTEND_PROFILE.toUpperCase(Locale.ROOT)
                                + " frontend fixtures, payload, and 4 decoder steps"
        );
        runButton.setAllCaps(false);
        runButton.setEnabled(BuildConfig.WHISPER_PAYLOAD_EMBEDDED);
        runButton.setOnClickListener(view -> runProbe(
                view,
                new RunContext(
                        newManualRunId(),
                        "manual_button",
                        System.currentTimeMillis(),
                        SystemClock.elapsedRealtimeNanos()
                )
        ));
        content.addView(runButton);

        status = new TextView(this);
        status.setTextSize(14);
        status.setTypeface(Typeface.MONOSPACE);
        status.setTextIsSelectable(true);
        status.setPadding(0, padding, 0, padding);
        status.setText(
                BuildConfig.WHISPER_PAYLOAD_EMBEDDED
                        ? "Ready. Execution stays disabled until every staged asset/runtime hash passes."
                        : "Stub build: no model/runtime payload embedded. Stage with "
                                + "scripts/stage_whisper_ort_qnn_probe.py and rebuild with "
                                + "WHISPER_PROBE_PAYLOAD_DIR."
        );
        content.addView(status);

        ScrollView scroll = new ScrollView(this);
        scroll.addView(content);
        setContentView(scroll);

        // Device automation may request the exact same path as the button,
        // once, without changing the default manual-launch behavior.
        if (state == null && getIntent().getBooleanExtra(EXTRA_RUN_PROBE, false)) {
            String automationRunId = getIntent().getStringExtra(EXTRA_RUN_ID);
            getIntent().removeExtra(EXTRA_RUN_PROBE);
            getIntent().removeExtra(EXTRA_RUN_ID);
            if (BuildConfig.WHISPER_PAYLOAD_EMBEDDED) {
                if (isValidRunId(automationRunId)) {
                    RunContext run = new RunContext(
                            automationRunId,
                            "automation_intent",
                            System.currentTimeMillis(),
                            SystemClock.elapsedRealtimeNanos()
                    );
                    runButton.post(() -> runProbe(runButton, run));
                } else {
                    status.setText(
                            "run_probe rejected: run_id must match "
                                    + "[A-Za-z0-9][A-Za-z0-9._-]{0,127}."
                    );
                }
            } else {
                status.setText("run_probe ignored: this is a payload-free stub build.");
            }
        }
    }

    private void runProbe(View ignored, RunContext run) {
        runButton.setEnabled(false);
        status.setText("Verifying and installing APK payload into app-private storage…");
        executor.submit(() -> {
            try {
                ProbePayload payload = ProbePayload.installAndVerify(this);
                WhisperDecodePolicy.Language semanticLanguage =
                        WhisperDecodePolicy.Language.fromCode(
                                BuildConfig.WHISPER_SEMANTIC_LANGUAGE
                        );
                WhisperFrontendProbe.Result frontend = null;
                if (BuildConfig.WHISPER_FRONTEND_FIXTURES_EMBEDDED) {
                    postStatus(
                            "Payload verified. Computing bundled "
                                    + BuildConfig.WHISPER_FRONTEND_PROFILE
                                    + " frontend tensors…"
                    );
                    frontend = WhisperFrontendProbe.run(
                            this,
                            semanticLanguage.code(),
                            BuildConfig.WHISPER_SEMANTIC_DECODE_ENABLED
                    );
                    if (frontend.featuresForLanguage("en") == null) {
                        JSONObject failure = new JSONObject();
                        failure.put("status", "failed_closed");
                        failure.put("error", "bundled WAV frontend exceeded numerical bound");
                        failure.put("bundled_audio_frontend", frontend.report);
                        failure.put("real_audio_transcription", false);
                        failure.put("accepted_packages_mutated", false);
                        persistReport(failure, run);
                        postStatus(failure.toString(2));
                        return;
                    }
                }
                postStatus("Frontend/payload verified. Loading QNN encoder and decoder sessions…");
                JSONObject report = WhisperQnnProbe.run(
                        payload,
                        frontend == null ? null : frontend.featuresForLanguage("en"),
                        frontend == null
                                ? null
                                : frontend.featuresForLanguage(semanticLanguage.code()),
                        semanticLanguage,
                        BuildConfig.WHISPER_SEMANTIC_DECODE_ENABLED
                );
                if (frontend != null) {
                    report.put("bundled_audio_frontend", frontend.report);
                } else {
                    report.put("bundled_audio_frontend", new JSONObject()
                            .put("status", "not_embedded")
                            .put("bounded_gate_passed", false)
                            .put("real_audio_frontend", false));
                }
                persistReport(report, run);
                postStatus(report.toString(2));
            } catch (Throwable error) {
                JSONObject failure = new JSONObject();
                try {
                    failure.put("status", "failed_closed");
                    failure.put("error_type", error.getClass().getName());
                    failure.put("error", String.valueOf(error.getMessage()));
                    failure.put("real_audio_transcription", false);
                    failure.put("accepted_packages_mutated", false);
                    persistReport(failure, run);
                    postStatus(failure.toString(2));
                } catch (Throwable reportError) {
                    postStatus("FAILED CLOSED\n" + error + "\nReport error: " + reportError);
                }
            } finally {
                runOnUiThread(() -> runButton.setEnabled(BuildConfig.WHISPER_PAYLOAD_EMBEDDED));
            }
        });
    }

    private File persistReport(JSONObject report, RunContext run) throws Exception {
        File reportDir = new File(getFilesDir(), "whisper_probe/reports");
        if (!reportDir.isDirectory() && !reportDir.mkdirs()) {
            throw new IllegalStateException("cannot create report directory");
        }
        File reportFile = new File(reportDir, "latest.json");
        File temporary = new File(reportDir, "latest.json.partial");
        if (temporary.exists() && !temporary.delete()) {
            throw new IllegalStateException("cannot clear stale partial report");
        }
        long completedAtUnixMs = System.currentTimeMillis();
        long completedAtElapsedNanos = SystemClock.elapsedRealtimeNanos();
        report.put("evidence_schema_version", 1);
        report.put("package", getPackageName());
        report.put("run_id", run.runId);
        report.put("run_trigger", run.trigger);
        report.put("started_at_unix_ms", run.startedAtUnixMs);
        report.put("started_at_utc", isoTimestamp(run.startedAtUnixMs));
        report.put("completed_at_unix_ms", completedAtUnixMs);
        report.put("completed_at_utc", isoTimestamp(completedAtUnixMs));
        report.put("elapsed_wall_ms", completedAtUnixMs - run.startedAtUnixMs);
        report.put(
                "elapsed_monotonic_ms",
                Math.round(
                        ((completedAtElapsedNanos - run.startedAtElapsedNanos)
                                / 1_000_000.0) * 1000.0
                ) / 1000.0
        );
        report.put("report_complete", true);
        report.put("completion_marker", REPORT_COMPLETION_MARKER);
        if (!report.has("real_audio_transcription")) {
            report.put("real_audio_transcription", false);
        }
        if (!report.has("accepted_packages_mutated")) {
            report.put("accepted_packages_mutated", false);
        }
        report.put("report_path", reportFile.getAbsolutePath());
        byte[] encoded = (report.toString(2) + "\n").getBytes(StandardCharsets.UTF_8);
        try (FileOutputStream output = new FileOutputStream(temporary)) {
            output.write(encoded);
            output.getFD().sync();
        }
        try {
            Files.move(
                    temporary.toPath(),
                    reportFile.toPath(),
                    StandardCopyOption.ATOMIC_MOVE,
                    StandardCopyOption.REPLACE_EXISTING
            );
        } catch (AtomicMoveNotSupportedException error) {
            throw new IllegalStateException("atomic report rename is unsupported", error);
        }
        return reportFile;
    }

    private static boolean isValidRunId(String runId) {
        return runId != null && RUN_ID_PATTERN.matcher(runId).matches();
    }

    private static String newManualRunId() {
        return "manual-" + System.currentTimeMillis() + "-"
                + UUID.randomUUID().toString().replace("-", "");
    }

    private static String isoTimestamp(long unixMilliseconds) {
        return Instant.ofEpochMilli(unixMilliseconds).toString();
    }

    private void postStatus(String message) {
        runOnUiThread(() -> status.setText(message));
    }

    @Override
    protected void onDestroy() {
        executor.shutdownNow();
        super.onDestroy();
    }

    private static final class RunContext {
        final String runId;
        final String trigger;
        final long startedAtUnixMs;
        final long startedAtElapsedNanos;

        RunContext(
                String runId,
                String trigger,
                long startedAtUnixMs,
                long startedAtElapsedNanos
        ) {
            if (!isValidRunId(runId)) {
                throw new IllegalArgumentException("invalid run_id");
            }
            this.runId = runId;
            this.trigger = trigger;
            this.startedAtUnixMs = startedAtUnixMs;
            this.startedAtElapsedNanos = startedAtElapsedNanos;
        }
    }
}
