package ai.liquid.lfmdemo.supertonicprobe;

import android.app.Activity;
import android.graphics.Color;
import android.graphics.Typeface;
import android.os.Bundle;
import android.os.SystemClock;
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
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.regex.Pattern;

/** Coexistable launcher for the isolated raw-ONNX Supertonic duration probe. */
public final class ProbeActivity extends Activity {
    public static final String EXTRA_RUN_PROBE = "run_probe";
    public static final String EXTRA_RUN_ID = "run_id";
    public static final String REPORT_COMPLETION_MARKER =
            "SUPERTONIC_DURATION_PROBE_REPORT_COMPLETE_V1";
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
        title.setText("Supertonic 3 · raw ONNX QNN canary");
        title.setTextSize(24);
        title.setTypeface(Typeface.DEFAULT, Typeface.BOLD);
        title.setTextColor(Color.rgb(33, 21, 79));
        content.addView(title);

        TextView boundary = new TextView(this);
        boundary.setText(
                "Duration predictor only. Offline, no microphone, no network, no audio. "
                        + "CPU EP fallback is disabled and accepted packages are untouched."
        );
        boundary.setTextSize(16);
        boundary.setPadding(0, padding / 2, 0, padding);
        content.addView(boundary);

        runButton = new Button(this);
        runButton.setText("Run duration predictor on QNN HTP");
        runButton.setAllCaps(false);
        runButton.setEnabled(BuildConfig.SUPERTONIC_PAYLOAD_EMBEDDED);
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
                BuildConfig.SUPERTONIC_PAYLOAD_EMBEDDED
                        ? "Ready. Every staged model, fixture, and runtime hash is checked first."
                        : "Stub build: stage a payload and set SUPERTONIC_PROBE_PAYLOAD_DIR."
        );
        content.addView(status);

        ScrollView scroll = new ScrollView(this);
        scroll.addView(content);
        setContentView(scroll);

        if (state == null && getIntent().getBooleanExtra(EXTRA_RUN_PROBE, false)) {
            String runId = getIntent().getStringExtra(EXTRA_RUN_ID);
            getIntent().removeExtra(EXTRA_RUN_PROBE);
            getIntent().removeExtra(EXTRA_RUN_ID);
            if (BuildConfig.SUPERTONIC_PAYLOAD_EMBEDDED && isValidRunId(runId)) {
                RunContext run = new RunContext(
                        runId,
                        "automation_intent",
                        System.currentTimeMillis(),
                        SystemClock.elapsedRealtimeNanos()
                );
                runButton.post(() -> runProbe(runButton, run));
            } else {
                status.setText("Automation rejected: missing payload or invalid run_id.");
            }
        }
    }

    private void runProbe(View ignored, RunContext run) {
        runButton.setEnabled(false);
        status.setText("Installing and hashing the app-owned payload…");
        executor.submit(() -> {
            try {
                SupertonicPayload payload = SupertonicPayload.installAndVerify(this);
                postStatus("Payload verified. Compiling/loading raw ONNX through QNN HTP…");
                JSONObject report = SupertonicQnnProbe.run(payload);
                persistReport(report, run);
                postStatus(report.toString(2));
            } catch (Throwable error) {
                try {
                    JSONObject failure = new JSONObject();
                    failure.put("schema_version", 1);
                    failure.put("gate_id", SupertonicPayload.EXPECTED_GATE_ID);
                    failure.put("status", "failed_closed");
                    failure.put("error_type", error.getClass().getName());
                    failure.put("error", String.valueOf(error.getMessage()));
                    failure.put("cpu_ep_fallback_disabled", true);
                    failure.put("full_tts", false);
                    failure.put("audio_output", false);
                    failure.put("production_integration", false);
                    failure.put("accepted_packages_mutated", false);
                    persistReport(failure, run);
                    postStatus(failure.toString(2));
                } catch (Throwable reportError) {
                    postStatus("FAILED CLOSED\n" + error + "\nReport error: " + reportError);
                }
            } finally {
                runOnUiThread(() -> runButton.setEnabled(
                        BuildConfig.SUPERTONIC_PAYLOAD_EMBEDDED
                ));
            }
        });
    }

    private void persistReport(JSONObject report, RunContext run) throws Exception {
        File reportDir = new File(getFilesDir(), "supertonic_probe/reports");
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
        report.put("started_at_utc", Instant.ofEpochMilli(run.startedAtUnixMs).toString());
        report.put("completed_at_unix_ms", completedAtUnixMs);
        report.put("completed_at_utc", Instant.ofEpochMilli(completedAtUnixMs).toString());
        report.put(
                "elapsed_monotonic_ms",
                Math.round(((completedAtElapsedNanos - run.startedAtElapsedNanos)
                        / 1_000_000.0) * 1000.0) / 1000.0
        );
        report.put("report_complete", true);
        report.put("completion_marker", REPORT_COMPLETION_MARKER);
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
    }

    private void postStatus(String message) {
        runOnUiThread(() -> status.setText(message));
    }

    private static boolean isValidRunId(String runId) {
        return runId != null && RUN_ID_PATTERN.matcher(runId).matches();
    }

    private static String newManualRunId() {
        return "manual-" + System.currentTimeMillis() + "-"
                + UUID.randomUUID().toString().replace("-", "");
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
            this.runId = runId;
            this.trigger = trigger;
            this.startedAtUnixMs = startedAtUnixMs;
            this.startedAtElapsedNanos = startedAtElapsedNanos;
        }
    }
}
