package ai.liquid.lfmdemo.supertonicchaindiagprobe;

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

/** Launcher for the isolated four-component Supertonic ORT-QNN probe. */
public final class ProbeActivity extends Activity {
    public static final String EXTRA_RUN_PROBE = "run_probe";
    public static final String EXTRA_RUN_ID = "run_id";
    public static final String REPORT_COMPLETION_MARKER =
            "SUPERTONIC_CHAIN_DIAGNOSTIC_REPORT_COMPLETE_V2";
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
        title.setText("Supertonic 3 · QNN stage diagnostic v2");
        title.setTextSize(24);
        title.setTypeface(Typeface.DEFAULT, Typeface.BOLD);
        title.setTextColor(Color.rgb(46, 24, 92));
        content.addView(title);

        TextView boundary = new TextView(this);
        boundary.setText(
                "Isolated offline compatibility scaffold. It executes duration, text "
                        + "encoder, four vector steps, and vocoder, but emits no file, "
                        + "plays no audio, and cannot promote audio quality."
        );
        boundary.setTextSize(16);
        boundary.setPadding(0, padding / 2, 0, padding);
        content.addView(boundary);

        runButton = new Button(this);
        runButton.setText("Run four raw ONNX components on QNN HTP");
        runButton.setAllCaps(false);
        runButton.setEnabled(BuildConfig.SUPERTONIC_CHAIN_DIAGNOSTIC_PAYLOAD_EMBEDDED);
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
                BuildConfig.SUPERTONIC_CHAIN_DIAGNOSTIC_PAYLOAD_EMBEDDED
                        ? "Ready. Every model, fixture, golden, and runtime hash is checked first."
                        : "Stub build: stage the external full-chain payload and rebuild."
        );
        content.addView(status);

        ScrollView scroll = new ScrollView(this);
        scroll.addView(content);
        setContentView(scroll);

        if (state == null && getIntent().getBooleanExtra(EXTRA_RUN_PROBE, false)) {
            String runId = getIntent().getStringExtra(EXTRA_RUN_ID);
            getIntent().removeExtra(EXTRA_RUN_PROBE);
            getIntent().removeExtra(EXTRA_RUN_ID);
            if (BuildConfig.SUPERTONIC_CHAIN_DIAGNOSTIC_PAYLOAD_EMBEDDED
                    && isValidRunId(runId)) {
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
        status.setText("Installing and hashing the app-owned chain payload…");
        executor.submit(() -> {
            SupertonicChainDiagnosticQnnProbe.Progress progress =
                    new SupertonicChainDiagnosticQnnProbe.Progress();
            try {
                progress.enterStage("payload_install", "install_and_hash");
                SupertonicChainDiagnosticPayload payload =
                        SupertonicChainDiagnosticPayload.installAndVerify(this);
                progress.completeStage(
                        "payload_install",
                        new JSONObject()
                                .put("component", "payload_install")
                                .put("manifest_verified", true)
                                .put("assets_verified", true)
                                .put("native_libraries_verified", true)
                                .put("model_storage", payload.root().getAbsolutePath())
                                .put(
                                        "runtime_storage",
                                        payload.nativeLibraryDir().getAbsolutePath()
                                )
                );
                postStatus(
                        "Payload verified. Running four raw ONNX components on QNN HTP…"
                );
                JSONObject report = SupertonicChainDiagnosticQnnProbe.run(
                        payload, progress
                );
                progress.appendSuccessEvidence(report);
                persistReport(report, run);
                postStatus(report.toString(2));
            } catch (Throwable error) {
                try {
                    JSONObject failure = new JSONObject();
                    failure.put("schema_version", 1);
                    failure.put("gate_id", SupertonicChainDiagnosticPayload.EXPECTED_GATE_ID);
                    failure.put("status", "failed_closed_stage_diagnostic");
                    failure.put("error_type", error.getClass().getName());
                    failure.put("error", String.valueOf(error.getMessage()));
                    failure.put("cpu_ep_fallback_disabled", true);
                    failure.put("full_neural_chain", false);
                    failure.put("chained_pcm_tensor", false);
                    failure.put("full_tts", false);
                    failure.put("audio_file_output", false);
                    failure.put("audio_playback", false);
                    failure.put("audio_quality_accepted", false);
                    failure.put("host_reference_comparisons_promoted", false);
                    failure.put("production_integration", false);
                    failure.put("accepted_packages_mutated", false);
                    progress.appendFailureEvidence(failure);
                    persistReport(failure, run);
                    postStatus(failure.toString(2));
                } catch (Throwable reportError) {
                    postStatus(
                            "FAILED CLOSED\n" + error + "\nReport error: " + reportError
                    );
                }
            } finally {
                runOnUiThread(() -> runButton.setEnabled(
                        BuildConfig.SUPERTONIC_CHAIN_DIAGNOSTIC_PAYLOAD_EMBEDDED
                ));
            }
        });
    }

    private File persistReport(JSONObject report, RunContext run) throws Exception {
        File reportDir = new File(
                getFilesDir(), "supertonic_chain_diagnostic_probe/reports"
        );
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
