package ai.liquid.lfmdemo.runtime;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.ByteArrayOutputStream;
import java.io.EOFException;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

final class AsrHttpServer implements AutoCloseable {
    private static final int PORT = 6010;
    private static final int MAX_WAV_BYTES = 320_044;
    private static final int MAX_HEADERS = 64;
    private static final long RUNNER_TIMEOUT_MS = 75_000L;
    private static final long RUNNER_STOP_TIMEOUT_MS = 3_000L;
    private static final String RUNNER_QUIT_COMMAND =
            "__SHERPA_ONNX_PERSISTENT_QUIT__";

    private final File nativeDir;
    private final File asrRoot;
    private final File requestRoot;
    private final File logsRoot;
    private final ExecutorService clients = Executors.newFixedThreadPool(4);
    private final Object inferenceLock = new Object();
    private final BackendExecution qnnExecution = new BackendExecution();
    private final BackendExecution russianExecution = new BackendExecution();
    private final BackendExecution parakeetExecution = new BackendExecution();

    private ServerSocket serverSocket;
    private Thread acceptThread;
    private volatile ResidentRunner residentRunner;
    private volatile boolean closed;

    AsrHttpServer(File nativeDir, File asrRoot, File requestRoot) {
        this.nativeDir = nativeDir;
        this.asrRoot = asrRoot;
        this.requestRoot = requestRoot;
        this.logsRoot = new File(asrRoot, "logs");
    }

    void start() throws IOException {
        if (!requestRoot.isDirectory() && !requestRoot.mkdirs()) {
            throw new IOException("could not create ASR request directory");
        }
        if (!logsRoot.isDirectory() && !logsRoot.mkdirs()) {
            throw new IOException("could not create ASR log directory");
        }
        ServerSocket socket = new ServerSocket();
        socket.setReuseAddress(true);
        socket.bind(new InetSocketAddress(InetAddress.getByName("127.0.0.1"), PORT));
        serverSocket = socket;
        acceptThread = new Thread(this::acceptLoop, "asr-loopback-accept");
        acceptThread.setDaemon(true);
        acceptThread.start();
    }

    @Override
    public void close() {
        closed = true;
        if (serverSocket != null) {
            try {
                serverSocket.close();
            } catch (IOException ignored) {
            }
        }
        clients.shutdownNow();
        synchronized (inferenceLock) {
            stopResidentRunnerLocked();
        }
    }

    private void acceptLoop() {
        while (!closed) {
            try {
                Socket socket = serverSocket.accept();
                socket.setSoTimeout(90_000);
                clients.execute(() -> handle(socket));
            } catch (IOException error) {
                if (!closed) error.printStackTrace();
            }
        }
    }

    private void handle(Socket socket) {
        try (socket;
             InputStream input = new BufferedInputStream(socket.getInputStream());
             OutputStream output = new BufferedOutputStream(socket.getOutputStream())) {
            String requestLine = readLine(input, 4096);
            if (requestLine == null) return;
            String[] fields = requestLine.trim().split(" +", 3);
            if (fields.length != 3) {
                respondError(output, 400, "bad_request", "Malformed request line");
                return;
            }
            String method = fields[0].toUpperCase(Locale.ROOT);
            String target = fields[1];
            Map<String, String> headers = readHeaders(input);

            if ("OPTIONS".equals(method)) {
                respondJson(output, 204, new JSONObject());
                return;
            }
            if ("GET".equals(method) && pathOf(target).equals("/health")) {
                JSONObject health = healthPayload();
                respondJson(output, health.optBoolean("ok") ? 200 : 503, health);
                return;
            }
            if ("POST".equals(method) && pathOf(target).equals("/prepare")) {
                String language = queryValue(target, "language");
                if (language == null || language.isEmpty()) language = "auto";
                JSONObject prepared = prepare(language);
                respondJson(output, prepared.optBoolean("ok") ? 200 : 503, prepared);
                return;
            }
            if (!"POST".equals(method) || !pathOf(target).equals("/asr")) {
                respondError(output, 404, "not_found", "Unknown endpoint");
                return;
            }

            int length = contentLength(headers);
            if (length < 0) {
                respondError(output, 411, "content_length_required", "Content-Length is required");
                return;
            }
            if (length < 46 || length > MAX_WAV_BYTES) {
                respondError(output, 413, "content_length_too_large", "WAV must fit the fixed 10-second ASR window");
                return;
            }
            byte[] wav = readExactly(input, length);
            if (!canonicalWav(wav)) {
                respondError(output, 400, "invalid_wav", "Expected mono 16 kHz PCM WAV");
                return;
            }
            String language = queryValue(target, "language");
            if (language == null || language.isEmpty()) language = "auto";
            JSONObject payload = recognize(wav, language);
            respondJson(output, payload.optBoolean("ok") ? 200 : 503, payload);
        } catch (Exception ignored) {
            // Each request owns its socket. The next health poll still reaches
            // the listener even if a malformed client disconnects mid-body.
        }
    }

    private JSONObject recognize(byte[] wav, String language) throws IOException, JSONException {
        File wavFile = File.createTempFile("request-", ".wav", requestRoot);
        Files.write(wavFile.toPath(), wav);
        try {
            JSONArray candidates = new JSONArray();
            if ("auto".equals(language)) {
                candidates.put(candidate("sensevoice_qnn", "qnn", runQnn(wavFile, "auto")));
                candidates.put(candidate("zipformer_ru_cpu", "cpu", runRussian(wavFile)));
            } else if ("qnn-auto".equals(language)) {
                candidates.put(candidate("sensevoice_qnn", "qnn", runQnn(wavFile, "auto")));
            } else if ("en".equals(language) || "zh".equals(language)
                    || "ja".equals(language) || "ko".equals(language)) {
                candidates.put(candidate("sensevoice_qnn", "qnn", runQnn(wavFile, language)));
            } else if ("ru".equals(language)) {
                candidates.put(candidate("zipformer_ru_cpu", "cpu", runRussian(wavFile)));
            } else if ("es".equals(language) || "fr".equals(language)) {
                // Spanish and French are CPU-only on purpose. The published
                // SM8850 Parakeet contexts embed QNN 2.40.0.251030114326_189385
                // while the programme pins 2.45.0.260326154327, and this process
                // already hosts the 2.40 SenseVoice provider. Exactly one QNN
                // provider may exist per process, so an accelerated ES/FR route
                // belongs in a separate package, not here. See out/f/F2.md.
                candidates.put(candidate("parakeet_v3_cpu", "cpu", runParakeet(wavFile)));
            } else {
                return errorPayload(
                        "unsupported_language",
                        "Use auto, qnn-auto, en, zh, ja, ko, ru, es, or fr"
                );
            }

            boolean anyOk = false;
            boolean allOk = candidates.length() > 0;
            for (int index = 0; index < candidates.length(); index++) {
                boolean ok = candidates.getJSONObject(index).optBoolean("ok");
                anyOk |= ok;
                allOk &= ok;
            }
            JSONObject response = new JSONObject();
            response.put("ok", anyOk);
            response.put("complete", allOk);
            response.put("schema_version", 1);
            response.put("requested_language", language);
            response.put("selection_required", "auto".equals(language));
            response.put("app_owned", true);
            response.put("candidates", candidates);
            return response;
        } finally {
            Files.deleteIfExists(wavFile.toPath());
        }
    }

    private JSONObject candidate(String backend, String provider, RunnerResult result)
            throws JSONException {
        JSONObject candidate = new JSONObject();
        candidate.put("backend", backend);
        candidate.put("provider", provider);
        candidate.put("ok", result.json != null);
        if (result.json != null) {
            candidate.put("result", result.json);
        } else {
            JSONObject error = new JSONObject();
            error.put("code", result.code);
            error.put("message", result.message);
            candidate.put("error", error);
        }
        return candidate;
    }

    private JSONObject prepare(String language) throws JSONException {
        RunnerSpec spec = runnerSpec(language);
        if (spec == null) {
            return errorPayload(
                    "unsupported_language",
                    "Use auto, qnn-auto, en, zh, ja, ko, ru, es, or fr"
            );
        }
        if (!spec.artifactsReady) {
            return errorPayload("backend_not_ready", spec.name + " assets are incomplete");
        }
        long startedAt = android.os.SystemClock.elapsedRealtime();
        synchronized (inferenceLock) {
            try {
                boolean reused = residentRunner != null
                        && residentRunner.isAlive()
                        && residentRunner.route.equals(spec.route);
                ResidentRunner prepared = ensureResidentRunnerLocked(spec);
                return new JSONObject()
                        .put("ok", true)
                        .put("schema_version", 1)
                        .put("app_owned", true)
                        .put("requested_language", language)
                        .put("route", prepared.route)
                        .put("resident", true)
                        .put("reused", reused)
                        .put("runner_pid", prepared.pid >= 0 ? prepared.pid : JSONObject.NULL)
                        .put("runner_uid", android.os.Process.myUid())
                        .put("load_ms", android.os.SystemClock.elapsedRealtime() - startedAt);
            } catch (IOException error) {
                stopResidentRunnerLocked();
                return errorPayload("prepare_failed", error.getMessage());
            }
        }
    }

    private RunnerResult runQnn(File wav, String language) throws IOException {
        if (!qnnArtifactsReady()) {
            RunnerResult result = RunnerResult.error(
                    "qnn_not_ready",
                    "SenseVoice QNN assets are incomplete"
            );
            qnnExecution.record(result);
            return result;
        }
        File paddedWav = fixedQnnWav(wav);
        try {
            RunnerResult result = run(qnnSpec(language), paddedWav);
            qnnExecution.record(result);
            return result;
        } finally {
            if (!paddedWav.equals(wav)) Files.deleteIfExists(paddedWav.toPath());
        }
    }

    private File fixedQnnWav(File source) throws IOException {
        byte[] input = Files.readAllBytes(source.toPath());
        if (input.length == MAX_WAV_BYTES) return source;
        byte[] padded = java.util.Arrays.copyOf(input, MAX_WAV_BYTES);
        writeU32(padded, 4, MAX_WAV_BYTES - 8L);
        writeU32(padded, 40, MAX_WAV_BYTES - 44L);
        File output = File.createTempFile("qnn-fixed-10s-", ".wav", requestRoot);
        Files.write(output.toPath(), padded);
        return output;
    }

    private void writeU32(byte[] bytes, int offset, long value) {
        bytes[offset] = (byte) (value & 0xff);
        bytes[offset + 1] = (byte) ((value >>> 8) & 0xff);
        bytes[offset + 2] = (byte) ((value >>> 16) & 0xff);
        bytes[offset + 3] = (byte) ((value >>> 24) & 0xff);
    }

    private RunnerResult runRussian(File wav) throws IOException {
        if (!russianArtifactsReady()) {
            RunnerResult result = RunnerResult.error(
                    "russian_not_ready",
                    "Russian Zipformer assets are incomplete"
            );
            russianExecution.record(result);
            return result;
        }
        RunnerResult result = run(russianSpec(), wav);
        russianExecution.record(result);
        return result;
    }

    /**
     * Spanish and French through the NeMo Parakeet-TDT-0.6B-v3 INT8 transducer
     * on the CPU provider.
     *
     * <p>Deliberately the same {@code libasrrunner.so} and the same argument
     * shape as {@link #runRussian}: sherpa-onnx detects the NeMo transducer
     * model type from the encoder's ONNX metadata, so no {@code --model-type}
     * flag is needed, and no QNN library is referenced anywhere on this path.
     *
     * <p>The encoder is 652 MiB, so only the selected recognizer remains
     * resident. Switching away from ES/FR fully stops this process before the
     * next route is created.
     */
    private RunnerResult runParakeet(File wav) throws IOException {
        if (!parakeetArtifactsReady()) {
            RunnerResult result = RunnerResult.error(
                    "parakeet_not_ready",
                    "Parakeet ES/FR CPU assets are incomplete"
            );
            parakeetExecution.record(result);
            return result;
        }
        RunnerResult result = run(parakeetSpec(), wav);
        parakeetExecution.record(result);
        return result;
    }

    private RunnerSpec runnerSpec(String language) {
        return switch (language) {
            case "auto", "qnn-auto" -> qnnSpec("auto");
            case "en", "zh", "ja", "ko" -> qnnSpec(language);
            case "ru" -> russianSpec();
            case "es", "fr" -> parakeetSpec();
            default -> null;
        };
    }

    private RunnerSpec qnnSpec(String language) {
        File senseVoice = new File(new File(asrRoot, "models"), "sensevoice");
        List<String> command = new ArrayList<>();
        command.add(runner().getAbsolutePath());
        command.add("--provider=qnn");
        command.add("--tokens=" + new File(senseVoice, "tokens.txt").getAbsolutePath());
        command.add("--sense-voice-language=" + language);
        command.add("--sense-voice-use-itn=true");
        command.add("--sense-voice.qnn-backend-lib="
                + new File(nativeDir, "libQnnHtp.so").getAbsolutePath());
        command.add("--sense-voice.qnn-system-lib="
                + new File(nativeDir, "libQnnSystem.so").getAbsolutePath());
        command.add("--sense-voice.qnn-context-binary="
                + new File(senseVoice, "model.bin").getAbsolutePath());
        Map<String, String> environment = new HashMap<>();
        // This is a packaged PIE child, so it inherits the runtime-app UID.
        // The standalone linker namespace still needs the public vendor RPC
        // shim and the APK-contained HTP skeleton explicitly.
        environment.put("LD_LIBRARY_PATH", "/vendor/lib64:" + nativeDir.getAbsolutePath());
        environment.put("ADSP_LIBRARY_PATH", nativeDir.getAbsolutePath());
        return new RunnerSpec(
                "qnn",
                "sensevoice_qnn:" + language,
                command,
                environment,
                1,
                qnnArtifactsReady()
        );
    }

    private RunnerSpec russianSpec() {
        File model = new File(new File(asrRoot, "models"), "russian");
        List<String> command = new ArrayList<>();
        command.add(runner().getAbsolutePath());
        command.add("--provider=cpu");
        command.add("--tokens=" + new File(model, "tokens.txt").getAbsolutePath());
        command.add("--encoder=" + new File(model, "encoder.int8.onnx").getAbsolutePath());
        command.add("--decoder=" + new File(model, "decoder.onnx").getAbsolutePath());
        command.add("--joiner=" + new File(model, "joiner.int8.onnx").getAbsolutePath());
        command.add("--num-threads=2");
        command.add("--decoding-method=greedy_search");
        return new RunnerSpec(
                "russian",
                "zipformer_ru_cpu",
                command,
                Collections.emptyMap(),
                2,
                russianArtifactsReady()
        );
    }

    private RunnerSpec parakeetSpec() {
        File model = new File(new File(asrRoot, "models"), "parakeet");
        List<String> command = new ArrayList<>();
        command.add(runner().getAbsolutePath());
        command.add("--provider=cpu");
        command.add("--tokens=" + new File(model, "tokens.txt").getAbsolutePath());
        command.add("--encoder=" + new File(model, "encoder.int8.onnx").getAbsolutePath());
        command.add("--decoder=" + new File(model, "decoder.int8.onnx").getAbsolutePath());
        command.add("--joiner=" + new File(model, "joiner.int8.onnx").getAbsolutePath());
        command.add("--num-threads=2");
        command.add("--decoding-method=greedy_search");
        return new RunnerSpec(
                "parakeet",
                "parakeet_v3_cpu",
                command,
                Collections.emptyMap(),
                1,
                parakeetArtifactsReady()
        );
    }

    private RunnerResult run(RunnerSpec spec, File wav) throws IOException {
        synchronized (inferenceLock) {
            try {
                ResidentRunner prepared = ensureResidentRunnerLocked(spec);
                RunnerResult result = runResidentLocked(prepared, wav);
                if (result.json != null) return result;
                // A transport/protocol failure should not break the demo. Stop
                // the bad resident child and use the established one-shot path
                // for this request; the next language prepare can retry.
                stopResidentRunnerLocked();
                return runOneShotLocked(spec, wav);
            } catch (IOException error) {
                stopResidentRunnerLocked();
                return runOneShotLocked(spec, wav);
            }
        }
    }

    private ResidentRunner ensureResidentRunnerLocked(RunnerSpec spec) throws IOException {
        if (residentRunner != null
                && residentRunner.isAlive()
                && residentRunner.route.equals(spec.route)) {
            return residentRunner;
        }
        stopResidentRunnerLocked();

        List<String> command = new ArrayList<>(spec.command);
        command.add("--persistent-server=true");
        command.add("--persistent-copies=" + spec.copies);
        File stderr = new File(logsRoot, "last-" + spec.name + ".stderr");
        ProcessBuilder builder = new ProcessBuilder(command);
        builder.directory(asrRoot);
        builder.environment().put("LD_LIBRARY_PATH", nativeDir.getAbsolutePath());
        builder.environment().putAll(spec.environment);
        builder.redirectError(stderr);
        Process process = builder.start();
        ResidentRunner candidate = new ResidentRunner(
                spec.name,
                spec.route,
                process,
                new BufferedReader(new InputStreamReader(
                        process.getInputStream(), StandardCharsets.UTF_8
                )),
                new BufferedWriter(new OutputStreamWriter(
                        process.getOutputStream(), StandardCharsets.UTF_8
                )),
                processPid(process)
        );
        String ready = readResidentJsonLine(
                candidate,
                RUNNER_TIMEOUT_MS,
                "persistent_ready"
        );
        JSONObject payload = parseJsonObject(ready);
        if (payload == null || !payload.optBoolean("persistent_ready")) {
            candidate.stop();
            throw new IOException("ASR resident runner did not report protocol readiness");
        }
        residentRunner = candidate;
        return candidate;
    }

    private RunnerResult runResidentLocked(ResidentRunner runner, File wav) {
        try {
            runner.writer.write(wav.getAbsolutePath());
            runner.writer.newLine();
            runner.writer.flush();
            String raw = readResidentJsonLine(runner, RUNNER_TIMEOUT_MS, "text");
            preserveResidentResult(runner.name, raw);
            JSONObject parsed = parseJsonObject(raw);
            if (parsed == null
                    || !parsed.has("text")
                    || parsed.optString("text", "").trim().isEmpty()) {
                return RunnerResult.error(
                        "invalid_result",
                        "ASR resident runner did not return a non-empty transcript"
                ).withProcess(runner.pid, android.os.Process.myUid());
            }
            return RunnerResult.ok(parsed).withProcess(
                    runner.pid,
                    android.os.Process.myUid()
            );
        } catch (IOException error) {
            return RunnerResult.error(
                    "resident_runner_failed",
                    error.getMessage()
            ).withProcess(runner.pid, android.os.Process.myUid());
        }
    }

    private RunnerResult runOneShotLocked(RunnerSpec spec, File wav) throws IOException {
        File stdout = File.createTempFile(spec.name + "-", ".stdout", requestRoot);
        File stderr = File.createTempFile(spec.name + "-", ".stderr", requestRoot);
        List<String> command = new ArrayList<>(spec.command);
        for (int index = 0; index < spec.copies; index++) {
            command.add(wav.getAbsolutePath());
        }
        try {
            ProcessBuilder builder = new ProcessBuilder(command);
            builder.directory(asrRoot);
            builder.environment().put("LD_LIBRARY_PATH", nativeDir.getAbsolutePath());
            builder.environment().putAll(spec.environment);
            builder.redirectOutput(stdout);
            builder.redirectError(stderr);
            Process process = builder.start();
            long runnerPid = processPid(process);
            boolean completed;
            try {
                completed = process.waitFor(RUNNER_TIMEOUT_MS, TimeUnit.MILLISECONDS);
            } catch (InterruptedException error) {
                Thread.currentThread().interrupt();
                process.destroyForcibly();
                return RunnerResult.error("interrupted", "ASR request was interrupted")
                        .withProcess(runnerPid, android.os.Process.myUid());
            }
            if (!completed) {
                process.destroyForcibly();
                preserve(spec.name, stdout, stderr);
                return RunnerResult.error("timeout", "ASR backend exceeded 75 seconds")
                        .withProcess(runnerPid, android.os.Process.myUid());
            }
            preserve(spec.name, stdout, stderr);
            if (process.exitValue() != 0) {
                return RunnerResult.error(
                        "runner_failed",
                        "ASR backend exited with code " + process.exitValue()
                ).withProcess(runnerPid, android.os.Process.myUid());
            }
            JSONObject parsed = lastJsonObject(
                    new String(Files.readAllBytes(stdout.toPath()), StandardCharsets.UTF_8)
            );
            if (parsed == null
                    || !parsed.has("text")
                    || parsed.optString("text", "").trim().isEmpty()) {
                return RunnerResult.error(
                        "invalid_result",
                        "ASR backend did not return a non-empty transcript JSON object"
                ).withProcess(runnerPid, android.os.Process.myUid());
            }
            return RunnerResult.ok(parsed).withProcess(
                    runnerPid,
                    android.os.Process.myUid()
            );
        } finally {
            Files.deleteIfExists(stdout.toPath());
            Files.deleteIfExists(stderr.toPath());
        }
    }

    private String readResidentJsonLine(
            ResidentRunner runner,
            long timeoutMs,
            String requiredKey
    ) throws IOException {
        long deadline = android.os.SystemClock.elapsedRealtime() + timeoutMs;
        while (android.os.SystemClock.elapsedRealtime() < deadline) {
            if (runner.reader.ready()) {
                String line = runner.reader.readLine();
                if (line == null) throw new EOFException("ASR resident runner closed stdout");
                JSONObject payload = parseJsonObject(line);
                // The QNN provider writes diagnostic warnings to stdout before
                // both the readiness record and inference result. They are not
                // protocol frames, so keep reading until a JSON line arrives.
                if (payload == null) continue;
                if (requiredKey == null || payload.has(requiredKey)) return line;
            }
            if (!runner.isAlive()) {
                throw new EOFException("ASR resident runner exited before responding");
            }
            android.os.SystemClock.sleep(10L);
        }
        throw new IOException("ASR resident runner exceeded 75 seconds");
    }

    private void stopResidentRunnerLocked() {
        ResidentRunner runner = residentRunner;
        residentRunner = null;
        if (runner != null) runner.stop();
    }

    private void preserveResidentResult(String name, String raw) {
        try {
            Files.write(
                    new File(logsRoot, "last-" + name + ".stdout").toPath(),
                    (raw + "\n").getBytes(StandardCharsets.UTF_8)
            );
        } catch (IOException ignored) {
        }
    }

    private JSONObject parseJsonObject(String raw) {
        if (raw == null) return null;
        String line = raw.trim();
        if (!line.startsWith("{") || !line.endsWith("}")) return null;
        try {
            return new JSONObject(line);
        } catch (JSONException ignored) {
            return null;
        }
    }

    private long processPid(Process process) {
        try {
            Object value = Process.class.getMethod("pid").invoke(process);
            return value instanceof Number ? ((Number) value).longValue() : -1L;
        } catch (Exception ignored) {
            return -1L;
        }
    }

    private void preserve(String name, File stdout, File stderr) {
        try {
            Files.copy(stdout.toPath(), new File(logsRoot, "last-" + name + ".stdout").toPath(), StandardCopyOption.REPLACE_EXISTING);
            Files.copy(stderr.toPath(), new File(logsRoot, "last-" + name + ".stderr").toPath(), StandardCopyOption.REPLACE_EXISTING);
        } catch (IOException ignored) {
        }
    }

    private JSONObject lastJsonObject(String raw) {
        String[] lines = raw.split("\\R");
        for (int index = lines.length - 1; index >= 0; index--) {
            String line = lines[index].trim();
            if (!line.startsWith("{") || !line.endsWith("}")) continue;
            try {
                return new JSONObject(line);
            } catch (JSONException ignored) {
            }
        }
        return null;
    }

    private JSONObject healthPayload() throws JSONException {
        boolean qnnArtifacts = qnnArtifactsReady();
        boolean russianArtifacts = russianArtifactsReady();
        boolean parakeetArtifacts = parakeetArtifactsReady();
        boolean qnnReady = qnnArtifacts && qnnExecution.verified();
        boolean russianReady = russianArtifacts && russianExecution.verified();
        boolean parakeetReady = parakeetArtifacts && parakeetExecution.verified();
        JSONObject backends = new JSONObject();
        JSONObject qnn = new JSONObject()
                .put("ready", qnnReady)
                .put("available", qnnArtifacts)
                .put("artifacts_ready", qnnArtifacts)
                .put("execution_verified", qnnExecution.verified())
                .put("provider", "qnn")
                .put("processor", "htp_v81")
                .put("input_contract", "fixed_10_seconds_zero_padded");
        qnnExecution.appendTo(qnn);
        backends.put("sensevoice_qnn", qnn);
        JSONObject russian = new JSONObject()
                .put("ready", russianReady)
                .put("available", russianArtifacts)
                .put("artifacts_ready", russianArtifacts)
                .put("execution_verified", russianExecution.verified())
                .put("provider", "cpu");
        russianExecution.appendTo(russian);
        backends.put("zipformer_ru_cpu", russian);
        JSONObject parakeet = new JSONObject()
                .put("ready", parakeetReady)
                .put("available", parakeetArtifacts)
                .put("artifacts_ready", parakeetArtifacts)
                .put("execution_verified", parakeetExecution.verified())
                .put("provider", "cpu")
                .put("languages", new JSONArray().put("es").put("fr"))
                // Naming the reason in the health payload keeps the CPU-only
                // choice auditable from the phone instead of only in a document.
                .put("npu_blocked_reason",
                        "published SM8850 contexts are QNN 2.40; this process pins 2.40 "
                                + "SenseVoice and the programme pins 2.45");
        parakeetExecution.appendTo(parakeet);
        backends.put("parakeet_v3_cpu", parakeet);
        ResidentRunner resident = residentRunner;
        boolean residentAlive = resident != null && resident.isAlive();
        return new JSONObject()
                // `ok` means at least one backend can be attempted. A backend's
                // `ready` flag is stricter: it becomes true only after this
                // service process has produced a non-empty transcript.
                .put("ok", qnnArtifacts || russianArtifacts || parakeetArtifacts)
                // `complete` and `artifacts_complete` keep their established
                // meaning: the five accepted languages (EN/ZH/JA/KO/RU).
                // scripts/install_app_owned_runtime.sh gates on them, and a
                // not-yet-accepted backend must not be able to redefine an
                // acceptance contract by being added to the process. ES/FR
                // readiness is reported separately and gated separately.
                .put("complete", qnnReady && russianReady)
                .put("artifacts_complete", qnnArtifacts && russianArtifacts)
                .put("accepted_languages_complete", qnnReady && russianReady)
                .put("esfr_artifacts_ready", parakeetArtifacts)
                .put("esfr_complete", parakeetReady)
                .put("schema_version", 1)
                .put("app_owned", true)
                .put("service_uid", android.os.Process.myUid())
                .put("port", PORT)
                .put("resident_runner", new JSONObject()
                        .put("active", residentAlive)
                        .put("route", residentAlive ? resident.route : JSONObject.NULL)
                        .put("pid", residentAlive && resident.pid >= 0
                                ? resident.pid : JSONObject.NULL)
                        .put("uid", residentAlive
                                ? android.os.Process.myUid() : JSONObject.NULL)
                        .put("selected_route_only", true)
                        .put("protocol", "stdin-json-v1"))
                .put("backends", backends);
    }

    private boolean qnnArtifactsReady() {
        File senseVoice = new File(new File(asrRoot, "models"), "sensevoice");
        return runner().isFile()
                && new File(nativeDir, "libonnxruntime.so").isFile()
                && new File(nativeDir, "libQnnHtp.so").isFile()
                && new File(nativeDir, "libQnnSystem.so").isFile()
                && new File(nativeDir, "libQnnHtpV81Stub.so").isFile()
                && new File(nativeDir, "libQnnHtpV81Skel.so").isFile()
                && new File(senseVoice, "model.bin").isFile()
                && new File(senseVoice, "tokens.txt").isFile();
    }

    private boolean russianArtifactsReady() {
        File model = new File(new File(asrRoot, "models"), "russian");
        return runner().isFile()
                && new File(nativeDir, "libonnxruntime.so").isFile()
                && new File(model, "encoder.int8.onnx").isFile()
                && new File(model, "decoder.onnx").isFile()
                && new File(model, "joiner.int8.onnx").isFile()
                && new File(model, "tokens.txt").isFile();
    }

    private boolean parakeetArtifactsReady() {
        File model = new File(new File(asrRoot, "models"), "parakeet");
        return runner().isFile()
                && new File(nativeDir, "libonnxruntime.so").isFile()
                && new File(model, "encoder.int8.onnx").isFile()
                && new File(model, "decoder.int8.onnx").isFile()
                && new File(model, "joiner.int8.onnx").isFile()
                && new File(model, "tokens.txt").isFile();
    }

    private File runner() {
        return new File(nativeDir, "libasrrunner.so");
    }

    private Map<String, String> readHeaders(InputStream input) throws IOException {
        Map<String, String> headers = new HashMap<>();
        for (int count = 0; count < MAX_HEADERS; count++) {
            String line = readLine(input, 8192);
            if (line == null) throw new EOFException("headers ended early");
            if (line.isEmpty()) return headers;
            int colon = line.indexOf(':');
            if (colon <= 0) throw new IOException("malformed header");
            headers.put(
                    line.substring(0, colon).trim().toLowerCase(Locale.ROOT),
                    line.substring(colon + 1).trim()
            );
        }
        throw new IOException("too many headers");
    }

    private int contentLength(Map<String, String> headers) {
        try {
            return Integer.parseInt(headers.getOrDefault("content-length", "-1"));
        } catch (NumberFormatException ignored) {
            return -1;
        }
    }

    private String readLine(InputStream input, int limit) throws IOException {
        ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        while (buffer.size() <= limit) {
            int value = input.read();
            if (value < 0) {
                return buffer.size() == 0
                        ? null
                        : new String(buffer.toByteArray(), StandardCharsets.US_ASCII);
            }
            if (value == '\n') {
                byte[] bytes = buffer.toByteArray();
                int length = bytes.length;
                if (length > 0 && bytes[length - 1] == '\r') length--;
                return new String(bytes, 0, length, StandardCharsets.US_ASCII);
            }
            buffer.write(value);
        }
        throw new IOException("line too long");
    }

    private byte[] readExactly(InputStream input, int length) throws IOException {
        byte[] bytes = new byte[length];
        int offset = 0;
        while (offset < length) {
            int count = input.read(bytes, offset, length - offset);
            if (count < 0) throw new EOFException("body ended early");
            offset += count;
        }
        return bytes;
    }

    private boolean canonicalWav(byte[] wav) {
        if (wav.length < 46) return false;
        return ascii(wav, 0, "RIFF")
                && ascii(wav, 8, "WAVE")
                && ascii(wav, 12, "fmt ")
                && ascii(wav, 36, "data")
                && u16(wav, 20) == 1
                && u16(wav, 22) == 1
                && u32(wav, 24) == 16_000
                && u16(wav, 34) == 16
                && u32(wav, 40) == wav.length - 44;
    }

    private boolean ascii(byte[] bytes, int offset, String expected) {
        if (offset + expected.length() > bytes.length) return false;
        for (int index = 0; index < expected.length(); index++) {
            if ((byte) expected.charAt(index) != bytes[offset + index]) return false;
        }
        return true;
    }

    private int u16(byte[] bytes, int offset) {
        return (bytes[offset] & 0xff) | ((bytes[offset + 1] & 0xff) << 8);
    }

    private long u32(byte[] bytes, int offset) {
        return (bytes[offset] & 0xffL)
                | ((bytes[offset + 1] & 0xffL) << 8)
                | ((bytes[offset + 2] & 0xffL) << 16)
                | ((bytes[offset + 3] & 0xffL) << 24);
    }

    private String pathOf(String target) {
        int query = target.indexOf('?');
        return query >= 0 ? target.substring(0, query) : target;
    }

    private String queryValue(String target, String key) throws IOException {
        int query = target.indexOf('?');
        if (query < 0 || query + 1 >= target.length()) return null;
        for (String pair : target.substring(query + 1).split("&")) {
            String[] parts = pair.split("=", 2);
            String name = URLDecoder.decode(parts[0], StandardCharsets.UTF_8.name());
            if (!key.equals(name)) continue;
            return parts.length == 2
                    ? URLDecoder.decode(parts[1], StandardCharsets.UTF_8.name())
                    : "";
        }
        return null;
    }

    private JSONObject errorPayload(String code, String message) throws JSONException {
        return new JSONObject()
                .put("ok", false)
                .put("schema_version", 1)
                .put("app_owned", true)
                .put("error", new JSONObject().put("code", code).put("message", message));
    }

    private void respondError(OutputStream output, int status, String code, String message)
            throws IOException, JSONException {
        respondJson(output, status, errorPayload(code, message));
    }

    private void respondJson(OutputStream output, int status, JSONObject payload) throws IOException {
        byte[] body = status == 204 ? new byte[0] : payload.toString().getBytes(StandardCharsets.UTF_8);
        String statusText = switch (status) {
            case 200 -> "OK";
            case 204 -> "No Content";
            case 400 -> "Bad Request";
            case 404 -> "Not Found";
            case 411 -> "Length Required";
            case 413 -> "Payload Too Large";
            case 503 -> "Service Unavailable";
            default -> "Internal Server Error";
        };
        String headers = "HTTP/1.1 " + status + " " + statusText + "\r\n"
                + "Content-Type: application/json; charset=utf-8\r\n"
                + "Content-Length: " + body.length + "\r\n"
                + "Cache-Control: no-store\r\n"
                + "Access-Control-Allow-Origin: *\r\n"
                + "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                + "Access-Control-Allow-Headers: Content-Type, Content-Length, Accept\r\n"
                + "Access-Control-Allow-Private-Network: true\r\n"
                + "Connection: close\r\n\r\n";
        output.write(headers.getBytes(StandardCharsets.US_ASCII));
        output.write(body);
        output.flush();
    }

    private static final class RunnerSpec {
        final String name;
        final String route;
        final List<String> command;
        final Map<String, String> environment;
        final int copies;
        final boolean artifactsReady;

        RunnerSpec(
                String name,
                String route,
                List<String> command,
                Map<String, String> environment,
                int copies,
                boolean artifactsReady
        ) {
            this.name = name;
            this.route = route;
            this.command = command;
            this.environment = environment;
            this.copies = copies;
            this.artifactsReady = artifactsReady;
        }
    }

    private static final class ResidentRunner {
        final String name;
        final String route;
        final Process process;
        final BufferedReader reader;
        final BufferedWriter writer;
        final long pid;

        ResidentRunner(
                String name,
                String route,
                Process process,
                BufferedReader reader,
                BufferedWriter writer,
                long pid
        ) {
            this.name = name;
            this.route = route;
            this.process = process;
            this.reader = reader;
            this.writer = writer;
            this.pid = pid;
        }

        boolean isAlive() {
            return process.isAlive();
        }

        void stop() {
            try {
                writer.write(RUNNER_QUIT_COMMAND);
                writer.newLine();
                writer.flush();
            } catch (IOException ignored) {
            }
            try {
                writer.close();
            } catch (IOException ignored) {
            }
            try {
                if (!process.waitFor(RUNNER_STOP_TIMEOUT_MS, TimeUnit.MILLISECONDS)) {
                    process.destroy();
                    if (!process.waitFor(RUNNER_STOP_TIMEOUT_MS, TimeUnit.MILLISECONDS)) {
                        process.destroyForcibly();
                        process.waitFor(RUNNER_STOP_TIMEOUT_MS, TimeUnit.MILLISECONDS);
                    }
                }
            } catch (InterruptedException error) {
                Thread.currentThread().interrupt();
                process.destroyForcibly();
            } finally {
                try {
                    reader.close();
                } catch (IOException ignored) {
                }
            }
        }
    }

    private static final class RunnerResult {
        final JSONObject json;
        final String code;
        final String message;
        final long runnerPid;
        final int runnerUid;

        private RunnerResult(
                JSONObject json,
                String code,
                String message,
                long runnerPid,
                int runnerUid
        ) {
            this.json = json;
            this.code = code;
            this.message = message;
            this.runnerPid = runnerPid;
            this.runnerUid = runnerUid;
        }

        static RunnerResult ok(JSONObject json) {
            return new RunnerResult(json, null, null, -1, -1);
        }

        static RunnerResult error(String code, String message) {
            return new RunnerResult(null, code, message, -1, -1);
        }

        RunnerResult withProcess(long pid, int uid) {
            return new RunnerResult(json, code, message, pid, uid);
        }
    }

    private static final class BackendExecution {
        private volatile boolean attempted;
        private volatile boolean verified;
        private volatile boolean lastAttemptOk;
        private volatile long lastAttemptEpochMs;
        private volatile long lastSuccessEpochMs;
        private volatile String lastErrorCode;
        private volatile String lastErrorMessage;
        private volatile long lastRunnerPid = -1;
        private volatile int lastRunnerUid = -1;

        synchronized void record(RunnerResult result) {
            long now = System.currentTimeMillis();
            attempted = true;
            lastAttemptEpochMs = now;
            lastAttemptOk = result.json != null;
            lastRunnerPid = result.runnerPid;
            lastRunnerUid = result.runnerUid;
            if (lastAttemptOk) {
                verified = true;
                lastSuccessEpochMs = now;
                lastErrorCode = null;
                lastErrorMessage = null;
            } else {
                lastErrorCode = result.code;
                lastErrorMessage = result.message;
            }
        }

        boolean verified() {
            return verified;
        }

        void appendTo(JSONObject payload) throws JSONException {
            String state;
            if (!attempted) {
                state = "untested";
            } else if (lastAttemptOk) {
                state = "passed";
            } else if (verified) {
                state = "degraded";
            } else {
                state = "failed";
            }
            payload.put("execution_state", state);
            payload.put("last_attempt_ok", attempted ? lastAttemptOk : JSONObject.NULL);
            payload.put("last_attempt_epoch_ms", attempted ? lastAttemptEpochMs : JSONObject.NULL);
            payload.put("last_success_epoch_ms", verified ? lastSuccessEpochMs : JSONObject.NULL);
            payload.put("last_error_code", lastErrorCode == null ? JSONObject.NULL : lastErrorCode);
            payload.put("last_error_message", lastErrorMessage == null ? JSONObject.NULL : lastErrorMessage);
            payload.put("last_runner_pid", lastRunnerPid >= 0 ? lastRunnerPid : JSONObject.NULL);
            payload.put("last_runner_uid", lastRunnerUid >= 0 ? lastRunnerUid : JSONObject.NULL);
            payload.put(
                    "runner_uid_source",
                    lastRunnerUid >= 0 ? "processbuilder_inherits_service_uid" : JSONObject.NULL
            );
            payload.put(
                    "runner_owned_by_service",
                    lastRunnerUid >= 0
                            ? lastRunnerUid == android.os.Process.myUid()
                            : JSONObject.NULL
            );
        }
    }
}
