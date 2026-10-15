package ai.liquid.lfmdemo.glasses

import ai.liquid.lfmdemo.AdaptiveEnergyVad
import ai.liquid.lfmdemo.tts.Candidate
import ai.liquid.lfmdemo.tts.CandidateRegistry
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothServerSocket
import android.bluetooth.BluetoothSocket
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.util.Log
import com.k2fsa.sherpa.onnx.GenerationConfig
import com.k2fsa.sherpa.onnx.OfflineTts
import com.k2fsa.sherpa.onnx.OfflineTtsConfig
import com.k2fsa.sherpa.onnx.OfflineTtsKokoroModelConfig
import com.k2fsa.sherpa.onnx.OfflineTtsMatchaModelConfig
import com.k2fsa.sherpa.onnx.OfflineTtsModelConfig
import com.k2fsa.sherpa.onnx.OfflineTtsSupertonicModelConfig
import com.k2fsa.sherpa.onnx.OfflineTtsVitsModelConfig
import com.k2fsa.sherpa.onnx.OfflineTtsZipVoiceModelConfig
import org.json.JSONObject
import java.io.BufferedReader
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileWriter
import java.io.IOException
import java.io.InputStreamReader
import java.net.HttpURLConnection
import java.net.URL
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import java.util.UUID
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Glasses bridge: turns the phone into the cascade server for a Rokid glasses
 * client connected over Bluetooth SPP.
 *
 * Audio in : glasses mic PCM16 16 kHz mono (SPP AUDIO_IN frames) -> the same
 *            AdaptiveEnergyVad + PcmRingBuffer pipeline the on-phone VAD uses.
 * Cascade  : utterance WAV -> 127.0.0.1:6010/asr -> raw-prompt
 *            /v1/completions stream (6020) -> clause chunking -> sherpa TTS.
 * Audio out: TTS PCM16 (SPP AUDIO_OUT frames, rate announced by TTS_START).
 *
 * Half-duplex: the glasses stop mic streaming between TTS_START and TTS_STOP,
 * so the phone never hears its own voice and no echo path exists.
 */
class GlassesBridgeService : Service() {

    private val tag = "LfmGlassesBridge"

    // Phone logcat is unusable on this device (OEM restriction); keep a
    // plain-text lifecycle trail in the app's private files dir. Lazy because
    // the service context is null during construction.
    private val bridgeLogFile by lazy { File(filesDir, "glasses-bridge.log") }

    private fun blog(message: String) {
        Log.i(tag, message)
        try {
            val stamp = SimpleDateFormat("HH:mm:ss.SSS", Locale.US).format(Date())
            FileWriter(bridgeLogFile, true).use {
                it.append(stamp).append(' ').append(message).append('\n')
            }
        } catch (_: Exception) {
        }
    }

    private var wakeLock: PowerManager.WakeLock? = null
    private var serverSocket: BluetoothServerSocket? = null
    private var acceptThread: Thread? = null
    @Volatile private var running = false
    @Volatile private var activeSession: Session? = null
    private val turnQueue = LinkedBlockingQueue<Utterance>(8)
    private val turnWorker = Thread({ drainTurnQueue() }, "lfm-glasses-turn")
    private val ttsPool = TtsPool()

    private val asrHealthUrl = "http://127.0.0.1:6010/health"
    private val llmUrl = "http://127.0.0.1:6020/v1/completions"
    private val llmHealthUrl = "http://127.0.0.1:6020/health"
    private val llmModel = "Qwen3-ASR-0.6B-Text-Q4_K_M"

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        val power = getSystemService(Context.POWER_SERVICE) as PowerManager
        wakeLock = power.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "ai.liquid.lfmdemo:glasses-bridge")
        wakeLock?.setReferenceCounted(false)
        wakeLock?.acquire()
        turnWorker.start()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        blog("onStartCommand (running=$running)")
        startForeground(NOTIFICATION_ID, buildNotification())
        if (!running) {
            running = true
            startAcceptLoop()
        }
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        running = false
        turnWorker.interrupt()
        try { serverSocket?.close() } catch (_: IOException) {}
        activeSession?.close("service_destroyed")
        ttsPool.shutdown()
        if (wakeLock?.isHeld == true) wakeLock?.release()
        super.onDestroy()
    }

    fun stopBridge() {
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    // ---- SPP server ----

    private fun startAcceptLoop() {
        val adapter = BluetoothAdapter.getDefaultAdapter()
        if (adapter == null) {
            blog("no Bluetooth adapter; stopping")
            stopBridge()
            return
        }
        if (Build.VERSION.SDK_INT >= 31 &&
            checkSelfPermission(android.Manifest.permission.BLUETOOTH_CONNECT)
                != PackageManager.PERMISSION_GRANTED
        ) {
            blog("BLUETOOTH_CONNECT permission missing; stopping")
            stopBridge()
            return
        }
        if (!adapter.isEnabled) {
            blog("Bluetooth adapter is OFF; stopping")
            stopBridge()
            return
        }
        val socket = openServerSocket(adapter) ?: run {
            stopBridge()
            return
        }
        serverSocket = socket
        acceptThread = Thread({
            var current = socket
            while (running) {
                val client: BluetoothSocket
                try {
                    client = current.accept()
                } catch (error: Exception) {
                    // The RFCOMM server socket can be left in a bad state after
                    // a link reset; drop it, re-open, and keep serving. Never
                    // exit the loop: the glasses client reconnects on its own.
                    blog("accept failed: ${error.javaClass.simpleName}: ${error.message}; re-opening")
                    try { current.close() } catch (_: IOException) {}
                    val next = openServerSocket(adapter)
                    if (next == null) {
                        try { Thread.sleep(1_000) } catch (_: InterruptedException) {}
                        continue
                    }
                    current = next
                    serverSocket = current
                    continue
                }
                val previous = activeSession
                if (previous != null) {
                    Log.i(tag, "new connection while one is active; replacing")
                    previous.close("replaced")
                }
                try {
                    val next = Session(client)
                    activeSession = next
                    next.start()
                } catch (error: Exception) {
                    Log.e(tag, "session setup failed", error)
                    try { client.close() } catch (_: IOException) {}
                }
            }
        }, "lfm-glasses-accept")
        acceptThread?.start()
    }

    private fun openServerSocket(adapter: BluetoothAdapter): BluetoothServerSocket? = try {
        adapter.listenUsingRfcommWithServiceRecord(SERVICE_NAME, UUID.fromString(SPP_UUID)).also {
            blog("SPP server listening as $SERVICE_NAME")
        }
    } catch (error: Exception) {
        blog("SPP listen FAILED: ${error.javaClass.simpleName}: ${error.message}")
        null
    }

    private fun buildNotification(): Notification {
        val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        manager.createNotificationChannel(
            NotificationChannel(CHANNEL_ID, "Glasses bridge", NotificationManager.IMPORTANCE_LOW)
        )
        val open = PendingIntent.getActivity(
            this, 0, Intent(this, ai.liquid.lfmdemo.MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE
        )
        return Notification.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setContentTitle("Glasses bridge active")
            .setContentText("Listening on Bluetooth SPP")
            .setOngoing(true)
            .setContentIntent(open)
            .build()
    }

    // ---- per-connection session ----

    private inner class Session(private val socket: BluetoothSocket) {
        private val input = socket.getInputStream()
        private val output = socket.getOutputStream()
        private val writerLock = Any()
        private val frameReader = SppProtocol.Reader(input)
        private val readerAlive = AtomicBoolean(true)

        @Volatile var language = "zh"
        @Volatile var muted = false

        private val history = ArrayDeque<Pair<String, String>>()
        fun historySnapshot(): List<Pair<String, String>> = ArrayList(history)
        fun historyAdd(user: String, assistant: String) {
            history.addLast(user to assistant)
            while (history.size > MAX_HISTORY) history.removeFirst()
        }

        private val vad = AdaptiveEnergyVad(vadConfig(), { event ->
            if (event.type == AdaptiveEnergyVad.EventType.ENDPOINT) {
                pendingEndpoint = event
            }
        })
        private val ring = PcmRing(RING_CAPACITY_SAMPLES)
        @Volatile private var pendingEndpoint: AdaptiveEnergyVad.Event? = null
        private val audioCarry = ByteArrayOutputStream()

        fun start() {
            try {
                vad.start()
                blog("session started for ${socket.remoteDevice?.name}")
                healthCheck()
            } catch (error: Exception) {
                blog("session start failed: ${error.javaClass.simpleName}: ${error.message}")
                close("start_failed")
                return
            }
            Thread({ readLoop() }, "lfm-glasses-reader").start()
        }

        fun send(type: Int, payload: ByteArray) {
            synchronized (writerLock) {
                // BluetoothSocket writes can be partial: loop until every byte
                // is queued, and split large payloads (TTS chunks) into small
                // frames to keep RFCOMM traffic smooth.
                var offset = 0
                while (offset < payload.size) {
                    val length = min(MAX_SEND_PAYLOAD, payload.size - offset)
                    val frame = SppProtocol.encode(type, payload.copyOfRange(offset, offset + length))
                    output.write(frame)
                    offset += length
                }
                output.flush()
            }
        }

        fun sendControl(control: String, details: Map<String, Any> = emptyMap()) {
            val json = JSONObject().put("t", control)
            for ((key, value) in details) json.put(key, value)
            send(SppProtocol.CONTROL, json.toString().toByteArray(Charsets.UTF_8))
        }

        private fun healthCheck() {
            val problems = ArrayList<String>()
            if (!httpGetOk(asrHealthUrl, 3000)) problems.add("asr(6010)")
            if (!httpGetOk(llmHealthUrl, 3000)) problems.add("llm(6020)")
            if (problems.isEmpty()) {
                sendControl("ready", mapOf("cascade" to "ok"))
                Log.i(tag, "cascade healthy; session ready")
            } else {
                sendControl("error", mapOf("msg" to "cascade down: " + problems.joinToString(",")))
                Log.e(tag, "cascade unhealthy: $problems")
            }
        }

        private fun readLoop() {
            while (readerAlive.get()) {
                try {
                    val frame = try {
                        frameReader.nextFrame()
                    } catch (error: IOException) {
                        blog("reader closed: ${error.javaClass.simpleName}: ${error.message}")
                        break
                    }
                    if (frame == null) break
                    when (frame.type) {
                        SppProtocol.AUDIO_IN -> feedAudio(frame.payload)
                        SppProtocol.CONTROL -> handleControl(String(frame.payload, Charsets.UTF_8))
                        else -> Log.w(tag, "unexpected frame type ${frame.type}")
                    }
                } catch (error: Exception) {
                    blog("readLoop thread died: ${error.javaClass.name}: ${error.message}")
                    break
                }
            }
            close("reader_done")
        }

        private fun handleControl(raw: String) {
            val json = try {
                JSONObject(raw)
            } catch (_: Exception) {
                return
            }
            when (json.optString("t")) {
                "hello" -> {
                    val requested = json.optString("lang", "zh")
                    if (requested.length in 1..8) language = requested
                    sendControl("hello_ack", mapOf("lang" to language))
                }
                "ping" -> sendControl("pong")
                "bye" -> close("client_bye")
            }
        }

        private var fedFrames = 0L

        private fun feedAudio(bytes: ByteArray) {
            if (muted) return
            audioCarry.write(bytes, 0, bytes.size)
            while (audioCarry.size() >= FRAME_BYTES) {
                val buffer = audioCarry.toByteArray()
                audioCarry.reset()
                audioCarry.write(buffer, FRAME_BYTES, buffer.size - FRAME_BYTES)
                val frame = ShortArray(FRAME_SAMPLES)
                val view = ByteBuffer.wrap(buffer, 0, FRAME_BYTES).order(ByteOrder.LITTLE_ENDIAN)
                for (index in 0 until FRAME_SAMPLES) frame[index] = view.getShort()
                feedVadFrame(frame)
                fedFrames += 1
                if (fedFrames % 250 == 0L) {
                    val m = vad.snapshotMetrics()
                    blog(
                        "heartbeat: fedFrames=$fedFrames vadFrames=${m.framesProcessed} " +
                            "samples=${m.samplesProcessed} endpoints=${m.endpoints} " +
                            "short=${m.rejectedShortBursts} impulse=${m.rejectedImpulseFrames} " +
                            "blocked=${m.outputBlockedFrames} floor=${"%.1f".format(m.noiseFloorDbfs)}dBFS"
                    )
                }
            }
        }

        private fun feedVadFrame(frame: ShortArray) {
            val sampleOffset = vad.snapshotMetrics().samplesProcessed
            ring.append(sampleOffset, frame, 0, frame.size)
            val result = vad.processFrame(frame)
            if (result.state == AdaptiveEnergyVad.State.ENDPOINT) {
                val endpoint = pendingEndpoint
                pendingEndpoint = null
                // The VAD parks in ENDPOINT after an endpoint; rearm keeps the
                // learned noise floor and resumes listening for the next turn.
                vad.rearm()
                if (endpoint == null) return
                val utterance = try {
                    ring.slice(endpoint.recommendedAudioStartSample, endpoint.decisionSample)
                } catch (error: IllegalArgumentException) {
                    Log.w(tag, "utterance slice failed: ${error.message}")
                    return
                }
                val seconds = utterance.size / SAMPLE_RATE.toDouble()
                if (seconds < MIN_UTTERANCE_SECONDS) {
                    blog("vad endpoint: dropping ${"%.2f".format(seconds)}s (too short)")
                    return
                }
                blog("vad endpoint: ${"%.2f".format(seconds)}s utterance queued")
                val copy = utterance.copyOf()
                if (!turnQueue.offer(Utterance(copy, language))) {
                    Log.w(tag, "turn queue full; dropping oldest")
                    turnQueue.poll()
                    turnQueue.offer(Utterance(copy, language))
                }
            }
        }

        fun close(reason: String) {
            if (!readerAlive.compareAndSet(true, false)) return
            blog("session closing: $reason")
            try { socket.close() } catch (_: IOException) {}
            try { vad.cancel() } catch (_: Exception) {}
            if (activeSession === this) activeSession = null
        }
    }

    // ---- turn cascade ----

    private class Utterance(val pcm: ShortArray, val language: String)

    private fun drainTurnQueue() {
        while (true) {
            val utterance = try {
                turnQueue.take()
            } catch (_: InterruptedException) {
                return
            }
            val active = activeSession ?: continue
            runTurn(active, utterance)
        }
    }

    private fun runTurn(session: Session, utterance: Utterance) {
        val startedAt = System.currentTimeMillis()
        val turnId = "g-" + turnCounter.incrementAndGet()
        blog("turn $turnId start: ${utterance.pcm.size / SAMPLE_RATE}s ${utterance.language}")
        try {
            val wav = makeWav(pcm16ToLittleEndian(utterance.pcm))
            try {
                File(filesDir, "turn-$turnId.wav").writeBytes(wav)
            } catch (_: Exception) {
            }
            val transcript = asrTranscript(wav, utterance.language)
            if (transcript.isEmpty()) {
                blog("turn $turnId: ASR empty")
                session.sendControl("error", mapOf("msg" to "asr_empty"))
                return
            }
            blog("turn $turnId heard: $transcript")
            session.sendControl("heard", mapOf("text" to transcript))

            val tts = ttsPool.engineFor(utterance.language)
            val history = synchronized(session) { session.historySnapshot() }
            val prompt = buildRawPrompt(history, transcript)
            val response = StringBuilder()
            val chunker = ClauseChunker(utterance.language)
            var ttsStarted = false

            fun speakClause(clause: String) {
                if (tts == null) return
                if (!ttsStarted) {
                    ttsStarted = true
                    session.muted = true
                    session.sendControl(
                        "tts_start",
                        mapOf("rate" to tts.sampleRate, "channels" to 1, "bits" to 16)
                    )
                }
                synthesizeClause(session, tts, clause)
            }

            streamLlm(prompt) { delta ->
                response.append(delta)
                for (clause in chunker.feed(delta)) speakClause(clause)
            }
            for (clause in chunker.flush()) speakClause(clause)
            if (ttsStarted) {
                session.sendControl("tts_stop")
                session.muted = false
            }
            val answer = response.toString().replace(Regex("\\s+"), " ").trim()
            if (answer.isEmpty()) {
                session.sendControl("error", mapOf("msg" to "llm_empty"))
            } else {
                synchronized(session) { session.historyAdd(transcript, answer) }
                session.sendControl("said", mapOf("text" to answer))
            }
            blog(
                "turn $turnId done in ${System.currentTimeMillis() - startedAt}ms, " +
                    "said: $answer"
            )
        } catch (error: Exception) {
            blog("turn $turnId FAILED: ${error.javaClass.simpleName}: ${error.message}")
            try {
                session.sendControl("error", mapOf("msg" to (error.message ?: "turn_failed")))
                session.muted = false
            } catch (_: IOException) {}
        }
    }

    // ---- ASR ----

    private fun asrTranscript(wav: ByteArray, language: String): String {
        val response = httpPost(
            "http://127.0.0.1:6010/asr?language=" + java.net.URLEncoder.encode(language, "UTF-8"),
            wav,
            mapOf("Content-Type" to "audio/wav", "Accept" to "application/json"),
            20_000
        )
        val json = JSONObject(response)
        if (!json.optBoolean("ok", false)) {
            throw IOException("asr not ok: ${json.optJSONObject("error")?.optString("message") ?: response.take(200)}")
        }
        val candidates = json.optJSONArray("candidates")
        if (candidates == null) {
            blog("asr raw (no candidates): ${response.take(400)}")
            return ""
        }
        for (index in 0 until candidates.length()) {
            val candidate = candidates.optJSONObject(index) ?: continue
            if (!candidate.optBoolean("ok", false)) continue
            // Companion schema nests the transcript under result (see
            // demo-ui normalizeCompanionCandidate); flat text kept as fallback.
            val text = normalizeTranscript(
                candidate.optJSONObject("result")?.optString("text")
                    ?: candidate.optString("text")
            )
            if (text.isNotEmpty()) return text
        }
        blog("asr raw (empty text): ${response.take(400)}")
        return ""
    }

    private fun normalizeTranscript(raw: String): String {
        var text = raw.replace(Regex("\\s+"), " ").trim()
        text = text
            .replace(Regex("^(?:transcript|transcription)\\s*:\\s*", RegexOption.IGNORE_CASE), "")
            .trim()
        if ((text.startsWith("\"") && text.endsWith("\"")) ||
            (text.startsWith("“") && text.endsWith("”"))
        ) {
            text = text.substring(1, text.length - 1).trim()
        }
        return text
    }

    // ---- LLM streaming ----

    private fun streamLlm(prompt: String, onDelta: (String) -> Unit) {
        val payload = JSONObject()
            .put("model", llmModel)
            .put("stream", true)
            .put("cache_prompt", true)
            .put("prompt", prompt)
            .put("temperature", 0)
            .put("top_k", 1)
            .put("max_tokens", 768)
        val connection = (URL(llmUrl).openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            setConnectTimeout(3000)
            setReadTimeout(60_000)
            doOutput = true
            setRequestProperty("Content-Type", "application/json")
            outputStream.use { it.write(payload.toString().toByteArray()) }
        }
        try {
            if (connection.responseCode != 200) {
                throw IOException("llm http ${connection.responseCode}")
            }
            val reader = BufferedReader(InputStreamReader(connection.inputStream, Charsets.UTF_8))
            var line: String? = reader.readLine()
            while (line != null) {
                if (line.isEmpty() || !line.startsWith("data:")) {
                    line = reader.readLine()
                    continue
                }
                val data = line.substring(5).trim()
                line = reader.readLine()
                if (data == "[DONE]") break
                val event = try {
                    JSONObject(data)
                } catch (_: Exception) {
                    continue
                }
                val choice = event.optJSONArray("choices")?.optJSONObject(0) ?: continue
                val delta = choice.optString("text").ifEmpty {
                    choice.optJSONObject("delta")?.optString("content") ?: ""
                }
                if (delta.isNotEmpty()) onDelta(delta)
            }
        } finally {
            connection.disconnect()
        }
    }

    // ---- raw prompt contract (mirrors demo-ui buildQwen3AsrPrompt) ----

    private fun buildRawPrompt(history: List<Pair<String, String>>, userText: String): String {
        val prompt = StringBuilder()
        prompt.append("下面是一段用户和助手之间的真实对话。\n")
        prompt.append("你现在扮演“助手”。\n")
        prompt.append("请站在助手的角色回答最后一个用户问题。\n")
        prompt.append("注意区分用户说的“我”和对助手说的“你”。\n")
        prompt.append("只回答问题，不要继续编写新的用户对话。\n\n")
        prompt.append("系统：").append(RAW_PROMPT_SYSTEM).append("\n")
        for ((user, assistant) in history) {
            prompt.append("\n用户：").append(user).append("\n助手回答：").append(assistant)
        }
        prompt.append("\n用户：").append(userText).append("\n助手回答：")
        return prompt.toString()
    }

    // ---- TTS ----

    private fun synthesizeClause(session: Session, entry: TtsEntry, clause: String) {
        var gain = 0.0f
        entry.tts.generateWithConfigAndCallback(clause, entry.config) { chunk ->
            if (gain == 0.0f) {
                var sum = 0.0
                var peak = 0.0f
                for (sample in chunk) {
                    sum += sample * sample
                    if (abs(sample) > peak) peak = abs(sample)
                }
                val rms = sqrt(sum / chunk.size).toFloat()
                gain = 0.9f / max(max(rms * 3.0f, peak), 0.01f)
            }
            val out = ByteArray(chunk.size * 2)
            val view = ByteBuffer.wrap(out).order(ByteOrder.LITTLE_ENDIAN)
            for (sample in chunk) {
                val scaled = (sample * gain).coerceIn(-0.95f, 0.95f)
                view.putShort((scaled * Short.MAX_VALUE).toInt().toShort())
            }
            try {
                session.send(SppProtocol.AUDIO_OUT, out)
            } catch (error: IOException) {
                return@generateWithConfigAndCallback 0
            }
            1
        }
    }

    // ---- TTS pool ----

    private class TtsEntry(
        val tts: OfflineTts,
        val candidate: Candidate,
        val config: GenerationConfig,
        val sampleRate: Int
    )

    private inner class TtsPool {
        private val lock = Any()
        private val engines = HashMap<String, TtsEntry>()
        private var loaded: List<Candidate>? = null

        fun engineFor(language: String): TtsEntry? = synchronized(lock) {
            engines[language] ?: loadFor(language)
        }

        private fun loadFor(language: String): TtsEntry? {
            val candidates = loaded ?: run {
                val bundleRoot = File(getExternalFilesDir(null) ?: filesDir, TTS_BUNDLE_DIR)
                val result = try {
                    CandidateRegistry.load(bundleRoot).first
                } catch (error: Exception) {
                    Log.e(tag, "TTS bundle load failed", error)
                    null
                }
                loaded = result
                result
            } ?: return null
            val candidate = candidates
                .filter { it.language == language || it.generationLanguage == language }
                .maxByOrNull { it.autoPriority }
                ?: candidates.maxByOrNull { it.autoPriority }
                ?: return null
            return try {
            val entry = buildEngine(
                candidate,
                File(getExternalFilesDir(null) ?: filesDir, TTS_BUNDLE_DIR)
            )
            engines[language] = entry
            blog("TTS engine for $language: ${candidate.id} @ ${entry.sampleRate}Hz")
            entry
            } catch (error: Exception) {
                Log.e(tag, "TTS engine load failed for $language", error)
                null
            }
        }

        private fun buildEngine(candidate: Candidate, bundleRoot: File): TtsEntry {
            val model = OfflineTtsModelConfig(
                numThreads = candidate.numThreads,
                debug = false,
                provider = "cpu",
            )
            when (candidate.engine) {
                "vits" -> model.vits = OfflineTtsVitsModelConfig(
                    model = candidate.path(bundleRoot, "model"),
                    lexicon = candidate.path(bundleRoot, "lexicon"),
                    tokens = candidate.path(bundleRoot, "tokens"),
                    dataDir = candidate.path(bundleRoot, "data_dir"),
                    dictDir = candidate.path(bundleRoot, "dict_dir"),
                )
                "matcha" -> model.matcha = OfflineTtsMatchaModelConfig(
                    acousticModel = candidate.path(bundleRoot, "acoustic_model"),
                    vocoder = candidate.path(bundleRoot, "vocoder"),
                    lexicon = candidate.path(bundleRoot, "lexicon"),
                    tokens = candidate.path(bundleRoot, "tokens"),
                    dataDir = candidate.path(bundleRoot, "data_dir"),
                    dictDir = candidate.path(bundleRoot, "dict_dir"),
                )
                "kokoro" -> model.kokoro = OfflineTtsKokoroModelConfig(
                    model = candidate.path(bundleRoot, "model"),
                    voices = candidate.path(bundleRoot, "voices"),
                    tokens = candidate.path(bundleRoot, "tokens"),
                    dataDir = candidate.path(bundleRoot, "data_dir"),
                    dictDir = candidate.path(bundleRoot, "dict_dir"),
                    lexicon = candidate.files["lexicon"]
                        ?.split(",")
                        ?.joinToString(",") { File(bundleRoot, it.trim()).absolutePath }
                        ?: "",
                )
                "supertonic" -> model.supertonic = OfflineTtsSupertonicModelConfig(
                    durationPredictor = candidate.path(bundleRoot, "duration_predictor"),
                    textEncoder = candidate.path(bundleRoot, "text_encoder"),
                    vectorEstimator = candidate.path(bundleRoot, "vector_estimator"),
                    vocoder = candidate.path(bundleRoot, "vocoder"),
                    ttsJson = candidate.path(bundleRoot, "tts_json"),
                    unicodeIndexer = candidate.path(bundleRoot, "unicode_indexer"),
                    voiceStyle = candidate.path(bundleRoot, "voice_style"),
                )
                "zipvoice" -> model.zipvoice = OfflineTtsZipVoiceModelConfig(
                    tokens = candidate.path(bundleRoot, "tokens"),
                    encoder = candidate.path(bundleRoot, "encoder"),
                    decoder = candidate.path(bundleRoot, "decoder"),
                    vocoder = candidate.path(bundleRoot, "vocoder"),
                    dataDir = candidate.path(bundleRoot, "data_dir"),
                    lexicon = candidate.path(bundleRoot, "lexicon"),
                )
                else -> throw IllegalArgumentException("Unknown TTS engine '${candidate.engine}'")
            }
            val tts = OfflineTts(null, OfflineTtsConfig(model = model))
            val config = GenerationConfig(
                silenceScale = 1.0f,
                speed = candidate.speed,
                sid = candidate.sid,
                referenceAudio = null,
                referenceSampleRate = 0,
                referenceText = candidate.referenceText ?: "",
                numSteps = candidate.numSteps,
                extra = candidate.extra,
            )
            return TtsEntry(tts, candidate, config, tts.sampleRate())
        }

        fun shutdown() {
            synchronized(lock) {
                for (entry in engines.values) {
                    try { entry.tts.release() } catch (_: Exception) {}
                }
                engines.clear()
            }
        }
    }

    // ---- helpers ----

    private fun vadConfig(): AdaptiveEnergyVad.Config =
        AdaptiveEnergyVad.Config.builder()
            .setPreRollMs(PRE_ROLL_MS)
            .setSilenceEndpointMs(ENDPOINT_HANGOVER_MS)
            .setMaxUtteranceMs(MAX_UTTERANCE_MS)
            .setBargeInPolicy(AdaptiveEnergyVad.BargeInPolicy.DISABLED)
            .build()

    private fun httpGetOk(url: String, timeoutMs: Int): Boolean {
        return try {
            val connection = (URL(url).openConnection() as HttpURLConnection).apply {
                setConnectTimeout(timeoutMs)
                setReadTimeout(timeoutMs)
            }
            val code = connection.responseCode
            connection.disconnect()
            code in 200..299
        } catch (_: Exception) {
            false
        }
    }

    private fun httpPost(url: String, body: ByteArray, headers: Map<String, String>, timeoutMs: Int): String {
        val connection = (URL(url).openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            setConnectTimeout(3000)
            setReadTimeout(timeoutMs)
            doOutput = true
            for ((key, value) in headers) setRequestProperty(key, value)
            outputStream.use { it.write(body) }
        }
        try {
            val code = connection.responseCode
            val stream = if (code in 200..299) connection.inputStream else connection.errorStream
            val text = stream?.bufferedReader()?.use { it.readText() } ?: ""
            if (code !in 200..299) throw IOException("http $code: ${text.take(200)}")
            return text
        } finally {
            connection.disconnect()
        }
    }

    private fun pcm16ToLittleEndian(samples: ShortArray): ByteArray {
        val out = ByteArray(samples.size * 2)
        val view = ByteBuffer.wrap(out).order(ByteOrder.LITTLE_ENDIAN)
        for (sample in samples) view.putShort(sample)
        return out
    }

    private fun makeWav(pcm: ByteArray): ByteArray {
        val out = ByteArrayOutputStream(44 + pcm.size)
        fun ascii(text: String) {
            out.write(text.toByteArray(Charsets.US_ASCII))
        }
        fun le32(value: Int) {
            out.write(value and 0xFF); out.write((value shr 8) and 0xFF)
            out.write((value shr 16) and 0xFF); out.write((value shr 24) and 0xFF)
        }
        fun le16(value: Int) {
            out.write(value and 0xFF); out.write((value shr 8) and 0xFF)
        }
        ascii("RIFF"); le32(36 + pcm.size); ascii("WAVEfmt "); le32(16); le16(1); le16(1)
        le32(SAMPLE_RATE); le32(SAMPLE_RATE * 2); le16(2); le16(16); ascii("data"); le32(pcm.size)
        out.write(pcm)
        return out.toByteArray()
    }

    companion object {
        const val CHANNEL_ID = "glasses-bridge"
        private const val NOTIFICATION_ID = 62
        private const val SERVICE_NAME = "LFM-Glasses"
        private const val SPP_UUID = "00001101-0000-1000-8000-0010a98c3f83"
        private const val TTS_BUNDLE_DIR = "tts-eval"
        private const val SAMPLE_RATE = 16_000
        private const val FRAME_SAMPLES = 320
        private const val FRAME_BYTES = FRAME_SAMPLES * 2
        private const val MAX_SEND_PAYLOAD = 8192
        private const val PRE_ROLL_MS = 240
        private const val ENDPOINT_HANGOVER_MS = 420
        private const val MAX_UTTERANCE_MS = 15_000
        private const val MIN_UTTERANCE_SECONDS = 0.4
        private const val MAX_HISTORY = 10
        private const val RING_CAPACITY_SAMPLES =
            (MAX_UTTERANCE_MS + PRE_ROLL_MS + ENDPOINT_HANGOVER_MS) * SAMPLE_RATE / 1000
        private const val RAW_PROMPT_SYSTEM =
            "你是一个有帮助的中文语音助手。请根据对话内容直接、简洁、准确地回答用户的问题。"
        private val turnCounter = AtomicLong(0)
    }
}

/**
 * Streaming clause chunker; mirrors demo-ui extractSafeTtsChunks (the TTS
 * phrasing rules verified on-phone) so the glasses hear the same chunking the
 * phone speaker hears.
 */
internal class ClauseChunker(private val language: String) {
    private var remainder = ""
    private val commaMinChars = 32
    private val cjkLeadCommaMinChars = 10
    private val minBoundedChars = 50
    private val maxChars = 70
    private val allowCjkLeadComma = language in setOf("zh", "ja", "ko")

    fun feed(delta: String): List<String> {
        remainder += delta.replace("\r", "")
        return extract(flush = false)
    }

    fun flush(): List<String> = extract(flush = true)

    private fun extract(flush: Boolean): List<String> {
        val chunks = ArrayList<String>()
        while (remainder.trim().isNotEmpty()) {
            var boundary = -1
            var index = 0
            while (index < remainder.length) {
                val character = remainder[index]
                var end = index + 1
                while (end < remainder.length && CLOSING.matches(remainder[end].toString())) end += 1
                val next: Char? = if (end < remainder.length) remainder[end] else null
                val isUnicodeStop = CLAUSE_STOP.matches(character.toString())
                val isAsciiStop = ASCII_STOP.matches(character.toString()) &&
                    (flush || next?.isWhitespace() == true)
                val isNewline = character == '\n'
                var isComma = false
                if (COMMA.matches(character.toString())) {
                    val commaPrefix = remainder.substring(0, end).trim()
                    val cjkLeadCommaReady = character == '，' ||
                        (language == "ja" && character == '、') ||
                        (character == ',' && next?.isWhitespace() == true)
                    val isCjkLeadComma = allowCjkLeadComma && chunks.isEmpty() &&
                        commaPrefix.length >= cjkLeadCommaMinChars && cjkLeadCommaReady
                    val isTightCjkComma = isCjkLeadComma && TIGHT_CJK.matches(character.toString())
                    isComma = (commaPrefix.length >= commaMinChars || isCjkLeadComma) &&
                        (isTightCjkComma || flush || next == null ||
                            next.isWhitespace())
                }
                if (isUnicodeStop || isAsciiStop || isNewline || isComma) {
                    boundary = end
                    break
                }
                index += 1
            }
            if (boundary < 0 && remainder.length > maxChars) {
                val candidate = remainder.substring(0, min(maxChars + 1, remainder.length))
                val whitespace = max(candidate.lastIndexOf(' '), candidate.lastIndexOf('\n'))
                if (whitespace >= minBoundedChars) boundary = whitespace + 1
            }
            if (boundary < 0) {
                if (!flush) break
                boundary = remainder.length
            }
            val chunk = remainder.substring(0, boundary).replace(Regex("\\s+"), " ").trim()
            remainder = remainder.substring(boundary).trimStart()
            if (chunk.isNotEmpty()) chunks.add(chunk)
        }
        return chunks
    }

    companion object {
        private val CLOSING = Regex("[\"'”’）)\\]】》」』]")
        private val CLAUSE_STOP = Regex("[。！？；：]")
        private val ASCII_STOP = Regex("[.!?;:]")
        private val COMMA = Regex("[,，、]")
        private val TIGHT_CJK = Regex("[，、]")
    }
}

/**
 * Minimal monotonic-offset PCM ring, mirroring the semantics of
 * NativeAudioBridge.PcmRingBuffer (package-private, not visible here):
 * append at the VAD's running sample offset, slice by absolute offsets.
 */
internal class PcmRing(capacitySamples: Int) {
    private val samples = ShortArray(
        capacitySamples.also { require(it > 0) { "capacitySamples must be positive" } }
    )
    private var firstIndex = 0
    private var size = 0
    private var firstSampleOffset = 0L
    private var nextSampleOffset = 0L

    fun append(sampleOffset: Long, source: ShortArray, offset: Int, length: Int) {
        require(sampleOffset == nextSampleOffset) {
            "non-contiguous PCM: expected $nextSampleOffset but received $sampleOffset"
        }
        for (index in 0 until length) {
            if (size < samples.size) {
                samples[(firstIndex + size) % samples.size] = source[offset + index]
                size += 1
            } else {
                samples[firstIndex] = source[offset + index]
                firstIndex = (firstIndex + 1) % samples.size
                firstSampleOffset += 1
            }
            nextSampleOffset += 1
        }
    }

    fun slice(startSample: Long, endSample: Long): ShortArray {
        require(startSample >= firstSampleOffset &&
            endSample >= startSample &&
            endSample <= nextSampleOffset
        ) {
            "PCM slice [$startSample, $endSample) outside retained " +
                "[$firstSampleOffset, $nextSampleOffset)"
        }
        val result = ShortArray((endSample - startSample).toInt())
        var readIndex = ((firstIndex + (startSample - firstSampleOffset)) % samples.size).toInt()
        for (index in result.indices) {
            result[index] = samples[readIndex]
            readIndex = (readIndex + 1) % samples.size
        }
        return result
    }
}
