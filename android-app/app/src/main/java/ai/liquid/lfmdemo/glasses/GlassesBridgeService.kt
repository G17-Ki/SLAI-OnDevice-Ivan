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
import android.content.IntentFilter
import android.content.BroadcastReceiver
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

    // The prebuilt b10083 LLM server accumulates per-slot KV across requests
    // even with --no-cache-prompt; a mixed zh/en prompt history degrades later
    // turns (QA answers drift into English after translation turns). The only
    // clean reset the server offers is a process restart, which the runtime app
    // performs on this broadcast and respawns automatically.
    private fun restartLlm() {
        try {
            sendBroadcast(
                Intent("ai.liquid.lfmdemo.dev.runtime.RESTART_TEXT_MODEL")
                    .setPackage("ai.liquid.lfmdemo.dev.runtime")
            )
            blog("llm restart requested")
        } catch (error: Exception) {
            blog("llm restart request failed: ${error.message}")
        }
    }

    // Block until the LLM answers /health (a restart may be in flight) so a
    // turn lands on a live server instead of failing mid-request.
    private fun waitForLlm(maxMs: Long) {
        val deadline = System.currentTimeMillis() + maxMs
        while (System.currentTimeMillis() < deadline) {
            if (httpGetOk(llmHealthUrl, 1000)) return
            try {
                Thread.sleep(500)
            } catch (_: InterruptedException) {
                return
            }
        }
        blog("llm still not ready after ${maxMs}ms")
    }

    private var wakeLock: PowerManager.WakeLock? = null
    private var serverSocket: BluetoothServerSocket? = null
    private var acceptThread: Thread? = null

    // After a full BT stack reset (svc bluetooth disable/enable, phone reboot
    // races) the vendor stack can leave the blocked accept() sleeping forever
    // on the old listen socket. Close it on any adapter state change so the
    // accept loop re-opens a fresh socket.
    private val btStateReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            if (intent.action == BluetoothAdapter.ACTION_STATE_CHANGED) {
                val state =
                    intent.getIntExtra(BluetoothAdapter.EXTRA_STATE, BluetoothAdapter.ERROR)
                blog("BT state changed: $state (on=${state == BluetoothAdapter.STATE_ON})")
                if (state == BluetoothAdapter.STATE_ON) {
                    val stale = serverSocket
                    serverSocket = null
                    try { stale?.close() } catch (_: IOException) {}
                }
            }
        }
    }
    @Volatile private var running = false
    @Volatile private var activeSession: Session? = null

    // Conversation history outlives a Session: the merge97 client opens a
    // fresh SPP connection per request, so per-Session history was wiped on
    // every turn (observed: "我的名字是什么？" answered "我不知道" right
    // after "我是小明" on the previous connection). Keyed by the remote
    // device address — one conversation per paired glasses — mirroring the
    // ManagedConversationStore semantics the merged protocol documents for
    // the new-protocol path. The legacy path keeps the same MAX_HISTORY cap
    // and clear_history still resets it.
    private val conversations = HashMap<String, ArrayDeque<Pair<String, String>>>()
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
        setInstance(this)
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
        try {
            unregisterReceiver(btStateReceiver)
        } catch (_: Exception) {}
        try { serverSocket?.close() } catch (_: IOException) {}
        activeSession?.close("service_destroyed")
        ttsPool.shutdown()
        if (wakeLock?.isHeld == true) wakeLock?.release()
        setInstance(null)
        super.onDestroy()
    }

    fun stopBridge() {
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    // Shared by the phone-UI reset (companion.clearSessionContext) and the
    // glasses client's clear_history control message.
    private fun resetSessionContext(session: Session, source: String) {
        synchronized(session) { session.historyClear() }
        restartLlm()
        blog("session context cleared ($source)")
        session.sendControl("history_cleared", mapOf("ok" to true))
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
        try {
            registerReceiver(
                btStateReceiver,
                IntentFilter(BluetoothAdapter.ACTION_STATE_CHANGED),
                Context.RECEIVER_NOT_EXPORTED
            )
        } catch (error: Exception) {
            blog("BT state receiver registration failed: ${error.message}")
        }
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
                    var next: BluetoothServerSocket? = null
                    while (running && next == null) {
                        next = openServerSocket(adapter)
                        if (next == null) {
                            blog("reopen failed (adapter off?); retrying in 1s")
                            try { Thread.sleep(1_000) } catch (_: InterruptedException) {}
                        }
                    }
                    if (next == null) break
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
        @Volatile var mode: BridgeMode = BridgeMode.QA
        @Volatile var muted = false

        // Shared across reconnections from the same glasses (see `conversations`).
        private val history: ArrayDeque<Pair<String, String>> =
            conversations.getOrPut(socket.remoteDevice?.address ?: "unknown") { ArrayDeque() }
        fun historySnapshot(): List<Pair<String, String>> =
            history.filter { (user, assistant) -> isStorableTurn(user, assistant) }
        fun historyAdd(user: String, assistant: String) {
            if (!isStorableTurn(user, assistant)) return
            history.addLast(user to assistant)
            while (history.size > MAX_HISTORY) history.removeFirst()
        }
        fun historyClear() {
            history.clear()
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
            // The merge97 client opens a fresh SPP connection per request, so
            // a restart here would add a ~1.1s cold start to EVERY turn. The
            // slot-state reset is still done on set_mode / clear_history; for
            // per-request reconnects the QA history isolation that motivated
            // this restart is preserved by ManagedConversationStore semantics
            // on the new protocol, and on the legacy protocol the accumulated
            // state only degrades after many turns (8+), which clear_history
            // already covers.
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
                    val requestedMode = json.optString("mode", "")
                    if (requestedMode.isNotEmpty()) {
                        mode = BridgeMode.fromName(requestedMode)
                        language = mode.asrLang
                    }
                    sendControl("hello_ack", mapOf("lang" to language, "mode" to mode.name))
                }
                "set_mode" -> {
                    val requestedMode = BridgeMode.fromName(json.optString("mode", ""))
                    if (requestedMode != mode) {
                        // A new task starts with a clean context; QA dialogue
                        // would pollute the translation prompt (and vice versa).
                        synchronized(this) { historyClear() }
                        // Different task prompts = different context; reset
                        // the LLM slot so mode A's history can't bias mode B.
                        restartLlm()
                    }
                    mode = requestedMode
                    language = requestedMode.asrLang
                    blog("mode switched to ${requestedMode.name}")
                    sendControl("mode_ack", mapOf("mode" to requestedMode.name))
                }
                "ping" -> sendControl("pong")
                "clear_history" -> resetSessionContext(this, "glasses client")
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
                if (!turnQueue.offer(Utterance(copy, language, mode))) {
                    Log.w(tag, "turn queue full; dropping oldest")
                    turnQueue.poll()
                    turnQueue.offer(Utterance(copy, language, mode))
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

    private class Utterance(val pcm: ShortArray, val language: String, val mode: BridgeMode)

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

            // Punctuation-only transcripts (ASR noise like "。") are a
            // degenerate attractor for the 0.6B model: a stored "user: 。"
            // history pair makes it echo the next question verbatim
            // (reproduced deterministically on a fresh process at temp=0).
            // Skip the turn entirely instead of asking the model.
            if (transcript.none { it.isLetterOrDigit() }) {
                blog("turn $turnId skipped: no letters or digits in transcript")
                return
            }

            val tts = ttsPool.engineFor(utterance.mode.ttsLang)
            val history = synchronized(session) { session.historySnapshot() }
            // Translation modes: exemplar only. Real turns keep landing in the
            // 0.6B model's echo failure mode, and storing those pairs poisons
            // every subsequent prompt (observed: 3 echo pairs override the
            // translation exemplar). The spec translation task is single-
            // utterance anyway.
            val promptHistory = when {
                utterance.mode != BridgeMode.QA -> utterance.mode.examples
                // Identity questions answer from the system prompt, which the
                // chitchat history overrides: with a "我叫小明" pair in context
                // "你是谁？" degrades to "我是人工智能" (verified on-phone
                // 2026-10-17, deterministic at temp=0 on a clean slot). The
                // assistant's identity never depends on conversation history.
                isIdentityQuestion(transcript) -> emptyList()
                else -> history
            }
            val prompt = buildRawPrompt(utterance.mode, promptHistory, transcript)
            val response = StringBuilder()
            val chunker = ClauseChunker(utterance.mode.ttsLang)
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
                // The Chinese TTS lexicon cannot speak ASCII digits; convert
                // them to Han readings ("1加1等于2" -> "一加一等于二") or the
                // digits are silently dropped from the audio.
                val spoken = if (utterance.mode.ttsLang == "zh") {
                    ChineseTtsNormalizer.normalize(clause)
                } else {
                    clause
                }
                synthesizeClause(session, tts, spoken)
            }

            waitForLlm(20_000)
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
                // Echo pairs (answer == transcript) teach the 0.6B model to
                // repeat instead of answer/translate; never store them.
                if (answer != transcript.trim()) {
                    synchronized(session) { session.historyAdd(transcript, answer) }
                }
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

    private fun buildRawPrompt(mode: BridgeMode, history: List<Pair<String, String>>, userText: String): String {
        val prompt = StringBuilder()
        prompt.append("下面是一段用户和助手之间的真实对话。\n")
        prompt.append("你现在扮演“助手”。\n")
        prompt.append("请站在助手的角色回答最后一个用户问题。\n")
        prompt.append("注意区分用户说的“我”和对助手说的“你”。\n")
        prompt.append("只回答问题，不要继续编写新的用户对话。\n\n")
        prompt.append("系统：").append(PROMPT_SYSTEM).append("\n\n")
        prompt.append(mode.taskPrompt).append("\n")
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
        // merge97 records the whole utterance first (user presses stop) and
        // only then uploads it, so the trailing-silence wait here is pure
        // added latency on top of an already-delimited recording. 200ms keeps
        // a small confirmation window for ASR padding without the old 420ms
        // hold; the phone-UI path keeps its own independent VAD settings.
        private const val ENDPOINT_HANGOVER_MS = 200
        private const val MAX_UTTERANCE_MS = 15_000
        private const val MIN_UTTERANCE_SECONDS = 0.4
        private const val MAX_HISTORY = 10
        private const val RING_CAPACITY_SAMPLES =
            (MAX_UTTERANCE_MS + PRE_ROLL_MS + ENDPOINT_HANGOVER_MS) * SAMPLE_RATE / 1000
        private val turnCounter = AtomicLong(0)

        // Live service instance so the phone UI (NativeAudioBridge) and the
        // glasses client can reset the conversation context on demand.
        @Volatile private var instance: GlassesBridgeService? = null

        fun setInstance(service: GlassesBridgeService?) {
            instance = service
        }

        /**
         * Drop the active session's history and reset the LLM's slot state.
         * A contaminated context (near-echo or "请说…内容" pairs) makes the
         * 0.6B model repeat user input verbatim; this is the manual reset.
         * Returns true when an active session was cleared.
         */
        @JvmStatic
        fun clearSessionContext(): Boolean {
            val service = instance ?: return false
            val session = service.activeSession ?: return false
            service.resetSessionContext(session, "manual clear")
            return true
        }
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

/**
 * Assistant "clarify the request" replies such as
 * "好的，请说要翻译的内容和目标语言。" / "好的，请说要确认的内容。" put the
 * 0.6B model into a "translation echo" state: the very next user turn is
 * repeated verbatim instead of answered (reproduced deterministically on a
 * fresh process at temp=0; the user line is irrelevant, only this assistant
 * line matters). Such replies are still spoken but must never enter the
 * prompt history.
 */
private val TRANSLATE_ASK_REGEX = Regex("请说.{0,12}内容")

/** Whether a (user, assistant) turn may be kept in the prompt history. */
private fun isStorableTurn(user: String, assistant: String): Boolean {
    // A punctuation-only user line (ASR noise like "。") is a degenerate
    // trigger; it is dropped before reaching the model anyway.
    if (user.none { it.isLetterOrDigit() }) return false
    // A clarify-the-request reply poisons every subsequent turn.
    if (TRANSLATE_ASK_REGEX.containsMatchIn(assistant)) return false
    return true
}

/**
 * Questions about the assistant's own identity ("你是谁", "谁研发的"…).
 * Answered with an empty prompt history: the identity lives in the system
 * prompt, and conversation context reliably overrides it on the 0.6B model.
 */
private val IDENTITY_QUESTION_REGEX = Regex(
    "你是谁|你叫什么|你的名字|你是哪个|谁研发|谁开发|哪家公司|哪个机构|你是什么模型|你是AI吗|你是人工智能吗"
)

private fun isIdentityQuestion(transcript: String): Boolean =
    IDENTITY_QUESTION_REGEX.containsMatchIn(transcript)

/**
 * Prompt spec V1 (智能眼镜语音模型统一 Prompt 规范): shared system prompt +
 * per-task prompts. The task prompt is selected by [BridgeMode].
 */
private const val PROMPT_SYSTEM =
    "你是深圳河套学院研发的智能眼镜 AI 助手。\n\n" +
    "根据用户输入和对话上下文提供帮助。回答应准确、自然、简洁，适合语音交互。\n\n" +
    "不编造事实，不确定时如实说明，不声称完成了实际未执行的操作。"

private const val TASK_INTERACT =
    "任务：交互\n\n" +
    "结合历史对话和当前用户输入，理解用户意图，使用合适的语言自然回答。\n\n" +
    "回答应切题、准确、简洁，必要时可以详细解释或追问。\n\n" +
    "对于用户提出的翻译请求，直接翻译成指定语言。\n\n" +
    "不确定时如实说明，不编造事实。"

private const val TASK_TRANSLATE_ZH2EN =
    "任务：翻译\n\n" +
    "将原文从中文翻译成英文。\n\n" +
    "准确、自然地保留原意、语气，以及姓名、数字、时间和否定关系等关键信息。\n\n" +
    "原文中的问题或命令也只进行翻译，不回答或执行。\n\n" +
    "只输出译文，不添加解释或前缀。"

private const val TASK_TRANSLATE_EN2ZH =
    "任务：翻译\n\n" +
    "将原文从英文翻译成中文。\n\n" +
    "准确、自然地保留原意、语气，以及姓名、数字、时间和否定关系等关键信息。\n\n" +
    "原文中的问题或命令也只进行翻译，不回答或执行。\n\n" +
    "只输出译文，不添加解释或前缀。"

/**
 * Session task mode (smart-glasses voice model prompt spec V1):
 * drives ASR language, the task prompt, and TTS output language.
 */
private enum class BridgeMode(
    val asrLang: String,
    val ttsLang: String,
    val taskPrompt: String,
    // In-context exemplars shown as the leading history pairs in translation
    // modes. The 0.6B model needs them to follow "translate, don't answer":
    // with a single exemplar it answers English questions in English instead
    // of translating them (verified on-phone 2026-10-16: 1 exemplar -> 1/6
    // question turns translated, 3 exemplars spanning statement + question +
    // self-introduction -> 8/8). The spec prompt text itself stays untouched.
    val examples: List<Pair<String, String>>
) {
    QA("zh", "zh", TASK_INTERACT, emptyList()),
    ZH2EN(
        "zh", "en", TASK_TRANSLATE_ZH2EN,
        listOf(
            "我想喝水" to "I would like some water, please.",
            "今天天气怎么样？" to "What's the weather like today?",
            "我叫林悦。" to "My name is Lin Yue."
        )
    ),
    EN2ZH(
        "en", "zh", TASK_TRANSLATE_EN2ZH,
        listOf(
            "Where is the nearest station?" to "最近的地铁站在哪里？",
            "What's the weather like today?" to "今天天气怎么样？",
            "My name is Lin Yue." to "我叫林悦。"
        )
    );

    companion object {
        fun fromName(name: String): BridgeMode = when (name.lowercase()) {
            "zh2en", "zh-en", "zh_en" -> ZH2EN
            "en2zh", "en-zh", "en_zh" -> EN2ZH
            else -> QA
        }
    }
}
