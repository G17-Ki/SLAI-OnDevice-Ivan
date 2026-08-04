# Phone integration and validation plan — offline neural TTS

Status: **historical design snapshot.** The feature-flagged backend described
here was subsequently implemented and exercised on the phone in the
experimental `0.5.0-tts-lab` series; see
[`DEMO_APP_TTS_LAB.md`](DEMO_APP_TTS_LAB.md) and
[`DEVICE_RESULTS.md`](DEVICE_RESULTS.md). The retained public release and APK
remain `0.4.13-expressive-lfm`.

## Where the seam already is

The app's non-English speech already goes through one narrow interface. The
WebView calls four methods on `NativeAudioBridge`
([`android-app/app/src/main/java/ai/liquid/lfmdemo/NativeAudioBridge.java`](../../../../android-app/app/src/main/java/ai/liquid/lfmdemo/NativeAudioBridge.java)):

```java
void initializeTts(String localeTag, String turnId, long generation);
void queueTtsChunk(String text, String turnId, String chunkId, long generation);
void finishTtsResponse(String turnId, long generation);
void cancelTts(String turnId, long generation, String reason);
boolean hasTtsEngine();
```

Today every one of those is implemented against `android.speech.tts.TextToSpeech`
pinned to `com.reecedunn.espeak` (`TTS_ENGINE_PACKAGE`). The turn/generation
bookkeeping, barge-in cancellation, utterance accounting, and the
`native_tts_*` telemetry events are all engine-agnostic already.

**The change is therefore a backend swap behind that interface, not a
re-architecture.** The JavaScript in `demo-ui/index.html` should not change at
all, which also means the streaming/chunking policy, barge-in behaviour, and
telemetry schema stay identical between old and new backends — which is what
makes an A/B comparison on the phone meaningful.

## Proposed design

Extract the current implementation into a `TtsBackend` interface with two
implementations:

| Implementation | Engine | Notes |
|---|---|---|
| `SystemTtsBackend` | `android.speech.tts.TextToSpeech` → eSpeak | Exactly today's behaviour. Unchanged. Remains the rollback target. |
| `SherpaOfflineTtsBackend` | sherpa-onnx `OfflineTts` via JNI | New. Synthesises PCM on a worker thread and hands buffers to the existing native audio path rather than to `TextToSpeech.speak`. |

Selection is a feature flag, defaulting **off**:

```
lfm.tts.backend = system | sherpa_offline      (default: system)
lfm.tts.sherpa.voice.zh = matcha_zh_baker | melotts_zh
lfm.tts.sherpa.voice.ru = supertonic3 | piper_ru_dmitri
lfm.tts.sherpa.threads  = 2
lfm.tts.sherpa.steps    = 4        # Supertonic only
```

`hasTtsEngine()` must return true only when the selected backend can actually
serve the requested locale, so the existing UI fallback path
(`responseMode === "tts" && !nativeTtsEngineAvailable()`) keeps working
unchanged when assets are missing.

### Non-negotiable behaviours for the new backend

These come directly from defects measured in this run, not from general caution.

1. **Validate every PCM buffer before enqueueing it.** Reject non-finite
   samples, reject all-silent buffers, and never hand a rejected buffer to the
   audio track. Kokoro-ZH produced *entirely* non-finite PCM for short Mandarin
   input in 15/15 calls; a backend that trusts the synthesiser will emit noise
   or silence to the speaker.
2. **Enforce a minimum first-chunk length per voice.** Measured floors:
   MeloTTS-ZH needs ≥ 2 Han characters (1 character gave RMS 0.003, inaudible);
   Kokoro-ZH needs ≥ 3 (below that, invalid PCM). Merge a too-short leading
   clause into the next one rather than synthesising it alone.
3. **Bound the first flush.** The first chunk should be a sentence or ~50–70
   characters, whichever is shorter. Measured worst-case first-chunk latency on
   a full four-sentence paragraph is 3.74 s for MeloTTS and 0.75 s even for
   Piper-ZH; the ≤ 0.9 s target only holds if the orchestrator does not hand the
   engine a paragraph.
4. **Apply per-voice makeup gain, and a DC blocker where required.** MeloTTS-ZH
   peaks at 0.05–0.41 (10–20 dB below Piper/Matcha) and needs gain. Kokoro
   carries a +0.021…+0.029 DC offset and would need a one-pole DC blocker if it
   were ever enabled.
5. **Keep one TTS model resident at a time**, and unload on locale switch. Peak
   standalone RSS ranged 287–519 MiB per candidate; LFM, Qwen, and SenseVoice
   residency is on top of that and is an open gate (below).

### Asset packaging

| Route | Assets | Size | Licence obligation |
|---|---|---:|---|
| Mandarin (demo) | `matcha-icefall-zh-baker` + `vocos-22khz-univ.onnx` + lexicon/dict | ~139 MiB | **non-commercial corpus** |
| Mandarin (licence-clean) | `vits-melo-tts-zh_en` + lexicon/dict | ~182 MiB | MIT |
| Russian (primary) | Supertonic 3 INT8 (already shipped for the multilingual fallback) | 139 MiB shared with fr/it/es/en | OpenRAIL-M |
| Russian (fallback) | `vits-piper-ru_RU-dmitri-medium` + `espeak-ng-data` | 77 MiB + ~17 MiB shared | CC0 corpus, but **espeak-ng-data is GPL-3.0** |

Note the asymmetry: MeloTTS-ZH, Matcha-ZH, and Supertonic phonemise from a
bundled lexicon or their own indexer, while every Piper voice and Kokoro require
GPL-3.0 `espeak-ng-data` at runtime. Choosing Piper for Russian therefore adds a
GPL-3.0 runtime dependency that the Supertonic route does not have. The app
already ships an eSpeak dependency today, so this is a continuation rather than
a new obligation — but it should be a deliberate decision, and
`THIRD_PARTY_NOTICES.md` must be updated either way.

## Rollback plan

1. The flag defaults to `system`. Shipping the code with the flag off is a
   no-op for the existing route.
2. If the new backend misbehaves on device, flip `lfm.tts.backend=system`. No
   reinstall, no data clear, no APK change.
3. If the app must be reverted entirely, reinstall
   `android-app/releases/lfm-audio-demo-0.4.13-expressive-lfm-debug.apk`
   (SHA-256 `b178c93f05d614f8156843755ea52697f55b299c5bb0b6812209d9ce5f139baf`),
   which is retained unmodified.
4. English is unaffected in every case: Expressive-LFM does not route through
   this bridge.

## Original open gates

These were the gates at design time. The device reports linked above close or
revise several of them; naturalness and full-stack co-residency remain open.

| # | Gate | Why it is open |
|---|---|---|
| 1 | **Acoustic naturalness** | No human has scored the blinded bundle. ASR and DNSMOS cannot answer it, and DNSMOS demonstrably misranks Mandarin (it puts eSpeak above Kokoro on P.808 while ASR puts eSpeak at 55.2% CER). |
| 2 | **First audible output, end of VAD to speaker** | All host timings start after the full input waveform exists and end at a scheduling timestamp, not at speaker output. The budget is ≤ 2.0 s median / ≤ 3.0 s P95. |
| 3 | **Resident memory with LFM, Qwen, and SenseVoice loaded** | Peak RSS here is standalone: 287 MiB (Matcha) to 519 MiB (Kokoro). The incremental cost alongside the existing models is unmeasured and is the most likely reason a candidate fails on device. |
| 4 | **Phone-side RTF and playback gaps** | The 1.5×/2.0× host-slowdown simulation is a proxy. MeloTTS-ZH already gapped at 1.0× on the host, so its phone behaviour is expected to be worse, not better. |
| 5 | **Thermal behaviour over a sustained session** | Untested. Supertonic at 0.286 RTF and Matcha at 0.146 leave headroom; MeloTTS at 0.715 does not. |
| 6 | **VAD interaction and barge-in** | The new backend must cancel in-flight synthesis on new user speech as promptly as `TextToSpeech.stop()` does. Synthesis is not interruptible mid-chunk, so chunk size bounds barge-in latency. |
| 7 | **Reboot and cold-start behaviour** | Model load measured at 0.70–2.52 s warm on the host. Cold start from app assets on device is unmeasured. |
| 8 | **Locale switching** | Unload/reload cost when a turn changes language is unmeasured, and matters because only one model may be resident. |
| 9 | **Mandarin NPU route** | See `NPU_STATUS.md`. Placement is proven; the Voice AI runtime is not integrated, and the Nubia previously reported QAIRT 2.37 against the package's 2.45. |

## Recommended next physical session

1. Install a debug build with `lfm.tts.backend=sherpa_offline`,
   `lfm.tts.sherpa.voice.ru=supertonic3`, steps 4, threads 2. Keep
   `0.4.13-expressive-lfm` installed as the rollback control.
2. Ten Russian turns and ten Mandarin turns, recording native WAV readiness,
   request start, first chunk queued, first `AudioTrack` playback-head movement,
   completion, gaps above 80 ms, thermal state, and incremental RSS with LFM
   resident.
3. Repeat the English Expressive-LFM acceptance matrix unchanged, to prove no
   regression.
4. Only then consider enabling the Mandarin route, and only after a human has
   scored the blinded bundle.
