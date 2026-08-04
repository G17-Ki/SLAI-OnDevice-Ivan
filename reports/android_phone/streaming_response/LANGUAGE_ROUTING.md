# Language routing for the phone demo

Date: 2026-07-23

## Decision

Keep **streamed integrated LFM for English**. For other languages, use a
fixed-locale modular route:

`ASR -> compact text model -> clause-chunked TTS`

No modular multilingual route below has been measured end to end on the Nubia.
The table deliberately separates four evidence classes:

- **PHONE**: measured on the Nubia. This currently exists only for the original
  integrated English LFM runner, and measures first PCM rather than acoustic
  speaker start.
- **MAC**: measured on the Apple M2 host.
- **AH-QRD**: account-owned Qualcomm AI Hub result on Snapdragon 8 Elite Gen 5
  QRD / SM8850 / HTP v81.
- **PUB-v81**: upstream or Qualcomm-published same-generation HTP-v81 evidence;
  it is not a Nubia measurement.

The repository also contains built but phone-unverified Android
`0.3.4-input-handoff`: a system-`TextToSpeech` control mode for auto plus
EN/ZH/ES/JA/KO/DE/IT/FR, with the redundant native-microphone Base64 round trip
removed. It depends on the voices installed on the phone, so it is not promoted
into the routing matrix as measured TTS evidence. The default installer remains
on `0.3.2` until physical acceptance.

## Routing matrix

| Language | Recommended ASR | Text model | Recommended TTS | Current evidence strength | Known blockers |
|---|---|---|---|---|---|
| **EN** | **Integrated LFM** for the primary route; SenseVoiceSmall INT8 for a modular fallback; X-ASR 480 ms is a streaming challenger | Integrated LFM; otherwise **LFM2.5-1.2B** | Stream integrated PCM immediately; Piper-EN or Supertonic are standalone fallbacks | **PHONE:** integrated first PCM 800/837 ms median/P95. **MAC:** SenseVoice and X-ASR each reached 15.8% WER on the shared 3 clips; X-ASR partials began after 1.6-3.1 s of audio. **AH-QRD:** Qualcomm Zipformer contexts reached 4.741/0.053/0.137 ms with all operations on NPU; the accumulated Supertonic English chain backchecked at 0.0 WER on one phrase despite strict waveform drift. Piper-EN release contexts run unchanged on exact SM8850: 34.337 ms acoustic core, 35.104 ms for one call each of all six graphs, and 2,089/2,089 HTP operations. **PUB-v81:** Piper full component sum 35.225 ms. | Updated playback has not been validated by the physical phone playback head. X-ASR is not yet phone/QNN validated and its Mandarin route failed. The Qualcomm Zipformer context lacks transcript-equivalence evidence. Piper has no exact-run golden, English audio, or phone integration yet. Supertonic still needs broader listening/language and phone gates. |
| **ES** | **Whisper Small**, fixed locale | **LFM2.5-1.2B**; Qwen budget-16 is optional, not necessary on the tiny diagnostic | **MeloTTS-ES** as the Qualcomm-native target; Supertonic CPU fallback | **MAC:** Whisper Small 19.0% WER on 3 clips; Supertonic 433/443 ms first PCM; LFM passed 9/9 text checks. **PUB-v81:** Melo component sum 138.213 ms, all reported layers on NPU. | No fresh Melo QRD run or Nubia runtime. Three ASR clips are only a screening set. |
| **DE** | **Whisper Small**, fixed locale | **LFM2.5-1.2B** | **Piper-DE** as the Qualcomm-native target; Supertonic CPU fallback | **MAC:** Whisper Small 4.8% WER on 3 clips; Supertonic 451/484 ms; LFM passed 9/9 text checks. **PUB-v81:** Piper component sum 35.181 ms, all reported layers on NPU. | No fresh full Piper-DE QRD chain and no Nubia measurement. |
| **FR** | **Whisper Small**, fixed locale | **LFM2.5-1.2B** | **Supertonic** | **MAC:** Whisper Small 25.0% WER on 3 clips; LFM passed 9/9 text checks; one four-step Supertonic synthesis completed in 367 ms. **AH-QRD:** shared Supertonic neural graphs are fully NPU placed and one English accumulated chain passed an ASR intelligibility sanity check. | French TTS has only a one-run host timing and no local intelligibility score. Language-specific QRD audio and human listening remain open. |
| **IT** | **Whisper Small**, fixed locale | **Qwen3-1.7B budget-16** quality route; LFM2.5-1.2B low-latency fallback | **Piper-IT** | **MAC:** Qwen passed 9/9 stability checks; LFM passed 6/9. **AH-QRD:** all four Piper neural components pass: encoder 16.958 ms, SDP 6.602 ms, Flow 8.930 ms, decoder 1.821 ms; 34.311 ms through the first decoder call and 1,449/1,449 NPU layers. The corrected 79-ID chain produced 1.614 seconds of finite audio in four decoder calls; final audio passed the 3%/3% gate, and Whisper Small recovered the exact sentence from local and remote WAVs at 0.0 WER/CER. **PUB-v81:** full Piper component sum 35.253 ms. | The ASR result is one phrase, not broad Italian quality. The chain retains a one-element Flow warning. Phonemization, orchestration, phone integration, queueing/playback, human listening, and physical Nubia audible onset remain unmeasured. |
| **ZH** | **SenseVoiceSmall INT8**, fixed locale | **LFM2.5-1.2B** for latency; Qwen budget-16 did not improve the diagnostic failure rate | **MeloTTS-ZH**; guarded Kokoro only as a secondary CPU fallback | **MAC:** SenseVoice 33.3% CER on 3 clips; LFM and Qwen each passed 6/9 text checks. Kokoro longer samples backchecked at 4.0% micro CER. **AH-QRD:** exact SenseVoice context passed at 10.871 ms with 2,597/2,597 HTP operations. Public MeloTTS-ZH SM8850-AD contexts ran unchanged on exact plain SM8850: BERT 2.679 ms, encoder 23.697 ms, flow 74.563 ms, decoder 43.900 ms; 144.839 ms total, 2,372/2,372 HTP operations, zero fallback. | SenseVoice Mandarin quality is not yet open-ended-demo grade. Kokoro produced invalid PCM on 10/31 short Mandarin safety calls, including after warm-up. Melo still needs a real Mandarin preprocessing-to-audio chain, numerical comparison, listening/ASR backcheck, and Nubia execution. |
| **JA** | **SenseVoiceSmall INT8**, fixed locale | **LFM2.5-1.2B** for latency; Qwen budget-16 had the same 6/9 diagnostic score | **LFM Japanese standalone TTS** as the current quality-evidence leader; Supertonic is the lighter challenger | **MAC:** SenseVoice 31.3% CER on 3 clips. LFM-JP standalone TTS reached 61.8/62.3 ms first PCM and backchecked exactly. Supertonic reached 385/467 ms, but broader Japanese backchecks showed 23.1% CER with Whisper Tiny and 30.8% CER in a separate SenseVoice check. **AH-QRD:** one exact-SM8850 accumulated Supertonic phrase backchecked identically across dynamic-local, fixed-local, and remote audio at 0.0 CER despite remote/fixed waveform NRMSE 0.751. **PUB-v81:** exact SenseVoice SM8850 context exists. | Never use the current persistent interleaved LFM-JP mode: turn 2 produced mixed Chinese/Japanese text and near-clipped PCM. The new Supertonic result is one short phrase, not a reversal of the broader errors; human A/B, representative phrases, and Nubia playback/RSS/thermal tests remain open. The standalone LFM bundle is heavy. |
| **KO** | **SenseVoiceSmall INT8**, fixed locale | **LFM2.5-1.2B** | **Supertonic** | **MAC:** SenseVoice 14.9% CER on 3 clips; LFM passed 9/9 text checks; Supertonic reached 543/628 ms and its short sample retranscribed exactly. **AH-QRD:** shared Supertonic graphs are fully NPU placed; one English accumulated chain retained exact ASR words. | Korean-specific QRD audio, human listening, and end-to-end phone execution remain open. |
| **RU** | **Whisper Small**, fixed locale | **Qwen3-1.7B budget-16** quality route; LFM2.5-1.2B low-latency fallback | **Supertonic** | **MAC:** Qwen passed 9/9 stability checks; LFM passed 6/9; Supertonic reached 466/535 ms and the short sample retranscribed exactly. **AH-QRD:** one English accumulated chain passed an ASR intelligibility sanity check. | Russian ASR was not included in the local audio fixture. No dedicated Qualcomm Piper/Melo route is exposed, and Russian-specific QRD audio/listening remains open. |

The EN/ZH/JA/KO SenseVoice rows now share one additional **AH-QRD** result:
the checksum-pinned ten-second SM8850 context passed job `j5w1ov24g` at
10.871 ms estimated inference and 126.223 MiB estimated peak memory, with all
2,597 runtime operations on Hexagon/HTP and zero fallback. It was generated
with QNN 2.40 and profiled under AI Hub QAIRT 2.45, so it is a compatibility
and placement result rather than a matching-runtime or transcript-equivalence
measurement. The physical Nubia QNN route remains open.

The ES/DE/FR/IT/RU Whisper rows now share an **AH-QRD** placement and latency
result. Pinned public W8A16 contexts generated for Galaxy S26 `sm8850-ad` ran
unchanged on exact plain SM8850: the fixed 30-second encoder reached
186.277 ms with 2,014/2,014 HTP operations, and each decoder call reached
3.932 ms with 2,853/2,853. Encoder plus 20 decoder calls is 264.917 ms of
neural-component arithmetic. This is cross-SKU context portability, not a
fresh compile, transcript-equivalence result, end-to-end ASR latency, Nubia
runtime, or streaming encoder; see [`whisper_v81/RESULT.md`](whisper_v81/RESULT.md).

The MeloTTS-ZH exact-QRD result is similarly scoped: it establishes that the
pinned public SM8850-AD QAIRT-2.45 contexts are portable to exact plain SM8850
and stay fully on HTP. Its 144.839 ms neural-component sum excludes host
preprocessing/attention, repeated decoder overlap calls, transfer, queueing,
and playback, and it is not Mandarin intelligibility evidence.

The Piper-EN release-context result is also profile-only. The public
`sm8850-ad` contexts ran unchanged on exact plain SM8850 with zero fallback:
34.337 ms and 1,448/1,448 HTP operations for encoder + SDP + flow + one audio-
decoder call; 35.104 ms and 2,089/2,089 HTP operations for one call each of all
six QNN graphs. One audio-decoder call exposes about 464.4 ms of new audio after
overlap slicing, which supports payload streaming. The package has no matching
inputs/goldens, so English equivalence and quality remain open; see
[`piper_en_v81/RESULT.md`](piper_en_v81/RESULT.md).

The Piper-IT result is stronger than a placement-only profile but remains
narrow. Its four-call production fixture passes the final-waveform gate and a
fixed-locale Small-ASR sentence check, while one of 294,912 Flow values misses
the combined 3%/3% tolerance. See
[`piper_v81/PRODUCTION_CHAIN_RESULT.md`](piper_v81/PRODUCTION_CHAIN_RESULT.md).

## Text-model policy

Use **LFM2.5-1.2B by default**. Its Q4 file is 664 MiB and the 28-case
diagnostic reached 85.7% with first speakable text at 111/215 ms median/P95 on
the M2. It is the smaller and substantially faster choice.

Use **Qwen3-1.7B with reasoning budget 16** only when its measured quality gain
matters, initially Italian and Russian. Across the same 28-case diagnostic it
reached 89.3%; the three-repeat stability run produced first speakable text at
509/707 ms. Budgets 32 and 64 had the same 89.3% score and were slower, so 16
is the finite-reasoning Pareto choice in the current evidence. Qwen still
failed one Chinese, one Japanese, and one Korean fact, so it is not a universal
quality upgrade.

These are tiny deterministic diagnostics, not general language benchmarks.

## Staged loading

The phone should not keep every model family resident:

1. Keep VAD, audio I/O, and the language/session selector always ready. Keep
   integrated LFM resident only while the English route is active.
2. At session activation, select and prewarm exactly one ASR family:
   SenseVoice for EN/JA/KO/ZH, or Whisper Small for ES/DE/FR/IT/RU. Cache the
   locale instead of detecting it on every utterance.
3. Keep one text model resident: LFM2.5-1.2B normally, or Qwen budget-16 for a
   selected quality route. Do not keep both merely to switch per turn.
4. Load one output engine for the session locale: integrated LFM for EN;
   Piper for EN/DE/IT; MeloTTS for ES/ZH; Supertonic as the shared broad
   fallback. Load the large LFM-JP bundle only for an explicit Japanese mode
   and unload it on exit.
5. Prewarm the chosen route before the first question, validate finite PCM for
   every TTS callback, and flush the first short speakable clause immediately.
   For Kokoro Mandarin, merge chunks below a provisional three-Han-character
   floor and retain fallback even above it.

Use the Android system-TTS mode only as a bring-up/control path until the phone
confirms the requested local voice is installed and acceptable. Its
`UtteranceProgressListener.onStart` event is a software start proxy, not an
acoustic-onset timestamp.

This policy is motivated by host memory evidence: SenseVoice and Whisper Small
each reached roughly 614-621 MiB incremental peak RSS, Supertonic reached about
300 MiB process peak in its main host run, Kokoro reached 572 MiB, and the
Japanese LFM ready server was about 1.44 GiB. Independent peaks must not be
added as a phone-RSS prediction, but they make simultaneous residency an
unnecessary risk.

## Planning arithmetic, not phone latency

Assumptions: timing starts after VAD has closed and the full input waveform is
available; locale and models are already warm; stages run serially; TTS starts
as soon as the first speakable text is ready; the current 140 ms playback
cushion is used; orchestration, Android audio-device startup, thermals, and any
phone slowdown are excluded.

| Planning case | Arithmetic from independent M2 P95 measurements | Sum |
|---|---:|---:|
| SenseVoice + default LFM + worst tested short-language Supertonic case | 174 + 215 + 628 + 140 ms | **1,157 ms** |
| SenseVoice + Qwen budget-16 + same Supertonic case | 174 + 707 + 628 + 140 ms | **1,649 ms** |
| Whisper Small on Mac CPU + default LFM + same Supertonic case | 2,382 + 215 + 628 + 140 ms | **3,365 ms** |
| Whisper Small W8A16 exact-QRD components + default LFM + same Supertonic case | 265 + 215 + 628 + 140 ms | **1,248 ms** |

The CPU row explains why the Latin-language ASR path needed HTP acceleration.
The W8A16 row is mixed component arithmetic: it substitutes exact-QRD encoder
plus 20 decoder calls for a phone ASR measurement and still excludes feature
extraction, host decode, transfers, and transcript finalization. None of these
sums is an end-to-end Nubia projection.

## Evidence sources

- [Integrated streaming and retained Nubia timing](REPORT.md)
- [SenseVoiceSmall host and SM8850 package evidence](sensevoice_asr_mac_m2/REPORT.md)
- [SenseVoice exact-SM8850 AI Hub profile](sensevoice_asr_mac_m2/AIHUB_V81_PROFILE.md)
- [Zipformer/X-ASR host gates and exact-SM8850 component profiles](zipformer_v81/RELEASE_CONTEXT_PROFILE.md)
- [Whisper multilingual host and published v81 evidence](multilingual_asr_mac_m2/README.md)
- [LFM2.5-1.2B multilingual diagnostic](lfm2_5_1_2b_q4_multilingual_voice_quality.json)
- [Qwen budget-16 stability diagnostic](qwen3_1_7b_q4_reasoning_budget16_stability.json)
- [Supertonic host report](supertonic3_mac_m2/REPORT.md)
- [Account-owned Supertonic and Piper QRD results](aihub_voice_v81/REPORT.md)
- [Piper Italian corrected production chain](piper_v81/PRODUCTION_CHAIN_RESULT.md)
- [MeloTTS-ZH exact-SM8850 portability result](melotts_zh_v81/RESULT.md)
- [Japanese Supertonic accumulated-chain result](supertonic_v81/ja_paris/RESULT.md)
- [Published Piper and MeloTTS v81 inventory](../npu/multilingual_tts_v81.md)
- [Android native-TTS source checkpoint](validation_2026-07-23_native_tts_source.md)
- [Android native-TTS APK validation](validation_2026-07-23_native_tts_apk.md)
- [Japanese LFM report](lfm_jp_q4_mac_m2/REPORT.md)
- [Kokoro Mandarin report](kokoro_int8_mac_m2/REPORT.md)
