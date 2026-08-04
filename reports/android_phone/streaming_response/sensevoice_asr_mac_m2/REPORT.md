# SenseVoiceSmall INT8: multilingual ASR challenger

Date: 2026-07-22<br>
Local host: Apple M2 MacBook Air, 8 GB RAM<br>
Languages in this diagnostic: English, Japanese, Korean, Mandarin Chinese

## Decision

Promote **SenseVoiceSmall INT8** to the leading ASR candidate for the
EN/JA/KO/ZH phone demo. On the same 12 supported-language Common Voice clips,
it was both more accurate than Whisper Small and dramatically faster on the
Mac CPU:

| Model | Threads | Final transcript median / P95 | Model files | Incremental peak RSS | EN WER | JA/KO/ZH micro CER | Macro primary error |
|---|---:|---:|---:|---:|---:|---:|---:|
| SenseVoiceSmall INT8 | 2 | **101 / 174 ms** | 228.5 MiB | 620.6 MiB max | **15.8%** | **24.3%** | **23.8%** |
| Whisper Base INT8 | 4 | 420 / 512 ms | 141.0 MiB | **315.2 MiB** | 36.8% | 39.2% | 41.2% |
| Whisper Small INT8 | 4 | 1,614 / 2,219 ms | 463.7 MiB | 614.0 MiB | 21.1% | 29.7% | 30.5% |

Quality is directly comparable because the rows use the same clips, fixed
locales, and normalizer. Host latency is useful engineering evidence but not a
strict apples-to-apples speed benchmark: SenseVoice used two threads and
Sherpa-ONNX, while the retained Whisper runs used four threads and
Faster-Whisper.

SenseVoice passes the exploratory gates previously defined for these four
languages: English WER at most 25%; combined CJK CER at most 25%; and no CJK
language above 40%. This is only three clips per language, so it is a candidate
selection result, not a product-quality claim.

The direct deployment route is unusually strong: Sherpa publishes an official,
pre-generated **SM8850 / HTP-v81 QNN context binary** for the exact canonical
model. That means the next Nubia test does not require waiting for an AI Hub
compile. The binary still needs a QNN-capable Sherpa executable and the matching
QNN 2.40 runtime libraries.

## What was measured

- Canonical Sherpa model:
  `sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17`.
- Runtime: `sherpa-onnx==1.13.4`, CPU provider.
- Quantization: the bundled exporter applies dynamic `QUInt8` weight
  quantization to `MatMul`; the release labels the package INT8.
- Fixed language selected separately for EN, JA, KO, and ZH.
- Inverse text normalization enabled.
- Greedy non-autoregressive CTC decoding.
- Two CPU threads, one worker process at a time, three timed repeats per clip.
- Audio pre-decoded to mono float32 PCM at 16 kHz before timing.
- Timed region includes Sherpa stream creation, waveform acceptance, feature
  extraction, model decode, and final transcript retrieval.

This path is offline rather than incremental. It exposes no useful partial
transcript, so `first_stable_ms` is the final transcript time. At roughly
100 ms median after VAD closure, partial ASR is not necessary for the first
2–3 second demo; VAD endpointing, LLM response policy, and first TTS chunk are
more likely to dominate.

The fixture is the CC0-1.0
`argmaxinc/common_voice_17_0-debug-zip` repository pinned at
`e77e410d41a77f733efbaf7307bd0ff7ebd6d0af`. It contains three clips per
language. SenseVoice does not support the Spanish, German, or French cells from
the original 21-clip fixture, so those nine clips were excluded.

## Quality by language

| Language | SenseVoice | Whisper Base | Whisper Small | Metric |
|---|---:|---:|---:|---|
| English | **15.8%** | 36.8% | 21.1% | WER |
| Japanese | **31.3%** | 47.9% | 37.5% | CER |
| Korean | **14.9%** | 25.4% | **14.9%** | CER |
| Mandarin Chinese | **33.3%** | 54.5% | 48.5% | CER |

The result is promising, not uniformly clean. Important failures include:

- English `six` became `S.`;
- Mandarin `否` became `房`;
- the longer third Japanese clip had substantial substitutions;
- the first Korean clip had multiple spacing and lexical errors.

For a public demo, keep the selected locale explicit, use a command-oriented
prompt set, and add a low-confidence/retry path. Before claiming open-ended
multilingual quality, run at least 100 representative utterances per language,
including room noise, accents, names, commands, and the professor's expected
demo vocabulary.

## Latency and memory

| Language | Final median / P95 | Median RTF |
|---|---:|---:|
| English | 131 / 155 ms | 0.023 |
| Japanese | 77 / 83 ms | 0.022 |
| Korean | 141 / 143 ms | 0.021 |
| Mandarin Chinese | 119 / 188 ms | 0.022 |

Compared with Whisper Small on the same supported subset, SenseVoice saves
about **1.51 seconds median** and **2.05 seconds P95** of host ASR time. Compared
with Whisper Base, it saves about 319 ms median and 338 ms P95.

Cold recognizer construction was 627 ms median. Keep the recognizer resident;
this is startup cost, not per-turn cost.

Memory is the main tradeoff. The conservative maximum incremental peak was
620.6 MiB, with fresh-worker values from 466.4 to 620.6 MiB. The maximum
post-load increment was 601.7 MiB. This is similar to Whisper Small and roughly
twice Whisper Base, despite the much smaller model file. Phone RSS must be
measured with the LLM and TTS resident before accepting the combined stack.

## Reproducible model provenance

Canonical CPU archive:

- GitHub release tag: `asr-models`;
- asset ID: `288366523`;
- archive size: 163,002,883 bytes;
- SHA-256:
  `7d1efa2138a65b0b488df37f8b89e3d91a60676e416f515b952358d83dfd347e`;
- extracted `model.int8.onnx` SHA-256:
  `c71f0ce00bec95b07744e116345e33d8cbbe08cef896382cf907bf4b51a2cd51`;
- extracted `tokens.txt` SHA-256:
  `f449eb28dc567533d7fa59be34e2abca8784f771850c78a47fb731a31429a1dc`.

The bundled exporter calls `SenseVoiceSmall.from_pretrained` with
`iic/SenseVoiceSmall` but does not pass an upstream revision. It references
SenseVoice commit `a80e676461b24419cf1130a33d4dd2f04053e5cc` for token IDs,
but that does not prove the downloaded weight revision. The report therefore
does not invent one: the archive and extracted-file hashes pin the runnable
artifact exactly.

The similarly named `2025-09-09` archive was inspected and deliberately
rejected for this comparison. Its README identifies
`ASLP-lab/WSYue-ASR/sensevoice_small_yue`, a Cantonese-specialized derivative,
not the canonical model above.

## Official SM8850 / HTP-v81 QNN package

Selected fixed-shape package:

- [Exact archive](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models-qnn-binary/sherpa-onnx-qnn-SM8850-binary-10-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17-int8.tar.bz2)
- release tag: `asr-models-qnn-binary`;
- GitHub asset ID: `326303090`;
- archive size: 162,023,574 bytes (154.5 MiB);
- archive SHA-256:
  `ecbc1ffba39f8e23582b79a199d12e8455425a22ef7b6b18c535ce25fcff2d64`;
- extracted `model.bin`: 254,193,664 bytes, SHA-256
  `522763135f536bc43e9d122bac39254f49df4afbb003462d6d8bcdb77051de30`;
- fixed context: 10 seconds / `num_frames=167`;
- target: Qualcomm SM8850, HTP v81;
- format: pre-generated QNN context binary.

QNN fixed shapes matter: audio shorter than ten seconds is padded to ten;
audio longer than ten seconds is truncated. Ten seconds is the safe first
package because the diagnostic fixture contains an 8.136-second clip. Once the
demo enforces a hard five-second VAD cap, the official five-second context is a
worthwhile lower-latency option.

Sherpa's published Xiaomi 17 Pro / SM8850 logs report:

- 10-second context: 1.214 s recognizer initialization and 290 ms decode for a
  5.592-second test file, padded to ten seconds;
- 5-second context: 52 ms decode, but the same file is truncated to five
  seconds.

These are upstream reference numbers, not measurements from this project's
Nubia. They nevertheless establish a credible v81 path and make physical-phone
testing the next highest-value experiment.

An account-owned profile of the checksum-pinned ten-second context subsequently
passed on the exact Snapdragon 8 Elite Gen 5 QRD: AI Hub job `j5w1ov24g`
reported **10.871 ms** estimated inference latency, **126.223 MiB** estimated
inference peak memory, and **2,597/2,597** runtime operations on Hexagon/HTP
with zero CPU or unknown fallback. The context was generated with QNN 2.40;
AI Hub does not expose API version 2.40, so this successful profile used QAIRT
2.45 and is specifically a QNN-2.40-context-on-QAIRT-2.45 compatibility result.

The AI Hub release-context run is profile-only. It excludes host preprocessing,
token handling, VAD, transfers, and playback, and the archive provides no
matching source golden for remote numerical or transcript comparison. See the
[exact-target profile report](AIHUB_V81_PROFILE.md) and
[`aihub_v81_profile_result.json`](aihub_v81_profile_result.json).

The context was generated with **QNN SDK 2.40.0.251030**. Sherpa explicitly
warns that changing the runtime SDK version requires regenerating `model.bin`.
The Nubia's OEM libraries previously reported QNN 2.37.0, so do not mix this
context with the vendor 2.37 stack. Push the matching 2.40 libraries from an
authorized SDK installation:

- application side: `libQnnHtp.so`, `libQnnHtpPrepare.so`,
  `libQnnSystem.so`, `libQnnHtpV81Stub.so`;
- DSP side: `libQnnHtpV81Skel.so` from
  `$QNN_SDK_ROOT/lib/hexagon-v81/unsigned`;
- Sherpa side: QNN-enabled `sherpa-onnx-offline` and `libonnxruntime.so`.

The semicolon in `ADSP_LIBRARY_PATH` is mandatory.

## Concrete phone deployment plan

The phone was disconnected during this run, so no ADB mutation or phone timing
was attempted. When it is connected:

```bash
export QNN_SDK_ROOT=/path/to/qnn/2.40.0.251030
export SHERPA_QNN_INSTALL=/path/to/qnn-enabled/sherpa-onnx/install

MODEL_DIR=$(scripts/sensevoice_asr_mac_m2/download_qnn_sm8850.sh)
PHONE_DIR=/data/local/tmp/sensevoice-qnn-v81
MODEL_NAME=$(basename "$MODEL_DIR")

adb devices -l
adb shell mkdir -p "$PHONE_DIR"
adb push "$MODEL_DIR" "$PHONE_DIR/"
adb push "$SHERPA_QNN_INSTALL/bin/sherpa-onnx-offline" "$PHONE_DIR/"
adb push "$SHERPA_QNN_INSTALL/lib/libonnxruntime.so" "$PHONE_DIR/"
adb push "$QNN_SDK_ROOT/lib/aarch64-android/libQnnHtp.so" "$PHONE_DIR/"
adb push "$QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpPrepare.so" "$PHONE_DIR/"
adb push "$QNN_SDK_ROOT/lib/aarch64-android/libQnnSystem.so" "$PHONE_DIR/"
adb push "$QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpV81Stub.so" "$PHONE_DIR/"
adb push "$QNN_SDK_ROOT/lib/hexagon-v81/unsigned/libQnnHtpV81Skel.so" "$PHONE_DIR/"

adb shell "cd $PHONE_DIR && chmod +x sherpa-onnx-offline && \
  export LD_LIBRARY_PATH=\$PWD && \
  export ADSP_LIBRARY_PATH=\"\$PWD;\${ADSP_LIBRARY_PATH:-}\" && \
  ./sherpa-onnx-offline \
    --provider=qnn \
    --tokens=./$MODEL_NAME/tokens.txt \
    --sense-voice-language=en \
    --sense-voice-use-itn=true \
    --sense-voice.qnn-backend-lib=./libQnnHtp.so \
    --sense-voice.qnn-system-lib=./libQnnSystem.so \
    --sense-voice.qnn-context-binary=./$MODEL_NAME/model.bin \
    ./$MODEL_NAME/test_wavs/en.wav"
```

After the smoke test:

1. Convert the 12 pinned clips to 16 kHz mono WAV and push them.
2. Run three warm repeats per clip with `--sense-voice-language` fixed to the
   reference locale.
3. Compare every phone transcript against `transcripts.csv` and the local raw
   JSON; investigate any QNN-vs-ONNX text regression.
4. Measure cold initialization, warm VAD-close-to-final transcript, process
   RSS/VmHWM, and thermal state while the LLM and TTS are resident.
5. Use a P95 ASR gate of 700 ms. If the ten-second context passes comfortably,
   retain it for open interaction; otherwise enforce a five-second utterance
   cap and test the five-second context.

## Licensing

- Model weights: custom **FunASR Model Open Source License Agreement v1.1**.
  It requires attribution and retention of the model name and is not represented
  here as an OSI-approved license.
- Sherpa-ONNX runtime: Apache-2.0.
- Qualcomm QNN/QAIRT libraries: Qualcomm's separate SDK terms; do not place
  redistributable binaries in a public repo without checking those terms.
- Diagnostic audio: CC0-1.0.

## Reproduction and raw artifacts

- Harness: [`scripts/sensevoice_asr_mac_m2`](../../../../scripts/sensevoice_asr_mac_m2/README.md)
- Raw local result: [`sensevoice_small_int8.json`](sensevoice_small_int8.json)
- Same-subset comparison: [`comparison.json`](comparison.json)
- Aggregate table: [`summary.csv`](summary.csv)
- Per-language latency: [`latency_by_language.csv`](latency_by_language.csv)
- References and hypotheses: [`transcripts.csv`](transcripts.csv)
- CPU model manifest: [`model_manifest.json`](model_manifest.json)
- SM8850 QNN inventory: [`qnn_sm8850_manifest.json`](qnn_sm8850_manifest.json)
- QNN runtime/build gate: [`QNN_RUNTIME_GATE.md`](QNN_RUNTIME_GATE.md)
- Exact-SM8850 AI Hub profile: [`AIHUB_V81_PROFILE.md`](AIHUB_V81_PROFILE.md)
- Exact-SM8850 profile JSON: [`aihub_v81_profile_result.json`](aihub_v81_profile_result.json)

The raw console log is retained locally for operator debugging but is excluded
from the public evidence links because repository policy ignores `*.log`.

Official references:

- [Sherpa QNN model catalog](https://k2-fsa.github.io/sherpa/onnx/qnn/models.html)
- [Sherpa SM8850 phone deployment guide](https://k2-fsa.github.io/sherpa/onnx/qnn/run-executables-on-your-phone-binary.html)
