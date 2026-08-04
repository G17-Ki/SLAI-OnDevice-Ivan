# LFM2.5-Audio unplugged phone demo

This repository contains an offline, on-device speech-agent demo for Qualcomm
phones plus the experiments used to evaluate ASR, language-model, and TTS
candidates. The current working deployment is a two-APK, app-owned stack on the
Nubia NX809J / SM8850:

| APK | Version | Responsibility |
|---|---|---|
| `ai.liquid.lfmdemo` | `0.6.0-unplugged` (code 26) | UI, microphone/VAD, LFM2.5-Audio, playback, and neural/offline TTS |
| `ai.liquid.lfmdemo.runtime` | `0.1.0-unplugged-runtime` (code 1) | SenseVoice/Zipformer ASR and Qwen2.5 response model |

Both APKs run under Android application UIDs and use model files in phone
storage. No inference process is launched by `adb shell`, and no ADB forwarding
is used by the app. Opening the main app starts both foreground services through
a signature-protected explicit intent.

This source state is newer than the retained public APK under
`android-app/releases/`. A checksum-pinned `0.6.0` release pair has not yet been
retained, so reproduce the current stack from source rather than treating the
older `0.4.13` APK as the current demo.

The physical-phone evidence below is bound to the previously accepted installed
pair identified in its provenance reports. Current source builds are not
byte-for-byte identical to that pair and require a fresh device acceptance run
before inheriting its unplugged or latency claims.

## Current phone stack

| Component | Runtime and storage owner | Processor | Device status |
|---|---|---|---|
| SenseVoiceSmall INT8 ASR, EN/ZH/JA/KO | Runtime-companion APK, `127.0.0.1:6010` | Qualcomm HTP v81 through QNN 2.40 | App-UID execution, exact fixtures, and full turns verified |
| Russian Zipformer INT8 ASR | Runtime-companion APK, `127.0.0.1:6010` | CPU | App-UID native-script fixture verified |
| Qwen2.5-1.5B-Instruct Q4_K_M | Runtime-companion APK, `127.0.0.1:6020` | CPU | Exact model identity and a 28-case phone sweep verified |
| LFM2.5-Audio-1.5B Q4 | Main APK, `127.0.0.1:8080` | CPU | App-owned English expressive route |
| Mandarin TTS, `matcha_zh_baker` | Main APK/Sherpa-ONNX; bundle in app external-files storage | CPU | Physical-phone recommendation; non-commercial corpus warning applies |
| RU/JA/KO TTS, Supertonic 3 five-step routes | Main APK/Sherpa-ONNX; bundle in app external-files storage | CPU | Physical routing, generation config, and playback verified |
| Other offline voices | eSpeak Android package | CPU | Functional fallback; voice quality is language-dependent |
| VAD, routing, and orchestration | Main APK | CPU | Phone-local |

All three HTTP services bind only to `127.0.0.1`. Only the EN/ZH/JA/KO
SenseVoice route is currently NPU-accelerated in the application. Russian ASR,
Qwen, LFM, the current neural TTS backends, VAD, and orchestration remain CPU
paths. AI Hub component profiles are research evidence, not proof that those
components execute end to end on the retail phone.

The neural TTS policy is language-aware: Mandarin prefers Matcha Baker;
Russian, Japanese, and Korean use language-specific Supertonic 3 five-step
configuration; and unsupported languages use the offline eSpeak fallback.
Concrete candidate IDs remain available as manual lab overrides. Matcha is the
best measured Mandarin demo voice but inherits a
non-commercial training-corpus restriction; review the TTS report before any
redistribution or commercial use.

Current physical ASR/full-turn acceptance covers English, Mandarin, Japanese,
Korean, and Russian. French, Italian, Spanish, German, and installed eSpeak
labels are not equivalent evidence of a robust native end-to-end route.

## What is physically verified

The checksum-aware runtime installer passed on the NX809J and produced:

- English QNN/HTP: `The tribal chieftain called for the boy and presented him with 50 pieces of gold.`
- Mandarin QNN/HTP: `开饭时间早上9点至下午5点。`
- Japanese QNN/HTP: `フランス の 首都 は どこ で すか？`
- Korean QNN/HTP: `프랑스의 수도는 어디인가요?`
- Russian CPU: `расскажи коротко о москве`

The ASR health contract separates artifacts on disk from successful execution;
a backend becomes ready only after a non-empty transcript in the current
service process. QNN logs include `Initializing HtpProvider`, and the validated
runner belongs to the companion app UID.

The app-owned Qwen2.5 phone sweep scored 24/28 after manual correction, with a
304 ms median first token and 672 ms median first speakable text. It passed the
small Mandarin and Russian subsets, but misses in French, Japanese, and Korean
mean it is not advertised as a universally robust multilingual model.

Earlier physical TTS sweeps measured about 118 ms end-to-speaker for Matcha
Mandarin and about 900 ms for five-step Supertonic Russian. Those measurements
remain valid candidate evidence. The latest serialized streaming engine passed
the seven-case physical queue/drain/cancellation suite, including automatic
Mandarin Matcha, automatic Russian Supertonic, unsupported-locale eSpeak,
manual pinning, ordered multi-chunk playback, and cancellation before and after
playback start.

The app-owned architecture removes the known `adbd` cgroup lifetime defect.
The previously accepted installed APK pair passed a cold reboot, first unlock,
main-only launch, and authenticated EN/ZH/RU/JA/KO full turns in airplane mode.
That same pair retained all four processes through a 70-second host-ADB shutdown
with zero forwards. Literal physical USB-cable removal remains an operator gate,
so do not turn host-disconnect evidence into a published physical-unplug claim.
There is intentionally no boot receiver: after a reboot, unlock the phone and
open the app manually. Final artifact hashes, fixture hashes, latency proxies,
and the remaining robustness boundary are recorded in
`reports/android_phone/full_turns/final_apk_a24e_33bf_cold_reboot_provenance_20260729.md`.

## Build and install

Requirements are JDK 17, an Android SDK containing API 35, `adb`, Python 3,
enough phone storage, and an arm64 SM8850 target. Heavy conversion, compilation,
training, and fine-tuning belong on Qualcomm AI Hub or Google Colab; local
Android builds are bounded to two workers.

Download and verify the four LFM Q4 files outside the checkout:

```bash
export LFM_DEMO_WORK_ROOT="$HOME/.cache/lfm-demo"
python3 -m pip install huggingface_hub
scripts/download_lfm_q4_models.sh \
  "$LFM_DEMO_WORK_ROOT/models/lfm25-audio-q4"
```

Build and install the current main APK without clearing its data:

```bash
scripts/build_android_apk.sh

scripts/install_native_phone_demo.sh \
  --apk android-app/app/build/outputs/apk/debug/app-debug.apk \
  --model-dir "$LFM_DEMO_WORK_ROOT/models/lfm25-audio-q4"
```

Always pass the current `--apk` path: the installer's default APK is the
historical retained `0.4.13` artifact.

Prepare the pinned external ASR, QNN, llama.cpp, and Qwen2.5 caches, then build,
install, provision, start, and execution-test the app-owned companion:

```bash
scripts/install_app_owned_runtime.sh --serial YOUR_ADB_SERIAL
```

The installer refuses a non-SM8850 target, verifies the native APK payload,
checks every private model copy by size and SHA-256, preserves existing package
data, and runs real EN/ZH/JA/KO/RU ASR fixtures before reporting success. See the
[Android deployment guide](docs/ANDROID_DEPLOYMENT.md) and the
[runtime-companion README](android-app/runtime-companion/README.md) for cache
preparation, storage paths, health semantics, and troubleshooting.

On a Nubia NX809J, installation is not complete until the one-time OEM launch
policy is provisioned for both `ai.liquid.lfmdemo` and
`ai.liquid.lfmdemo.runtime`. In **Settings > Battery > Apps AI-control > Manual
control**, set **Auto-start**, **Secondary launch**, and **Background running**
to **Allowed** for both apps. The deployment guide gives the required
`dumpsys AutoLaunch` checks before and after reboot; installer success alone
does not prove that this persistent policy is enabled.

The main-app neural TTS bundle belongs at
`/sdcard/Android/data/ai.liquid.lfmdemo/files/tts-eval`. Use the checksum-aware
main-app bundle installer documented in the deployment guide; do not use the
standalone benchmark app's push script for the demo package.

On the audited NX809J, the repaired manifest authenticates all 437 referenced
assets, including the 16 text resources that the legacy manifest checked only
for existence. The metadata-only repair transferred zero model files, updated
the stale registry/manifest atomically, and a second verify-only run passed all
437 assets plus all three metadata files.

## Host validation

Run the lightweight source gates after changes:

```bash
node scripts/test_streaming_audio_ui.mjs
python3 scripts/test_android_native_tts_bridge.py
python3 scripts/test_profile_text_llm_streaming.py
PYTHONPATH=scripts python3 scripts/test_android_streaming_telemetry.py
python3 -m unittest -v scripts/test_phone_acceptance_harness.py
bash scripts/test_adaptive_energy_vad.sh
```

Build the current main APK with:

```bash
env JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home \
  ANDROID_HOME="$HOME/Library/Android/sdk" \
  android-app/gradlew --max-workers=2 \
  -p android-app :app:assembleDebug
```

Host tests do not establish phone latency, HTP execution, acoustic quality,
thermal stability, or unplug survival.

## Evidence

- Current app-owned ASR/QNN audit:
  `reports/progress_20260729/ASR_QNN_AUDIT.md`
- Current backbone/LLM audit: `reports/progress_20260729/LLM_AUDIT.md`
- [Autonomous deployment stop report](reports/android_phone/model_sweep_20260729/DIRECT_PHONE_NPU_STOP_20260730.md)
- [Physical-phone TTS results](reports/android_phone/streaming_response/tts_research_20260726/DEVICE_RESULTS.md)
- [Language-routing evidence](reports/android_phone/streaming_response/LANGUAGE_ROUTING.md)
- [QAIRT/HTP v81 support memo](reports/android_phone/npu/qairt_v81_support.md)
- [Experiment tracker](docs/EXPERIMENTS.md)

Historical reports retain their original measurements and conclusions. Where a
later app-owned implementation supersedes an old operational limitation, the
historical file carries a dated status banner rather than rewritten evidence.

## Repository layout

- `android-app/app/` — main Android app and LFM/TTS integration.
- `android-app/runtime-companion/` — app-owned ASR/Qwen foreground runtime.
- `scripts/` — preparation, deployment, profiling, and acceptance tools.
- `reports/` — measurements, audits, and technical conclusions.
- `docs/` — current deployment and experiment guides.
- `vendor/` — pinned upstream Liquid Audio source.
- `output/` — generated public technical report.

## Limits and licensing

The physical result is from a Nubia NX809J / SM8850 phone, not AR1 or glasses.
AI Hub QRD timings are same-chipset reference evidence, not retail-phone or
glasses latency. No end-to-end glasses power, thermal, or always-on endurance
result is claimed.

First-party code is Apache-2.0. `vendor/`, native libraries, model weights,
voice bundles, and Qualcomm runtime packages have their own terms. Review
`android-app/THIRD_PARTY_NOTICES.md`, the LFM Open License, each model card, and
the Matcha Baker corpus restriction before redistribution or commercial use.
