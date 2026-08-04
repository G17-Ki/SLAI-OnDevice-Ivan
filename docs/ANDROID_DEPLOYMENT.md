# Android deployment — current app-owned stack

This guide builds and provisions the current two-APK source deployment for the
Nubia NX809J / SM8850:

- main app: `ai.liquid.lfmdemo`, version code 26,
  `0.6.0-unplugged`;
- runtime companion: `ai.liquid.lfmdemo.runtime`, version code 1,
  `0.1.0-unplugged-runtime`.

The architecture no longer launches ASR or Qwen through `adb shell`. Both are
owned by the runtime companion's Android UID; LFM and TTS are owned by the main
app. A USB cable is needed for the one-time build/install/provision workflow,
not for loopback inference. The previously accepted installed pair identified
by the provenance reports passed cold reboot, first-unlock, main-only launch,
authenticated EN/ZH/RU/JA/KO turns in airplane mode, and a 70-second host-ADB
shutdown with unchanged PIDs and zero forwards. Current source builds are not
byte-for-byte identical to that pair and require fresh device acceptance.
Literal physical cable removal remains open before publishing the narrower
physical-unplug claim.

The retained APK under `android-app/releases/` is the historical
`0.4.13-expressive-lfm` main app. It is not the current two-APK deployment.

## Runtime map

| Owner | Component | Endpoint | Processor | Verified scope |
|---|---|---:|---|---|
| Main app | LFM2.5-Audio-1.5B Q4 | `127.0.0.1:8080` | CPU | App-owned English expressive path |
| Runtime companion | SenseVoiceSmall INT8 | `127.0.0.1:6010` | QNN 2.40 / HTP v81 | English, Mandarin, Japanese, and Korean exact fixtures/full turns |
| Runtime companion | Russian Zipformer INT8 | `127.0.0.1:6010` | CPU | Russian native-script fixture |
| Runtime companion | Qwen2.5-1.5B-Instruct Q4_K_M | `127.0.0.1:6020` | CPU | Exact identity and 28-case phone sweep |
| Main app | Matcha Baker Mandarin TTS | internal | CPU | Prior phone recommendation; corpus restriction applies |
| Main app | Supertonic 3 five-step multilingual TTS | internal | CPU | Russian, Japanese, and Korean routing/playback verified |
| eSpeak package | Other installed offline voices | Android TTS API | CPU | Functional fallback |
| Main app | VAD, language routing, playback | internal | CPU | Phone-local |

All HTTP endpoints bind only to loopback. The app does not require an ADB
forward. Only SenseVoice English/Mandarin/Japanese/Korean is integrated on the NPU. Russian
ASR, Qwen, LFM, neural TTS, eSpeak, VAD, and orchestration are CPU paths.

The two apps must be signed with the same key because the main app starts the
exported runtime service through the signature-level
`ai.liquid.lfmdemo.permission.START_RUNTIME` permission. The debug builds from
one checkout satisfy this requirement.

## Storage and lifecycle

The current deployment keeps executable native payloads in each installed
APK's immutable native-library directory and weights in package-owned storage:

| Package | Data |
|---|---|
| `ai.liquid.lfmdemo` private files | Four LFM Q4 files under `files/models/` |
| `ai.liquid.lfmdemo` external files | Neural TTS bundle under `tts-eval/` |
| `ai.liquid.lfmdemo.runtime` private files | Qwen2.5 under `files/text/models/`; SenseVoice and Zipformer under `files/asr/models/` |
| Runtime-companion APK | llama.cpp server/libraries, Sherpa runner, ONNX Runtime, QNN libraries, and `libQnnHtpV81Skel.so` |

Opening the main app explicitly starts the runtime companion and the LFM
foreground service. The companion is `START_STICKY` and monitors/restarts the
Qwen subprocess while its service is alive. There is no boot receiver: after a
phone reboot, unlock the device and open the main app manually.

Do not clear either package's data during an update. Use `adb install -r`, and
use the checksum-aware installers below so already-correct large files are
reused.

## Requirements

- Nubia NX809J or another arm64 Android phone whose SoC reports `SM8850`;
- Android 16 was used for the physical acceptance recorded here;
- USB debugging and an authorized `adb` connection for provisioning;
- JDK 17 and an Android SDK containing API 35;
- Python 3 and `huggingface_hub` for initial pinned downloads;
- enough phone storage for roughly 1 GiB of LFM, 1.1 GiB of Qwen2.5, ASR/QNN
  assets, the APK native payload, and the optional 1.4 GiB TTS lab bundle.

The runtime installer rejects a non-arm64 or non-SM8850 phone because the QNN
context and HTP v81 payload are target-specific.

Heavy training, fine-tuning, graph conversion, or broad model compilation must
run on Google Colab or Qualcomm AI Hub. Local work should remain limited to
downloads, checksum verification, source tests, and Android builds capped at
two workers.

## 1. Prepare the LFM bundle

Keep mutable downloads outside the repository:

```bash
export LFM_DEMO_WORK_ROOT="$HOME/.cache/lfm-demo"
python3 -m pip install huggingface_hub

scripts/download_lfm_q4_models.sh \
  "$LFM_DEMO_WORK_ROOT/models/lfm25-audio-q4"
```

The downloader pins the LiquidAI revision and verifies all four files against
`android-app/model-files.sha256`.

## 2. Prepare the runtime-companion caches

The final installer is offline and fail-closed. Seed these external caches once
before running it.

Prepare the exact-SM8850 SenseVoice/QNN bundle and the Russian ASR assets:

```bash
python3 scripts/npu_multilingual/prepare_sensevoice_sm8850_bundle.py

LMF_BUILD_JOBS=2 \
  scripts/npu_multilingual/build_sherpa_qnn_android.sh

python3 scripts/android_asr_companion/prepare_assets.py
```

If a verified Sherpa QNN runner cache already exists, reuse it and skip the
local build. Do not increase the build-job cap on the Mac.

Prepare the pinned official llama.cpp Android runtime:

```bash
python3 scripts/android_text_companion/prepare_runtime.py
```

Download the exact Qwen2.5 file named in the checked-in candidate manifest:

```bash
python3 - <<'PY'
import json
from pathlib import Path
from huggingface_hub import hf_hub_download

manifest = json.loads(Path(
    "reports/android_phone/streaming_response/"
    "model_candidate_followup/candidate_manifest.json"
).read_text())
model = next(x for x in manifest["models"]
             if x["id"] == "qwen2_5_1_5b_q4_k_m")
target = Path.home() / ".cache/lmf-demo/android-text-companion/models"
target.mkdir(parents=True, exist_ok=True)
hf_hub_download(
    repo_id=model["repository"],
    filename=model["filename"],
    revision=model["revision"],
    local_dir=target,
)
PY
```

The deployment installer independently enforces the pinned byte count and
SHA-256 before using the file.

For an offline machine, seed the exact files into the same cache roots and use
the preparers' `--offline`/`--seed-dir` options. Model weights, Qualcomm runtime
objects, and generated native payloads must remain outside the repository.

## 3. Build and install the current main app

Build with JDK 17 and at most two Gradle workers:

```bash
scripts/build_android_apk.sh
```

Install the current APK and the verified LFM files without clearing package
data:

```bash
scripts/install_native_phone_demo.sh \
  --apk android-app/app/build/outputs/apk/debug/app-debug.apk \
  --model-dir "$LFM_DEMO_WORK_ROOT/models/lfm25-audio-q4"
```

The explicit `--apk` argument is required for the current source deployment;
without it, the legacy installer defaults to the retained `0.4.13` artifact.
Its final text about shell-owned companions is also historical and does not
describe the runtime installed in the next step.

## 4. Build, install, and verify the app-owned runtime

With the target connected and authorized:

```bash
scripts/install_app_owned_runtime.sh --serial YOUR_ADB_SERIAL
```

This command:

1. verifies the pinned Qwen2.5 model and all external native inputs;
2. stages an `arm64-v8a` JNI tree outside the checkout;
3. builds the runtime companion with `--max-workers=2`;
4. verifies that the APK contains the Qwen, llama.cpp, ASR, ONNX, QNN, and HTP
   v81 payloads;
5. installs with `adb install -r` and preserves package data;
6. stops only the known legacy shell lifecycle scripts;
7. reuses or atomically copies every verified model into companion-private
   storage;
8. starts the companion and checks Qwen model identity;
9. executes real English, Mandarin, Japanese, Korean, and Russian ASR fixtures; and
10. confirms that the Qwen process and ASR runners belong to the companion UID.

Successful English, Mandarin, Japanese, and Korean validation requires the
exact transcript, expected SenseVoice language tag, and QNN provider.
Successful Russian validation requires the expected Zipformer CPU transcript.
File presence or exit status alone is not accepted.

Useful options:

```text
--skip-build     verify and reuse an existing companion APK
--skip-install   preserve the currently installed APK
--skip-assets    leave the current private models untouched
--no-start       stage only; do not run health/execution gates
--keep-forwards  retain temporary host diagnostic forwards
```

See `android-app/runtime-companion/README.md` for the exact cache layout,
health schema, and resolved QNN linker requirements.

### Mandatory Nubia Apps AI-control provisioning

The NX809J requires a one-time persistent OEM launch-policy setup for **both**
installed packages. Without it, the main app's cold explicit start of the
runtime companion can be blocked even when both APKs are correctly signed and
installed.

On the phone:

1. Open **Settings > Battery > Apps AI-control**.
2. Turn off **All AI-control** and confirm **OK** so **Manual control** becomes
   available.
3. Open **Manual control**.
4. Configure both **LFM Audio** (`ai.liquid.lfmdemo`) and **LFM Offline
   Runtime** (`ai.liquid.lfmdemo.runtime`).
5. For each app, set **Auto-start**, **Secondary launch**, and **Background
   running** to **Allowed**.

Verify both packages immediately after provisioning, before the reboot test:

```bash
for package_name in ai.liquid.lfmdemo ai.liquid.lfmdemo.runtime; do
  adb -s YOUR_ADB_SERIAL shell dumpsys AutoLaunch -p "$package_name"
done
```

Each package must report:

```text
selfStartPolicy=1
relatedStartPolicy=1
selfStartMode=3
relatedStartMode=3
```

Any policy value `0`, or either mode at `1` or `2`, means provisioning is not
complete. Also confirm in the UI that **Background running** remains
**Allowed**.

Then reboot, unlock the phone once, and repeat the same read-only query before
opening either app:

```bash
adb -s YOUR_ADB_SERIAL reboot
adb -s YOUR_ADB_SERIAL wait-for-device
# Unlock the phone manually, then rerun the two-package loop above.
```

The same policy values and modes must survive the reboot. After verification,
open the main app manually; there is intentionally no boot receiver. The
`dumpsys AutoLaunch -wl` and `-a` debug switches are not substitutes for this
UI step because their allowlists are memory-only and do not survive reboot.

## 5. Install the main-app TTS bundle

The demo package reads neural TTS candidates from:

```text
/sdcard/Android/data/ai.liquid.lfmdemo/files/tts-eval
```

Use the demo-specific checksum-aware installer:

```bash
scripts/install_main_app_tts_bundle.sh \
  --serial YOUR_ADB_SERIAL \
  --bundle /path/to/verified/tts-eval
```

The bundle must contain internally consistent `candidates.json` and
`bundle_manifest.json`. The installer verifies referenced files by
size/SHA-256, skips already valid files for resumable updates, installs metadata
atomically with `candidates.json` last, and does not clear or delete unrelated
app data. Use `--manifest-only` when all model files are already resident: it
verifies those assets and transfers metadata only. Use `--verify-only` for a
strict no-change audit, or `--dry-run` to validate the source bundle and print
the plan without ADB.

The 2026-07-29 NX809J repair used a clean staged manifest that authenticates all
437 referenced assets, including the 16 token/lexicon resources that the legacy
manifest checked only for existence. `--manifest-only` verified every resident
asset, transferred zero model files, atomically replaced the stale metadata,
and a second `--verify-only` pass authenticated all assets and all three
metadata files. Do not use this shortcut on another installation until its
resident assets pass the same verification.

Do not use `tools/push_tts_bundle.sh`: it targets the standalone benchmark app,
not `ai.liquid.lfmdemo`.

The automatic policy prefers:

- `zh` → `matcha_zh_baker`;
- `ru` → `supertonic3_s5`;
- other languages → installed offline eSpeak voice.

Matcha Baker's corpus restriction must stay visible in the UI and release
notes. Manual candidate selection is a lab override and should fail visibly if
its files are incomplete rather than silently attributing fallback audio to the
wrong model.

## 6. Install the offline fallback voice

Install the audited eSpeak package and required language data:

```bash
scripts/install_espeak_phone.sh --serial YOUR_ADB_SERIAL
```

eSpeak is an offline compatibility fallback, not the Mandarin or Russian
quality winner. Its available languages depend on the installed package and
voice data.

## 7. Launch and verify the pair

Confirm package identities:

```bash
adb -s YOUR_ADB_SERIAL shell dumpsys package ai.liquid.lfmdemo \
  | grep -E 'versionCode|versionName'

adb -s YOUR_ADB_SERIAL shell dumpsys package ai.liquid.lfmdemo.runtime \
  | grep -E 'versionCode|versionName'
```

Open `LFM Audio`. It starts the companion explicitly; the companion can also be
opened once from its `LFM Offline Runtime` launcher icon. Keep both foreground
notifications enabled and exempt both packages from aggressive battery
optimization during sustained testing.

Wait for route-specific readiness, then run at least:

- a short English SenseVoice → LFM expressive turn;
- a Mandarin SenseVoice → Qwen2.5 → Matcha turn;
- a Russian Zipformer → Qwen2.5 → Supertonic turn;
- a direct unsupported-locale TTS probe confirming the offline eSpeak fallback;
  and
- a manual TTS candidate turn confirming that the override remains pinned.

The runtime installer already proves all five ASR fixture transcripts and Qwen
identity. A UI green light without an actual full turn is not sufficient for
end-to-end acceptance.

## 8. Cable and reboot acceptance

As of 2026-07-29, the final APK pair has completed these gates:

- package hashes, PIDs, process owners, private model paths, and boot identity
  were recorded;
- a true reboot left all demo processes absent, then first unlock plus launching
  only the main app started main, runtime, LFM, and Qwen from phone storage;
- English, Mandarin, Russian, Japanese, and Korean full turns passed 5/5 on
  that cold-started airplane-mode stack;
- Home/return testing kept the four PIDs unchanged and showed the expected
  binding/wake-lock release behavior; and
- lifecycle, memory, thermal, freezer, crash, and low-memory evidence was
  captured.

The exact final main/runtime pair retained all four processes through a
70-second host-ADB shutdown with zero forwards. After reconnect, the runtime
remained foreground and started, with its important binding and active-client
wake lock healthy.

The remaining literal cable and operator gates are:

1. Physically remove the USB cable.
2. From the phone UI, repeat representative EN/ZH/RU/JA/KO turns.
3. Reconnect only after the turns, confirm ownership/PID survival, and collect
   logs and telemetry.

Airplane-mode turns, repeated cold-reboot soak, and an external microphone
measurement remain useful robustness gates but are separate from the eliminated
`adbd` process-lifetime dependency. Reopen-to-start after reboot is the intended
behavior, not unsolicited boot startup.

## Build and host gates

```bash
node scripts/test_streaming_audio_ui.mjs
python3 scripts/test_android_native_tts_bridge.py
python3 scripts/test_profile_text_llm_streaming.py
PYTHONPATH=scripts python3 scripts/test_android_streaming_telemetry.py
PYTHONPATH=scripts python3 -m unittest -v \
  scripts/test_android_asr_companion.py \
  scripts/test_phone_acceptance_harness.py
bash scripts/test_adaptive_energy_vad.sh
scripts/android_text_companion/run_host_tests.sh
```

Build the modules directly when needed:

```bash
env JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home \
  ANDROID_HOME="$HOME/Library/Android/sdk" \
  android-app/gradlew --max-workers=2 -p android-app :app:assembleDebug
```

The runtime-companion build is normally performed by its installer because the
required JNI directory is generated from verified external caches.

Host gates do not prove retail-phone performance, transcript parity, acoustic
quality, HTP execution, unplug survival, or reboot behavior.

## Interaction limits

- The main app accepts up to 30 seconds of recorded input.
- SenseVoice uses a fixed ten-second graph. Short canonical 16 kHz mono PCM is
  zero-padded to exactly ten seconds; longer recordings are segmented into
  requests of at most 9.5 seconds.
- Qwen uses a 2,048-token context, one slot, four generation/prompt threads,
  batch 128, micro-batch 64, no GPU layers, and a response ceiling set by the
  main app.
- Conversation memory is opt-in and bounded; revalidate behavior after any
  context or response-budget change.
- Current neural TTS model switching avoids retaining two large candidate
  models at once. The latest build passed automatic ZH/RU switching, manual
  pinning, three-chunk ordering/drain, and cancellation before and after
  playback start on the physical phone.

## Troubleshooting

### Runtime installer says a cache is missing

Run the relevant preparation command from step 2. Do not copy an unverified
same-named file into the cache: the installer enforces byte counts and hashes.

### QNN artifacts exist but ASR is not ready

`artifacts_ready` means files are present. `ready` and `execution_verified`
become true only after the current service process returns a non-empty
transcript. Rerun `scripts/install_app_owned_runtime.sh`; it exercises both QNN
fixtures and prints the execution-aware health payload.

QNN HTP v81 additionally requires the public vendor `libcdsprpc.so` declaration
and the immutable APK copy of `libQnnHtpV81Skel.so`. The current companion
contains both fixes. A custom APK missing either can fail device creation even
when the other QNN libraries are present.

### Qwen is missing or reports the wrong model

The expected alias is `Qwen2.5-1.5B-Instruct-Q4_K_M`, backed by
`qwen2.5-1.5b-instruct-q4_k_m.gguf` in companion-private storage. The installer
checks `/health`, `/v1/models`, file integrity, and process UID. Do not substitute
the legacy Qwen3 shell companion and call it the current deployment.

### Main app is current but the runtime stays unavailable

Confirm that both APKs came from the same checkout/signing key. Open the
runtime-companion launcher once and keep its foreground notification enabled.
The main app fails closed per route when the companion or an asset bundle is
missing.

### TTS falls back or a voice is absent

Run the TTS installer with `--verify-only`. Confirm that `candidates.json` was
installed last, every referenced file matches its manifest, and the requested
locale is supported. Automatic-mode load failure may use eSpeak; a manually
pinned candidate should surface an error instead.

### A service is killed in the background

Set both apps to unrestricted battery use and keep both foreground
notifications enabled. Android process policy can still reclaim an app under
pressure; this is different from the eliminated `adb shell` cgroup defect.
Reopen the main app to restart the intended services.

## Evidence boundary

The current app-owned ASR/QNN proof is recorded in
`reports/progress_20260729/ASR_QNN_AUDIT.md`. The Qwen selection, package
ownership proof, phone sweep, and current NPU limitations are in
`reports/progress_20260729/LLM_AUDIT.md`. TTS candidate measurements are in
`reports/android_phone/streaming_response/tts_research_20260726/DEVICE_RESULTS.md`.

The device is a retail Nubia NX809J / SM8850, not AR1 or glasses. Qualcomm AI
Hub QRD component profiles remain a separate evidence class. No full LFM,
Qwen, or TTS NPU deployment is claimed. Cold-reboot/main-only acceptance is
complete for the exact previously accepted installed APK pair; current source
builds require fresh device acceptance, and literal physical USB removal is not
yet proven.
