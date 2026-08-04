# Android bundled-WAV Whisper frontend milestone

## Outcome

The isolated package `ai.liquid.lfmdemo.whisperprobe` now contains a real-audio
frontend that consumes fixed, APK-bundled English and Japanese PCM16 WAVs and
emits UINT16 `[1,80,3000]` tensors through JNI. The implementation uses exact
Torch/QAI framing and reflection, pinned Hann and Hugging Face mel tables, one
persistent 400-point real KissFFT plan, explicit FMA mel accumulation, and the
QAI float64-division/ties-to-even quantizer.

This is still a verification-only milestone:

- the manifest has no `INTERNET` or `RECORD_AUDIO` permission;
- microphone input and semantic transcription remain disabled;
- the fixed decode remains four tokens, not a 199-token decode;
- no ADB command, phone install, or device mutation was performed here;
- the accepted `ai.liquid.lfmdemo` and `ai.liquid.lfmdemo.runtime` packages
  were not touched;
- no training, fine-tuning, model conversion, or heavy host inference ran.

## Native contract

The new core is shared by the Android JNI library and the host unit runner.
It fails closed unless all of the following hold:

1. RIFF/WAVE input is mono PCM16 at 16 kHz with the exact pinned PCM frame
   count;
2. PCM is converted as float32 `int16 / 32768.0` and right-padded to 480,000
   samples;
3. frame start is `frame*160-200`, with Torch reflection (`-1 -> 1`);
4. the periodic Hann `[400]` and mel `[201,80]` tables match their pinned
   byte sizes and SHA-256 identities before JNI execution;
5. one KissFFT configuration is allocated with `pthread_once` and reused;
6. power is `hypotf(real, imag)^2`, and mel accumulation uses explicit
   `fmaf`;
7. postprocessing is clamp-to-`1e-10`, `log10`, global max-minus-8, then
   `(x+4)/4`;
8. each float32 feature is promoted to double before division by
   `4.677007018472068e-05`, rounded ties-to-even, shifted by zero point 32072,
   clamped, and emitted little-endian UINT16;
9. output must contain exactly 240,000 values in mel-major/frame-minor order.

The JNI binary is cross-compiled only for `arm64-v8a` with NDK
`27.0.12077973`, API 28, `-O2 -fno-fast-math -ffp-contract=off`, and a pinned
vendored KissFFT source set. Its FFT plan persists for the app process and a
mutex serializes access to the shared plan.

## Bundled assets

The deterministic staging manifest SHA-256 is
`aea54e5c751f9a2b489c709e5ca149db7c39cbdabea3ed4675483c00e236693f`.

| Asset | Bytes | SHA-256 |
|---|---:|---|
| English WAV | 105,682 | `b5fa81a74e608624b52fb3e4cf895d6982a455022597a0da0ad3dc52866f8b69` |
| Japanese WAV | 79,464 | `0da6ee3b9514a96597bf9225a1bcdf7cae86608e095e157490899a443948b761` |
| Hann `[400]` float32 LE | 1,600 | `c692d19f0b2bd745f0dfc967261c3e3d21a2473e485a677d219e9b4a73e0218e` |
| Mel `[201,80]` float32 LE | 64,320 | `d18b48f33345d958ea71098bb990b5e32cebc78d9a2cf65ca0d7302f34ebe8a9` |
| EN UINT16 golden | 480,000 | `29c478d40bb98b7782280dcfbc98662cee514df7e62c180c5ba849db24a2f6a1` |
| Corrected JA UINT16 golden | 480,000 | `8cf37ba9a60f5cd91453dae2d721e9fd7fec69919582223bf02a3f76f8421a80` |

The rejected Japanese float32-division tensor beginning `05ec7e5f` is not
present in the staged manifest, Java gate, native implementation, or Android
build gate.

## Host numerical gate

`python3 scripts/test_whisper_android_frontend.py` compiled the exact Android
core in portable host mode and passed 5/5 tests. It reproduces the completed
native audit result exactly:

| Fixture | Produced tensor SHA-256 | Differing UINT16 | Fraction | Max delta |
|---|---|---:|---:|---:|
| EN | `2f959ac4b0c75b4c67087df5e5ff3a530f37c65f54afa4480d7b54ce480b15c1` | 105 / 240,000 | 0.0004375 | 1 |
| JA | `5e434a8a99650a3248a6c69a807daabc6179e0bee7c52a771346e2b3a385d96e` | 70 / 240,000 | 0.000291667 | 1 |

The test also proves that staging rejects a tampered corrected-JA golden and
that the native core rejects a truncated WAV. The host bound is maximum delta
1 and at most 240 differing values (`0.001`). It is not promoted to target
acceptance until the same comparator runs on SM8850.

## Probe integration and timing evidence

When the frontend bundle is present, the probe now:

1. verifies the unchanged schema-2 model/runtime payload;
2. verifies the frontend manifest and every bundled asset SHA-256;
3. computes and compares EN and corrected JA tensors;
4. persists produced SHA-256, differing count/fraction, maximum delta, exact
   hash match, and per-fixture/aggregate frontend timing;
5. fails closed before QNN execution if either fixture exceeds the numerical
   bound;
6. passes the generated EN tensor to the existing QNN encoder only after both
   fixtures pass;
7. records encoder execute time, four per-decoder-step times, aggregate decoder
   model time, and decoder-loop wall time while preserving the prior timing
   fields.

The report continues to set `real_audio_transcription=false` and
`semantic_transcript_claimed=false`.

Every persisted report is now bound to a strict automation `run_id`, records
UTC/unix start and completion timestamps, exact wall elapsed milliseconds, and
an independently measured monotonic elapsed time. Publication is fail-closed:
the app fsyncs `latest.json.partial` and then performs a same-directory atomic
rename to `latest.json`; a completed report carries
`WHISPER_PROBE_REPORT_COMPLETE_V1`. Manual button runs receive their own unique
IDs, so automation cannot accidentally accept a stale manual result.

## Android build and local APK gate

Two independent clean JDK 17, maximum-two-worker builds and lint runs passed:

```sh
LMF_CACHE_ROOT="${XDG_CACHE_HOME:-$HOME/.cache}/lmf-demo"
JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home \
ANDROID_HOME="$LMF_CACHE_ROOT/android-sdk" \
ANDROID_SDK_ROOT="$LMF_CACHE_ROOT/android-sdk" \
WHISPER_PROBE_PAYLOAD_DIR=/private/tmp/lmf-whisper-ort-qnn-probe-payload-en29c-v4 \
WHISPER_FRONTEND_ASSET_DIR=/private/tmp/lmf-whisper-frontend-assets-v1 \
./gradlew :whisper-probe:clean :whisper-probe:assembleDebug \
  :whisper-probe:lintDebug \
  --max-workers=2 --no-daemon
```

Result:

- APK: `android-app/whisper-probe/build/outputs/apk/debug/whisper-probe-debug.apk`
- bytes: `413123865`
- SHA-256: `319fbd224be175f64dcd85cf888d240ed135db4a6b481b5a7518ee343eba37f4`
- reproducibility: both clean builds were byte-for-byte identical (`cmp` pass)
- frontend JNI bytes: `23048`
- frontend JNI SHA-256:
  `475cad57685c04f3027c3c07a5dbfc9b67a33e8f0669fd4f2da36eefd48e9a6d`
- frontend JNI ELF: 64-bit little-endian AArch64
- lint: passed

The independent whole-APK verifier passed and is recorded in
`whisper_android_frontend_local_verify.json`. It requires the exact unchanged
schema-2 base payload, exact base signing/manifest/alignment policy, the six
frontend assets, canonical frontend manifest, and the pinned frontend AArch64
JNI binary. It checks the APK byte count and SHA-256 before opening the ZIP,
rejects duplicate entries, and pins the complete asset, native-library, and
non-payload member sets. The original schema-2 verifier remains unchanged and
intentionally continues to reject extra archive members.

## Host collector and device-evidence contract

The host collector is implemented but was not executed in this milestone. It
first runs the independent local APK verifier, installs only the exact pinned
APK, ignores stale app reports by `run_id`, and archives a matching
`failed_closed` report before raising. It publishes no `*_current.json` alias.
Successful device evidence must additionally prove:

- exact installed whole-APK identity and unchanged accepted-package hashes;
- exact `{en, ja}` frontend fixture inventory and corrected JA golden;
- bound-respecting tensor differences, with boolean values rejected from all
  numeric timing/fraction fields;
- top-level EN/JA frontend timings identical to their fixture timings;
- QNN requested with CPU EP fallback disabled, four decoder steps, and the
  native frontend-generated EN tensor as encoder input;
- package-UID mappings for ORT/QNN/frontend libraries, an open CDSP FastRPC FD,
  and Qualcomm DMA-buffer FDs;
- archived raw `/proc/<pid>/status`, `/proc/<pid>/maps`, and `/proc/<pid>/fd`
  snapshots whose byte counts and SHA-256 hashes re-parse to the summarized
  process evidence;
- app-private resident models/runtime and the pinned frontend JNI hash.

The SoC collector uses `ro.soc.model` first and has fail-closed fallbacks for
QTI vendor and sysfs identities when that property is empty.

## Validation commands

```sh
python3 scripts/test_whisper_android_frontend.py
python3 scripts/test_whisper_ort_qnn_probe_source.py
python3 -m unittest -v \
  scripts.test_verify_whisper_android_frontend_apk \
  scripts.test_verify_whisper_android_frontend_device \
  scripts.test_run_whisper_android_frontend_device_probe
python3 -m py_compile \
  scripts/stage_whisper_android_frontend_assets.py \
  scripts/test_whisper_android_frontend.py \
  scripts/verify_whisper_android_frontend_apk.py \
  scripts/verify_whisper_android_frontend_device.py \
  scripts/run_whisper_android_frontend_device_probe.py
python3 scripts/verify_whisper_android_frontend_apk.py \
  --apk android-app/whisper-probe/build/outputs/apk/debug/whisper-probe-debug.apk \
  --report reports/android_phone/streaming_response/whisper_v81/whisper_android_frontend_local_verify.json
```

All listed local gates passed, including 38 focused trust-root, device-verifier,
and collector regressions. The frontend APK above was not installed. The phone
therefore still has the older accepted isolated base probe
`3a620198ec050fda6f6c8f8e1e04f1ee0f7764368c141c3aadb799b1a47bb64e`.
Target EN/JA hashes, target differing counts, target timing, QNN execution, and
any decoded-token result for this new APK remain deliberately unclaimed until
a later isolated device run.

## Target-device addendum — 2026-07-30

The preceding uninstalled status is superseded by the isolated device gate for
run `frontend-20260730T010200Z-root319f-r3`. The exact APK above was installed
on NX809J / SM8850 in airplane mode and the machine verifier passed.

| Fixture | Target-produced SHA-256 | Differing UINT16 | Fraction | Max delta | Compute |
|---|---|---:|---:|---:|---:|
| EN | `b5539fa1c0b55a198889e01b936ca2b895ee5ffdd5dbc4c02537610fdf10a8b2` | 104 / 240,000 | 0.000433333 | 1 | 32.986 ms |
| JA | `f61bf92865b4c444cbd66875ef8c43369e4d99dff5a81351abb034958fb58ec1` | 71 / 240,000 | 0.000295833 | 1 | 32.204 ms |

The EN tensor was passed directly into the QNN encoder. Encoder execution was
195.829 ms; the four decoder calls were 6.644, 4.420, 4.018, and 4.003 ms.
CPU EP fallback remained disabled. UID 10510 / PID 27511 held the required
ORT/QNN/frontend mappings, an open CDSP FastRPC descriptor, and 16 Qualcomm DMA
buffers. Both accepted package hashes were unchanged before and after the run.

Verified evidence:

- `device_frontend_run_20260730T010200Z_root319f_r3/device_verify_frontend-20260730T010200Z-root319f-r3.json`
  (`e3dc1ffef422fee768acfcd312326ea122ed32f769e48a03246bb96e2d8756e1`);
- matching app report SHA-256
  `15483105463fd5f11d350d81a030ebf21e22ef26a41fdbc7df2e72a615a0e2f9`;
- matching evidence envelope SHA-256
  `2e0d8ad715bcc2988be01b111aa39ee9506644a7fb557e9efb9e77754c177ce4`.

This remains a bounded frontend plus four-step compatibility probe. It does not
claim a semantic transcript, microphone capture, or production runtime
integration.
