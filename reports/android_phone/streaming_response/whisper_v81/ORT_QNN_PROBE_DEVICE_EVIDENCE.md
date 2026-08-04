# Nubia Whisper Small ORT-QNN probe device evidence

- Date: 2026-07-29
- Device: Nubia NX809J / SM8850 / arm64-v8a / HTP v81
- Package: `ai.liquid.lfmdemo.whisperprobe`
- Result: **isolated compatibility probe pass**

## Claim boundary

This proves that the pinned Whisper Small W8A16 EPContext encoder and decoder
can create QNN sessions and execute through the HTP backend from a normal app
UID on this phone. It is feasibility evidence for later runtime integration.
It does not yet prove a production audio frontend, microphone transcription,
multilingual accuracy, or integration into `ai.liquid.lfmdemo.runtime`.

## Local trust gate

- APK: `android-app/whisper-probe/build/outputs/apk/debug/whisper-probe-debug.apk`
- Bytes: `412376700`
- SHA-256: `3a620198ec050fda6f6c8f8e1e04f1ee0f7764368c141c3aadb799b1a47bb64e`
- Embedded feature tensor SHA-256:
  `29c478d40bb98b7782280dcfbc98662cee514df7e62c180c5ba849db24a2f6a1`
- Exact embedded members: 15/15 verified.
- Signing certificate SHA-256:
  `9ad7b2846b7c18853b99e444d4d05323321b3a9118c13c5a8fc242e58b5859e4`
- Manifest: no `INTERNET` or `RECORD_AUDIO`; arm64-v8a only; optional
  `libcdsprpc.so`.
- Hardened verifier report:
  `nubia_whisper_ort_qnn_probe_local_verify.json`.
- Focused Python tests: 31/31 pass. Ruff, `py_compile`, JSON parsing, and
  `git diff --check` pass.
- The separate device-evidence verifier authenticates the local verifier
  report and app-private result, installed APK identity, accepted-package
  hashes, package/process UID, QNN/fallback flags, required library maps,
  FastRPC FD, DMA buffers, resident payload inventory, decoder semantics, and
  timings. Its 8/8 tests pass;
  machine result: `nubia_whisper_ort_qnn_probe_device_verify.json`.

The verifier audit also fixed two gate defects before installation: a
65-character `aapt2` digest typo in the schema-2 trust root and an `aapt2`
parser that ignored indented optional-feature records. Regression tests now
cover both failures.

## Installation and ownership

- Device was in airplane mode (`airplane_mode_on=1`).
- Installed base APK SHA-256 exactly matches the verified local APK:
  `3a620198ec050fda6f6c8f8e1e04f1ee0f7764368c141c3aadb799b1a47bb64e`.
- Package UID: `10510` / `u0_a510`.
- Successful rerun PID: `23378`, with all four UID fields equal to `10510`.
- The app-private report is owned by `u0_a510:u0_a510` and was read with
  `run-as ai.liquid.lfmdemo.whisperprobe`.
- The copied app-private payload remains resident after the probe process is
  stopped: `352027 KiB` under
  `/data/user/0/ai.liquid.lfmdemo.whisperprobe/files/whisper_probe/payload-v1`.
  Encoder/decoder wrappers, both EPContext binaries, metadata, and the exact
  feature fixture all match the schema-pinned byte counts.

## QNN / HTP execution result

Both the cold installation run and a force-stop/relaunch run reported
`compatibility_probe_pass`.

- ORT: `1.25.0`.
- Available providers: CPU and QNN.
- Requested provider: `QNNExecutionProvider`.
- Backend: absolute app-private `libQnnHtp.so` path.
- `session.disable_cpu_ep_fallback=1`: confirmed in the package-authored report.
- Encoder session created and pinned UINT16 fixture executed.
- Decoder session created and four calls executed.
- Attention-mask updates: `199, 198, 197, 196`.
- Returned token IDs: `50259, 50259, 50259, 50259`.
- No semantic transcript is claimed for this four-token compatibility probe.

Rerun timing:

- encoder session load: `314.391 ms`;
- decoder session load: `106.648 ms`;
- encoder plus four decoder executions: `295.119 ms`;
- total: `739.743 ms`.

The first run, including first-time payload materialization and colder runtime
state, completed in `1271.234 ms`.

## Process-level HTP evidence

`/proc/23378/maps` showed the package UID loading:

- app-owned `libQnnHtp.so`;
- app-owned `libQnnHtpV81Stub.so`;
- app-owned `libQnnSystem.so`;
- app-owned `libonnxruntime.so` and `libonnxruntime4j_jni.so`;
- vendor `/vendor/lib64/libcdsprpc.so`.

`/proc/23378/fd` showed an open `/dev/fastrpc-cdsp` handle plus multiple
`/dmabuf:qcom,system` allocations. Together with successful QNN sessions,
disabled CPU fallback, HTP-v81 stub loading, and successful encoder/decoder
execution, this is direct process-level evidence that the probe used the CDSP
HTP route. No probe-specific QNN logcat line was retained, so logcat is not
used as an independent claim.

## Accepted-package integrity

Both protected packages were re-hashed after probe installation and execution:

- `ai.liquid.lfmdemo`:
  `a24e203a30dcaaf9215dc3f0d5b99fa4b893a9e0ceca71c99c13e49bdb1b020a`;
- `ai.liquid.lfmdemo.runtime`:
  `33bf0d6d284d6eaad96308f22a8d10c3879ba6d0b792f893486be03f0a17e244`.

Both remain byte-identical to their accepted hashes.

After evidence capture, only the probe process was force-stopped to release its
working memory; its APK and 352027 KiB private payload remain installed. The
accepted demo returned to the foreground with main/runtime/LFM/Qwen PIDs
`10556/10663/10690/10718`. Phone-local `6010/health` reported both app-owned
ASR backends ready and execution-verified, while `6020/health` returned
`{"status":"ok"}`. No ADB forward was needed.

## Preserved reports

- `nubia_whisper_ort_qnn_probe_device_report_20260729.json`
- `nubia_whisper_ort_qnn_probe_device_rerun_20260729.json`

## Next gates

1. Implement the exact QAI/HF-compatible real-audio frontend and target-test
   its EN and JA tensors.
2. Run real EN, JA, KO, ZH, and RU fixtures and measure WER/CER plus latency.
3. Integrate the proven route into a candidate runtime package without
   replacing the accepted runtime until the full gate passes.
4. Repeat the offline, force-stop, unplug, and reboot acceptance tests.
