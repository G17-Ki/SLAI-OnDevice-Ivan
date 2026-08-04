# Supertonic duration raw-ONNX QNN result

Date: 2026-07-30 (Asia/Shanghai)

## Result

The exact Supertonic 3 INT8 duration-predictor source ONNX executed on the
NX809J through ORT 1.25 and the app-owned QAIRT 2.42 QNN HTP backend with CPU
EP fallback disabled. Independent device verification passed. This proves one
component only; it is not yet a full TTS or audio-quality result.

## Pinned identities

- Package: `ai.liquid.lfmdemo.supertonicprobe`
- APK bytes: 55,486,252
- APK SHA-256:
  `698940112aa2c197315f1cc996554ca9171ab46c7a8ff65623166f3c850a8801`
- Source model SHA-256:
  `c3eb91414d5ff8a7a239b7fe9e34e7e2bf8a8140d8375ffb14718b1c639325db`
- ORT: 1.25.0
- QAIRT build: `2.42.0.251225135753_193295`
- Run ID: `st-duration-20260730T0038-698940`

The installed phone APK was hashed before launch and matched the external
whole-APK trust root. The accepted main and runtime APKs remained unchanged.

## Execution

- QNN session created: yes
- CPU EP fallback disabled: yes
- Symbolic overrides: batch 1, text length 32
- Expected duration: 2.127588987350464
- Actual duration: 2.12890625
- Absolute error: 0.0013172626495361328
- Session load: 245.144 ms
- Execute: 1.320 ms
- Total model probe: 260.893 ms

## NPU evidence

The package UID mapped app-owned ORT, `libQnnHtp.so`,
`libQnnHtpV81Stub.so`, and `libQnnSystem.so`, plus vendor
`libcdsprpc.so`. It held `/dev/fastrpc-cdsp` and 16 QCOM DMA-buffer file
descriptors. RSS was 236.934 MiB and PSS was 85.517 MiB. The resident private
model payload was 3.569 MiB.

## Evidence

- `supertonic_duration_ort_qnn_probe_spec.json`
- `supertonic_duration_ort_qnn_apk_local_verify.json`
- `supertonic_duration_device_report_st-duration-20260730T0038-698940.json`
- `supertonic_duration_device_evidence_st-duration-20260730T0038-698940.json`
- `supertonic_duration_device_verify_st-duration-20260730T0038-698940.json`
- Raw correlated process status, maps, fd, and smaps-rollup snapshots in this
  directory with the same run ID.

## Next gate

Build a second isolated chain candidate containing the exact text encoder,
vector estimator, and vocoder source graphs. Each session must keep CPU
fallback disabled and show the same app-UID HTP/FastRPC/DMA evidence. Only
after all four stages pass should the experiment emit PCM and measure RTF,
first audio, clipping, silence, language identity, storage, and reboot/unplugged
durability.
