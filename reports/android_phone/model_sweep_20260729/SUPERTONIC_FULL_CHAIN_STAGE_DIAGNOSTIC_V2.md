# Supertonic full-chain stage diagnostic v2

Status: host gate complete; physical r1 run complete and independently verified.
No ADB or device mutation occurred while producing the host gate. The later
physical run was executed by the isolated device lane.

## Purpose

The accepted r1 full-chain probe failed with ORT CPU fallback disabled, but its
report did not identify the component or phase. Diagnostic v2 preserves the
same raw ONNX models, ORT-QNN runtime, QNN HTP settings, fixtures, goldens, and
disabled CPU fallback while recording the exact active stage and phase plus all
successfully completed prior-stage evidence.

This package is diagnostic only. A pass or localized failure cannot promote
full TTS, audio quality, language identity, latency, unplugged durability, or
production integration.

## Frozen host artifacts

- Package: `ai.liquid.lfmdemo.supertonicchaindiagprobe`
- Activity: `ai.liquid.lfmdemo.supertonicchaindiagprobe.ProbeActivity`
- Gate: `supertonic3_full_chain_stage_diagnostic_raw_onnx_ort_qnn_sm8850_probe_v2`
- Version: `0.2.0-stage-diagnostic-raw-onnx-qnn-probe`
- APK: `android-app/supertonic-chain-diagnostic-probe/build/outputs/apk/debug/supertonic-chain-diagnostic-probe-debug.apk`
  - bytes: `197695292`
  - SHA-256: `b854439ce533027b447822e86944d67eed27f1deb0d57bbf147b7c6e189bb14c`
- Spec: `supertonic_chain_diagnostic_ort_qnn_probe_spec.json`
  - bytes: `4890`
  - SHA-256: `030982d0fc1a6f6c7c5e636008ff7309fbbfe87028d7ba04c3874791645b8040`
- Local verification: `supertonic_chain_diagnostic_ort_qnn_apk_local_verify.json`
  - bytes: `11325`
  - SHA-256: `283220900873d0f0a3a3da672fe5e48eae5e32409927db8edfe3c85ea1c1ca40`
- Embedded payload manifest:
  - bytes: `19969`
  - SHA-256: `96a9254d1f793147869da198a717901e90ae2609694b58fce5b6a689fae77c29`
- Diagnostic stager:
  - bytes: `8291`
  - SHA-256: `1096b9ba4ef6882a0d6468f97f8b697da0bbb1b4f2cb8bda4b010f6de0399438`
- Pinned r1 source spec:
  - bytes: `25413`
  - SHA-256: `239a83e940e22f0289af5059900f5b815ea2b865da4abac3c698558fd0494dc5`
- Pinned r1 payload manifest:
  - bytes: `19378`
  - SHA-256: `735e61b070c7ad199a67bc18829036f4575551b5533a49e9ead5b36392f0819f`

The APK was produced twice from clean diagnostic-module builds with JDK 17 and
two Gradle workers. Both builds were byte-identical and lint passed both times.
The accepted r1 module was not cleaned or rebuilt.

## Stage evidence

The report preserves an ordered prefix of these successful stages:

1. `payload_install`
2. `runtime_initialization`
3. `duration_predictor`
4. `text_encoder`
5. `vector_estimator`
6. `vocoder`

Runtime initialization distinguishes native-library loading, ORT environment
creation, provider validation, and fixture loading. Each neural component marks
`session_create`, `contract_validation`, `execute`, and `output_validation`
before the corresponding failure boundary. Vector execution additionally
records `failure_vector_step`, completed vector-step count, and completed
vector-step reports.

Failure reports include `failure_stage`, `failure_phase`, ordered
`completed_stages`, `completed_stage_reports`, session-created flags, vector
progress, `cpu_ep_fallback_disabled=true`, run ID, and both completion markers.
Reports are fsynced to `latest.json.partial` and atomically renamed to
`latest.json`.

Static QAIRT 2.42 documentation makes a vector-stage compatibility failure
plausible because the graph contains DynamicQuantizeLinear, MatMulInteger, and
Softplus patterns with unfavorable HTP support evidence. This is only a prior;
the physical diagnostic report decides the observed failure stage.

## Host gates completed

- Diagnostic source and claim-boundary gate: pass.
- Diagnostic stager tests: 4 pass.
- Host APK verifier tests: 2 pass.
- No-ADB device verifier tests: 3 pass.
- Collector launch-hygiene gate: pass.
- Python compilation: pass.
- Ruff: pass.
- Whole-APK verification: pass.
- Android lint: pass twice.
- Synthetic complete failed-report verification: pass.

The earlier host checkpoint stated that the collector had not yet been
physically exercised. That statement was accurate at handoff and is now
superseded by the verified physical r1 result below.

## Physical r1 result

Run ID: `st-chain-diag-20260730-r1`

- Archived app report:
  - bytes: `6446`
  - SHA-256: `79b6119da8c31c61b3ef1e6d3886cf8aed3f08da30730b957eeb1b7229a5e6c4`
- Archived collector evidence:
  - bytes: `4027`
  - SHA-256: `fa0d62c8a87923ec45215c5951a05337eb59ce6ad952e9dd62d05e7f68236f36`
- Archived independent verification:
  - bytes: `2274`
  - SHA-256: `f7200f24321c290dc33c248de1a454af822a2c77eb729175e4e21861dab132ba`
- Verification status: `diagnostic_failure_localized_verified`

The verified ordered completion prefix is:

1. `payload_install`
2. `runtime_initialization`
3. `duration_predictor`
4. `text_encoder`

The vector estimator then failed at `session_create`, before vector step 0. Its
session-created flag is false, the completed vector-step count is zero, and the
vocoder was not reached. ORT reported that the vector session contained nodes
assigned to the default CPU EP while CPU EP fallback was explicitly disabled.

The completed duration output was `2.12890625` seconds. It derives a waveform
length of `93884` samples and remains in the required 31-of-32 latent bucket.
The text encoder completed with a finite `1 x 256 x 32` output and its recorded
host comparison remained diagnostic-only.

The collector confirmed before launch that both `latest.json` and its partial
file were absent. The archived report matches the requested run ID and both
completion markers, and no partial remained after capture. The installed
diagnostic APK exactly matched the frozen host APK. Airplane mode was enabled
on the NX809J/SM8850 target.

The accepted packages were byte-identical before and after the run:

- `ai.liquid.lfmdemo`:
  `a24e203a30dcaaf9215dc3f0d5b99fa4b893a9e0ceca71c99c13e49bdb1b020a`
- `ai.liquid.lfmdemo.runtime`:
  `33bf0d6d284d6eaad96308f22a8d10c3879ba6d0b792f893486be03f0a17e244`

A fresh no-ADB verifier replay produced the same semantic verification document
as the archived result.

### Physical evidence boundaries

- This is a localized compatibility failure, not a full-chain or full-TTS pass.
- It does not prove vocoder execution, PCM output, audio quality, language
  identity, promoted latency, or production readiness.
- It proves execution through the exact QNN-configured package with CPU fallback
  disabled; the three archived JSON files do not contain process-level NPU/HTP
  profiling or mapped-library evidence.
- The error localizes the incompatibility to vector-estimator session creation.
  It does not identify a particular ONNX operator as the device-proven culprit.
- Post-run process-stop and accepted-runtime health are not archived inside these
  three diagnostic JSON files and require their separate device-lane evidence.
- Host collection wall time is `1011` ms earlier than the phone report's wall
  completion time because they are different clock domains. Cross-host wall
  timestamps must not be used for latency. The phone's `3411.431` ms monotonic
  elapsed value is a diagnostic total only and is not promoted latency.

The collector invocation used for this archived run was:

```bash
python3 scripts/collect_supertonic_chain_diagnostic_ort_qnn_probe_device.py \
  --serial <SERIAL> \
  --run-id st-chain-diag-20260730-r1 \
  --output-dir reports/android_phone/model_sweep_20260729/device_chain_diag_run_20260730_r1 \
  --timeout-seconds 180
```

The collector intentionally accepts either a fully verified diagnostic chain
pass or a structurally complete, stage-localized failed-closed outcome. It does
not publish a `current` alias.
