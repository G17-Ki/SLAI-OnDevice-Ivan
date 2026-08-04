# Supertonic full-chain raw-ONNX QNN r1 failure

## Outcome

The first physical four-component Supertonic chain candidate failed closed. It
did not fall back to CPU and did not promote a full-chain, PCM, latency, or TTS
claim.

- Package: `ai.liquid.lfmdemo.supertonicchainprobe`
- Installed APK bytes: `197687829`
- Installed APK SHA-256:
  `4929a85fbb72da8ee83152e6b0258f223e5d9c4692724be7c14c0eb42538c981`
- Run ID: `st-chain-20260729T175400Z-root4929-r1`
- App elapsed time: `3420 ms`
- App status: `failed_closed`
- ORT error: `This session contains graph nodes that are assigned to the
  default CPU EP, but fallback to CPU EP has been explicitly disabled by the
  user.`

The matching app-private report was archived before the collector raised:

`device_chain_run_20260730T015400Z_root4929_r1/supertonic_chain_device_report_st-chain-20260729T175400Z-root4929-r1.json`

Report SHA-256:
`e271a9dce0980ddcebb9bc6938ba8eedccf9dced07918c1564d4d04ffbcd2394`.

## Integrity and phone state

The candidate was installed under a separate package and never replaced either
accepted APK. Post-run hashes remained exact:

- `ai.liquid.lfmdemo`:
  `a24e203a30dcaaf9215dc3f0d5b99fa4b893a9e0ceca71c99c13e49bdb1b020a`
- `ai.liquid.lfmdemo.runtime`:
  `33bf0d6d284d6eaad96308f22a8d10c3879ba6d0b792f893486be03f0a17e244`

Airplane mode remained `1`, the ADB forward list was empty, and the failed
candidate process was force-stopped after report and hash capture.

## Component localization

The r1 app did not persist progressive stage state, so its report alone cannot
identify the exact session that failed. A distinct diagnostic trust root is
required before making an exact stage claim.

A static QAIRT 2.42 source/operator audit narrows the likely incompatibility:

- The identical 1,086-node duration graph already passed the physical HTP gate
  with CPU fallback disabled.
- The text encoder introduces `Split`, `Tanh`, and `Tile`; these are documented
  in the pinned QAIRT 2.42 HTP FP16 supported-op material.
- The vector estimator contains 22 `DynamicQuantizeLinear` and 38
  `MatMulInteger` nodes, neither of which appears in the pinned QAIRT 2.42 ONNX
  converter supported-op table used by the audit.
- The vector estimator also contains one `Softplus`; the pinned QAIRT 2.42
  OpDef table marks HTP BF16, FP16, INT16, INT8, and OTHER support as `NO`.

This makes the vector estimator the structural leading suspect, but it remains
an inference until the stage-diagnostic candidate reports the failing session.

## Next bounded gates

1. Preserve the r1 APK, trust root, local verifier, and failed evidence.
2. Build a distinct diagnostic candidate that records `failure_stage` and the
   completed-stage inventory while retaining
   `session.disable_cpu_ep_fallback=1`.
3. Run duration, text encoder, vector estimator, and vocoder as individually
   trust-rooted sessions before retrying the chained path.
4. If the vector graph is confirmed, use exact QAIRT 2.42 remotely to replace
   dynamic integer quantization and unsupported `Softplus` with a numerically
   verified HTP-compatible graph. Do not substitute a QAIRT 2.45 context into
   the accepted 2.42 runtime.
5. Re-run the full chain only after all four component sessions pass with
   app-UID QNN/HTP, FastRPC, DMA-buffer, resident-hash, and protected-APK
   evidence.

## Remote-build status

The pinned QAIRT 2.42 Qwen/Genie Colab controller and a matching remote
preflight/worker are ready, but no Colab VM was allocated in this continuation.
The designated Colab CLI OAuth cache requires an interactive authorization
refresh, which was unavailable during this run. The build
therefore remains `not_started`; no local conversion or heavy inference was
attempted.
