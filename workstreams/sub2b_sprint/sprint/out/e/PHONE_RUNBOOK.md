# Nubia NX809J phone runbook

> **PRECONDITION — NO PHONE NPU EXECUTION WITHOUT A VERSION-MATCHED COMPILED ARTIFACT AND RUNTIME.** Compile build and device runtime must both equal exact QAIRT/QNN `2.45.0.260326154327`. Missing, older, newer, mixed or unknown pairs stop at CPU. Never substitute the staged 2.40 Parakeet contexts, QAIRT 2.48, a system/vendor QNN library, `qnn-net-run` under the shell UID, or silent CPU fallback.

## Current stop state

This bundle is intentionally `BLOCKED_APP_RUNNER_MISSING` and `CPU_STAGE_ONLY`. It has no APK implementing the app-private import/request/result protocol, no complete CPU stack, and zero exact-2.45 NPU contexts. The exact 2.45 runtime members are staged and verified but are not incorporated into an app. Running `./install.sh` today exits `20` before any `adb` call; a direct NPU request exits `24` before any `adb` call.

Do not remove either blocker just to get past the scripts. The app integration task must first:

1. Build the two-package/app-UID harness or an approved isolated service APK, with no shell-owned inference process and no adb forward.
2. Incorporate exactly one QAIRT provider: the staged raw 2.45 members or GenieX v0.3.18, never both.
3. Add `apks/harness.apk`, record its SHA-256 in `config/app.env`, and implement `--es lmf_request_path <app-private request.psv>` on the configured Activity.
4. Make the app write `results/<request-id>/status`, `placement.psv` and `run.log` under its private directory. `placement.psv` must report `actual_backend`, `cpu_fallback`, `compile_qairt` and `runtime_qairt`.
5. For NPU, add hash-pinned SM8850 contexts compiled at exact `2.45.0.260326154327`, install them app-private, and mark only those exact rows `ENABLED` in `manifest/npu_pairs.psv`.

## Exact future command sequence

The following commands are for the authorized later phone session, after the activation boundary above is complete. They were not run during this sprint.

```sh
cd workstreams/sub2b_sprint/sprint/out/e/artifacts/nubia_bundle

# 1. Resolve one authorized target and prove it is NX809J / SM8850 / arm64-v8a.
./verify.sh --identity

# 2. Hash every model/reference file and all 11 staged QAIRT members.
# SHA-256 is streamed through Android toybox, so no Python/jq/host sha tool is required.
./verify.sh --bundle

# 3. Install only the hash-pinned APK and app-private non-NPU payload.
# Current bundle stops before this command because the APK is missing.
./install.sh --mode cpu
./verify.sh --installed

# 4. Run a CPU-only scripted stage after a real CPU model and runner are registered.
# run.sh forces and verifies airplane mode, launches only the Activity, checks every
# package process UID, and requires the app to attest that QNN was not loaded.
./run.sh --backend cpu --stage asr --lang en \
  --fixture files/lmf/fixtures/parakeet_v3_3s/0.wav

# 5. Run a full prerecorded turn only after all selected CPU stages exist.
./run.sh --backend cpu --turn --lang ru \
  --fixture files/lmf/fixtures/ru/turn_01.wav

# 6. Collect logcat, package/device identity, PSS inputs, runtime mappings,
# app-private outputs and placement evidence without an adb forward.
COLLECT_ROOT="$PWD/collections" ./collect.sh
```

For an NPU session, first complete all exact-pair activation steps, then use this stricter sequence:

```sh
./install.sh --mode npu
./verify.sh --npu-ready
./run.sh --backend npu --stage asr --lang en \
  --fixture files/lmf/fixtures/parakeet_v3_3s/0.wav
COLLECT_ROOT="$PWD/collections" ./collect.sh
```

`--backend auto` is fail-safe: it selects NPU only when exact pairs are enabled and their installed runtime and context hashes pass; otherwise it records `CPU_STAGE_ONLY` and asks the app for CPU explicitly. A direct `--backend npu` never downgrades to CPU.

## Mechanical acceptance checks

Every script requires exactly one adb target, state `device`, with primary ABI `arm64-v8a`, an `SM8850` property and an `NX809J` product property. Any other attached/offline/unauthorized device fails selection. Any entry in `adb forward --list` fails the session, and failure to query the forward list also fails closed.

After installation, the APK hash must match `config/app.env`; `run-as <package> id -u` must equal the package manager UID and must be neither root (`0`) nor shell (`2000`). Every process whose name is the package or a package suffix is checked against that UID. Models, requests, results and DSP libraries remain app-private; inference is never executed from `/data/local/tmp`.

For NPU, the scripts independently verify the installed hashes of `libQnnHtp.so`, `libQnnHtpV81Stub.so`, `libQnnSystem.so`, `libQnnHtpV81.so` and `libQnnHtpV81Skel.so`, plus app-private `runtime.version`. Every enabled context is hashed in app-private storage. All compile/runtime identities must equal the pin byte-for-byte before launch.

The session is invalid if any log contains `Using newer context binary on old SDK`, `err 0x1388`, `Qnn API version ... mismatch`, `Can't create context from binary. Error 5000`, or `Qnn System library version ... mismatch`. `collect.sh` also rejects mappings of QNN libraries from `/system` or `/vendor`.

## Airplane-mode turn and collection notes

`run.sh` sets `airplane_mode_on=1`, broadcasts the state and reads it back before dispatching the request. This is a scripted offline check with the USB cable still attached for adb; it is not the later physical-unplug acceptance. Application results must stay private until `collect.sh` streams them after the run.

Software timestamps, logcat, PSS and placement gathered here become phone evidence only after schema and session review. They are not acoustic-onset latency, native-speaker quality evidence, or a glasses result.
