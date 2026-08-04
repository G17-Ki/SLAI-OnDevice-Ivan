# Android ASR companion

> **Deployment status (2026-07-29): superseded.** This directory is retained as
> the pinned-asset preparer, protocol reference, and legacy shell-owned
> diagnostic path. The current standalone demo packages ASR in
> `ai.liquid.lfmdemo.runtime`; see
> [`../../android-app/runtime-companion/README.md`](../../android-app/runtime-companion/README.md).
> The app-owned service has physically verified EN/ZH/JA/KO QNN/HTP transcripts
> and language tags plus the Russian CPU fallback. Do not launch this shell
> lifecycle for the current app-owned architecture.

This directory can deploy the older phone-local ASR diagnostic service for the
Nubia SM8850 without putting model weights, Qualcomm runtime objects, or Android
executables in the repository. Its preparation manifests are still reused by
the app-owned runtime installer.

- English, Mandarin, Japanese, and Korean use the verified ten-second
  SenseVoiceSmall INT8 QNN context on HTP v81.
- Russian uses `sherpa-onnx-zipformer-ru-int8-2025-04-20` on the phone CPU.
- Auto runs SenseVoice Auto and Russian Zipformer concurrently and returns both
  candidates. It does not invent a cross-model confidence score.
- The service binds only `127.0.0.1:6010`, so the phone WebView can call it and
  it is not exposed on Wi-Fi.

The HTTP wrapper is intentionally used instead of the optional Sherpa WebSocket
binary: it gives the app one fixed WAV contract, strict size limits, CORS, and a
single response schema across the QNN and CPU models.

## Historical shell-wrapper blocker

The 2026-07-27 device-log audit found that this shell handler's SenseVoice QNN
runner completed, but the wrapper rejected the parsed result on 9/9 requests.
The context and HTP execution were real; correct English/Mandarin transcript
delivery through this checked-in shell handler was not accepted. Instrument
the four `json_result_valid` failure branches and preserve failed request
output before loosening validation. The Russian CPU route is independent of
this defect.

This is not the status of the current Java app-owned server. On 2026-07-29 the
runtime-companion path fixed native-library exposure, packaged the HTP v81 DSP
skeleton, applied exact ten-second zero-padding, and passed the real EN/ZH/JA/KO
transcript fixtures and tags under the application UID. Evidence is in
`reports/progress_20260729/ASR_QNN_AUDIT.md`.

The pinned Android CLI has two output quirks handled explicitly by the wrapper:
QNN provider diagnostics can appear on stdout around the JSON line, and the
Russian transducer returns an empty result for a one-item CLI batch while a
duplicated two-item batch returns two identical correct results. The handler
extracts the final JSON result line and intentionally supplies the Russian WAV
twice. This adds redundant Russian decoder work but avoids inventing text or
shipping a known-empty single-item route.

## Pinned external assets

[`assets.json`](assets.json) pins the Russian archive at 60,239,942 bytes and
SHA-256
`d6a651569aacc9a177259fa54705dd76acae23f6a4d62ea6797bd220d4b57163`.
SenseVoice, QNN 2.40, ONNX Runtime 1.27, and Sherpa-ONNX 1.13.4 reuse the pinned
manifest in [`../npu_multilingual/assets.json`](../npu_multilingual/assets.json).
The Russian archive identifies
[`alphacep/vosk-model-ru`](https://huggingface.co/alphacep/vosk-model-ru/tree/main)
as its source; its model card declares Apache-2.0. SenseVoice retains its
upstream custom model license, which deployment stages beside the model when it
is present in the pinned archive.

The preparer rejects external roots inside the repository. Upstream model and
runtime licenses continue to govern the downloaded assets; do not commit or
republish them from this repository.

## Legacy prepare and shell deploy

Prerequisites:

- one SM8850 Android phone visible to `adb`;
- the Android QNN CLI runner built by
  `scripts/npu_multilingual/build_sherpa_qnn_android.sh`, or pass
  `--build-runner`;
- enough phone storage for the SenseVoice context, QNN runtime, Russian model,
  and runner.

From the repository root:

```bash
python3 scripts/android_asr_companion/prepare_assets.py \
  --seed-dir /path/to/optional/pinned-downloads

scripts/android_asr_companion/deploy.sh \
  --serial YOUR_ADB_SERIAL \
  --seed-dir /path/to/optional/pinned-downloads
```

`prepare_assets.py` verifies URL metadata, byte sizes, archive hashes, extracted
SenseVoice hashes, QNN runtime hashes, and the runner manifest. `deploy.sh`
repeats that verification immediately before staging. Use `--offline` to fail
instead of downloading. Use `--build-runner` only when the default external
runner cache is absent; it uses the existing resource-bounded build script.

The deployment stops an earlier service only through its recorded PID after
checking `/proc/PID/cmdline`. It never uses `pkill`, `killall`, or a process-name
match. It stages the files, writes fixed phone paths to `env.sh`, and starts:

```text
nohup toybox nc -s 127.0.0.1 -p 6010 -L sh handler.sh
```

The legacy process is phone-local, but it is launched through `adb shell` and
remains in `adbd`'s cgroup on the tested phone. Removing USB restarts `adbd`
and terminates the process; `nohup` only detaches descriptors and does not
provide cable independence. Keep USB connected when using this diagnostic
path, and run `start.sh start` again after a disconnect, `adbd` restart, or
phone reboot. For the current demo, use
`scripts/install_app_owned_runtime.sh` instead.

Lifecycle commands:

```bash
adb -s YOUR_ADB_SERIAL shell sh /data/local/tmp/lmf-asr-companion/start.sh status
adb -s YOUR_ADB_SERIAL shell sh /data/local/tmp/lmf-asr-companion/start.sh restart
adb -s YOUR_ADB_SERIAL shell sh /data/local/tmp/lmf-asr-companion/start.sh log 100
adb -s YOUR_ADB_SERIAL shell sh /data/local/tmp/lmf-asr-companion/start.sh stop
```

Add `--forward` to deployment only when host-side debugging is useful. The
phone app itself must use `http://127.0.0.1:6010` and needs no adb forwarding.

## HTTP contract

### Health and CORS

`GET /health` returns readiness and the exact input bounds. It is HTTP 200 only
when both backends are present, otherwise HTTP 503. `OPTIONS /asr` returns 204.
All responses include `Access-Control-Allow-Origin: *`, an exact byte
`Content-Length`, private-network preflight permission, `Cache-Control:
no-store`, and `Connection: close`.

### Recognition

```text
POST /asr?language=zh|en|ru|auto|qnn-auto
Content-Type: audio/wav
Content-Length: BYTES

<raw WAV bytes>
```

The body must be a canonical 44-byte RIFF header followed by signed PCM16:

- 16,000 Hz;
- mono;
- 16 bits per sample;
- matching RIFF size, data size, and HTTP Content-Length;
- at least one PCM sample;
- at most 320,044 bytes total, exactly ten seconds plus the WAV header.

Chunked transfer encoding and duplicate/missing Content-Length are rejected.
The hard ten-second bound protects the fixed-shape QNN context. Longer turns
must be segmented before this endpoint; raising an environment variable cannot
override the hard maximum.

Explicit-language success:

```json
{
  "ok": true,
  "schema_version": 1,
  "requested_language": "zh",
  "candidates": [
    {
      "backend": "sensevoice_qnn",
      "provider": "qnn",
      "ok": true,
      "result": {"lang": "<|zh|>", "text": "..."}
    }
  ]
}
```

The stable transcript location is `candidates[i].result.text`. SenseVoice's raw
`result.lang` is retained when upstream emits it. Other raw Sherpa fields such
as tokens, timestamps, emotion, or event are preserved without normalization.
The CLI does not provide a calibrated confidence value, so this service does
not expose one.

Auto success:

```json
{
  "ok": true,
  "complete": true,
  "schema_version": 1,
  "requested_language": "auto",
  "selection_required": true,
  "candidates": [
    {"backend": "sensevoice_qnn", "provider": "qnn", "ok": true, "result": {"text": "..."}},
    {"backend": "zipformer_ru_cpu", "provider": "cpu", "ok": true, "result": {"text": "..."}}
  ]
}
```

Auto returns HTTP 200 when at least one backend succeeds; `complete` says
whether both did. A failed candidate remains in the array with `ok: false` and
an error object. If both fail, the response is HTTP 503. The caller should pick
Chinese-script SenseVoice output for Mandarin, Cyrillic-script Zipformer output
for Russian, and use SenseVoice's language tag as supporting evidence for
English/Mandarin. This script gate is more honest than comparing uncalibrated
scores from unrelated recognizers.

`qnn-auto` is the low-latency first stage used by the phone UI. It runs only
SenseVoice in automatic language mode and returns the ordinary one-candidate
explicit envelope. The UI accepts that early result only when it contains
clear Han script or positive English lexical evidence. Otherwise it makes a
second explicit `ru` request and applies the conservative two-candidate gate.
This keeps common English/Mandarin Auto turns near QNN latency without silently
turning uncertain Latin phonetics into an English answer. The all-at-once
`auto` endpoint remains available for diagnostics.

Protocol errors use the same JSON envelope:

```json
{"ok": false, "error": {"code": "invalid_wav", "message": "..."}}
```

## Tests

The host test uses fake fixed-output runners; it exercises HTTP framing, exact
Content-Length, CORS, explicit Mandarin/Russian routing, concurrent Auto, WAV
validation, and the ten-second bound without adb or model inference:

```bash
python3 -m unittest -v scripts/test_android_asr_companion.py
bash -n scripts/android_asr_companion/handler.sh
bash -n scripts/android_asr_companion/start.sh
bash -n scripts/android_asr_companion/deploy.sh
```

Physical transcript quality and HTP execution still require deployment to the
target phone; these host tests do not claim either.
