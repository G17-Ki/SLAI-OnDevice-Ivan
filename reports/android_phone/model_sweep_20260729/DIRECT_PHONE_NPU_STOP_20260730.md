# Direct-phone NPU final stop — 2026-07-30

Status: **stopped at the direct-phone boundary**.

At 2026-07-30 09:23 CST, the remaining candidates were audited against the
artifacts already present on the Nubia NX809J/SM8850 phone and in the local
workspace/cache. There is no safe full-stack NPU candidate that can be launched
without first producing or acquiring a compiled artifact. Per the narrowed
instruction, no Colab/Hub acquisition, conversion, download, APK build, install,
or further inference was attempted.

## Decision matrix

| Component | Direct phone result | Decision |
|---|---|---|
| SenseVoice ASR, EN/ZH/JA/KO | Already proven under the production app UID on QNN/HTP | Keep in production |
| Whisper v2 ASR | Physically executed on HTP with CPU fallback disabled; KO passed twice, while ZH and RU failed the frozen quality gates | Do not replace SenseVoice |
| Qwen2.5-1.5B backbone | Phone contains only the exact GGUF and a llama.cpp CPU runtime; no Genie runner/library/config or generated Qwen container/context exists | NPU **no-go** |
| Supertonic 3 TTS, raw ONNX through QAIRT 2.42 | Duration predictor and text encoder completed; vector estimator failed at `session_create` with CPU-assigned nodes while CPU fallback was disabled; zero vector steps ran and vocoder was not reached | Full-chain NPU **no-go** |
| Supertonic 3 TTS, QAIRT 2.45 contexts | Four successful Hub objects are authenticated, but none is downloaded and no matching 2.45 Android runtime/candidate APK is local or installed | Not runnable under current scope |

## Qwen direct-phone boundary

The app-private runtime inventory contains:

```text
files/text/models/qwen2.5-1.5b-instruct-q4_k_m.gguf
files/text/qwen-server.log
```

The exact GGUF is 1,117,320,736 bytes with SHA-256
`6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e`.
The installed runtime APK contains llama.cpp libraries and `libqwenserver.so`;
its launch contract pins `--n-gpu-layers 0`. Neither the phone nor the searched
workspace, local cache, and system temporary directory contains the
required generated Qwen deliverables:

- `genie-t2t-run` and `libGenie.so` in a phone candidate;
- `gen_ai_config.json`, `backend_extensions.json`, or `models/use_cases.json`;
- Qwen `models/split_*/model.bin`, a Qwen QNN context, or a saved
  `LLMContainer`;
- `container.SHA256SUMS` proving a completed deliverable.

The cached QAIRT 2.42 archive supplies build/runtime inputs, but it is not a
compiled Qwen model. A raw GGUF cannot be submitted directly to HTP. The
prepared high-RAM Colab conversion remains `not_started_auth_blocked`, so there
is no honest phone command to run.

## Supertonic direct-phone boundary

The isolated QAIRT 2.42/ORT-QNN physical diagnostic completed this ordered
prefix with CPU fallback disabled:

1. payload install;
2. runtime initialization;
3. duration predictor;
4. text encoder.

The vector-estimator session was not created because ORT assigned nodes to the
default CPU EP. Completed vector steps were `0`, and the vocoder was never
reached. The phone inventories for the duration, chain, and diagnostic probes
contain only raw `.onnx` models, fixtures, support files, and reports; they do
not contain replacement QNN contexts.

QAIRT 2.45 Hub metadata identifies four SM8850/HTP-v81 context binaries, but
their SHA-256 identities are unavailable until authenticated download. No
matching QAIRT 2.45 runtime, isolated JNI package, or
`ai.liquid.lfmdemo.ttsprobe.qairt245` installation exists. Mixing those contexts
with a 2.42 process is prohibited and was not attempted.

## Final phone state

All commands in this final audit were read-only. The accepted phone APKs still
match their protected identities:

- `ai.liquid.lfmdemo`:
  `a24e203a30dcaaf9215dc3f0d5b99fa4b893a9e0ceca71c99c13e49bdb1b020a`
- `ai.liquid.lfmdemo.runtime`:
  `33bf0d6d284d6eaad96308f22a8d10c3879ba6d0b792f893486be03f0a17e244`

Airplane mode is `1`; `adb forward --list` and `ip route` are empty. The
Whisper and three Supertonic probe processes are stopped. No accepted package,
model, or phone file was changed by this final audit.

## Stop boundary

The production phone-resident demo remains usable offline with SenseVoice on
HTP for EN/ZH/JA/KO, Zipformer CPU for RU, Qwen2.5-1.5B CPU, and the accepted
CPU TTS routes. Moving the remaining backbone or complete TTS chain onto HTP
requires a new compiled, version-matched artifact. That work is outside the
current direct-phone-only instruction, so execution stops here.

Related evidence:

- `QAIRT242_QWEN25_GENIE_BOUNDARY.md`
- `SUPERTONIC_FULL_CHAIN_STAGE_DIAGNOSTIC_V2.md`
- `QAIRT245_SUPERTONIC_ISOLATED_LANE.md`
- `../streaming_response/whisper_v81/WHISPER_MULTILANG_V2_DEVICE_RESULT_20260730.md`
- `../full_turns/POST_SWEEP_FULL_TURN_ACCEPTANCE_20260730.md`
