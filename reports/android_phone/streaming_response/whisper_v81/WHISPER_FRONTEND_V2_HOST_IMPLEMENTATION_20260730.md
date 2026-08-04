# Whisper frontend-v2 multilingual host implementation

Date: 2026-07-30

## Outcome

The isolated `ai.liquid.lfmdemo.whisperprobe` lane now has an additive
five-language frontend profile for English, Chinese, Russian, Korean, and
Japanese. The v2 profile bundles every WAV, QAI UINT16 golden, and frontend
table needed by the probe in the APK, so this lane no longer depends on a cable
or host filesystem after installation.

This milestone completes host implementation and whole-APK authentication. It
does **not** claim that a v2 APK has run on the phone, that ZH/KO/RU semantic
decoding has passed on SM8850, or that the new candidates should replace an
accepted route. No ADB command, APK installation, network request, model
conversion, training, fine-tuning, or heavy host inference was performed in
this implementation phase.

Post-implementation independent audit is closed. It found and fixed policy-only
trust-root drift, forged/minimal local-verifier evidence, v2 collector profile
selection, and a v1 `semantic_decode_overlay: null` plus forged semantic-claim
bypass. The APK verifier now requires exact top-level and nested contracts; the
device verifier revalidates the complete local report and trust roots; and the
collector carries an explicit or exact-inferred `v1`/`v2` profile through local
and device verification before installation.

## Compatibility and execution policy

- `WHISPER_FRONTEND_PROFILE` defaults to `v1`.
- v1 remains the accepted EN/JA contract and always computes both fixtures.
- v2 admits exactly `en`, `zh`, `ru`, `ko`, and `ja`.
- A v2 verification build computes all five fixtures.
- A v2 semantic build authenticates all 12 v2 assets but computes only English
  plus the selected semantic language. English remains the input to the
  unchanged four-step ORT-QNN compatibility probe.
- Semantic decode remains default-off and requires the exact build flag
  `WHISPER_SEMANTIC_DECODE_ENABLED=true`.
- Torch reference generation explicitly uses one intra-op and one inter-op
  thread. Android builds use JDK 17 and at most two Gradle workers.

The implementation leaves the existing v1 stager, v1 manifest identity,
EN/JA specs, base ORT-QNN payload, package isolation, and fail-closed CPU
fallback policy unchanged.

## Canonical v2 frontend bundle

The additive stager is
`scripts/stage_whisper_android_frontend_assets_multilang.py`.

- stager bytes: `11,188`
- stager SHA-256:
  `4a89e1ffcbdebd62368d0ed95274da586c758b9e70380ad19323dfe7067f16a7`
- canonical manifest bytes: `5,110`
- canonical manifest SHA-256:
  `acb5c46f3bbbe3c11620bd662242157b30695500c1a0e8b6e7510acdc3157736`
- authenticated assets: `12`
- authenticated asset bytes: `2,850,126`

The 12 assets are five canonical 16 kHz mono PCM16 WAVs, five little-endian
UINT16 `[1,80,3000]` QAI goldens, the periodic Hann table, and the pinned
Slaney mel table. The v2 asset prefix is `whisper_frontend/v2`; v1 assets are
not rewritten.

## Host frontend numerical evidence

The exact Android C frontend was compiled in portable host mode and compared
against the checked-in QAI reference path. Every language passed the existing
bound of maximum UINT16 delta `1`, at most `240` differing values, and differing
fraction no greater than `0.001`.

| Language | QAI golden SHA-256 | Portable native SHA-256 | Differing UINT16 | Fraction | Max delta |
|---|---|---|---:|---:|---:|
| EN | `29c478d40bb98b7782280dcfbc98662cee514df7e62c180c5ba849db24a2f6a1` | `2f959ac4b0c75b4c67087df5e5ff3a530f37c65f54afa4480d7b54ce480b15c1` | 105 | 0.0004375000 | 1 |
| ZH | `6d18da5e1ec42bbd8161881a598e835c365138377b3d42364ff360e4849d4263` | `3449ffb63541de502bbfb3b0e347ae46f304342f8447522c1be06c147941fd55` | 136 | 0.0005666667 | 1 |
| RU | `0e0c25a24157edabe3c1d193e60e8b59886aa9b2ba9b1581fb5c90c2eaf3bbc4` | `ee1090f0991af648392497fea66f8ec28e86abc711c24b935355857ddfd38065` | 40 | 0.0001666667 | 1 |
| KO | `808b38d3c4198c15566116562eb7ad61fbafc01912a747ae439047fb9d7eb1c5` | `acc2bf56a89f45efab2c58b098ff06b3dc04bc44fe11b68736bf08862d64a439` | 93 | 0.0003875000 | 1 |
| JA | `8cf37ba9a60f5cd91453dae2d721e9fd7fec69919582223bf02a3f76f8421a80` | `5e434a8a99650a3248a6c69a807daabc6179e0bee7c52a771346e2b3a385d96e` | 70 | 0.0002916667 | 1 |

## Multilingual semantic safety

The Android vocabulary decoder now concatenates raw token bytes before one
strict UTF-8 decode. It rejects malformed or unmappable UTF-8, U+FFFD, NUL,
surrogates, and disallowed control characters. Dependency-free JDK 17 host
tests pin successful ZH, KO, and RU token vectors.

The semantic verifier accepts the following separately pinned build contracts:

| Language | Language token | Forced prompt | WER gate | CER gate | Compute gate |
|---|---:|---|---:|---:|---:|
| ZH | 50260 | `[50258,50260,50359,50363]` | 1.00 | 0.20 | 2,500 ms |
| KO | 50264 | `[50258,50264,50359,50363]` | 0.34 | 0.20 | 2,500 ms |
| RU | 50263 | `[50258,50263,50359,50363]` | 0.25 | 0.10 | 2,500 ms |

Generated IDs remain restricted to base vocabulary `0..50256` plus EOT
`50257`. Language, control, task, and timestamp IDs `50258..51864` remain
suppressed. The decoder retains cumulative right-aligned UINT16 attention
unmasking, a maximum of 199 decoder calls, and a maximum of 196 generated
tokens.

## Reproducible APK and trust-root results

Each full APK includes the unchanged app-private ORT-QNN model/runtime payload
and all 12 v2 frontend assets. Each configuration was built twice from clean
state with JDK 17 and `--max-workers=2`; the two outputs for each configuration
were byte-for-byte identical. The Gradle build and lint gates passed.

| Build | APK bytes | APK SHA-256 | Spec SHA-256 | Local gate |
|---|---:|---|---|---|
| v2 verify-all | 414,786,946 | `aad2f5959de6d9f9ae77db028e50ea1e51fe545fee2bb2f206fddf018452bcc9` | `1bf369d7efa093d664c3e4f29a75723cd3de77936ae00e98212c9504fc16a291` | pass |
| ZH semantic | 414,786,946 | `b33bf51b988c1def433d42d37fc00d7df1a2de788c5758efab0be68ef0331d3c` | `1a77a5804df42ee58d38d93220e98929696896c9d780249d990fb45be1ac78b4` | pass |
| KO semantic | 414,786,946 | `81b75cdb237d00acffe9696f1b463d841a036f2fd8208917af6b072be47325e7` | `8564002d4e2ddb624410f40b53afd7e9ca0dd70794d2c046fbe7b07f7ad3b54a` | pass |
| RU semantic | 414,786,946 | `e058489e6946e21a32f28aff458330cddc686f43a61d4f939f5cada5ceaebb9e` | `07b2f69f60a2d78dc9f6fc07a02a3249ab90300333b33523b6aa1a643a7fef51` | pass |

The frontend JNI is unchanged across these builds:

- bytes: `23,048`
- SHA-256:
  `475cad57685c04f3027c3c07a5dbfc9b67a33e8f0669fd4f2da36eefd48e9a6d`
- format: 64-bit little-endian AArch64 ELF

All four committed local verification reports pass whole-APK identity,
complete archive inventory, duplicate-member rejection, signer identity,
binary manifest policy, alignment, unchanged base payload identity, frontend
manifest identity, all 12 frontend asset hashes, and JNI identity.

Only the RU semantic APK currently remains at
`android-app/whisper-probe/build/outputs/apk/debug/whisper-probe-debug.apk`.
The verify-all, ZH, and KO binaries are represented by their authenticated
specs and local reports but are not retained as extra 415 MB copies. Before any
future install, the requested configuration must be rebuilt, match its exact
whole-APK byte count and SHA-256, and pass the independent local verifier.

## Final host validation

After the final spec assertions and README edits, the following validation was
rerun:

- Ruff check and format check passed for 15 lane-specific Python files.
- 96 focused unit tests passed: 8 native-audit, 5 v1 frontend, 5 v2 frontend,
  13 APK-verifier, 33 device-verifier, 19 semantic-verifier, and 13 collector
  cases.
- `VocabDecoderHostTest` passed under pinned JDK 17.
- `WhisperDecodePolicyHostTest` passed under pinned JDK 17.
- the Android/ORT-QNN source gate passed.
- all eight v2 spec/local-verification JSON documents parsed successfully.
- `git diff --check` passed.
- the current RU APK was independently re-verified; the new report was
  byte-for-byte identical to the committed RU local-verification report.
- historical v1 frontend plus EN and JA semantic device evidence re-verified
  under the hardened validators.

## Remaining device-evidence phase

The host verifier/collector gate is now frozen. The remaining work is target
evidence rather than implementation promotion:

1. Rebuild and locally authenticate the exact v2 verify-all APK, install only
   that isolated package, and prove all five frontend fixtures meet the bound
   on NX809J/SM8850.
2. Rebuild and locally authenticate each ZH, KO, and RU semantic APK.
3. Run each semantic APK twice in isolated runs, requiring identical APK/spec,
   normalized transcript, and generated-token sequence across repetitions.
4. Require CPU fallback disabled, app-UID ORT/QNN/HTP mappings, an open CDSP
   FastRPC descriptor, Qualcomm DMA buffers, exact resident payload hashes,
   and accepted-package hashes unchanged before and after every run.
5. Record frontend, encoder, per-token decoder, semantic-compute, and wall
   timing. Do not promote multilingual quality or latency until the declared
   WER/CER and compute gates pass on the phone.
6. Force-stop the probe at the end of each lane, preserve archived host
   evidence, and leave the accepted application/runtime packages unchanged.

Until those runs exist, v2 is an authenticated host-ready candidate and not an
accepted on-device multilingual ASR route.

The exact sequential rebuild, local verification, collector, semantic
verification, force-stop, identity, and repetition commands are recorded in
`WHISPER_FRONTEND_V2_DEVICE_RUNBOOK_20260730.md`. That runbook was prepared but
no ADB command from it was executed in this milestone.
