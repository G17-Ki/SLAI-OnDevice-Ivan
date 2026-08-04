# Supertonic 3 full-chain raw-ONNX QNN scaffold

Last updated: 2026-07-30 01:55 (Asia/Shanghai)

## Outcome

The host-only four-component Supertonic 3 candidate is complete and ready for
an isolated phone gate. It packages the duration predictor, text encoder,
four-step vector estimator, and vocoder together with ORT 1.25 and the exact
QAIRT `2.42.0.251225135753_193295` QNN runtime. CPU EP fallback is disabled.

No ADB command, install, launch, or other device mutation was performed in this
lane. The accepted duration probe package and its evidence were not modified.

## Immutable package identity

- Package: `ai.liquid.lfmdemo.supertonicchainprobe`
- Gate: `supertonic3_full_chain_raw_onnx_ort_qnn_sm8850_probe_v1`
- Version: `0.1.0-full-chain-raw-onnx-qnn-probe`
- APK bytes: `197687829`
- APK SHA-256:
  `4929a85fbb72da8ee83152e6b0258f223e5d9c4692724be7c14c0eb42538c981`
- Signing certificate SHA-256:
  `9ad7b2846b7c18853b99e444d4d05323321b3a9118c13c5a8fc242e58b5859e4`

Two clean JDK 17 builds with `--max-workers=2`, including Android lint, were
byte-for-byte identical. The APK has no network, microphone, or storage
permission. Its binary manifest keeps `allowBackup=false`,
`usesCleartextTraffic=false`, and optional `libcdsprpc.so` access only.

## Pinned inputs

| Component | Bytes | SHA-256 |
| --- | ---: | --- |
| Duration predictor | 3,700,147 | `c3eb91414d5ff8a7a239b7fe9e34e7e2bf8a8140d8375ffb14718b1c639325db` |
| Text encoder | 36,416,150 | `c7befd5ea8c3119769e8a6c1486c4edc6a3bc8365c67621c881bbb774b9902ff` |
| Vector estimator | 78,400,833 | `20cd86fa5c6effedfda0e7cffe5b0569ca401c440a0c3a1d72bf39286c0db3fd` |
| Vocoder | 25,991,073 | `e923d60f53f95eb1ce235f1dc33ec56d9c057823c96fa6f8acf98f32b0da6152` |

The deterministic fixture uses one thread, NumPy 1.26.4, ORT 1.28.0, seed 42,
fixed text length 32, fixed latent length 32, 31 real latent positions, four
vector steps, and a 44.1 kHz / 93,826-sample host reference.

- Fixture NPZ SHA-256:
  `1a9f30f1e31c85777e335d9c68cde1ff35344fcf8153292bebd23121b56868e4`
- Fixture manifest SHA-256:
  `277bc5b31320a06bbcff587df33a2338c8d080520ee0356d476cc9fc50649163`
- Embedded payload manifest SHA-256:
  `735e61b070c7ad199a67bc18829036f4575551b5533a49e9ead5b36392f0819f`
- Stager SHA-256:
  `831b442d6a62a02a58b6c8ba352441937eee73bed904503840ea5692f94ca5c5`

## Evidence boundary

The app chains actual stage outputs and fails closed unless all four QNN
sessions are created, all tensors are finite and exact-shape, every masked
latent padding position is zero, all four vector steps execute, and the
vocoder returns a non-silent 98,304-sample PCM tensor.

Host-reference comparisons are recorded only as diagnostics. During review, a
duration-stage contradiction was found: its tolerance had been promoted even
though the payload policy marked every reference comparison diagnostic-only.
That gate was removed structurally; the exact 31-of-32 latent bucket remains a
real shape-safety gate. No phone-calibrated tolerance, audio quality, language
identity, playback, promoted latency, unplugged durability, full-TTS, or
production integration claim is made.

## Independent verification

- Authoritative spec: `supertonic_chain_ort_qnn_probe_spec.json`
  - bytes: `25413`
  - SHA-256:
    `239a83e940e22f0289af5059900f5b815ea2b865da4abac3c698558fd0494dc5`
- Local verification report: `supertonic_chain_ort_qnn_apk_local_verify.json`
  - bytes: `10749`
  - SHA-256:
    `ff4c899298a66b4d8cb82b07a329f24c4a42134d78e360adeab27bce8d32d0b9`

The host verifier checks the whole APK before opening it, rejects duplicate or
extra ZIP members, pins every asset and native library, requires all assets to
be stored, and verifies the binary manifest, exact signer, and four-byte ZIP
alignment. It also pins the unchanged accepted packages:

- `ai.liquid.lfmdemo`:
  `a24e203a30dcaaf9215dc3f0d5b99fa4b893a9e0ceca71c99c13e49bdb1b020a`
- `ai.liquid.lfmdemo.runtime`:
  `33bf0d6d284d6eaad96308f22a8d10c3879ba6d0b792f893486be03f0a17e244`

The future read-only collector archives a matching failed-closed app report
before raising, publishes no current alias, cross-checks Android 16 UID sources,
requires the app-private partial report to be absent, hashes all four resident
models, and binds raw `/proc/status`, `maps`, `fd`, and `smaps` evidence. The
independent device verifier re-parses those artifacts and requires app-UID
QNN/ORT mappings, vendor FastRPC, `/dev/fastrpc-cdsp`, QCOM DMA buffers, exact
run/timestamp/publication correlation, unchanged accepted packages, and the
diagnostic-only claim boundary.

## Host validation

- 21 focused gates passed: one source gate plus five tests each for the stager,
  APK verifier, device verifier, and collector.
- Python bytecode compilation passed.
- Ruff checks passed; all newly added Python files are Ruff-formatted.
- Android `assembleDebug` and `lintDebug` passed twice under JDK 17 with two
  Gradle workers.
- Local whole-APK verification passed after the second clean build.

## Next gate

Install only this isolated package, launch it with a fresh strict `run_id`, and
run the read-only collector. Promotion requires four real QNN sessions, four
vector calls, finite/non-silent PCM, app-owned QNN/ORT mappings, FastRPC/DMA
evidence, a complete app-private report with no remaining partial, a resident
four-model inventory, airplane mode, and both accepted package hashes unchanged.
