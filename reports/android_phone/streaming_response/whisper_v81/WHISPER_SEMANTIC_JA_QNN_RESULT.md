# Whisper Small Japanese semantic decode on SM8850 QNN/HTP

## Outcome

The isolated Japanese candidate passed its predeclared bounded CER gate twice
on Nubia NX809J / SM8850 with ORT 1.25 QNN EP and CPU EP fallback disabled.

Reference:

```text
フランスの首都は、どこですか？
```

Both runs produced the same candidate:

```text
フランスの主とは、どこですか?
```

This is a deterministic homophone/orthography error (`首都` -> `主と`), not a
run-stability failure. Normalized CER was `2/13 = 0.153846`, below the
predeclared `0.20` gate. Normalized token-like WER was `0.5` and is reported but
is not the primary metric for Japanese.

The result supports Whisper as a Japanese NPU sweep candidate, but the accepted
SenseVoice QNN route remains the production choice because its current Japanese
fixture preserves `首都` correctly.

## Reproducible trust root

Two clean JDK 17, maximum-two-worker builds with
`WHISPER_SEMANTIC_DECODE_ENABLED=true` and
`WHISPER_SEMANTIC_LANGUAGE=ja` were byte-for-byte identical:

- APK bytes: `413135969`;
- APK SHA-256:
  `52972173c76a935c942fdb2dd0fcda7c712f387e914f589a410b881eba82ddd2`;
- semantic/frontend spec SHA-256:
  `bcfcea2c7a718fd6badeb82a0a5e350e52d1b7a3b9a231dda27016710f405858`;
- local APK verifier SHA-256:
  `a1aa4beeba763ff9d00eea8e88c23e3c648ec4c49b6c9bc789332e620b3103fc`.

The exact package, signer, manifest, alignment, ZIP inventory, QNN
model/runtime assets, frontend manifest, corrected Japanese golden, and
AArch64 JNI library passed before installation.

## Device results

| Metric | Run 1 | Run 2 |
|---|---:|---:|
| Run ID | `semantic-ja-20260729T175404Z-root529-r1` | `semantic-ja-20260729T175619Z-root529-r2` |
| Encoder execute | 193.350 ms | 192.114 ms |
| Decoder model execute, 17 calls | 70.489 ms | 67.701 ms |
| Decoder loop wall | 84.402 ms | 81.253 ms |
| Semantic encoder + decoder-loop | 277.752 ms | 273.367 ms |
| Whole compatibility + semantic probe | 990.631 ms | 963.814 ms |
| CER | 15.3846% | 15.3846% |
| EOT reached | yes | yes |

Both runs used language token `50266`, reached EOT after 17 calls, generated
the same 14 IDs including EOT, and reported cumulative first-unmasked indices
199..183. UID 10510 mapped the app-owned ORT/QNN/v81/frontend stack, held an
open CDSP FastRPC FD, and exposed 16 Qualcomm DMA-buffer FDs. Run 2 RSS was
299,692 KiB and high-water RSS was 463,320 KiB.

Both protected APK hashes remained exact. The isolated process was stopped
after evidence capture, and the accepted Qwen plus five-language ASR runtime
health gate passed without build, install, or asset copy.

## Evidence

Run 1 directory: `device_semantic_ja_run_20260730T015404Z_root529_r1/`

- app report SHA-256:
  `e506e5f79349d45ff54893ad15928a38431b2ef34d4c9a0807482d2c4e7a501e`;
- evidence SHA-256:
  `0503de80d01518719889ce4a66119cfe0580c5b46261fecd004b65bbf9da1dc4`;
- semantic verifier SHA-256:
  `c81fce3f8eccf51cffd58346301759cd678f7b095a14b667e1f1708cbf9a508a`.

Run 2 directory: `device_semantic_ja_run_20260730T015619Z_root529_r2/`

- app report SHA-256:
  `bfc7121ca8d59c26c93da92d9070ec57ab3d290ce09b2d7ebeebc40e740dbaf5`;
- evidence SHA-256:
  `1f27ea943ae5bd40806b33ad7ebbdcb83bc0b52f0c4c09b7581f138dd1c15595`;
- semantic verifier SHA-256:
  `005d7df616c35016f6672fa25a6a90fa838c88cc155924d81a8354bcc686d889`.

## Claim boundary

This proves a fixed bundled Japanese semantic candidate and QNN/HTP execution.
It does not enable microphone input, replace the accepted runtime, or establish
general Japanese WER/CER. Additional varied Japanese utterances are required
before any promotion.
