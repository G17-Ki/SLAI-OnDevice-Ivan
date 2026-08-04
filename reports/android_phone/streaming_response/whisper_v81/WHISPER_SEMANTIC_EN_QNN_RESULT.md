# Whisper Small English semantic decode on SM8850 QNN/HTP

## Outcome

The isolated package `ai.liquid.lfmdemo.whisperprobe` produced the exact
normalized English reference twice from the APK-bundled WAV while running the
Whisper Small W8A16 encoder and decoder through ORT 1.25 QNN EP with CPU EP
fallback disabled.

This closes the bounded English semantic-candidate gate:

- normalized WER: `0.0` on both runs;
- normalized CER: `0.0` on both runs;
- transcript: `What is the capital of France? Answer in one short sentence.`;
- EOT reached after 17 decoder calls and 14 generated IDs including EOT;
- cumulative attention progression: first unmasked indices `199..183`;
- semantic model compute: `274.805 ms` and `275.007 ms`;
- app report still sets `semantic_transcript_claimed=false` and
  `real_audio_transcription=false`.

The result is evidence for one fixed English bundled fixture in a coexistable
probe. It is not microphone acceptance, production runtime integration, or a
multilingual Whisper quality claim.

## Reproducible APK and trust root

Two independent clean builds used JDK 17, `--max-workers=2`, and the exact
build flag `WHISPER_SEMANTIC_DECODE_ENABLED=true`. The resulting APKs were
byte-for-byte identical:

- APK bytes: `413135137`;
- APK SHA-256:
  `9dc3862abe755d986e9d233c3627dccaa7f3caa268598318cf5ae5c96e8669a2`;
- package: `ai.liquid.lfmdemo.whisperprobe`;
- signer certificate SHA-256:
  `9ad7b2846b7c18853b99e444d4d05323321b3a9118c13c5a8fc242e58b5859e4`.

The separate semantic/frontend trust root is
`nubia_whisper_android_frontend_semantic_en_probe_spec.json`, 6,691 bytes,
SHA-256
`1efbfeaa2c05627c7dc4ca86fe7e86fe8af0b41da95ae3f143ebd893cc4cd430`.
The independent local APK report is 8,255 bytes with SHA-256
`a39036565bc39c9ade6476c74be57bd2ad718c578ff6ba7fe57e1961f7c2d675`.
It verifies the whole APK before ZIP inspection, the complete member sets,
signer, binary manifest, Android alignment, exact model/runtime assets, and the
pinned AArch64 frontend JNI library.

## Decoder contract

The semantic route is separate from the unchanged four-step compatibility
route and uses:

- English prompt `[50258, 50259, 50359, 50363]`;
- decoder positions `0..198`;
- at most 196 generated IDs;
- cumulative right-aligned UINT16 attention suffixes;
- allowed generated IDs `0..50256` plus EOT `50257`;
- suppressed control/language/task/timestamp IDs `50258..51864`;
- deterministic lower-ID selection on equal allowed logits.

For both device runs, generated IDs were exactly:

```text
[708, 307, 264, 4238, 295, 6190, 30, 24545, 294, 472, 2099, 8174, 13, 50257]
```

The independent semantic verifier checks the call/token invariant, input-token
feedback, cumulative-mask trace, EOT placement, token domain, timing sums,
UTF-8 transcript integrity, and predeclared WER/CER/latency bounds. Twelve
deterministic verifier tests pass, alongside the pure-Java JDK 17 policy test,
native frontend tests, stager tests, and the existing APK/device/collector
regressions.

## Physical run results

| Metric | Run 1 | Run 2 |
|---|---:|---:|
| Run ID | `semantic-en-20260729T173721Z-root9dc-r1` | `semantic-en-20260729T173925Z-root9dc-r2` |
| Encoder execute | 193.675 ms | 194.201 ms |
| Decoder model execute, 17 calls | 67.697 ms | 67.342 ms |
| Decoder loop wall | 81.130 ms | 80.806 ms |
| Semantic encoder + decoder-loop | 274.805 ms | 275.007 ms |
| Whole compatibility + semantic probe | 994.459 ms | 990.159 ms |
| Normalized WER | 0.0 | 0.0 |
| Normalized CER | 0.0 | 0.0 |
| EOT reached | yes | yes |

The target was Nubia NX809J / SM8850 / arm64-v8a in airplane mode. Probe UID
10510 mapped app-owned ORT, QNN HTP, QNN System, v81, and frontend libraries,
plus vendor `libcdsprpc.so`. `/dev/fastrpc-cdsp` was open and 16 Qualcomm DMA
buffer FDs were present. The resident app-private payload was 352,027 KiB. On
run 2 the process reported 295,724 KiB RSS and 460,684 KiB high-water RSS.

Both protected APKs remained exact before and after each run:

- main:
  `a24e203a30dcaaf9215dc3f0d5b99fa4b893a9e0ceca71c99c13e49bdb1b020a`;
- runtime:
  `33bf0d6d284d6eaad96308f22a8d10c3879ba6d0b792f893486be03f0a17e244`.

After evidence capture, the probe process was force-stopped. The accepted
runtime then passed its no-build/no-install/no-copy health gate: exact
Qwen2.5-1.5B identity, EN/ZH/JA/KO SenseVoice QNN transcripts, and RU Zipformer
CPU transcript. Final state retained airplane mode `1`, no ADB forwards, and no
running Whisper probe process.

## Evidence

Run 1 directory:
`device_semantic_en_run_20260730T013721Z_root9dc_r1/`

- app report SHA-256:
  `d13c791712677d24790e41488df936479e5cb51c642a0dae65e007763bd7c15e`;
- evidence envelope SHA-256:
  `b4c0660d2fe6d3f7c84e6e4f27515a8d176f84708c03bd5985eaea5f9c2ff8e8`;
- semantic verifier SHA-256:
  `43688940a914fbf7ce6809b214fd2dd517ddfd63756f98eccbafcbfa517fcd51`.

Run 2 directory:
`device_semantic_en_run_20260730T013925Z_root9dc_r2/`

- app report SHA-256:
  `732ad29ab105537d1d8cf81c535311f2127847cdfe1b25749d6d84531f8ec8c0`;
- evidence envelope SHA-256:
  `4661942dc2ebcbc645cadc334f61b6ea52d6f6ed493b301d80de77d51abd342f`;
- semantic verifier SHA-256:
  `4c08a0f3d566caf2fb0211f7bbb490a3d2829649d34dbd05d88a0a735accd140`.

## Next bounded ASR step

Generalize the semantic build to select a separately pinned language/fixture,
then add Japanese, Mandarin, Korean, and Russian one at a time with predeclared
CER/WER and latency gates. The accepted SenseVoice/Zipformer production route
remains active until each Whisper language passes its own isolated evidence
gate and a candidate runtime survives unplugged/reboot acceptance.
