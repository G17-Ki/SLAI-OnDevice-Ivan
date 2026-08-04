# Whisper frontend-v2 multilingual device result

Date: 2026-07-30

## Outcome

The isolated `ai.liquid.lfmdemo.whisperprobe` lane completed its physical
five-language frontend gate and the predeclared ZH/KO/RU semantic sweep on
Nubia NX809J / SM8850 / HTP-v81.

| Lane | Physical result | Decision |
|---|---|---|
| Five-language frontend | Pass | The bundled EN/ZH/RU/KO/JA native frontend is verified on device. |
| ZH semantic | Reject after r1 | Traditional output exceeded the frozen Simplified-Chinese CER gate; no r2 was run. |
| KO semantic | Pass r1 and r2 | Exact transcript, tokens, EOT, WER, and CER repeated; retain as an isolated NPU candidate. |
| RU semantic | Reject after r1 | WER exceeded the frozen gate; no r2 was run. |

These are bundled-fixture candidate results, not microphone, acoustic,
automatic-language-selection, production-integration, or end-to-end speech
agent acceptance. The accepted SenseVoice QNN routes remain production for
EN/ZH/JA/KO, and the accepted Zipformer CPU route remains production for RU.

## Frozen binaries and host gates

All four configurations were clean-built with JDK 17 and
`--max-workers=2`, reproduced their predeclared whole-APK identities, passed
lint, and produced local verifier reports byte-identical to the committed
reports before installation.

| Lane | APK bytes | APK SHA-256 |
|---|---:|---|
| v2 verify-all | 414,786,946 | `aad2f5959de6d9f9ae77db028e50ea1e51fe545fee2bb2f206fddf018452bcc9` |
| ZH semantic | 414,786,946 | `b33bf51b988c1def433d42d37fc00d7df1a2de788c5758efab0be68ef0331d3c` |
| KO semantic | 414,786,946 | `81b75cdb237d00acffe9696f1b463d841a036f2fd8208917af6b072be47325e7` |
| RU semantic | 414,786,946 | `e058489e6946e21a32f28aff458330cddc686f43a61d4f939f5cada5ceaebb9e` |

The hardened host gate passed 96 focused tests plus the pinned JDK vocabulary,
semantic-decode-policy, and Android/ORT-QNN source gates. The collector used
the explicit v2 profile and rejected profile/spec drift before ADB install.

## Five-language frontend physical gate

Run `frontend-v2-20260729T194745Z-rootaad-r1` passed with CPU EP fallback
disabled. App-UID evidence contained ORT, QNN HTP/v81, FastRPC CDSP, and 16
Qualcomm DMA buffers. All five tensors stayed within maximum UINT16 delta 1.

| Language | Differing values / 240,000 | Native frontend ms |
|---|---:|---:|
| EN | 104 | 32.874 |
| ZH | 135 | 32.115 |
| RU | 39 | 31.538 |
| KO | 92 | 31.477 |
| JA | 71 | 31.473 |

Aggregate frontend compute was 159.477 ms. Compatibility execution also
passed: encoder execution was 197.240 ms, four decoder executions totaled
18.966 ms, and total probe time was 645.610 ms.

Evidence directory:
`device_frontend_v2_run_20260729T194745Z_rootaad_r1/`.

- device report SHA-256:
  `1ff8a5285a9f1ad8dafd45e3913ec3f916aa8a223376b555efd31353af460631`
- collector evidence SHA-256:
  `01a2b3c8e4533086d69de94d05c946c160691f5266a816f8e4c7c8c8853c73b9`
- independent device verification SHA-256:
  `263d8d62d6e27c230ad386adb95b084b9e755cce2642ce3ee69d9507193af149`

## Chinese semantic r1: rejected without threshold changes

Run `semantic-zh-v2-20260730-b33bf5-r1` reached EOT after 14 decoder calls.
Semantic compute was 260.938 ms: 193.035 ms encoder plus 67.903 ms decoder
loop. The exact candidate was:

`請用中文簡單介紹一下北京`

against the frozen Simplified-Chinese reference:

`请用中文简单介绍一下北京。`

Normalized CER was `4/12 = 0.333333`, above the predeclared `0.20` limit;
normalized WER was `1.0`. The differences are Traditional forms
`請/簡/單/紹`. The gate deliberately does not normalize Traditional Chinese
into the Simplified trust root after observing the result. The semantic
verifier therefore failed closed, r2 was not run, and the candidate is not
promotable.

Evidence directory:
`device_semantic_zh_v2_run_20260730_b33bf5_r1/`.

- device report SHA-256:
  `18c1e606aaa2bfa6c85f3af9d6141f061a0d7cb46562a964d85d798bfdb5fc72`
- collector evidence SHA-256:
  `1a72047ebf5e52b7ed4dadeecfdf6df516f19e76ead5d8887ecc307affb7a8bc`
- collector device verification SHA-256:
  `599c72427373d9c0fc13ab6ec1634f36906852e88997b9887e886a44d5c84158`

## Korean semantic r1/r2: accepted isolated candidate

Both repetitions reached EOT after 14 decoder calls and produced the exact
same transcript and generated-token sequence:

`프랑스의 수도는 어디인가요?`

Generated IDs in both runs were
`[32051, 9143, 7785, 2785, 23455, 1098, 20879, 4215, 28517, 30, 50257]`.
Normalized WER and CER were both 0.0. Semantic compute was 261.311 ms in r1
and 261.651 ms in r2. Both semantic verifier reports passed, with CPU fallback
disabled and matching QNN/HTP/FastRPC/DMA and protected-package evidence.

Evidence directories:

- `device_semantic_ko_v2_run_20260730_81b75c_r1/`
  - device report SHA-256:
    `22134a79bbe691fa038886e6bed41a610df126d96c236973227233b211bd7005`
  - semantic verification SHA-256:
    `44206c5e6626367c0af151b71d22e04645d7f6edbe9579405171028f8491c8f7`
- `device_semantic_ko_v2_run_20260730_81b75c_r2/`
  - device report SHA-256:
    `ce48000bae3814ec4f30994a503741a53eccfa07bf574ac38fac001aa197eca7`
  - semantic verification SHA-256:
    `737a103be06072406b7e8eaf7f2c188e431bc763094f34a12430722fcc37319c`

This promotes only isolated candidate feasibility. SenseVoice QNN remains the
production Korean route because it is already integrated, exact on the same
fixture, and does not require replacing the accepted runtime.

## Russian semantic r1: rejected without threshold changes

Run `semantic-ru-v2-20260730-e05848-r1` reached EOT after 14 decoder calls.
Semantic compute was 261.660 ms: 193.719 ms encoder plus 67.941 ms decoder
loop. The exact candidate was:

`Расскажи короткая Москве.`

against reference `Расскажи коротко о Москве.`. Normalized WER was `2/4 =
0.50`, above the frozen `0.25` limit. CER was `2/22 = 0.090909`, within its
separate limit, but both gates are required. The semantic verifier failed
closed, r2 was not run, and the candidate is not promotable.

Evidence directory:
`device_semantic_ru_v2_run_20260730_e05848_r1/`.

- device report SHA-256:
  `59c1a3d4f35094043d28fd413e76be866f916bfb8d02ca42aab581516b79acf3`
- collector evidence SHA-256:
  `3c99e872fffac2bc9de609b5bae9cc2e9b8812f3c6b5827b6f7bdef4b01322f0`
- collector device verification SHA-256:
  `134b1a19651db89d6cfaa5b884c97bc3ceed4746ec82adeaf4e73f176f60a681`

## Final accepted-runtime and phone state

After the sweep, the no-build/no-install/no-copy accepted-runtime gate passed:

- exact Qwen2.5-1.5B-Instruct-Q4_K_M identity;
- EN/ZH/JA/KO SenseVoice QNN/HTP transcripts;
- RU Zipformer CPU transcript;
- companion service UID 10514 with complete app-owned artifacts.

The isolated Whisper process was force-stopped. The final read-only audit found
airplane mode `1`, an empty IP route, no ADB forwards, main/runtime PIDs alive,
and exact protected APK hashes:

- main:
  `a24e203a30dcaaf9215dc3f0d5b99fa4b893a9e0ceca71c99c13e49bdb1b020a`
- runtime:
  `33bf0d6d284d6eaad96308f22a8d10c3879ba6d0b792f893486be03f0a17e244`

No model training, fine-tuning, conversion, or heavy host inference was used.
