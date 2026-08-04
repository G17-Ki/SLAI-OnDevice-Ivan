# Overnight cap ledger

Both overnight demos now have complete accepted phone inventories and normalized untrimmed handoff films. Demo 2 is physically below 3B; its filmed cold runner receipts passed EN at 1,988.922 ms and ZH at 1,740.289 ms under the request-to-in-app-audio-start proxy gate. Demo 1 is physically below 2B and completed EN/ZH/ES/FR/RU in one uninterrupted cold offline engineering process. Its historical RU answer remains coverage-only, while the separate final film ends with the factually correct answer “Москва.”

| Target | Unique learned basis | Conservative resident basis | Exclusive cap | Conservative headroom | Arithmetic verdict | Installed verdict |
|---|---:|---:|---:|---:|---|---|
| **Demo 2 — installed EN+ZH inventory** | **1,974,795,009** | **2,208,452,225** | 3,000,000,000 | **791,547,775** | **PASS** | **PHONE ACCEPTED** |
| Demo 2 — intentionally overcounting host target | 1,988,326,684 | **2,221,983,906** | 3,000,000,000 | 778,016,094 | PASS | Safe bound; counts absent Vocos 24 kHz |
| Demo 2 — remove LFM only, retain full TTS laboratory | not promoted | **2,672,875,114** | 3,000,000,000 | 327,124,886 | PASS, conditional on LFM absence | Analytical fallback; not the accepted installed scope |
| **Demo 1 — installed five-language inventory** | **1,810,531,015** | **1,820,982,302** | 2,000,000,000 | **179,017,698** | **PASS** | **PHONE ACCEPTED** |
| Demo 1 — intentionally overcounting host target | 1,824,062,690 | **1,834,513,983** | 2,000,000,000 | 165,486,017 | PASS | Safe bound; counts absent Vocos 24 kHz |

The complete machine-readable ledger, per-file hashes, parse receipts, and arithmetic assertions are in [`cap_ledger.json`](cap_ledger.json).

## What the two bases mean

`learned_unique` is the requested cap basis: unique learned logical tensor elements. Proven tied weights are de-duplicated; fixed integer buffers, quantization scales, graph constants, and non-neural rule tables do not count.

`conservative_resident` is intentionally harsher. It counts the running GGUF server's serialized tensor-element metadata and every ONNX initializer or Constant element in resident exports, including quantization auxiliaries, fixed buffers, explicitly retained unused variants, and style/voice banks. It can over-count learned parameters. Therefore:

- a conservative total below the cap is sufficient evidence that the learned total is also below it;
- the conservative number must never be relabelled as an exact learned-parameter total;
- the arithmetic becomes a phone claim only after a full post-install inventory proves that excluded files are absent.

Licences are recorded in the lane evidence but do not gate this research demo. None of these counts proves CPU, GPU, or NPU placement.

## Demo 2 — “IT'S GOOD”

### Accepted physical inventory

The final 42-row receipt covers two APKs, fifteen private files, and twenty-five external TTS files. It proves one Qwen GGUF only, no app-private LFM weights, no stale TTS tree, no Silero weights, and no LFM process or port 8080. The installed TTS tree omits the staged but unreferenced Vocos 24 kHz file.

| Component | Learned unique | Conservative resident |
|---|---:|---:|
| Qwen2.5 | 1,543,714,304 | 1,777,088,000 |
| SenseVoiceSmall | 233,999,167 | 233,999,167 |
| RU Zipformer, resident but unused by Demo 2 | 65,610,041 | 65,610,041 |
| Installed Supertonic EN + Matcha/Vocos ZH | 131,471,497 | 131,755,017 |
| `AdaptiveEnergyVad` | 0 | 0 |
| **Total** | **1,974,795,009** | **2,208,452,225** |
| **Headroom below 3B** | **1,025,204,991** | **791,547,775** |

Primary receipt: `evidence/demo2/final_pruned/inventory_receipt.json`, SHA-256 `73f5e9d6e5fb8e4becb529de19f0dceee8519284a1b90586a446d5ab363f302b`.

### Intentionally overcounting host target

| Component | Role/scope | Unique learned | Conservative resident | Evidence | Current state |
|---|---|---:|---:|---|---|
| Qwen2.5-1.5B-Instruct Q4_K_M | EN+ZH backbone | 1,543,714,304 | 1,777,088,000 | `EXACT` logical count; `PHONE-MEASURED` running GGUF metadata | Executed in committed warm smoke |
| SenseVoiceSmall | EN+ZH ASR | 233,999,167 | 233,999,167 | `EXACT` | Executed in committed EN/ZH evidence |
| RU Zipformer INT8 | resident but out of Demo 2 language scope | 65,610,041 | 65,610,041 | `EXACT` | Counted conservatively; not exercised by the EN/ZH run |
| Supertonic EN + Matcha/Vocos ZH host-bound tree | EN+ZH TTS | 145,003,172 | 145,286,698 | `EXACT` logical rows; direct-export `UB` | Safe overcount; physical tree omits Vocos 24 kHz |
| `AdaptiveEnergyVad` | dependency-free Java VAD | 0 | 0 | `EXACT` source identity | Deployed app code; no Silero model artifact exists |
| **Total** |  | **1,988,326,684** | **2,221,983,906** |  | **PASS under 3B** |

Arithmetic:

```text
learned:
1,543,714,304 + 233,999,167 + 65,610,041 + 145,003,172 + 0
= 1,988,326,684

conservative:
1,777,088,000 + 233,999,167 + 65,610,041 + 145,286,698 + 0
= 2,221,983,906
```

This host bound retains an unused `vocos-24khz.onnx`; the complete phone inventory proves that file is absent, so the accepted physical total above is smaller.

### Minimal remove-LFM-only scope

This is the fallback that changes the fewest phone files. It retains the entire nineteen-candidate TTS laboratory inventory and proves the cap using a deliberately conservative bound.

| Component | Conservative resident elements | Class |
|---|---:|---|
| Qwen2.5 GGUF metadata | 1,777,088,000 | `PHONE-MEASURED` |
| SenseVoiceSmall | 233,999,167 | `EXACT` |
| RU Zipformer INT8 | 65,610,041 | `EXACT` |
| Full TTS laboratory | 596,177,906 | `UB` |
| `AdaptiveEnergyVad` | 0 | `EXACT` |
| **Total** | **2,672,875,114** | **PASS under 3B** |

The full TTS bound is:

```text
582,600,946 ONNX initializer/Constant elements
+ 13,447,680 one unique Kokoro voice bank
+    129,280 Supertonic style bank
= 596,177,906
```

This scope intentionally does not publish a learned-unique TTS total because it retains duplicated FP32/INT8 variants and quantization auxiliaries. The exported-element upper bound is enough to establish the cap.

### LFM must actually be absent

The four LFM artifacts contain 1,644,992,171 learned elements and 1,644,992,196 total tensor elements. Adding them back yields:

- pruned Demo 2 plus LFM: **3,866,976,102** conservative elements;
- remove-LFM-only scope plus LFM: **4,317,867,310** conservative elements.

Both fail the under-3B target. The older warm smoke proves LFM was present then; the final 42-row receipt supersedes that deployment state and proves the four weights, process, and port 8080 are absent.

The exact removal targets are:

- `files/models/LFM2.5-Audio-1.5B-Q4_0.gguf`
- `files/models/mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf`
- `files/models/tokenizer-LFM2.5-Audio-1.5B-Q4_0.gguf`
- `files/models/vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf`

The old phone evidence used source that started `LfmService` and required port 8080. The accepted final main APK is `712632b7987557f02bd3bf67b77c00b49025207178f1719094bbb6a81cb2f53f`; the physical receipt proves no LFM model, process, or listener remains.

## Demo 1 — “IT FITS”

### Accepted physical inventory

The final receipt covers **61 rows**: two APKs, thirty-four private files, and twenty-five external TTS files. It proves exactly one Qwen3.5 GGUF, Qwen2.5 absent, all four LFM weights absent, no extra GGUF, no Silero weights, no unused Vocos 24 kHz export, no stale TTS candidate tree, no partial incoming file, no Parakeet QNN context/runtime payload, no LFM process, and no enumerated port-8080 listener. The unprivileged listener probe could not expose owning PIDs, so the supported network-state claim is the narrow port-8080 result—not “no sockets.”

| Component | Languages | Unique learned | Conservative resident | Evidence/current state |
|---|---|---:|---:|---|
| Qwen3.5-0.8B Q4_0, exact Lane B text-only artifact | all five | 752,393,024 | 752,393,024 | Exact private hash; Qwen2.5 removed; all five turns accepted |
| SenseVoiceSmall | EN, ZH | 233,999,167 | 233,999,167 | Hash-verified; QNN/HTP executed in accepted EN/ZH turns |
| Parakeet-TDT-0.6B-v3 INT8 CPU export | ES, FR | 627,057,286 | 637,225,053 | Hash-verified CPU-only payload; ES/FR turns accepted |
| RU Zipformer INT8 | RU | 65,610,041 | 65,610,041 | Hash-verified CPU route; execution accepted, retained answer quality failed |
| Installed Matcha ZH + shared Supertonic EN/ES/FR/RU | all five | 131,471,497 | 131,755,017 | Five candidates; 22 referenced + 3 metadata files verified; Vocos 24 kHz absent |
| `AdaptiveEnergyVad` Java code | all five | 0 | 0 | `EXACT`; deployed source, no model artifact |
| **Total** |  | **1,810,531,015** | **1,820,982,302** | **PHONE ACCEPTED under 2B** |
| **Headroom below 2B** |  | **189,468,985** | **179,017,698** |  |

Arithmetic:

```text
learned:
752,393,024 + 233,999,167 + 627,057,286 + 65,610,041 + 131,471,497 + 0
= 1,810,531,015

conservative:
752,393,024 + 233,999,167 + 637,225,053 + 65,610,041 + 131,755,017 + 0
= 1,820,982,302
```

Primary receipt: `evidence/demo1/final_inventory/inventory_receipt.json`, SHA-256 `7da4c77a757c9f1dfab63316470b0986233866427312734a3b75fbf1ab546b4a`, committed at `95bd4fc`. It binds this inventory to five completed app turns and five completed playbacks in one cold boot, airplane mode, with empty ADB-forward lists and the same main/runtime PIDs. Two SenseVoice turns carry the narrow matched-QNN execution claim; Parakeet, Zipformer, Qwen, and TTS remain CPU routes. The retained historical RU answer is execution coverage only; the final film replaced it with the separate factually correct “Москва” fixture, without reusing the historical latency.

### Intentionally overcounting host target

The earlier host target retained the uninstalled Vocos 24 kHz export. It remains a valid safe bound, but it is not the accepted physical total:

| Scope | Unique learned | Conservative resident | Conservative headroom |
|---|---:|---:|---:|
| Host target with absent Vocos 24 kHz retained | 1,824,062,690 | 1,834,513,983 | 165,486,017 |
| Accepted physical inventory | **1,810,531,015** | **1,820,982,302** | **179,017,698** |

The exact required Qwen3.5 artifact is:

| Field | Value |
|---|---|
| Repository | `unsloth/Qwen3.5-0.8B-GGUF` |
| Revision | `6ab461498e2023f6e3c1baea90a8f0fe38ab64d0` |
| File | `Qwen3.5-0.8B-Q4_0.gguf` |
| Bytes | 507,154,688 |
| SHA-256 | `1f2002727f6f9c9ea39bdb0eefb58703d5d2652d55a185902aab5dcf1b4ad701` |

A different ggml-org artifact shares that filename but has different bytes and hash. It cannot inherit Lane B's quality evidence.

The exact payload was downloaded and independently header-parsed during the overnight run. It contains 320 tensors and exactly 752,393,024 tensor elements, with zero MTP/draft tensors and zero vision/visual/mmproj tensors. App-private installation, on-device hash verification, and five-language execution are now separately closed by the final inventory and checkpoint receipts. The header receipt is `evidence/demo1/qwen35_08b_gguf_inventory.json`.

Demo 1 required replacement, not addition. The final listing proves Qwen2.5 is absent beside Qwen3.5 and the full TTS laboratory was replaced by the five-candidate physical tree. Adding Qwen2.5 back to the accepted physical inventory would raise the conservative scope to **3,598,070,302**, invalidating the claim.

## Direct export receipts

### Parakeet INT8 CPU export

| File | SHA-256 | Initializer/Constant elements |
|---|---|---:|
| `encoder.int8.onnx` | `acfc2b4456377e15d04f0243af540b7fe7c992f8d898d751cf134c3a55fd2247` | 619,096,503 |
| `decoder.int8.onnx` | `179e50c43d1a9de79c8a24149a2f9bac6eb5981823f2a2ed88d655b24248db4e` | 11,807,384 |
| `joiner.int8.onnx` | `3164c13fc2821009440d20fcb5fdc78bff28b4db2f8d0f0b329101719c0948b3` | 6,321,166 |
| **Total** |  | **637,225,053** |

Lane A proves the logical model has 627,057,286 learned parameters. The larger export total counts quantization auxiliaries and fixed tensors. Parakeet is CPU-only in this demo; these rows support no NPU claim.

### Pruned TTS export

| Resident row | Learned unique used by ledger | Conservative elements |
|---|---:|---:|
| Matcha acoustic model | 18,664,582 | 18,665,803 |
| Vocos 22 kHz | 13,460,015 | 13,460,021 |
| Vocos 24 kHz, retained but unused | 13,531,675 | 13,531,681 |
| Supertonic graphs | 99,217,620 | 99,499,913 |
| Supertonic style bank | 129,280 | 129,280 |
| **Total** | **145,003,172** | **145,286,698** |

The JSON ledger records the SHA-256 of every ONNX file and the style bank. This table is the intentionally overcounting host export; the accepted physical inventory excludes Vocos 24 kHz and therefore counts 131,471,497 learned / 131,755,017 conservative for TTS.

## What is installed and what is not

Demo 2 accepted snapshot, retained by its receipt:

Proven installed and executed:

- Qwen2.5 CPU companion, one GGUF only;
- SenseVoiceSmall in final cold EN/ZH runs;
- verified two-candidate Supertonic EN + Matcha ZH tree;
- main APK `712632…2f53f`, byte-identical to its corresponding accepted installed main artifact;
- runtime APK `33bf0d…7e244`, byte-identical to its corresponding accepted installed runtime artifact;
- the zero-parameter `AdaptiveEnergyVad` implementation.

Proven absent from the covered model roots/state:

- LFM weights and relevant/8080 listener;
- extra GGUFs, Silero weights, Vocos 24 kHz, and sibling stale TTS trees.

Current Demo 1 accepted snapshot:

- exact Qwen3.5 `444406…c652c` installed under the runtime UID;
- Qwen2.5 removed; exactly one GGUF remains;
- main APK `cab69190…a140d`, runtime APK `d43c0423…46ae1`;
- five-candidate TTS tree verified 22/22 plus 3 metadata files;
- CPU-only Parakeet payload hash-verified and executed in ES/FR, with no partial or QNN payload;
- complete 61-row inventory accepted at 1,810,531,015 learned / 1,820,982,302 conservative;
- all five language routes completed in one cold, airplane-mode, no-forward process.

Host staged but not phone proven for either accepted scope: none.

## Acceptance status

Demo 2: **PASS engineering and capture.** Complete inventory, exclusions, final APK identities, and indirectly receipt-bound engineering acceptance are recorded. The separate film captures preserve cold EN and ZH runs with automated runner acceptance below the request-to-in-app-audio-start proxy gate; the normalized MP4 passed automated decode/audio/volume/hash checks. Detailed manual film review remains owner-pending.

Demo 1: **PASS and filmed.** Complete 61-row inventory, exclusions, package/model identities, and five same-boot engineering turns are recorded. The normalized untrimmed film uses a separate final RU fixture and ends with the factually correct visible answer “Москва”; no historical RU latency is reused for that fixture.

The honest statement is: **Demo 2 has an accepted inventory below 3B and indirectly receipt-bound n=1 EN/ZH request-to-first-audio proxies below 3 s; Demo 1 is physically accepted below 2B, completed all five offline app routes, and has a media-validated untrimmed film with a factually correct final Russian answer.**

## Primary source hashes

| Source | SHA-256 |
|---|---|
| `out/a/ledger.json` | `c20710a43a2d1c8f9ff6ad9693fdeb67437aaf68055c734703c64e2d88383e15` |
| `out/b/b1_prereg.json` | `6107258882449913b9136ad9eba930e88952192b6df08a301eafc00f03f6255b` |
| `out/b/B3.json` | `eb4b3af642bdc7f196b7a1236555a41ba374aad8141c0d2ccacd4e4a54f27f03` |
| `out/c/asr_table.json` | `62172907d9d27415c316c2daea41ff03d79de59d9b41e2ac343b30f3b1e44179` |
| `out/d/tts_table.json` | `e44a77791eff385f258208e9f440ac6d4e6bb32d72e772efddbac9689ea64762` |
| `out/overnight/evidence/demo2/warm/smoke.json` | `c7ec74c774c753c57add08f952203be0e93c9f53daadf17e89e5976dc8aea474` |
| `out/overnight/evidence/demo2/no_lfm_build.json` | `91083ea0f83ecc2289723c123bd23e2a14954b0476ed5421e86c3b7680a6dc85` |
| `out/overnight/evidence/demo2/final_pruned/inventory_receipt.json` | `73f5e9d6e5fb8e4becb529de19f0dceee8519284a1b90586a446d5ab363f302b` |
| `out/overnight/evidence/demo2/final_pruned/device_inventory.tsv` | `14380fa29ca09fd96eb04b8566aee69a8b93460fbf59e3860c6bad18df22952a` |
| `out/overnight/evidence/demo2/final_pruned/device_state.txt` | `959c42492bf47a4e08faffefb5bb90d0ceac325109e9655ffdad0ebc90042369` |
| `out/overnight/evidence/demo2/final_pruned/tts_verify.txt` | `3b8f0c1c6d9654218cbcf64277cd327fa916eebe713dd9e5166dca4a1fb5383f` |
| `out/overnight/device_bundles/demo2/accepted_apk_pair.json` | `a1b823dd66395343f1bd8786adaa83103139a5009685146653f7cfcb1bf65720` |
| `out/overnight/device_bundles/demo2/app-main-accepted-712632b7.apk` | `712632b7987557f02bd3bf67b77c00b49025207178f1719094bbb6a81cb2f53f` |
| `out/overnight/device_bundles/demo2/app-runtime-accepted-33bf0d6d.apk` | `33bf0d6d284d6eaad96308f22a8d10c3879ba6d0b792f893486be03f0a17e244` |
| `out/overnight/evidence/demo1/qwen35_08b_gguf_inventory.json` | `f13a84cdbf946f4ab883c7f3f20738542728ff9f2dfb9c175d6c8b4c5d967bbb` |
| `out/overnight/evidence/demo1/qwen35_runtime_install.json` | `1f2002727f6f9c9ea39bdb0eefb58703d5d2652d55a185902aab5dcf1b4ad701` |
| `out/overnight/evidence/demo1/provisioning_receipt.json` | `6ad57edca522b96e7ec1cc684a060d2a06d728bc691694b590b915df79749d30` |
| `out/overnight/evidence/demo1/final_inventory/inventory_receipt.json` | `7da4c77a757c9f1dfab63316470b0986233866427312734a3b75fbf1ab546b4a` |
| `out/overnight/evidence/demo1/final_inventory/device_inventory.tsv` | `2c8fc77faa6c4a0f4b3cbceb32e302d8b6923a1112e90fa77d36026426e892bf` |
| `out/overnight/evidence/demo1/final_inventory/device_state.txt` | `88d849e3e4f7e38ae551bb34630debc5814fad359efb42dc91ed8a5d4f761e85` |
| `out/overnight/evidence/demo1/five_language_cold/final/summary.json` | `365565af07a432351a7b39c04cc8e9d662d07df714b1ceb67f6d54c966a9de7e` |
| `out/overnight/evidence/demo1/five_language_cold/final/telemetry.jsonl` | `fac08614ea4b5916d9c3a6d613ae426379e6852bcf1f70fa9951c0dfe2267ece` |
| `out/overnight/device_bundles/demo2/bundle_manifest.json` | `323e384bdda0129e4d0d464ffff43123c3ff2ac77c5665a503800cd32d265c09` |
| `out/f/.../nubia_esfr_bundle/manifest/bundle_manifest.json` | `a79f329b5ff369df480dfc3b6f7af632cae5362e17b4093c06e7a36bd6c96ea3` |
| `out/a/tools/onnxr.py` | `52ee606b115820db69041dffcb14cfde36541b28507600aaee3ffa3311b93c31` |

Demo 2's pruned manifests are promoted into its durable final receipt. Demo 1's five-language TTS and CPU Parakeet payloads are promoted into `provisioning_receipt.json`, and the final two-package/external inventory is accepted by `evidence/demo1/final_inventory/inventory_receipt.json`.
