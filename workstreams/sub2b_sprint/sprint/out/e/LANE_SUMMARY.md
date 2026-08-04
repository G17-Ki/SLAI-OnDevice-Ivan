# Lane E summary — device preparation and AI Hub

## Headline

Qualcomm AI Hub is usable, but this account cannot select SM8850 for compilation. A genuine SM8850 Parakeet-TDT 0.6B v3 prebuilt exists and verified, yet its three contexts embed exact QNN `2.40.0.251030114326_189385`. Lane E pins the programme to exact QAIRT/QNN `2.45.0.260326154327`, so the programme-level Parakeet verdict is **EXPORT**; the downloaded artifact is **REPACKAGE only inside an isolated exact-2.40 runtime process**.

The Nubia bundle and phone procedure are staged, but execution is deliberately blocked as `BLOCKED_APP_RUNNER_MISSING` / `CPU_STAGE_ONLY`. There is no app-UID runner, no complete five-language CPU stack, and zero exact-2.45 model contexts. No phone was connected and no version mismatch was attempted.

## E1–E6 results

- **E1 — AI Hub:** `qai-hub` 0.53.0 authenticated and enumerated 79 devices. SM8850/S26 targets were discoverable, but three selector forms all returned `No devices match the given OS name, version, and attributes.` A 119-byte ONNX probe compiled successfully for SM8750 at QAIRT `2.48.0.260626120635` in 70.951 seconds **HOST**. Numeric account quota remains **UNKNOWN**.
- **E2 — sherpa inventory:** the three release tags contain exactly 2,238 assets: 2,237 QNN archives plus one checksum file. Exactly 236 target SM8850. Parakeet v3 has all 12 claimed 3–30 second windows; Nemotron streaming reaches 80 ms. Counts and release metadata are **EXACT**; filename-derived graph precision remains **VENDOR**.
- **E3 — one prebuilt:** asset `465352256` verified at 437,668,456 bytes and SHA-256 `9935bbeecab3583516bbf437fa750cf2d2daf9959f2702f3b18781d6c5988b2f`. All extracted files verified. The three context headers independently contain SM8850, HTP minimum architecture 81, and the same exact QNN 2.40 build. A pinned FP32 ONNX equivalent produced CPU transcript/token/timestamp references for all three test WAVs; this was not QNN execution or phone performance.
- **E4 — Qualcomm registry:** Zipformer ZH-EN's usable path is float-only and its mixed/quantized QNN paths are disabled after Qualcomm CI failures. The expected “no TTS HTP” result is stale: all six currently published TTS models have VoiceAI/QNN NPU profiles and SM8850-for-Galaxy assets at QAIRT 2.45. Qwen3-0.6B has a disabled SM8850 w4a16 GenieX-QAIRT package after an accuracy failure. Qwen3.5-0.8B has q4_0 GenieX-llama.cpp only; its vendor SM8850 QRD row reports CPU decode 59.414 tok/s versus NPU 41.909 tok/s, while NPU wins prefill. Different-device rows were not summed.
- **E5 — one pin:** compile and device runtime are both fixed to exact `2.45.0.260326154327`. The official 1,657,471,677-byte QAIRT archive verified at SHA-256 `aff2f154319d88db81a4125c35882395cd88c024959ca3cc3cda3ba442ef2df0`; all 11 allowlisted runtime/identity/licence members match their pinned hashes. `artifact_registry.json` contains 36 honest rows and explicit unmet dependencies. Exactly one QNN runtime provider may exist per APK/process.
- **E6 — staged bundle:** the local bundle contains 46 files / 760,928,954 bytes, including the ignored model/audio/runtime payloads and committed control plane. The scripts require one authorized NX809J/SM8850 arm64 target, reject adb forwards and unreadable forward state, verify hashes, require app UID ownership, refuse missing or mixed versions before QNN load, require a readable non-empty run log, force airplane mode, and collect app-private evidence. Bash syntax, shellcheck, 19 payload/runtime hashes, the external registry pointer, seven mocked-adb safety cases, and an independent fail-closed audit pass. Real phone/adb calls: zero.

## Confidence and boundary

Confidence is **HIGH** for API inventory, file identities, hashes, embedded version strings, registry presence, the 2.45 runtime supply, and the conclusion that the current bundle must not load its 2.40 contexts. Confidence is **NONE** for phone latency, numerical QNN parity, device placement, thermals, memory, end-to-end quality, or on-device success because the phone was absent.

The acceleration path exists in published SM8850 artifacts and Qualcomm profiles, but Lane E did not prove an executable complete stack. Plugging in the phone permits the runbook's identity check; it does not authorize NPU execution until the missing activation inputs are supplied.

## Next authorized actions

1. Obtain SM8850 compilation entitlement or acquire independently hashed Parakeet contexts compiled at exact QAIRT `2.45.0.260326154327`.
2. Build and hash an app-UID runner that incorporates exactly one 2.45 runtime provider and implements the documented import/request/result contract.
3. Integrate the selected complete CPU/NPU five-language artifacts from lanes A–D with immutable revisions, hashes and licences.
4. Only then follow `PHONE_RUNBOOK.md`: identity → bundle verification → app-private install → installed verification → airplane-mode scripted turn → collection.

Lane storage at completion is approximately 731 MiB, below the 5 GB cap. Recoverable CPU and QAIRT download archives were deleted after verification. No model weights, audio, APK, AAR, `.so`, or Qualcomm runtime binary was committed.

---

# Lane H addendum — report and demo package

## Headline

Lane H assembled the available A/D/E record into a defensible report package without filling gaps from memory. All six measured stack subtotals and Lane A's recorded known-row UB subtotals are below the cap, but five stacks still contain `UNKNOWN` rows; only H has closed arithmetic. Licensing is the cross-stack discriminator, and Q/F additionally require a binding zero-parameter punctuation/ITN design to remain below cap.

The tested TTS set still has no general shippable Mandarin route. Kokoro ZH is waveform-unsafe on the reproduced `HOST` probes; Matcha produces finite host audio but can drop Latin/OOV spans and uses a commercially blocked Baker checkpoint. No native-speaker acceptance was run.

The device record remains staging-only. SM8850 is discoverable but not compile-selectable for the Lane E account. The verified Parakeet contexts embed exact QNN `2.40.0.251030114326_189385`; selecting exact QAIRT/QNN `2.45.0.260326154327` as the programme pin is `PROJECTED`, so the contexts remain blocked from programme NPU use. No phone execution, NPU parity, or glasses evidence exists.

## H1–H5 closeout

- **H1:** `out/h/evidence_index.json` indexes 108 claims: 105 landed A/D/E/H claims and three explicit B/C/F pending rows. Claim IDs are unique; required fields, JSON structure, source-file SHA-256s, and evidence-class counts were validated.
- **H2:** `out/h/REPORT.md` contains the requested six sections, all six ledger rows, explicit evidence classes and source labels, open-question closure conditions, and clearly marked B/C/F slots.
- **H3:** `out/h/DEMO_SHOTLIST.md` separates host, phone, and pending shots; supplies exact EN/ZH/ES/FR/RU lines; includes the Matcha dropped-token and Kokoro non-finite failure demonstrations; and never presents the blocked phone bundle as runnable.
- **H4:** `out/h/CLAIMS_AND_LIMITS.md` pairs every supported sentence with the adjacent claim it does not support, including the mandatory no-glasses, no-native-acceptance, no-NPU-parity, non-commercial-ZH, and small-`n` host-screen limits.
- **H5:** the final re-scan found no `out/b/backbone_verdict.json`, no `out/c/asr_table.json`, and no Lane F result file. All three slots remain `PENDING`; no result or number was inferred.

## Demo-media and phone boundary

Lane D's validated listening bundle is git-ignored and is not present in this checkout. Audio shots require Lane D's original artifact workspace and D5 verification; otherwise the shot-list requires a `PENDING — LANE D MEDIA` slate and committed evidence cards only.

The only future phone shot currently authorized by the package is identity capture with `verify.sh --identity`. Installation and CPU/NPU inference remain blocked by the missing app runner, incomplete five-language payload, and zero exact-pin NPU contexts.

## Confidence and storage

Confidence is **high** that the report preserves the landed lane values, source hashes, and evidence classes, and **none** for absent B/C/F verdicts, on-device success, NPU parity, native-speaker acceptance, or glasses performance. The `out/h` package is approximately 132 KiB, below Lane H's 0.5 GB cap, and contains no weights or audio.
