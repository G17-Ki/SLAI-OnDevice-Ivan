# Natural multilingual TTS research — 2026-07-26

> **Deployment update — 2026-07-29.** The measurements and model rankings in
> this research index remain historical evidence. Operational references to
> ADB-launched ASR/Qwen companions are superseded by the app-owned
> `0.6.0-unplugged` two-APK source stack. Matcha Baker remains the Mandarin demo
> choice (with its corpus restriction) and Supertonic 3 five-step remains the
> Russian choice; final physical regression of the latest serialized automatic
> TTS route is still pending.

Replacing the robotic Android system-TTS fallback with a natural offline voice,
Mandarin first and Russian second, without touching the validated English
Expressive-LFM route.

The host study ran with the phone disconnected. An overnight on-device run on
**NX809J / SM8850 / Android 16** has since measured every candidate on real
hardware — see [`DEVICE_RESULTS.md`](DEVICE_RESULTS.md), which takes precedence
over host figures wherever the two disagree.

| Document | What it answers |
|---|---|
| [`REPORT.md`](REPORT.md) | Candidate ranking, measured evidence, licensing and footprint, why each recommendation follows |
| [`NPU_STATUS.md`](NPU_STATUS.md) | The MeloTTS-ZH real-text repeated-decoder chain on exact SM8850 / HTP v81, and what it does and does not prove |
| [`PHONE_INTEGRATION.md`](PHONE_INTEGRATION.md) | Feature-flagged backend design, asset packaging, rollback, and the gates a phone session must still close |
| Private listening bundle | Blinded, loudness-equalized clips and score sheets retained outside Git so restricted voice outputs and the unblinding map are not published |
| [`melotts_chain_audio/`](melotts_chain_audio/) | Labelled A/B/C audio: host golden vs exact-QRD fixed-local vs exact-QRD accumulated |
| [`DEVICE_RESULTS.md`](DEVICE_RESULTS.md) | **Physical NX809J/SM8850 measurements — supersedes the host numbers wherever they disagree** |
| [`DEMO_APP_TTS_LAB.md`](DEMO_APP_TTS_LAB.md) | Every candidate switchable inside the experimental demo UI, driven through the app's own bridge |
| [`FIX_PLAN_UNPLUGGED.md`](FIX_PLAN_UNPLUGGED.md) | **Why the ADB-launched companions die on unplug, and the ordered path to standalone operation** — including a possible NPU-ASR result-discard bug |
| [`RELAXED_BUDGET_ADDENDUM.md`](RELAXED_BUDGET_ADDENDUM.md) | Re-profile against a 1-2 s first-audio cap; corrects the Supertonic step choice and the Kokoro INT8 finding |
| [`../../../../android-tts-bench/`](../../../../android-tts-bench/README.md) | Standalone on-device bench app for listening to and timing every candidate on the phone |
| [`evidence/`](evidence/) | Sanitized machine-readable results |

## Headline

- **Mandarin (demo):** `matcha-icefall-zh-baker` — the only Mandarin candidate
  passing every measured gate. Blocked commercially by a non-commercial corpus.
- **Mandarin (licence-clean):** MeloTTS-ZH. The two-thread host run made it look
  NPU-only, but the physical phone reaches a 663 ms first chunk and 0.44 RTF at
  eight threads. Its exact-QRD chain also passes end to end.
- **Russian (primary):** Supertonic 3 at **five** steps. *(Corrected from
  four - the original call rested on a five-repeat P95 that was a single
  outlier; see the addendum.)* Five steps ties six for the best measured
  intelligibility at 0.529 s P95 to first chunk.
- **Russian (fallback):** Piper `ru_RU-dmitri` — the only candidate in the whole
  run with no blocking failure and no missed target, and the only compact
  Russian voice found with a CC0 corpus.
- **eSpeak is now disqualified by measurement**, not by description: Mandarin
  micro-CER 55.2% and Russian micro-WER 61.7%, both above their hard limits.

**Device headlines:** Matcha reaches **118 ms end-to-speaker** on the phone;
MeloTTS-ZH turns out to be **CPU-viable at 4-8 threads** (the host's two-thread
cap hid this); Supertonic does not scale past two threads, so five steps sits at
900 ms; no thermal throttling over 25 consecutive long utterances; and **memory,
not latency, is now the binding constraint** (0.5-1.2 GiB per candidate).

**Naturalness is deliberately still open.** No human has scored the blinded
bundle. Everything above is a screen that disqualifies candidates and orders the
survivors; it cannot decide which voice sounds human.

## Reproduce

Harness and instructions:
[`scripts/tts_research_20260726/`](../../../../scripts/tts_research_20260726/).
Weights, raw logs, signed URLs, and large audio batches are kept outside this
repository under the external run root.
