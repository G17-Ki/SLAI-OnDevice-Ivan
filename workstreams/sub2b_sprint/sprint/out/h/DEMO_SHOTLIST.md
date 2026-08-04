# Lane H demo shot-list

This cut is an evidence demo, not a completed-product demo. Film the host results that
exist, show the failure cases on purpose, and leave the phone and unlanded-lane inserts
visibly pending. Target length: about 2 minutes 30 seconds.

> **MEDIA BLOCKER IN THIS CHECKOUT:** Lane D's 278-clip audio bundle is git-ignored and
> the expected listening path contains **zero WAVs in this checkout**. Shots 2–7
> require the original Lane D artifact workspace mounted or copied back to the exact
> `out/d/artifacts/listening/` path and
> verified against D5 before filming. Do not imply that those clips can be played from
> this branch as it stands. If the artifact workspace is unavailable, film only the
> committed evidence cards and keep audio playback labelled `PENDING — LANE D MEDIA`.

> **PHONE BLOCKER:** the Lane E bundle is currently
> `BLOCKED_APP_RUNNER_MISSING` / `CPU_STAGE_ONLY`. It is a verified staging and refusal
> demo, not a runnable phone demo. No APK, complete five-language CPU stack, or exact-2.45
> NPU context is present. **Source:** Lane E, task E6,
> `workstreams/sub2b_sprint/sprint/out/e/E6.json` (`hard_blocks`, `app`,
> `version_contract`) and `workstreams/sub2b_sprint/sprint/out/e/E6.md`.

## Preflight: do not improvise the evidence

1. Restore or mount the original ignored Lane D artifact workspace, then use only the
   listening directory named in `D5.json`:
   `workstreams/sub2b_sprint/sprint/out/d/artifacts/listening/`. The audited bundle is
   278/280 WAVs and its clip-manifest SHA-256 is
   `dac26b1fb522c6c730435054c33be122f56bcf78b2b0813464cab4b193dcaed6`.
   If that bundle or its audit is unavailable, replace playback with the committed
   evidence card and a `PENDING — LANE D MEDIA` slate; do not substitute newly generated
   audio. **Source:** Lane D, task D5,
   `workstreams/sub2b_sprint/sprint/out/d/D5.json` (`inventory`, `artifacts`).
2. Put the exact source text and exact WAV filename on screen during every language
   shot. A clip playing is evidence of a host-produced, waveform-valid file; it is not
   native-speaker acceptance. **Source:** Lane D, tasks D1/D5,
   `workstreams/sub2b_sprint/sprint/out/d/D1.md` (Limits) and
   `workstreams/sub2b_sprint/sprint/out/d/D5.md`.
3. Keep an evidence-class bug on screen: `EXACT`, `HOST`, `VENDOR`, or `PENDING`.
   Never label the Qualcomm registry rows as an exact Nubia profile. **Source:** Lane E,
   task E4, `workstreams/sub2b_sprint/sprint/out/e/E4.json` (`evidence_policy`,
   `limitations`).
4. Do not film an NPU inference success. The staged Parakeet contexts embed exact QNN
   2.40, while choosing exact QAIRT/QNN 2.45 as the programme pin is a `PROJECTED`
   policy decision; zero exact pairs are enabled under that policy.
   **Source:** Lane E, tasks E3/E5/E6,
   `workstreams/sub2b_sprint/sprint/out/e/E3.md`,
   `workstreams/sub2b_sprint/sprint/out/e/E5.json`, and
   `workstreams/sub2b_sprint/sprint/out/e/E6.md`.

## Final cut, in order

The quoted claim in each row is the exact narrator line. Do not strengthen it in the edit.

| # | Where | What to film and exact spoken text | Visible success or failure condition | Exact supported claim | Evidence trace |
|---:|---|---|---|---|---|
| 1 | **HOST** | Open on the six-stack ledger table, then highlight the licence traffic-light and unresolved-row columns. No synthesized speech. | All six measured subtotals and recorded known-row UB subtotals are visibly below 2,000,000,000; only H has no `UNKNOWN` row, and only Stack M is green on Lane A's licence sheet. | “The cap did not eliminate any measured candidate; only H has closed arithmetic, while licensing is the cross-stack discriminator and Q and F still require a zero-parameter punctuation design.” | **Machine-computed ledger arithmetic from `EXACT` measured fields, recorded `UB` fields, and `UNKNOWN` open rows; no aggregate class added here.** Lane A, tasks A5/A7, `workstreams/sub2b_sprint/sprint/out/a/ledger.json` (`stacks`) and `workstreams/sub2b_sprint/sprint/out/a/A7.md` (six verdicts and open rows). |
| 2 | **HOST — EN; media workspace required** | From the restored Lane D artifact workspace, show and play `en_supertonic_01.wav`. On-screen source text: **“The warehouse received 48 boxes, and each box weighed 12 kilograms.”** | The exact filename is visible, playback advances, and the D5 row remains labelled `comparison_candidate`; do not add “native” or “production” captions. Without the restored workspace, show `PENDING — LANE D MEDIA` instead. | “On the host, Supertonic produced a playable, waveform-valid English clip for this exact prompt.” | **`HOST`.** Lane D, task D5, `workstreams/sub2b_sprint/sprint/out/d/D5_prompts.json` (`languages.en`, index 01), `workstreams/sub2b_sprint/sprint/out/d/D5.json` (`route_summary`, EN/Supertonic), and `workstreams/sub2b_sprint/sprint/out/d/LISTEN.md` (English row 01). |
| 3 | **HOST — ES; media workspace required** | From the restored Lane D artifact workspace, show and play `es_supertonic_01.wav`. On-screen source text: **“El almacén recibió 48 cajas, y cada caja pesaba 12 kilogramos.”** | The exact filename is visible, playback advances, and the D5 row remains labelled `comparison_candidate`. Without the restored workspace, show `PENDING — LANE D MEDIA` instead. | “On the host, Supertonic produced a playable, waveform-valid Spanish clip for this exact prompt.” | **`HOST`.** Lane D, task D5, `workstreams/sub2b_sprint/sprint/out/d/D5_prompts.json` (`languages.es`, index 01), `workstreams/sub2b_sprint/sprint/out/d/D5.json` (`route_summary`, ES/Supertonic), and `workstreams/sub2b_sprint/sprint/out/d/LISTEN.md` (Spanish row 01). |
| 4 | **HOST — FR; media workspace required** | From the restored Lane D artifact workspace, show and play `fr_supertonic_01.wav`. On-screen source text: **“L'entrepôt a reçu 48 cartons, et chaque carton pesait 12 kilogrammes.”** | The exact filename is visible, playback advances, and the D5 row remains labelled `comparison_candidate`. Without the restored workspace, show `PENDING — LANE D MEDIA` instead. | “On the host, Supertonic produced a playable, waveform-valid French clip for this exact prompt.” | **`HOST`.** Lane D, task D5, `workstreams/sub2b_sprint/sprint/out/d/D5_prompts.json` (`languages.fr`, index 01), `workstreams/sub2b_sprint/sprint/out/d/D5.json` (`route_summary`, FR/Supertonic), and `workstreams/sub2b_sprint/sprint/out/d/LISTEN.md` (French row 01). |
| 5 | **HOST — RU; media workspace required** | From the restored Lane D artifact workspace, show and play `ru_supertonic_01.wav`. On-screen source text: **“На склад поступило 48 коробок, каждая весила 12 килограммов.”** | The exact filename is visible, playback advances, and the D5 row remains labelled `comparison_candidate`. Without the restored workspace, show `PENDING — LANE D MEDIA` instead. | “On the host, Supertonic produced a playable, waveform-valid Russian clip for this exact prompt.” | **`HOST`.** Lane D, task D5, `workstreams/sub2b_sprint/sprint/out/d/D5_prompts.json` (`languages.ru`, index 01), `workstreams/sub2b_sprint/sprint/out/d/D5.json` (`route_summary`, RU/Supertonic), and `workstreams/sub2b_sprint/sprint/out/d/LISTEN.md` (Russian row 01). |
| 6 | **HOST — ZH, bounded positive; media workspace required** | From the restored Lane D artifact workspace, show and play `zh_matcha_01.wav`. On-screen source text: **“仓库收到了48个箱子，每个箱子重12公斤。”** Keep a permanent `DIAGNOSTIC / NON-COMMERCIAL CHECKPOINT` bug. | The exact file exists and plays; the screen also shows `d3_pass_text_coverage_limited` and the Baker/checkpoint commercial blocker. Without the restored workspace, show the evidence card only. | “Matcha produced finite host audio for this Mandarin prompt, but this is a diagnostic result from a commercially blocked checkpoint, not a shippable Mandarin voice.” | **`HOST` synthesis evidence + `EXACT` pinned licence/provenance observation.** Lane D, tasks D4/D5, `workstreams/sub2b_sprint/sprint/out/d/D5_prompts.json` (`languages.zh`, index 01), `workstreams/sub2b_sprint/sprint/out/d/D5.json` (`route_summary`, ZH/Matcha), and `workstreams/sub2b_sprint/sprint/out/d/D4.md` (Provenance and commercial replacement). |
| 7 | **HOST — ZH failure: dropped Latin spans; media workspace required for playback** | From the restored Lane D artifact workspace, show and play `zh_matcha_13.wav` while displaying the exact requested text: **“API返回了一个包含24条记录的JSON文件。”** Then cut to the captured frontend evidence showing `Ignore OOV` for `API` and `JSON`. If the raw probe or audio workspace is unavailable, show the committed D1/D5 evidence cards; never fabricate a terminal replay or imply playback occurred here. | With the artifact workspace, the WAV is finite and playable **while** the evidence panel shows that the frontend ignored the Latin tokens. Without it, the visible success condition is only the committed trace, labelled `HOST EVIDENCE — MEDIA PENDING`. | “Matcha can return finite PCM while ignoring unsupported Latin tokens, so a finite waveform and a scored CER point estimate do not prove mixed-script coverage.” | **`HOST`.** Lane D, task D1, `workstreams/sub2b_sprint/sprint/out/d/D1.json` (`frontend_findings[0]`, probe SHA-256 `b4c838a70db4a2d9955f5c58c4470b91d708ea08fcfd23f0ae0d6fbc0d0496a2`); task D5, `workstreams/sub2b_sprint/sprint/out/d/D5.json` (`diagnostic_classification.matcha_zh_text_coverage_risk`, index 13); exact probe construction in `workstreams/sub2b_sprint/sprint/out/d/probe_matcha_frontend.py`. |
| 8 | **HOST — ZH failure: non-finite PCM** | Put **“你好”** on screen and show the D3 three-run result for Kokoro v1.1 ZH. There should be no fake playback: all three outputs were all-NaN. Optional insert: show D5's missing `zh_kokoro_14.wav` row instead of inventing a WAV. | The evidence card visibly says `failed_runs: 3` and `all samples non-finite`; silence/no serialized WAV is presented as the caught failure, not an editing error. | “Kokoro v1.1 ZH is waveform-unsafe on the tested short prompt: ‘你好’ failed three out of three runs with all samples non-finite.” | **`HOST`.** Lane D, task D3, `workstreams/sub2b_sprint/sprint/out/d/D3.json` (`named_failures`, Kokoro/`你好`) and `workstreams/sub2b_sprint/sprint/out/d/D3.md`; optional long-input evidence from task D5, `workstreams/sub2b_sprint/sprint/out/d/D5.json` (`missing_expected`). |
| 9 | **HOST — deployment guardrail** | In `workstreams/sub2b_sprint/sprint/out/e/artifacts/nubia_bundle/`, request the staged NPU ASR path exactly as documented: `./run.sh --backend npu --stage asr --lang en --fixture files/lmf/fixtures/parakeet_v3_3s/0.wav`. Film the refusal, not an override. | The request exits before any device access because the only staged contexts are exact QNN 2.40 against the exact 2.45 pin; there is no silent CPU fallback presented as NPU. | “The staged deployment package fails closed before QNN load when the compiled context and runtime are not an exact version match.” | **`HOST` safety validation.** Lane E, task E6, `workstreams/sub2b_sprint/sprint/out/e/E6.md` (mocked-adb validation); exact current stop and command contract in `workstreams/sub2b_sprint/sprint/out/e/PHONE_RUNBOOK.md`. |
| 10 | **PHONE — `PENDING` tonight** | After the handset is physically available, film only `./verify.sh --identity`. Keep the phone and terminal in the same frame. | Required future condition: exactly one authorized device, state `device`, properties `NX809J`, `SM8850`, and `arm64-v8a`, with no adb forward. Until a captured lane output records this, leave the card `PENDING — PHONE IDENTITY`. | **No present claim.** If the capture passes and is archived, the narrow future line is: “The attached authorized test handset identified as Nubia NX809J / SM8850 / arm64-v8a.” | **`PENDING`; no phone evidence exists yet.** Acceptance contract: Lane E task E6, `workstreams/sub2b_sprint/sprint/out/e/PHONE_RUNBOOK.md` (Mechanical acceptance checks). Non-performance is explicit in `workstreams/sub2b_sprint/sprint/out/e/E6.json` (`phone_activity`). |
| 11 | **HOST — pending evidence slate** | Show three empty inserts: `PENDING — Lane B backbone verdict`; `PENDING — Lane C ASR table`; `PENDING — Lane F ES/FR integration`. Do not fill from memory or prior reports. | Each slot either shows a landed source filename or remains visibly `PENDING`; there are no placeholder numbers. | “Backbone, ASR, and ES/FR integration conclusions are intentionally pending until their lane outputs land.” | **`PENDING`.** Expected sources: `workstreams/sub2b_sprint/sprint/out/b/backbone_verdict.json`, `workstreams/sub2b_sprint/sprint/out/c/asr_table.json`, and `workstreams/sub2b_sprint/sprint/out/f/`; no claim is authorized while absent. |
| 12 | **HOST — closing truth card** | End on `CLAIMS_AND_LIMITS.md`, with the phone identity and B/C/F slots still visibly pending. | Final frame reads: `HOST EVIDENCE + FAIL-CLOSED STAGING; NOT AN END-TO-END PHONE OR GLASSES RESULT`. | “This package demonstrates audited host evidence, caught failure modes, and version-safe phone staging; it does not yet demonstrate a complete five-language product on the phone.” | Lane A `workstreams/sub2b_sprint/sprint/out/a/LANE_SUMMARY.md`; Lane D `workstreams/sub2b_sprint/sprint/out/d/LANE_SUMMARY.md`; Lane E `workstreams/sub2b_sprint/sprint/out/e/LANE_SUMMARY.md`. |

## Phone boundary for tonight

The only currently defensible phone shot is identity capture, and even that remains
`PENDING` until recorded. Do **not** film installation, CPU inference, NPU inference,
latency, placement, parity, thermals, or memory as a success. Lane E records no APK, no
complete CPU stack, zero exact-2.45 NPU contexts, and zero real phone calls. Plugging in
the phone permits the identity check; it does not lift the activation boundary.

**Source:** Lane E, tasks E5/E6,
`workstreams/sub2b_sprint/sprint/out/e/E5.md`,
`workstreams/sub2b_sprint/sprint/out/e/E6.json`, and
`workstreams/sub2b_sprint/sprint/out/e/PHONE_RUNBOOK.md`.

## Inserts to fill at H5, only if sources land

- **`PENDING — Lane B`:** replace the empty backbone card only from
  `workstreams/sub2b_sprint/sprint/out/b/backbone_verdict.json`; carry its original
  evidence class and test scope.
- **`PENDING — Lane C`:** add ASR transcription footage only from
  `workstreams/sub2b_sprint/sprint/out/c/asr_table.json`; use its exact fixture text,
  output, target, and evidence class.
- **`PENDING — Lane F`:** add integrated ES/FR footage only from a landed file under
  `workstreams/sub2b_sprint/sprint/out/f/`; do not turn Lane D's standalone TTS clips
  into an integration claim.
