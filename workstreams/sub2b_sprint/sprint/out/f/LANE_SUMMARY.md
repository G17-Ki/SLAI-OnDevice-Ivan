# Lane F summary — ES/FR integration and the phone bundle

## Headline

**Spanish and French now exist end to end in the demo, offline, on the CPU, and the bundle
that adds them to the Nubia is staged and hash-pinned.** All five tasks F1–F5 are complete.

Confidence is **HIGH** that the code is correct and the artifacts are what they claim to
be: both Android modules compile, every documented host gate passes, 69/69 ES/FR synthesis
runs cleared the app's own waveform gate, ES/FR transcripts were byte-identical across
every repeat, 26 registry rows carry locally computed SHA-256s, and 20 mock-adb cases prove
the installer fails closed. Confidence is **NONE** for anything on the phone: no phone was
connected, so every device latency figure is `PROJECTED` and none has been observed.

The one number to carry into the room: an ES or FR turn is expected to take **7–11 seconds**
`PROJECTED`, most of it model load, not recognition. That is honest and it is filmable —
but it should be planned for, not discovered on camera.

## Two premises in the brief that were wrong

1. **Lane E's `BLOCKED_APP_RUNNER_MISSING` is wrong about this repository** — as the brief
   said. A two-package app-UID runner is checked in, builds, and has passed cold-reboot
   acceptance. Lane F extended it and never wrote a runner.
2. **There is no 0.4.14 APK.** The brief names it as the known-good rollback.
   `android-app/releases/` retains exactly one artifact,
   `lfm-audio-demo-0.4.13-expressive-lfm-debug.apk` (code 21, SHA-256 `b178c93f…5f139baf`),
   and `android-app/releases/README.md` says plainly that it predates the runtime companion
   and must not be described as the current deployment. `F5.md` §6 replaces it with a
   four-step ladder whose first rung — `rollback_esfr.sh`, deleting four files — is better
   than any APK swap, because the ES/FR route is purely additive.

## F1–F5

- **F1 — ground truth.** Four language pins exist and disagreed: the companion's server
  allow-list (`AsrHttpServer.java:156-171`), the WebView's companion gate
  (`index.html:4394`), the UI locale set (already eight languages wide), and the TTS
  auto-priority whitelist (`CandidateRegistry.kt:135-144`). ES/FR needed **seven additive
  edits**, all named with file and line. Before this lane, choosing Spanish silently fell
  through to LFM2.5-Audio on CPU.
- **F2 — route decided: CPU.** Option (b), Parakeet-TDT-0.6B-v3 INT8 on the CPU provider
  inside the existing companion. Option (a) is the follow-up. The published SM8850 Parakeet
  contexts embed QNN `2.40.0.251030114326_189385` against a `2.45.0.260326154327` programme
  pin, and exactly one QNN provider may live in a process — so a second NPU consumer belongs
  in its own package, like the four existing probes, not in the companion on the night of a
  shoot. **No QNN context is loaded, pushed, or referenced anywhere in this lane.**
- **F3 — TTS: a configuration change, not an integration.** `supertonic3_es_s5` and
  `supertonic3_fr_s5` carry a `files` map byte-identical to the shipped RU/JA/KO rows, so
  `sharesModelWith()` matches and the phone gains **zero model bytes** and reloads nothing
  on a language switch. **69/69 runs passed** both `PcmGate` (ported faithfully from the
  app) and Lane D's exact D3 gate, with **zero chunks dropped**. Two host gates were
  extended so the shared-graph property is now enforced rather than assumed.
- **F4 — bundle staged.** `artifacts/nubia_esfr_bundle/`: 16 installable files,
  671,938,556 bytes, all hashed; 26 registry rows, all with `npu_claim: false`; four
  explicit unmet dependencies. Lane E's discipline kept (one authorized NX809J/SM8850
  target, zero adb forwards, app-UID ownership, atomic hash-verified placement), its QAIRT
  pin machinery dropped as inapplicable, and a new refusal added: any payload row that looks
  like a QNN context or Qualcomm runtime object is rejected with exit `24` before a device
  is contacted. `shellcheck -x` clean; 20/20 mock-adb cases pass; zero real adb calls.
- **F5 — deployment delta.** Which of the eight steps change (three), what to run, what
  working looks like per language with its failure signatures, the rollback ladder, and a
  shot list that refuses to cut the ES/FR wait out of the film.

## Three findings worth keeping

1. **The Parakeet CPU INT8 and the SM8850 QNN prebuilt share one token table** — both
   `tokens.txt` files are 93,939 bytes with SHA-256 `d5854467…37e3c35d`, matching Lane E's
   `E3.json:61`. A transcript from today's CPU route is therefore directly comparable to one
   from a future NPU route. `EXACT`.
2. **The bottleneck is architectural, not the model.** The companion spawns its recognizer
   per request (`AsrHttpServer.java:294-352`), so every ES/FR turn re-creates an ONNX
   session for a 652 MiB encoder — roughly four times the cost of the decode it enables,
   with a warm page cache. The fix needs no new native code:
   `com.k2fsa.sherpa.onnx.OfflineRecognizer` is already inside the `sherpa-onnx-1.13.4.aar`
   the main app links. The blocker is a `libonnxruntime.so` name collision with the
   companion's QNN build — the same class of problem that created the two-APK split. Half a
   day, and it turns a 10-second turn into a 2-second one.
3. **Supertonic cannot produce the short-input failure that killed Piper.** Piper DaveFX ES
   returned 0.081 s for `A`, below the 0.15 s floor; Supertonic returned 1.317 s for the
   same text, and every single-character input landed near 1.3 s. The lowest RMS seen
   anywhere across 69 runs was 0.0479, ten times the 0.005 floor. Nothing was marginal.

## What I could not do, and one measurement I have to disown

- **No phone.** Staging only, as instructed. `install_esfr.sh`, `verify_esfr.sh`,
  `rollback_esfr.sh` and `collect_esfr.sh` are proven to *refuse* correctly and to be
  statically clean. They are not proven to *succeed*.
- **Host latency in this lane is not usable and I am not going to pretend otherwise.** The
  measuring Mac carried load average 29–53 on eight cores with unrelated applications at
  40–90% CPU. Identical ES/FR synthesis configurations differed by up to **35×** across
  runs (RTF 0.75 to 26.5). `F3.json` records `host_timing_quality.usable: false` and classes
  that latency **`UNKNOWN`**, not `HOST`. The gate verdicts and the transcript determinism
  are arithmetic over samples and are unaffected — those stand. F2's phone projection
  inherits this contamination on top of a weak cross-workload transfer; it answers "about
  one second, about ten, or about a minute?" and nothing finer.
- **No accuracy and no listening result.** Four vendor clips decoding correctly is an
  integration proof, not a word error rate. A finite, unclipped waveform is a catastrophe
  screen, not intelligibility. Lane D ranked Supertonic ES/FR `UNSTABLE — no rank`.
- **Italian was left alone.** It was in the selector before this lane and still falls
  through to the LFM CPU path. Wiring it was not in scope and pretending otherwise would
  have been the easy, wrong thing.
- One redundant re-timing run was killed mid-flight once the machine's load made it
  pointless. The retained `f3_supertonic_esfr.json` is a complete 69/69 dataset and the
  staged fixture hashes were re-verified against the manifest afterwards: 16/16 match.

## The one command that converts projection into measurement

```bash
workstreams/sub2b_sprint/sprint/out/f/artifacts/nubia_esfr_bundle/verify_esfr.sh
```

It prints one round-trip time per language. Those two numbers replace every `PROJECTED`
figure in `F2.md`.

## Storage and hygiene

Lane storage **781 MiB** against the 3 GB cap; no allowance was borrowed. Both source
archives (487 MB Parakeet, 129 MB Supertonic) were verified against their published or
repo-pinned SHA-256 and then deleted. No model weight, WAV, APK, `.so`, or Qualcomm runtime
object is committed — only the control plane, matching how Lane E committed its own. Three
commits on `sprint/f-integration`; nothing was pushed and no other branch was touched.

## Verification actually run

Both Android modules compile (`:app` and `:runtime-companion`, JDK 17, AGP 8.7.3). Every
host gate in `docs/ANDROID_DEPLOYMENT.md:397-407` passes, plus
`test_tts_device_bundle_metadata.py`, `test_tts_candidate_registry_contract.py`,
`test_phone_acceptance_evidence.mjs`, and the bundle's own 20-case suite. Three of those
gates caught real mistakes in my own changes before I committed them, which is the reason
to run them.
