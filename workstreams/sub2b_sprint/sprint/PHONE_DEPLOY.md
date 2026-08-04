# After the sprint — connecting the Nubia NX809J

This runs when the five lanes are merged and you plug the phone in. It is **not** part of
the 10–20 hour window.

## Preconditions (all four, checked mechanically before anything is installed)

| # | Condition | Source | If it fails |
|---|---|---|---|
| 1 | `ledger.json` says the chosen stack is COMPLIANT — not UNVERIFIED | Lane A | Stop. Close the UNKNOWN row first. |
| 2 | `backbone_verdict.json` names a surviving backbone | Lane B | Stop. Deploying an unmeasured backbone wastes the phone session. |
| 3 | For every stage intended to run on the NPU, a compiled context **and** a runtime of the **same** QAIRT version exist | Lane E `E5.json` | That stage runs on CPU. Say so in the log; do not improvise a version pairing. |
| 4 | Every artifact hash matches `artifact_registry.json` | Lane E | Stop. An unverified artifact makes every downstream measurement unattributable. |

## Sequence

```bash
# 1 identity
adb devices                       # expect the NX809J, authorised
adb shell getprop ro.build.fingerprint
adb shell getprop ro.soc.model    # expect SM8850-class

# 2 install and verify
cd workstreams/sub2b_sprint/sprint/out/e/artifacts/nubia_bundle
./install.sh                      # pushes models + runtime, verifies sha256 on-device
./verify.sh                       # asserts every hash matches artifact_registry.json

# 3 app-UID execution, no shell dependency
./run.sh --stage asr --lang en --fixture fixtures/en_01.wav
./run.sh --stage tts --lang zh --text "你好"
./run.sh --turn --lang en --fixture fixtures/en_01.wav   # full ASR -> LLM -> TTS

# 4 offline acceptance
adb shell settings put global airplane_mode_on 1
adb shell am broadcast -a android.intent.action.AIRPLANE_MODE
./run.sh --turn --lang ru --fixture fixtures/ru_01.wav   # must still pass
adb reboot && sleep 60
./run.sh --turn --lang zh --fixture fixtures/zh_01.wav   # cold-reboot survival

# 5 collect
./collect.sh                      # logcat, timings, PSS/RSS, placement, output PCM
```

## What to measure in the first phone session

- Per-stage latency and the full-turn time, **as software timestamps** — and label them
  that way. True acoustic onset needs a two-channel rig and is out of scope here.
- Peak and steady PSS per process. The prior stack measured ~4.3 GiB PSS with dual
  backbones; anything near that is a failure regardless of quality.
- Whether the intended HTP placement actually held, with CPU fallback disabled — and
  whether a stage silently fell back.
- Installed bytes on device, counting APK payload and private storage separately. An asset
  shipped inside the APK and unpacked at first launch exists on disk twice, permanently.

## What NOT to claim from this session

- Not glasses evidence. SM8850 latency, memory and thermals do not transfer to AR1-class
  silicon — different HTP generation, roughly a quarter of the RAM, and a fraction of the
  power budget.
- Not "credible native" anything. That needs the native-speaker panel this sprint cut.
- Not end-to-end latency from summed component timings. Measure the turn, or say you didn't.

## Then

Use the curated deployment evidence and bounded claims under `out/overnight/`
for the surviving stack. The internal planning programme is intentionally
omitted from the public-research snapshot.
