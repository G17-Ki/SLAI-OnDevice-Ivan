# Demo app TTS lab — every candidate in the demo UI

> **Historical integration snapshot — superseded 2026-07-29.** The candidate
> measurements below remain valid for the recorded `0.5.0-tts-lab` builds. The
> current source is the two-APK `0.6.0-unplugged` stack with app-owned ASR and
> Qwen2.5 plus a serialized, locale-aware TTS design. Its final rapid-queue,
> cancellation, language-switch, and full-turn phone regression remains open;
> do not reinterpret this historical 15/15 smoke as that acceptance.

`0.5.0-tts-lab` is an experimental on-device series that turns the demo app
into a switchable TTS lab: the same ASR → LLM → TTS path, with the voice
selected at runtime instead of hard-wired to the Android system engine. The
measurements below are from versionCode 23; the retained source has advanced to
versionCode 25. No `0.5.0` APK is published, and the retained public APK
remains `0.4.13-expressive-lfm`.

This is the *integration* counterpart to
[`DEVICE_RESULTS.md`](DEVICE_RESULTS.md). The standalone bench proves a model
runs on this phone; this proves it runs **through the app's own bridge, gate,
gain and playback path** — which is where asset paths, teardown and streaming
faults actually show up.

## What is installed

| | |
|---|---|
| Package | `ai.liquid.lfmdemo`, versionName `0.5.0-tts-lab`, versionCode 23 |
| Install verified | md5 `f0d19edc…` compared against the local APK **after** install |
| Bundle | `/sdcard/Android/data/ai.liquid.lfmdemo/files/tts-eval` — 440 files, 1.4 GiB |
| Picker | 15 entries: eSpeak control + 7 Mandarin + 7 Russian, grouped by language |
| Previous build | `0.4.14-expressive-default-guard`, retained locally as an ignored comparison artifact, sha256 `47bc3b39…` |

Selecting **System eSpeak NG (control)** restores the previous behaviour exactly,
without reinstalling — the control route is preserved as a menu entry, not as a
separate build.

## Every voice, driven through the app's bridge

One Mandarin and one Russian sentence, synthesized through
`initializeTts` → `queueTtsChunk` → `finishTtsResponse`, i.e. the same calls a
live LLM turn makes. `load` is cold model load; `first` is time to the first
PCM chunk reaching the player.

| Engine | Lang | Threads | Load (ms) | First chunk (ms) | RTF | Dropped |
|---|---|---|---|---|---|---|
| `system` | any | - | 204 | — | — | — |
| `matcha_zh_baker` | zh | 1 | 803 | 81 | 0.037 | 0 |
| `melotts_zh` | zh | 8 | 1203 | 270 | 0.210 | 0 |
| `melotts_zh_int8` | zh | 8 | 1003 | 1206 | 0.940 | 0 |
| `piper_zh_huayan` | zh | 8 | 804 | 146 | 0.042 | 0 |
| `kokoro_zh_fp32` | zh | 4 | 1004 | 772 | 0.300 | 0 |
| `kokoro_zh_int8` | zh | 4 | 1003 | 1074 | 0.432 | 0 |
| `zipvoice_zh` | zh | 4 | 1203 | 1604 | 0.326 | 0 |
| `supertonic3_s4` | ru | 4 | 603 | 722 | 0.182 | 0 |
| `supertonic3_s5` | ru | 4 | 403 | 456 | 0.115 | 0 |
| `supertonic3_s6` | ru | 4 | 402 | 1026 | 0.258 | 0 |
| `supertonic3_s8` | ru | 4 | 402 | 1352 | 0.340 | 0 |
| `piper_ru_dmitri` | ru | 8 | 602 | 131 | 0.047 | 0 |
| `piper_ru_irina` | ru | 8 | 603 | 127 | 0.033 | 0 |
| `piper_ru_ruslan` | ru | 8 | 602 | 112 | 0.041 | 0 |
**15/15 loaded and spoke. Zero dropped chunks across the whole matrix** — the
PCM gate passed every candidate on these two sentences, including both Kokoro
builds. That does not retract the Kokoro INT8 defect: it reproduces on very
short input (2-character Mandarin), not on a full sentence, so a clean run here
is consistent with the earlier finding rather than a contradiction of it.

These are single-utterance smoke tests. The experimental app backend still
starts independent synthesis tasks for streamed chunks; serialization,
cumulative playback-drain accounting, and cancellation safety are not accepted.
Do not interpret the 15/15 result as production readiness.

Threads are the **device-measured** optima, not the host's flat cap of two. The
staged `candidates.json` had been carrying the host value, which is exactly the
setting that made MeloTTS look unviable and Matcha look slower than it is;
`stage_device_bundle.py` now applies `DEVICE_THREADS` from the thread-scaling
table in `DEVICE_RESULTS.md`.

### What stands out

- **Matcha 81 ms** — reproduces the standalone bench figure exactly, through the
  full app path. At one thread.
- **Piper is uniformly quick** in both languages: 112–146 ms.
- **INT8 is a pessimization again, in the app as on the bench.** MeloTTS INT8 is
  4.5× slower to first chunk than fp32 (1206 vs 270 ms) and Kokoro INT8 is 1.4×
  slower than fp32 (1074 vs 772 ms). The INT8 builds cost latency *and* carry
  the numerical defect; nothing recommends them on this SoC.
- **Supertonic's step count is not monotonic in wall time here** — s5 (456 ms)
  beat s4 (722 ms). Do not read a step ranking off a single utterance; the
  ranking in the addendum rests on repeats, and this run is a smoke test.

## Observed request completion: ASR transcript not independently verified

The table above exercises the TTS leg alone. These are complete turns — a WAV
in, on-device ASR, Qwen3-1.7B-Q4 on CPU, and the selected voice out:

| | Mandarin | Russian |
|---|---|---|
| Input | `zh_001.wav` | `ru_001.wav` |
| ASR resolved language | `zh` | `ru` |
| LLM first text | 2478 ms | 2881 ms |
| Voice | `matcha_zh_baker` | `piper_ru_dmitri` |
| TTS first chunk | **98 ms** | **90 ms** |
| RTF / audio | 0.039 / 5.2 s | 0.065 / 1.4 s |

A later log audit found that the SenseVoice QNN result-validation branch
rejected 9/9 requests, and the transcripts from these two turns were not
independently recorded or checked. The request chain completed and the TTS
measurements are real, but the source and correctness of the text passed to
Qwen are unresolved. Do not treat this table as multilingual ASR acceptance;
see [`FIX_PLAN_UNPLUGGED.md`](FIX_PLAN_UNPLUGGED.md).

**The voice is not the bottleneck in the assembled app.** Time to first text
from the LLM is 2.5–2.9 s; the TTS adds ~0.1 s on top of it. Every latency
argument in this research — the 1 s cap, the 1–2 s relaxed cap, the Supertonic
step count — concerns a term that is an order of magnitude smaller than the
language model in front of it. That does not make the TTS work pointless (a
voice that gaps or drops chunks still ruins the turn, and the gates exist for
that), but it does mean **first-audio latency should not be the deciding
criterion between the surviving candidates. Naturalness should.**

### Both companions must be running — and a cable unplug kills them

Non-English ASR and the LLM are phone-local HTTP services, **not part of the
APK**. Nothing here touches the internet — every URL is loopback — but the
services must be alive, and their lifetime is the sharp edge of this build.

**Measured on NX809J.** Every process started by `adb shell` is placed in
adbd's own cgroup, `/system/uid_0/pid_<adbd>`. Unplugging the USB cable
restarts adbd, init tears that cgroup down, and the companions die with it:

| Defence | Present? | Saved them? |
|---|---|---|
| `nohup`, reparented to init (PPID 1) | yes | **no** |
| `oom_score_adj = -1000` (never-kill) | yes | **no** — it was not a memory kill |
| `setsid` | no — and measured not to help: a `setsid` child stays in the same cgroup | **no** |
| Moving the pid to another cgroup | — | **not permitted**: `EACCES` writing `cgroup.procs` as shell |

The evidence: the phone had been up 130 h while adbd had been up 7 m 44 s, and
the companions died inside that window with ~7.5 GB free and their logs ending
on a cleanly-completed request — no crash, no OOM.

**There is no root-free way for these shell-launched processes to survive an
unplug.** Start them after every replug and every reboot, then keep USB
connected:

```bash
./tools/start_phone_companions.sh
```

Once they are up, inference traffic is phone-local and internet-free, but the
complete stack is not cable-independent.

### The UI now names the missing service instead of hanging

`waitForModel()` polled three health URLs every 900 ms for ever, with no
timeout and no error surface, so a dead companion presented as a permanent
"Starting model / Loading…". Worse, the mic is gated on
`modelReady = asrModelReady && lfmModelReady`, so with port 6010 down *nothing*
worked — not even the fully in-app English route, because the gate fires before
any language logic.

After ~13 s of failed polls the status line now reads
`Not running: speech recognition (port 6010)`. Verified both directions on
device: stopping the companion produces the message, and restarting it returns
the app to `Offline stack ready` on its own — **no app restart needed**.


Non-English ASR and the LLM are phone-local HTTP services, not part of the APK.
With them stopped the turn fails with a clear message
("The phone multilingual ASR companion is not running") rather than silently
degrading — but it does fail:

```bash
adb shell "cd /data/local/tmp/lmf-asr-companion && sh start.sh start"
```
```bash
adb shell "cd /data/local/tmp/lmf-text-companion && sh start.sh start"
```

## Reproducing without a microphone

The debug build exposes a WebView devtools socket, so a turn can be driven
straight through the page:

```bash
adb forward tcp:9222 localabstract:$(adb shell cat /proc/net/unix | grep -o 'webview_devtools_remote_[0-9]*' | tail -1)
```

Three things cost time to discover and are worth stating plainly:

1. The bridge object is `window.NativeAudio`. `NativeAudioBridge` is only the
   Java class name and does not exist in the page.
2. `listTtsEngines()` returns `{"engines": [...]}`, not a bare array.
3. **The `generation` argument must strictly increase.** `initializeTtsOnUi`
   discards any command whose generation does not advance
   (`NativeAudioBridge.java:1183`). This is correct barge-in behaviour, but a
   driver that reuses `generation = 1` sees every engine after the first sit
   silent until it times out.

A WAV can be injected into the page's file input with a `DataTransfer` so the
app's own change handler runs a real turn. `DOM.setFileInputFiles` does *not*
work here — the renderer cannot read `/sdcard`.

Harness: `scripts/tts_research_20260726/`. Raw results:
[`evidence/device/demo_app_engine_sweep.json`](evidence/device/demo_app_engine_sweep.json),
[`evidence/device/demo_app_end_to_end.json`](evidence/device/demo_app_end_to_end.json).

## Still open

**Naturalness.** This page contains no opinion about which voice sounds human;
every number here is a latency, a footprint or a gate. The private blinded
bundle still needs a human.
