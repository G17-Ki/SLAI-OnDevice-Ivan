# Companion lifetime and standalone-mode fix plan

> **Superseded operational status — 2026-07-29.** This report accurately
> documents the earlier shell-owned failure and is preserved as historical
> evidence. The current two-APK source moves ASR and Qwen2.5 into
> `ai.liquid.lfmdemo.runtime`; real EN/ZH QNN/HTP and RU CPU transcripts now
> pass under its app UID. Physical unplug and reboot full-turn acceptance of the
> latest pair is still pending. See
> `reports/progress_20260729/ASR_QNN_AUDIT.md`.

Prepared with **no phone attached**, from the repository and the 2026-07-27
device-log pull. Everything needing hardware confirmation is marked
**[device]**.

Archive: `<RUN_ROOT>/device-logs/LMF_demo_device_logs_20260727.tar.gz`
(22 MB, sha256 `1476d557…`), 123 files — 8 bench sweeps (693 rows), 93 device
WAVs, both companion logs, device info.

## 1. What broke

**Two independent causes, not one.**

**(a) The companions died on unplug.** ASR (6010) and Qwen (6020) are shell-uid
processes under `/data/local/tmp`, started by `adb shell`, so they live in
adbd's cgroup `/system/uid_0/pid_<adbd>`. Unplugging restarts adbd and init
tears that cgroup down. Measured: `nohup`+PPID 1 and `oom_score_adj = -1000`
were already in place and did not help; a `setsid` child stays in the same
cgroup; shell gets `EACCES` writing `cgroup.procs`. **No root-free escape.**
Corroborated by the log signature — four `listening on 127.0.0.1:6020` sessions
and **zero** `cleaning up before exit` lines: no session ever ran its shutdown
handler. No OOM, no bind failure, no model error.

English survived because `liblfmserver.so` is inside the APK and the app execs
it itself ([`LfmService.java:110`](../../../../android-app/app/src/main/java/ai/liquid/lfmdemo/LfmService.java)).

**(b) A separate, live bug was discarding every NPU ASR result** — while the
companions were alive and the cable was in.

## 2. Bugs, ranked

**1 — QNN result discarded on every request.**
`scripts/android_asr_companion/handler.sh:231`. The runner exits **0** and
completes the decode (log shows `Done!`, RTF 0.006), then `json_result_valid`
rejects the parsed stdout, `run_qnn` returns 4, and the request 503s.
**9 of 9 requests**, request PIDs 30206–31365 — all from the 2026-07-27 session,
including 2 with `--sense-voice-language=zh`. **0** failures on the
byte-identical Russian CPU path.

> **Evidence correction.** The two end-to-end turns were observed to complete
> and TTS spoke, but transcript correctness was not independently checked. The
> logs indicate that every Mandarin ASR result was rejected, so the source of
> the text that reached the LLM is unresolved. The chain ran; transcription
> correctness was not verified. **[device]** to identify what served those
> turns.

**2 — The failure branch destroys its own evidence.** `handler.sh:232` logs only
the runner's *stderr*, and the `trap cleanup 0 1 2 15` at `handler.sh:62`
`rm -rf`s the request dir including `result.json.raw`. All four checks in
`json_result_valid` (`handler.sh:198-203`) return 1 silently, so **which one
tripped is not determinable from the logs**. Bug 1 cannot be root-caused until
this is fixed.

**3 — The mic gate couples the in-APK English route to a dead companion.**
`index.html:5192` `modelReady = asrModelReady && lfmModelReady`, consumed at
`:2151`. With 6010 down at page load the record button is disabled and *nothing*
works — including the fully in-app route, which has no technical need of the
companion.

**4 — The health poll self-terminates.** `index.html:5217` reschedules only
while something is unhealthy; once all three are green it stops forever and
never notices a later death. This is why English kept working after the unplug
(`modelReady` stayed latched true) — and it makes a *half*-healthy unplug worse
than a full one.

**5 — A QNN 503 aborts the whole `auto` turn.** `index.html:3985` throws before
`selected` is assigned, so the working Russian CPU stage at `:4023` is never
reached.

### Stale-log red herrings — do not chase

- `text-companion/server.log:4` **`ERR: Missing --mmproj argument`** — this is
  **not** the Qwen server. It was written by the Liquid LFM2.5-Audio runner
  (build 7641 / `68d8edf2`); the pinned Qwen runtime is b10083 / `846e991ec`.
  Different program, stale directory. **The Qwen error is just "the companion is
  dead, restart it" — there is no second bug in the text path.**
- `server-b10083.log` — a hand-run bring-up (`n_ctx_slot = 1024` vs the deployed
  2048), not the deployed config.
- The pre-fix deployment guide and ASR README said the services survived USB
  removal. Those claims were factually wrong and are corrected in the current
  tree.

## 3. The fix, in order

| # | Step | Effort | Phone? |
|---|---|---|---|
| 0 | Correct the docs and the operator procedure | **done** | no |
| 1 | Instrument the ASR failure branch | ~30 min write | to run |
| 2 | Fix the QNN discard | ~1 h | to verify |
| 3 | Decouple the mic gate; keep polling; fail closed per-language | ~half day | to verify |
| 4 | Move ASR into the APK | ~1–2 days | yes |
| 5 | Move Qwen into the APK | ~1–2 days | yes |

**Step 0 — docs + procedure (completed).** The public instructions now use the
measured mechanism. For the current build, the supported sequence is:
*connect USB → start both companions → open the app → wait for "Offline stack
ready" → keep USB connected.* After any disconnect or reboot, reconnect and
restart the companions before reopening the app. Also note: keep memory off and
avoid "remember"/"recall" phrasing when Qwen is unavailable —
`index.html:4639` sets `requiresExactText` on a store cue even with memory
disabled, rerouting an English turn to the missing Qwen companion.

**Step 0b — do NOT build a respawn supervisor.** A supervisor launched by
`adb shell` lands in the same adbd cgroup as the processes it watches and dies
in the same teardown. It only covers a crash while the cable is in, and the logs
show no crash to cover. Same verdict for wireless adb: the cgroup is adbd's own,
so TCP changes *which event* restarts adbd, not the binding — and a Wi-Fi drop
restarts it too. Its prerequisite is also unmet (port 5555 timed out under
network client isolation).

**Step 1 — instrument.** In `handler.sh`: log `wc -c` of `$raw_output` and
`$result_file` plus a bounded `od -c` tail; make each of the four `return 1`s in
`json_result_valid` print its own reason; gate `cleanup` on an `ASR_KEEP_FAILED`
env var so the request dir survives a failure.

**Step 2 — fix the discard.** Root cause first (step 1), then the likely repair
is the extraction at `handler.sh:230`
(`toybox grep '^{' | tail -n 1`) and/or the `head -c 1` / `tr | tail -c 1`
idioms in `json_result_valid`. Do not "fix" it by loosening validation until the
cause is known.

**Step 3 — make the app degrade honestly.** Drop `asrModelReady` from the mic
gate so the in-app English route works whenever the LFM server is up; keep the
health poll running after everything goes green so a later death is noticed; and
fail closed *per language* with an accurate message rather than disabling input.

**Steps 4–5 — move the companions into the APK.** This is the only real fix, and
the repo already proves the pattern: `liblfmserver.so` is a **PIE executable**
given a `lib*.so` name so `PackageManager` extracts it, exec'd from
`nativeLibraryDir` with `extractNativeLibs="true"` and `useLegacyPackaging`.
- **ASR is the cheap half** — few runner files, and `libonnxruntime.so` is
  already byte-identical inside the APK via the sherpa AAR. Model data (~328 MB)
  stays in `getExternalFilesDir`, not the APK. Blocker to settle first: QNN/HTP
  has never been measured from an app uid here — **not known blocked, just not
  measured**.
- **Qwen is the expensive half** — the blocker is *filename collisions, not
  size*: 8 of the b10083 libs already exist in `jniLibs` at different versions
  (`libllama.so` 2.1 MB vs 34.6 MB), and `jniLibs` is flat per ABI. Preferred fix
  is a static single-`.so` rebuild; the 1.06 GB GGUF stays outside the APK.
- Neither step creates a boot service — the app must still be *opened* after a
  reboot. Say so in the docs rather than implying otherwise.

## 4. What this buys

- **After 0+3:** unplugged, from a cold launch, English Expressive works
  end-to-end; zh/ru fail closed with an accurate message instead of a dead mic.
- **After 1+2:** multilingual ASR actually works *while tethered* **[device]**.
- **After 4:** multilingual ASR works unplugged after the app is reopened
  following a reboot.
- **After 5:** the full stack works unplugged after the app is opened. The cable
  is then needed only for installation and debugging.
- **Never, under this plan:** zh/ru through the in-app LFM ASR at 8080 — a
  measured model limitation (LFM ASR returns Latin phonetics for Mandarin and
  Russian), not a routing choice.

## 5. Do first when the phone returns

1. Confirm identity: versionCode **25** / `0.5.0-tts-lab`, and that the deployed
   `handler.sh` is byte-identical to the repo copy — several conclusions assume it.
2. **Root-cause the QNN discard.** Run the `handler.sh:218` command line by hand
   with stdout to a *regular file* (the working `direct.stdout` capture looks
   line-buffered; the companion's is block-buffered — a plausible cause). Then
   `grep -c '^{'` and `od -c | tail`. Repeat ~10× across audio lengths; 9/9
   argues against a race.
3. **Re-run the two end-to-end turns and inspect the transcripts.** The prior
   run did not independently record or validate them; this identifies what
   actually served those turns.
4. **Prove or kill app-uid QNN for free** — sideload the hash-pinned upstream
   SM8850 SenseVoice QNN APK and speak one utterance. This settles the only open
   blocker on step 4 in minutes, with zero code written. Gate step 4 on it.
5. Only then build step 3 and re-validate the unplug sequence end to end.
