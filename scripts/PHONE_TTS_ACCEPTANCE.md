# Physical phone TTS acceptance over CDP

`run_phone_tts_acceptance.mjs` drives the installed app's native TTS bridge
inside its debuggable WebView. It does not invoke ADB, install an APK, copy a
model, or change app source. Before running it, expose the WebView DevTools
socket by your preferred wired or wireless workflow so its CDP HTTP endpoint
is reachable from the host.

Run the complete acceptance suite:

```sh
node scripts/run_phone_tts_acceptance.mjs \
  --host 127.0.0.1 \
  --port 9224 \
  --output reports/android_phone/tts_acceptance/latest.json
```

Run selected cases during diagnosis:

```sh
node scripts/run_phone_tts_acceptance.mjs \
  --port 9224 \
  --cases auto_en_supertonic_voice4,auto_zh_matcha,auto_ru_piper_ruslan,auto_fr_espeak_fallback \
  --output reports/android_phone/tts_acceptance/routes.json
```

List all case names with `--list-cases`. The process exits nonzero if a
preflight check or an acceptance assertion fails.

The suite verifies:

- automatic Mandarin routing to `matcha_zh_baker`;
- automatic Russian routing to `supertonic3_s5`;
- automatic French fallback to the pinned offline eSpeak package;
- a concrete Matcha selection staying pinned for a Russian locale request;
- three chunks retaining queue, start, and completion order, with the terminal
  event arriving only after all three speaker-drain completions;
- immediate cancellation and cancellation after the first native PCM/start
  proxy plus a 500 ms playback window each producing exactly one cancelled
  terminal and no late start/done callbacks.

The harness records every matching `onNativeTtsEvent` payload plus host capture
times, native monotonic times, bridge catalog snapshots, assertions, and any
driver exception. A final suite-wide grace window reattributes callbacks to
their owning turns, so a late callback cannot disappear between cases. JSON is
written atomically before connection, after preflight, after every case, after
the final event audit, and after cleanup.

The page's own turn coordinator allocates generation numbers. This is
important: inventing a timestamp-sized generation outside the coordinator can
make later normal UI turns fail the bridge's stale-command guard. The original
native event handler and the pre-run TTS policy are restored during cleanup.

Run the host-only self-test without a phone:

```sh
node scripts/test_phone_tts_acceptance_harness.mjs
```
