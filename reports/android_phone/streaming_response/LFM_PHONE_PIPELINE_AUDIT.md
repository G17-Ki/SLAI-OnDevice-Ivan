# Phone-only LFM streaming-path audit

Date: 2026-07-23<br>
Scope: exact microphone-to-playback data flow, remaining buffering, LFM stream
semantics, and the smallest safe standalone-TTS handoff

## Decision

The current English phone path should remain the primary demo path. It already
receives text token pieces and generated PCM before the LFM response completes,
and the `0.3.1-telemetry` UI schedules each PCM delta immediately. The retained
full output WAV is now only a replay/fallback artifact; it no longer gates first
audio.

For the modular multilingual route, use a **separate, warm TTS runtime** and
feed it stable sentence/clause chunks from the text LLM stream. Do not send
those chunks back to the same LFM server process: its request worker is serial,
so TTS would wait behind the active LLM request and provide no first-audio
latency benefit. A second full LFM instance is also incompatible with the
measured phone memory envelope.

The repository now contains a bounded multilingual chunker and a serialized
host TTS queue implementation. Android integration is intentionally not folded
into the `0.3.1-telemetry` APK; a phone-native standalone TTS runtime has not
yet been selected and validated.

## Exact current phone path

| Phase | Current behavior | Buffers the complete item? | Blocks first response audio? |
|---|---|---:|---:|
| Microphone capture | Android `AudioRecord`, mono PCM16 at 16 kHz | **Yes.** PCM is retained until the user/VAD endpoint closes. | Expected: inference starts after endpoint close. |
| WAV creation and bridge | Java creates one WAV, Base64-encodes it, and sends it to the WebView | **Yes.** The full utterance crosses the JavaScript bridge. | Small post-endpoint copy/encoding cost. |
| HTTP request | The page embeds the full Base64 WAV in one JSON chat request | **Yes.** There is no streaming request-body/audio-input protocol. | The server cannot begin ASR before the request is complete. |
| LFM prefill / input audio | The native server decodes the supplied WAV and performs audio prefill | **Yes.** Complete-WAV input is the API contract. | This is part of post-VAD model latency. |
| LFM text output | SSE `delta.content` strings are appended as they arrive | No full-text wait. | **No.** First text is observable before completion. |
| LFM audio output | SSE PCM deltas are decoded and queued on one WebAudio timeline | No full-audio wait for playback. | **No.** The first PCM packet can be scheduled immediately. |
| Replay/fallback | Every PCM delta is also retained, concatenated, and wrapped as a WAV after completion | **Yes.** Full generated audio remains in memory for replay and fallback. | **No.** This happens alongside/after streamed playback. |

Source trace:

- [`NativeAudioBridge.java`](../../../android-app/app/src/main/java/ai/liquid/lfmdemo/NativeAudioBridge.java)
  lines 65-69 collect PCM; lines 81-116 stop, copy, create the WAV, Base64 it,
  and call JavaScript; lines 129-144 append microphone buffers.
- [`index.html`](../../../android-app/app/src/main/assets/demo-ui/index.html)
  lines 642-647 decode the complete Java bridge payload; lines 821-839 embed
  the complete WAV in JSON; lines 847-905 consume SSE incrementally; lines
  870-901 queue each PCM delta; lines 907-915 concatenate the retained deltas
  only for replay/fallback.
- [`LfmService.java`](../../../android-app/app/src/main/java/ai/liquid/lfmdemo/LfmService.java)
  lines 83-100 launch one local native server on `127.0.0.1:8080`.

The fallback browser-recording path has the same endpoint behavior: it retains
all `MediaRecorder` chunks, decodes/resamples the complete blob, then constructs
one 16 kHz WAV before submission. Therefore the present system is
**VAD/end-of-turn gated input plus streamed output**, not duplex streaming ASR.

## Does the packaged LFM runtime stream tokens?

**Yes, at the HTTP interface it streams decoded text token pieces.** It does
not expose token IDs, logits, or a stable tokenizer callback API to the Android
application; the contract is SSE strings in `choices[0].delta.content`.

Evidence tied to the exact pinned artifact revision:

- The official client at revision
  [`7d525f8`](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-GGUF/blob/7d525f883a077e20afb782f2ff618edcae0e39e4/liquid_audio_chat.py#L225)
  sends `stream=True` for single-shot and chat requests.
- Its stream loop consumes `delta.content` immediately and independently
  consumes `delta.audio_chunk`, forwarding each audio packet to a non-blocking
  playback queue
  ([lines 245-282](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-GGUF/blob/7d525f883a077e20afb782f2ff618edcae0e39e4/liquid_audio_chat.py#L245)).
- The pinned Android runner hash is recorded in
  [`runner-files.sha256`](../../../android-app/runner-files.sha256). The binary
  contains the `audio_chunk`, `chat.completion.chunk`, `text/event-stream`, and
  `non streaming API is not implemented` markers.
- The measured Nubia profile observed many SSE events per response and first
  text/audio well before completion. For `question_chat`, median first text was
  677 ms, first generated audio 800 ms, and completion 6,786 ms
  ([profile.json](../native_profile/20260716t120118z/profile.json)).

The model card says the runners come from
[`ggml-org/llama.cpp#18641`](https://github.com/ggml-org/llama.cpp/pull/18641).
The branch already contained the one-thread HTTP pool and single request worker
in ancestor commit
[`ac98513`](https://github.com/tdakhran/llama.cpp/blob/ac985130a2a5aa594f4e5b819eabd100b4c453fa/tools/liquid-audio/server.cpp#L143-L154).
The current public PR head shows the same architecture directly: a single
request worker
([server.cpp lines 160-173](https://github.com/tdakhran/llama.cpp/blob/53b3c5474ef2e02c3a83fdbc166af76ffa22d542/tools/liquid-audio/server.cpp#L160-L173)),
per-token-piece text callbacks and per-audio-block callbacks
([runner.cpp lines 278-308](https://github.com/tdakhran/llama.cpp/blob/53b3c5474ef2e02c3a83fdbc166af76ffa22d542/tools/liquid-audio/runner.cpp#L278-L308)),
and immediate SSE emission
([server.cpp lines 191-233](https://github.com/tdakhran/llama.cpp/blob/53b3c5474ef2e02c3a83fdbc166af76ffa22d542/tools/liquid-audio/server.cpp#L191-L233)).
That PR head postdates the packaged Hugging Face revision and uses a newer PCM
schema, so it is implementation-lineage evidence rather than proof of the
exact binary source commit. The exact pinned client and measured binary behavior
remain the authoritative packaged-runtime evidence.

## Why the same LFM server cannot be the standalone TTS worker

The server keeps one `Runner` and one worker that pops and completes one work
item at a time. If the app starts a text-only LFM request, extracts a clause,
and submits a TTS request to the same endpoint, that TTS request sits behind the
still-active generation. It starts after full text completion, recreating the
latency problem.

Aborting the text request at the first clause is not a safe workaround. It
throws away the remaining answer, complicates context continuation, and lets
the TTS turn mutate the same runner/context that would be needed to resume the
answer. Running a second full LFM process would roughly duplicate the measured
model residency; the retained phone profile already reached 2,169 MiB (about
2.12 GiB) combined sampled peak RSS.

Therefore the standalone TTS path must be one of the lighter independent
runtimes already under evaluation (for example Supertonic, Piper/Melo, or a
temporary Android system-TTS control), kept warm while the text LLM streams.

## Implemented bounded host path

[`streaming_text_chunks.py`](../../../scripts/streaming_text_chunks.py) adds an
incremental chunker with these rules:

1. Prefer a completed sentence and include closing quotation/bracket marks.
2. Require one character of lookahead for ASCII sentence punctuation so `3.`
   is not spoken before a later `14` arrives.
3. Protect decimal numbers, ellipses, single-letter initials, and common
   abbreviations.
4. Permit comma/semicolon/colon clause flushes only after a minimum useful
   length.
5. Flush at a word boundary at the configured maximum length; CJK text without
   spaces is bounded by Unicode character count.
6. Flush the final tail only after the upstream LLM completion event.

[`profile_cascade_voice_pipeline.py`](../../../scripts/profile_cascade_voice_pipeline.py)
now feeds only newly visible SSE text into that chunker. Every ready chunk is
submitted to a one-worker executor: the first TTS job starts while the LLM is
still generating, later jobs remain ordered, and the full response tail is no
longer dropped after the first measured chunk. Running plus waiting jobs are
bounded (four by default); a slow TTS applies backpressure instead of growing
an unlimited executor queue. Outside explicit sample capture, completed jobs
retain timing metadata rather than full PCM arrays. The result JSON records
every chunk's ready time, TTS start, synthesis time, audio duration, and any
SSE-reader time spent applying TTS-queue backpressure. New runs use result
schema version 2 because completion now covers the serialized full-response
TTS queue instead of only the first chunk.

Deterministic validation is in
[`test_streaming_text_chunks.py`](../../../scripts/test_streaming_text_chunks.py)
and
[`test_streaming_tts_queue.py`](../../../scripts/test_streaming_tts_queue.py).
It covers fragmented tokens, decimal lookahead, abbreviations, closing quotes,
CJK punctuation, clause boundaries, hard bounds, ordering, final-tail behavior,
fragmentation invariance, invalid policies, first-chunk overlap with an open LLM
stream, serialized full-response synthesis, and bounded-queue backpressure. All
13 tests pass without loading a model.

## Smallest safe Android implementation

Do this as a new versioned APK after choosing the phone-native TTS runtime:

1. Keep `0.3.1-telemetry` unchanged for the integrated English baseline.
2. Add a modular route that performs VAD-gated ASR, then starts a streaming text
   LLM request.
3. Port the tested chunker state machine to the app layer and feed it only new
   visible text. Never synthesize open `<think>` content.
4. Keep one dedicated TTS worker/model warm. Enqueue chunks serially and reject
   stale generation IDs after barge-in/cancel.
5. Forward each TTS PCM callback into the existing WebAudio queue immediately;
   do not wait for a WAV container or later chunks.
6. Record `chunk_ready`, `tts_start`, `first_pcm`, scheduled start,
   output-head crossing, queue depth, and underflow count. Bound queued text and
   synthesized-but-unplayed audio so a slow TTS cannot grow memory indefinitely.
7. Accept the route only after ten fixed-file and ten microphone turns on the
   Nubia show end-of-speech to audible output at or below 2.0 s median / 3.0 s
   P95 with no audible gap above 80 ms.

The host change establishes the handoff logic, not an Android performance
claim. The next implementation decision is which standalone TTS runtime to
embed first; the exact-SM8850 Piper/Melo and Supertonic experiments should make
that choice, followed by a separately versioned APK and physical-phone run.
