# Android streaming telemetry summary

Generated: 2026-08-02T19:15:16.032585+00:00

This trace uses Android elapsed-realtime timestamps. For microphone turns,
`input_stop_requested` is the user's/manual stop proxy; it is not an
acoustically measured last-speech sample or a production VAD-close event.

## Result

- Turns: 2
- Complete turns: 1
- Turns with queue underflow: 0
- Total underflows: 0
- Native Android TTS turns: 1
- Native TTS playback-completed turns: 1
- Native TTS playback-failed turns: 0
- Native TTS turns cancelled/flushed by user lifecycle: 0
- Turns whose ASR executed on NPU: 1
- Turns that reached the generation cap: 0
- Turns with output-quality warnings: 0
- Turns with repetition warnings: 0

| Metric | Count | Median (ms) | P95 (ms) | Max (ms) |
|---|---:|---:|---:|---:|
| `manual_input_stop_to_request_start` | 0 | — | — | — |
| `manual_input_stop_to_wav_ready` | 0 | — | — | — |
| `request_to_complete` | 1 | 15058.4 | 15058.4 | 15058.4 |
| `request_to_first_buffer_queued_event` | 0 | — | — | — |
| `request_to_first_pcm_sse` | 0 | — | — | — |
| `request_to_first_text_sse` | 1 | 1870.5 | 1870.5 | 1870.5 |
| `request_to_first_voiced_audio_scheduled` | 0 | — | — | — |
| `request_to_first_voiced_output_head_estimate` | 0 | — | — | — |
| `request_to_native_tts_first_audio_start_proxy` | 1 | 2329.6 | 2329.6 | 2329.6 |
| `request_to_native_tts_first_chunk_queued` | 1 | 2211.1 | 2211.1 | 2211.1 |
| `request_to_native_tts_init_ready` | 1 | 1999.4 | 1999.4 | 1999.4 |
| `request_to_native_tts_last_utterance_done` | 1 | 15057.3 | 15057.3 | 15057.3 |
| `request_to_native_tts_playback_terminal` | 1 | 15058.4 | 15058.4 | 15058.4 |
| `request_to_native_tts_queue_closed` | 1 | 2504.2 | 2504.2 | 2504.2 |
| `request_to_native_tts_utterance_start` | 1 | 2329.3 | 2329.3 | 2329.3 |
| `request_to_output_head` | 0 | — | — | — |
| `request_to_scheduled_first_buffer` | 0 | — | — | — |
| `request_to_tts_text_sse_complete` | 1 | 2504.5 | 2504.5 | 2504.5 |

The WebAudio output-head timestamp excludes the downstream Android
mixer/HAL, DAC, speaker, and acoustic propagation. External recording is
still required for the final audible-onset claim. Likewise, Android TTS
`UtteranceProgressListener.onStart` is only a native playback-start proxy;
it is not a microphone-observed acoustic onset.
For native TTS, `tts_text_sse_complete` means only that model text ended,
and `native_tts_queue_closed` means only that no more clauses will be
submitted. A turn is complete only when `native_tts_playback_terminal`
has status `completed`.
