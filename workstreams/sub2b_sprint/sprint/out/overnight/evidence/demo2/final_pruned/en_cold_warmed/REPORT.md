# Android streaming telemetry summary

Generated: 2026-08-02T19:53:59.915172+00:00

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
| `request_to_complete` | 1 | 6532.9 | 6532.9 | 6532.9 |
| `request_to_first_buffer_queued_event` | 0 | — | — | — |
| `request_to_first_pcm_sse` | 0 | — | — | — |
| `request_to_first_text_sse` | 1 | 2716.0 | 2716.0 | 2716.0 |
| `request_to_first_voiced_audio_scheduled` | 0 | — | — | — |
| `request_to_first_voiced_output_head_estimate` | 0 | — | — | — |
| `request_to_native_tts_first_audio_start_proxy` | 1 | 4040.4 | 4040.4 | 4040.4 |
| `request_to_native_tts_first_chunk_queued` | 1 | 2972.4 | 2972.4 | 2972.4 |
| `request_to_native_tts_init_ready` | 1 | 1329.1 | 1329.1 | 1329.1 |
| `request_to_native_tts_last_utterance_done` | 1 | 6531.9 | 6531.9 | 6531.9 |
| `request_to_native_tts_playback_terminal` | 1 | 6532.9 | 6532.9 | 6532.9 |
| `request_to_native_tts_queue_closed` | 1 | 2973.1 | 2973.1 | 2973.1 |
| `request_to_native_tts_utterance_start` | 1 | 4040.0 | 4040.0 | 4040.0 |
| `request_to_output_head` | 0 | — | — | — |
| `request_to_scheduled_first_buffer` | 0 | — | — | — |
| `request_to_tts_text_sse_complete` | 1 | 2973.5 | 2973.5 | 2973.5 |

The WebAudio output-head timestamp excludes the downstream Android
mixer/HAL, DAC, speaker, and acoustic propagation. External recording is
still required for the final audible-onset claim. Likewise, Android TTS
`UtteranceProgressListener.onStart` is only a native playback-start proxy;
it is not a microphone-observed acoustic onset.
For native TTS, `tts_text_sse_complete` means only that model text ended,
and `native_tts_queue_closed` means only that no more clauses will be
submitted. A turn is complete only when `native_tts_playback_terminal`
has status `completed`.
