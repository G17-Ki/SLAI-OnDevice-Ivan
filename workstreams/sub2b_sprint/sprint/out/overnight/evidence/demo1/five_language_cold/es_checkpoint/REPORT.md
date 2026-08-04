# Android streaming telemetry summary

Generated: 2026-08-02T22:48:51.354658+00:00

This trace uses Android elapsed-realtime timestamps. For microphone turns,
`input_stop_requested` is the user's/manual stop proxy; it is not an
acoustically measured last-speech sample or a production VAD-close event.

## Result

- Turns: 5
- Complete turns: 4
- Turns with queue underflow: 0
- Total underflows: 0
- Native Android TTS turns: 4
- Native TTS playback-completed turns: 4
- Native TTS playback-failed turns: 0
- Native TTS turns cancelled/flushed by user lifecycle: 0
- Turns whose ASR executed on NPU: 2
- Turns that reached the generation cap: 0
- Turns with output-quality warnings: 0
- Turns with repetition warnings: 0

| Metric | Count | Median (ms) | P95 (ms) | Max (ms) |
|---|---:|---:|---:|---:|
| `manual_input_stop_to_request_start` | 0 | — | — | — |
| `manual_input_stop_to_wav_ready` | 0 | — | — | — |
| `request_to_complete` | 4 | 8697.9 | 15605.1 | 15605.1 |
| `request_to_first_buffer_queued_event` | 0 | — | — | — |
| `request_to_first_pcm_sse` | 0 | — | — | — |
| `request_to_first_text_sse` | 4 | 1426.6 | 2348.7 | 2348.7 |
| `request_to_first_voiced_audio_scheduled` | 0 | — | — | — |
| `request_to_first_voiced_output_head_estimate` | 0 | — | — | — |
| `request_to_native_tts_first_audio_start_proxy` | 4 | 2841.3 | 4437.5 | 4437.5 |
| `request_to_native_tts_first_chunk_queued` | 4 | 1959.5 | 2648.4 | 2648.4 |
| `request_to_native_tts_init_ready` | 4 | 1820.6 | 2646.8 | 2646.8 |
| `request_to_native_tts_last_utterance_done` | 4 | 8697.0 | 15604.1 | 15604.1 |
| `request_to_native_tts_playback_terminal` | 4 | 8697.9 | 15605.1 | 15605.1 |
| `request_to_native_tts_queue_closed` | 4 | 2082.0 | 2945.0 | 2945.0 |
| `request_to_native_tts_utterance_start` | 4 | 2840.8 | 4437.0 | 4437.0 |
| `request_to_output_head` | 0 | — | — | — |
| `request_to_scheduled_first_buffer` | 0 | — | — | — |
| `request_to_tts_text_sse_complete` | 4 | 2074.1 | 2945.3 | 2945.3 |

The WebAudio output-head timestamp excludes the downstream Android
mixer/HAL, DAC, speaker, and acoustic propagation. External recording is
still required for the final audible-onset claim. Likewise, Android TTS
`UtteranceProgressListener.onStart` is only a native playback-start proxy;
it is not a microphone-observed acoustic onset.
For native TTS, `tts_text_sse_complete` means only that model text ended,
and `native_tts_queue_closed` means only that no more clauses will be
submitted. A turn is complete only when `native_tts_playback_terminal`
has status `completed`.
