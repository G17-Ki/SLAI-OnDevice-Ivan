# Android streaming telemetry summary

Generated: 2026-08-02T18:09:00.857827+00:00

This trace uses Android elapsed-realtime timestamps. For microphone turns,
`input_stop_requested` is the user's/manual stop proxy; it is not an
acoustically measured last-speech sample or a production VAD-close event.

## Result

- Turns: 22
- Complete turns: 20
- Turns with queue underflow: 0
- Total underflows: 0
- Native Android TTS turns: 22
- Native TTS playback-completed turns: 20
- Native TTS playback-failed turns: 0
- Native TTS turns cancelled/flushed by user lifecycle: 2
- Turns whose ASR executed on NPU: 22
- Turns that reached the generation cap: 0
- Turns with output-quality warnings: 0
- Turns with repetition warnings: 0

| Metric | Count | Median (ms) | P95 (ms) | Max (ms) |
|---|---:|---:|---:|---:|
| `manual_input_stop_to_request_start` | 0 | — | — | — |
| `manual_input_stop_to_wav_ready` | 0 | — | — | — |
| `request_to_complete` | 22 | 8536.8 | 13744.9 | 13752.7 |
| `request_to_first_buffer_queued_event` | 0 | — | — | — |
| `request_to_first_pcm_sse` | 0 | — | — | — |
| `request_to_first_text_sse` | 22 | 555.3 | 655.0 | 834.5 |
| `request_to_first_voiced_audio_scheduled` | 0 | — | — | — |
| `request_to_first_voiced_output_head_estimate` | 0 | — | — | — |
| `request_to_native_tts_first_audio_start_proxy` | 22 | 1031.3 | 1221.0 | 1603.0 |
| `request_to_native_tts_first_chunk_queued` | 22 | 931.5 | 1133.6 | 1510.4 |
| `request_to_native_tts_init_ready` | 22 | 519.9 | 624.7 | 1493.4 |
| `request_to_native_tts_last_utterance_done` | 20 | 8698.8 | 13743.8 | 13751.3 |
| `request_to_native_tts_playback_terminal` | 22 | 8536.8 | 13744.9 | 13752.7 |
| `request_to_native_tts_queue_closed` | 22 | 961.8 | 1192.8 | 1527.1 |
| `request_to_native_tts_utterance_start` | 22 | 1030.8 | 1220.8 | 1602.7 |
| `request_to_output_head` | 0 | — | — | — |
| `request_to_scheduled_first_buffer` | 0 | — | — | — |
| `request_to_tts_text_sse_complete` | 22 | 961.9 | 1192.9 | 1519.7 |

The WebAudio output-head timestamp excludes the downstream Android
mixer/HAL, DAC, speaker, and acoustic propagation. External recording is
still required for the final audible-onset claim. Likewise, Android TTS
`UtteranceProgressListener.onStart` is only a native playback-start proxy;
it is not a microphone-observed acoustic onset.
For native TTS, `tts_text_sse_complete` means only that model text ended,
and `native_tts_queue_closed` means only that no more clauses will be
submitted. A turn is complete only when `native_tts_playback_terminal`
has status `completed`.
