# SenseVoice exact-SM8850 AI Hub profile

Status: **strict-NPU compatibility pass** on 2026-07-23.

## Purpose

Profile the checksum-pinned upstream SenseVoiceSmall INT8 ten-second QNN
context on the account-owned Snapdragon 8 Elite Gen 5 QRD (Android 16,
SM8850, HTP v81). This is a lightweight compatibility, latency, memory, and
operator-placement test. It does not run the model locally and it does not
claim transcript equivalence because the release archive has no matching
source graph or local QNN golden.

## Pinned input

- Context: `model.bin`, 254,193,664 bytes.
- SHA-256:
  `522763135f536bc43e9d122bac39254f49df4afbb003462d6d8bcdb77051de30`.
- Upstream generator: QNN SDK `2.40.0.251030`.
- Fixed input policy: up to ten seconds of audio, padded or truncated by the
  Sherpa integration.

The binary remains outside the repository. Persistent AI Hub state and raw
profile outputs are written to an external work directory and only sanitized,
small result files will be copied here.

The first profile request selected QAIRT API version `2.40` to match the
context generator. AI Hub rejected that request before creating a profile job:
`QAIRT SDK version 2.40 is not valid`. The upload itself completed as model
`mmxkykorn`. A second, separately persisted compatibility run reuses that
upload and requests the account's supported QAIRT `2.45`; any pass will be
reported explicitly as a QNN-2.40-context-on-QAIRT-2.45 compatibility result,
not an exact-runtime match.

The QAIRT 2.45 profile is persisted as job `j5w1ov24g`. It was submitted from
detached tmux after the job ID was written to external `state.json`; polling can
resume by ID without another upload or submission.

## Result

| Field | Measured result |
|---|---:|
| AI Hub profile job | `j5w1ov24g` |
| Estimated inference latency | **10.871 ms** |
| Estimated inference peak memory | **126.223 MiB** |
| First context load | 159.548 ms |
| Warm context load | 159.479 ms |
| Hexagon/HTP runtime operations | **2,597 / 2,597** |
| CPU fallback | **0** |
| Unknown/unsupported placement | **0** |

The twenty profiler iterations ranged from 10.871 to 12.189 ms. AI Hub
accepted the QNN 2.40-generated context under QAIRT 2.45 and reported every
runtime operation on Hexagon/HTP. This is a strong compatibility and latency
result for the exact account target.

The machine-readable summary is
[`aihub_v81_profile_result.json`](aihub_v81_profile_result.json).

## Acceptance labels

- `profile_passed_strict_npu`: the context loads on the exact QRD and every
  reported runtime operation is placed on Hexagon/HTP with zero CPU or unknown
  fallback.
- `profile_completed_with_fallback_or_unknown`: the job runs but placement is
  not fully NPU.
- `profile_failed`: the QNN 2.40 context is incompatible with the requested AI
  Hub runtime or another remote error is reported.

Even a strict-NPU pass remains QRD evidence, not a Nubia or glasses result.
Transcript, sustained-session, memory-residency, and end-to-end latency gates
still require the physical phone or an equivalent controlled runtime.

In particular, 10.871 ms is the context execution estimate after input
preparation and loading. It excludes audio decoding/resampling, feature
preparation performed by the host integration, token decoding, app scheduling,
VAD endpointing, transfers, and playback. The upstream archive does not include
a matching source golden for AI Hub inference, so this run cannot establish
transcript equivalence or numerical accuracy.
