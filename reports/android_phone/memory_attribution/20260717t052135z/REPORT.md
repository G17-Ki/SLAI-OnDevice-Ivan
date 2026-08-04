# Android LFM memory attribution

Created: 2026-07-17T05:21:44.292233+00:00

All values are KiB-derived MiB from `/proc/<pid>/smaps`. The active capture is the highest-RSS sample observed while one request was executing.

| Capture | smaps RSS | VmRSS | VmHWM | PSS |
|---|---:|---:|---:|---:|
| idle | 1690.8 | 1690.3 | 1690.3 | 1688.0 |
| active_peak | 1749.2 | 1748.2 | 1748.2 | 1746.3 |
| after | 1745.6 | 1744.7 | 1748.2 | 1742.8 |

## Active-peak attribution

| Category | RSS MiB | PSS MiB |
|---|---:|---:|
| anonymous | 1041.2 | 1041.2 |
| gguf_weights | 697.6 | 697.6 |
| shared_libraries | 9.2 | 7.5 |
| other_file_backed | 0.9 | 0.1 |
| device_mappings | 0.3 | 0.0 |
| special_mappings | 0.0 | 0.0 |

## Per-GGUF active residency

| File | RSS MiB | PSS MiB |
|---|---:|---:|
| `LFM2.5-Audio-1.5B-Q4_0.gguf` | 652.3 | 652.3 |
| `tokenizer-LFM2.5-Audio-1.5B-Q4_0.gguf` | 45.4 | 45.4 |

## Request

- Mode: `chat`.
- Input: `question.wav`.
- First text: 320.4 ms.
- First audio: 434.1 ms.
- Total: 4601.6 ms.

The raw peak smaps file has app-private installation paths normalized. `mapping_groups.csv` contains every grouped mapping for independent checking.
