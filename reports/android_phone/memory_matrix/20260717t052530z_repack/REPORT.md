# Android repack memory matrix

Each row is one fresh model-process launch followed by one request. Values are exploratory single-run measurements; they are not latency distributions.

| Configuration | Args | Idle RSS MiB | Active RSS MiB | VmHWM MiB | Anonymous MiB | Total ms | Exact ASR |
|---|---|---:|---:|---:|---:|---:|---|
| repack_default | `-c 512` | 1646.4 | 1702.4 | 1701.6 | 994.2 | 493.1 | True |
| no_repack | `-c 512 --no-repack` | 1109.3 | 1164.2 | 1163.4 | 446.5 | 816.0 | True |

For failed configurations, inspect that configuration's `failure.json`. Every successful configuration retains its smaps tables and server load log.
