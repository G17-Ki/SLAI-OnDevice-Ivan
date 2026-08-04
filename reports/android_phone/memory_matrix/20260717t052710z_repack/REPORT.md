# Android repack memory matrix

Each row is one fresh model-process launch followed by one request. Values are exploratory single-run measurements; they are not latency distributions.

| Configuration | Args | Idle RSS MiB | Active RSS MiB | VmHWM MiB | Anonymous MiB | Total ms | Exact ASR |
|---|---|---:|---:|---:|---:|---:|---|
| repack_default | `-c 512` | 1646.1 | 1705.8 | 1705.1 | 997.9 | 3188.0 | — |
| no_repack | `-c 512 --no-repack` | 1106.9 | 1163.0 | 1162.5 | 445.4 | 5589.5 | — |

For failed configurations, inspect that configuration's `failure.json`. Every successful configuration retains its smaps tables and server load log.
