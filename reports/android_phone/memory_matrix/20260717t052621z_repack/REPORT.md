# Android repack memory matrix

Each row is one fresh model-process launch followed by one request. Values are exploratory single-run measurements; they are not latency distributions.

| Configuration | Args | Idle RSS MiB | Active RSS MiB | VmHWM MiB | Anonymous MiB | Total ms | Exact ASR |
|---|---|---:|---:|---:|---:|---:|---|
| repack_default | `-c 512` | 1646.4 | 1827.4 | 1826.7 | 1119.2 | 1635.3 | True |
| no_repack | `-c 512 --no-repack` | 1106.9 | 1279.8 | 1278.7 | 562.3 | 2523.8 | True |

For failed configurations, inspect that configuration's `failure.json`. Every successful configuration retains its smaps tables and server load log.
