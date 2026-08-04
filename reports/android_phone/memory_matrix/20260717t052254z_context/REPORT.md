# Android context memory matrix

Each row is one fresh model-process launch followed by one request. Values are exploratory single-run measurements; they are not latency distributions.

| Configuration | Args | Idle RSS MiB | Active RSS MiB | VmHWM MiB | Anonymous MiB | Total ms | Exact ASR |
|---|---|---:|---:|---:|---:|---:|---|
| ctx_512 | `-c 512` | 1646.3 | 1702.1 | 1701.0 | 994.1 | 482.9 | True |
| ctx_1024 | `-c 1024` | 1654.3 | 1708.9 | 1708.1 | 1001.0 | 528.3 | True |
| ctx_2048 | `-c 2048` | 1666.9 | 1723.5 | 1722.8 | 1015.6 | 496.5 | True |
| ctx_4096 | `-c 4096` | 1690.8 | 1745.0 | 1744.3 | 1037.1 | 511.3 | True |
| ctx_8192 | `-c 8192` | 1739.8 | 1795.2 | 1795.0 | 1087.1 | 492.5 | True |

For failed configurations, inspect that configuration's `failure.json`. Every successful configuration retains its smaps tables and server load log.
