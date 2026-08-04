# Android context memory matrix

Each row is one fresh model-process launch followed by one request. Values are exploratory single-run measurements; they are not latency distributions.

| Configuration | Args | Idle RSS MiB | Active RSS MiB | VmHWM MiB | Anonymous MiB | Total ms | Exact ASR |
|---|---|---:|---:|---:|---:|---:|---|
| ctx_512 | `-c 512` | 1646.1 | 1827.3 | 1826.4 | 1119.2 | 1639.0 | True |
| ctx_1024 | `-c 1024` | 1654.4 | 1834.3 | 1833.5 | 1126.4 | 1651.2 | True |
| ctx_2048 | `-c 2048` | 1666.9 | 1847.9 | 1847.1 | 1139.9 | 1631.2 | True |
| ctx_4096 | `-c 4096` | 1690.7 | 1870.6 | 1869.9 | 1162.8 | 1654.3 | True |
| ctx_8192 | `-c 8192` | 1739.7 | 1920.9 | 1920.0 | 1212.7 | 2579.2 | True |

For failed configurations, inspect that configuration's `failure.json`. Every successful configuration retains its smaps tables and server load log.
