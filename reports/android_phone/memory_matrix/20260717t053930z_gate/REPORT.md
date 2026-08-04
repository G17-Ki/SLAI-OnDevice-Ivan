# Android gate memory matrix

Each row is one fresh model-process launch followed by one request. Values are exploratory single-run measurements; they are not latency distributions.

| Configuration | Args | Idle RSS MiB | Active RSS MiB | VmHWM MiB | Anonymous MiB | Total ms | Exact ASR |
|---|---|---:|---:|---:|---:|---:|---|
| no_repack_default | `-c 512 --no-repack` | 1108.7 | 1348.7 | 1347.9 | 631.3 | 7872.7 | — |
| no_repack_b128_ub32 | `-c 512 --no-repack -b 128 -ub 32` | 1081.6 | 1305.9 | 1305.0 | 588.5 | 8798.5 | — |
| no_repack_b64_ub16 | `-c 512 --no-repack -b 64 -ub 16` | 1081.3 | 1303.6 | 1302.8 | 586.0 | 9524.2 | — |
| no_repack_b64_ub16_q4kv | `-c 512 --no-repack -b 64 -ub 16 -ctk q4_0 -ctv q4_0` | 1077.7 | 1301.9 | 1301.0 | 584.3 | 8862.3 | — |

For failed configurations, inspect that configuration's `failure.json`. Every successful configuration retains its smaps tables and server load log.
