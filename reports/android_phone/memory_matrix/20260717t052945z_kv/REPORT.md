# Android kv memory matrix

Each row is one fresh model-process launch followed by one request. Values are exploratory single-run measurements; they are not latency distributions.

| Configuration | Args | Idle RSS MiB | Active RSS MiB | VmHWM MiB | Anonymous MiB | Total ms | Exact ASR |
|---|---|---:|---:|---:|---:|---:|---|
| kv_f16 | `-c 512 --no-repack -ctk f16 -ctv f16` | 1106.9 | 1274.4 | 1273.8 | 556.9 | 4157.5 | True |
| kv_q8_0 | `-c 512 --no-repack -ctk q8_0 -ctv q8_0` | 1105.4 | 1277.0 | 1276.1 | 559.3 | 4137.7 | True |
| kv_q4_0 | `-c 512 --no-repack -ctk q4_0 -ctv q4_0` | 1113.2 | 1273.9 | 1273.1 | 556.3 | 4113.5 | True |

For failed configurations, inspect that configuration's `failure.json`. Every successful configuration retains its smaps tables and server load log.
