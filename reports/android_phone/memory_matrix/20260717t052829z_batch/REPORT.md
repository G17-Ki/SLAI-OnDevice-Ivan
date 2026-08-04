# Android batch memory matrix

Each row is one fresh model-process launch followed by one request. Values are exploratory single-run measurements; they are not latency distributions.

| Configuration | Args | Idle RSS MiB | Active RSS MiB | VmHWM MiB | Anonymous MiB | Total ms | Exact ASR |
|---|---|---:|---:|---:|---:|---:|---|
| batch_default | `-c 512 --no-repack` | 1107.1 | 1280.0 | 1278.9 | 562.3 | 2491.7 | True |
| batch_1024_ubatch_256 | `-c 512 --no-repack -b 1024 -ub 256` | 1105.3 | 1268.0 | 1267.0 | 550.6 | 3971.6 | True |
| batch_512_ubatch_128 | `-c 512 --no-repack -b 512 -ub 128` | 1091.3 | 1249.8 | 1249.0 | 532.3 | 4297.7 | True |
| batch_256_ubatch_64 | `-c 512 --no-repack -b 256 -ub 64` | 1085.2 | 1240.2 | 1239.1 | 522.7 | 4286.9 | True |

For failed configurations, inspect that configuration's `failure.json`. Every successful configuration retains its smaps tables and server load log.
