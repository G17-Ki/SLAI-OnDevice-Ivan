# Qualcomm AI Hub component results

Runs use AI Hub QCS8550 (Proxy), Android 12, Hexagon v73. Placement is transfer evidence; latency and memory are not AR1 measurements.

| Component | Runtime | Status | NPU | CPU | Other | Latency ms | Peak MB | Golden comparison |
|---|---|---|---:|---:|---:|---:|---:|---|
| backbone-attention-cached-decode | strict-npu | passed | 61 | 0 | 0 | 2.452 | 87.008 | output: pass=True, max=0.0006004, cos=1.000000; updated_key_cache: pass=True, max=0.002435, cos=1.000000; updated_value_cache: pass=True, max=5.817e-05, cos=1.000000 |
| backbone-attention-prefill | strict-npu | passed | 62 | 0 | 0 | 2.444 | 91.547 | output: pass=True, max=0.001128, cos=1.000000 |
| backbone-conv-cached-decode | strict-npu | passed | 32 | 0 | 0 | 2.695 | 91.883 | output: pass=True, max=0.0003621, cos=1.000000; updated_conv_cache: pass=True, max=0.0002784, cos=1.000000 |
| backbone-conv-prefill | strict-npu | passed | 36 | 0 | 0 | 2.693 | 92.242 | output: pass=True, max=0.000857, cos=1.000000 |
| depth-decoder | ort-diagnostic | passed | 4024 | 0 | 0 | 26.610 | 379.645 | tokens: exact=True; next_audio_embedding: pass=True, max=7.626e-05, cos=1.000000 |
| depth-decoder | strict-npu | passed | 2953 | 0 | 0 | 25.540 | 91.980 | tokens: exact=True; next_audio_embedding: pass=True, max=7.626e-05, cos=1.000000 |
| detokenizer-t4 | ort-diagnostic | numerical_mismatch | 445 | 0 | 0 | 2.521 | 170.543 | log_abs: pass=False, max=9.916, cos=0.419949; angle: pass=False, max=95.15, cos=0.015959 |
| detokenizer-t4 | strict-npu | numerical_mismatch | 371 | 0 | 0 | 1.880 | 87.984 | log_abs: pass=False, max=7.942, cos=0.490982; angle: pass=False, max=86.99, cos=-0.023606 |
| detokenizer-t8 | ort-diagnostic | numerical_mismatch | 445 | 0 | 0 | 2.798 | 195.230 | log_abs: pass=False, max=10.43, cos=0.846288; angle: pass=False, max=100.8, cos=0.280647 |
| detokenizer-t8 | strict-npu | numerical_mismatch | 371 | 0 | 0 | 2.577 | 87.395 | log_abs: pass=False, max=10.15, cos=0.870809; angle: pass=False, max=73.85, cos=0.264855 |
| fastconformer | ort-diagnostic | numerical_mismatch | 1232 | 0 | 0 | 4.945 | 306.871 | adapted: pass=False, max=0.08194 |
| fastconformer | strict-npu | numerical_mismatch | 1215 | 0 | 0 | 4.786 | 87.004 | adapted: pass=False, max=0.08194 |
