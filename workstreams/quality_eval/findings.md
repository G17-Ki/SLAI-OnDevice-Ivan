# Local quality audit

This audit reuses the pinned local and Qualcomm AI Hub artifacts; it submits no cloud jobs.

## Decision summary

- Full BF16 ASR normalized exact match: **True** (WER 0.000, CER 0.000).
- Strict-NPU depth-decoder token match: **True**.
- FastConformer feature cosine: **0.998424**; NRMSE **0.0698**.
- All 10 remote feature frames retrieve the matching local frame: **100%** accuracy.
- Frozen-backbone common-context top-1 agreement: **16/17 (94.1%)**; the first prediction matches, but the first divergence occurs at step 1.
- The golden choice remains in the remote branch's top 5 on **100.0%** of compared steps.
- Generated response WAVs are finite 24 kHz mono audio with hard-clip rates 0.000% and 0.000%.
- NPU detokenizer reconstruction remains unusable: T=4 SI-SDR **-22.62 dB**, T=8 SI-SDR **-28.67 dB**.

## What the new downstream check changes

The AI Hub FastConformer output is close enough to preserve the initial LFM token, but not close enough to claim exact sequence equivalence. Under a shared golden-token context, a later top-1 choice changes even though the competing distributions remain close.

Golden path: `It seems like your message was cut off. How can I assist you today?<|im_end|>`

Remote argmax under that same context: `It looks like your message was cut off. How can I assist you today?<|im_end|>`

This makes downstream validation after quantization mandatory. Feature cosine alone would have hidden a real decision-boundary crossing.

## Measurement boundaries

- The FastConformer downstream probe contains only the first 80 mel frames of one question.wav sample. It is a sensitivity test, not a corpus benchmark.
- The downstream sensitivity comparison runs the same pinned BF16 LFM backbone on CPU for both branches; only the ten injected audio embeddings differ.
- The exact ASR match is the full BF16 Colab golden versus Liquid Audio's published reference; it does not test AI Hub FastConformer substitution.
- Generated-response audio has no clean reference. The reported WAV checks establish technical integrity, not naturalness, intelligibility, MOS, PESQ, or STOI.
- Detokenizer waveform fidelity is reported separately and remains a blocker for NPU use.
- AI Hub used QCS8550 Proxy / Hexagon v73, not AR1; none of these are AR1 quality or latency claims.

## Reproduce

```bash
work/venvs/lfm/bin/python outputs/lfm-feasibility/scripts/evaluate_quality_locally.py
```

The machine-readable details, including every compared token distribution, are in `results.json`.
