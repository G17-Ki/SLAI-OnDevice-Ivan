# LFM2.5-Audio full BF16 golden on Colab L4

## Reproducibility

- Model: `LiquidAI/LFM2.5-Audio-1.5B`
- Model revision: `c362a0625dfe45aa588dce5f0ada28a7e5707628`
- Liquid Audio source commit: `19e65845923a7f136442c95137884ec61eb386aa`
- GPU: NVIDIA L4, BF16 supported
- PyTorch / torchaudio: 2.8.0 + CUDA 12.8
- Transformers: 4.56.1
- Seed: 20260715

## Model load and memory

| Metric | Result |
|---|---:|
| Load time | 16.104 s |
| Steady allocated after load | 3.034 GB |
| Peak allocated during load | 6.216 GB |
| Reserved CUDA memory | 6.388 GB |

The unquantized BF16 system is therefore far outside a 2 GB glasses envelope.
This confirms that quantization and component/runtime partitioning are required,
not optional optimization work.

## Sequential ASR

| Metric | Result |
|---|---:|
| Input duration | 18.356 s |
| Preprocess | 0.559 s |
| First token | 0.502 s |
| Generation | 1.593 s |
| Text tokens | 58 |
| Generation rate | 36.40 tokens/s |

The transcript matches the official Liquid Audio README reference for
`asr.wav`, with only the decoded `<|im_end|>` control suffix retained in this
raw golden.

## Persistent two-turn interleaved interaction

| Metric | Turn 1 | Turn 2 |
|---|---:|---:|
| First text | 97.97 ms | 56.49 ms |
| First audio token | 339.67 ms | 225.87 ms |
| Generated steps | 190 | 151 |
| Generation time | 12.070 s | 9.238 s |
| Generated steps/s | 15.74 | 16.35 |
| Text tokens | 41 | 38 |
| Valid audio frames | 148 | 112 |
| Decoded waveform | 11.84 s | 8.96 s |

The same chat state was retained across both turns. The second user turn was:

> My business specializes in chairs. Can you give me something related to that?

The second answer correctly adapted the woodworking slogan suggestions to
chairs, demonstrating functional persistent-session context.

## Important deployment observations

- Full official BF16 inference is functionally healthy on L4.
- The persistent-session usage mode works across multiple turns.
- The generation loop sustains roughly 16 interleaved steps/s in this setup.
- The audio detokenizer reports `torch.float32`, even though the main model is
  BF16. This is a separate precision/deployment target.
- L4 timing is a golden-reference measurement, not an AR1 performance estimate.
- Compact token goldens, transcripts, metrics, and response WAVs are preserved
  under `reports/colab_bf16/`; model weights and caches were not downloaded.
