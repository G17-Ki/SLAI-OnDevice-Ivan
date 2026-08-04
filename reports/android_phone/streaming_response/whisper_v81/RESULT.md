# Whisper Small W8A16 on SM8850 / HTP v81

## Result

Qualcomm's public W8A16 Whisper Small contexts generated for the Samsung
Galaxy S26 `sm8850-ad` variant load and profile unchanged on AI Hub's exact
plain-SM8850 target. Both graphs stay entirely on Hexagon/HTP v81.

| Component | Exact-QRD latency | Estimated inference peak | HTP placement |
|---|---:|---:|---:|
| Encoder, fixed 30-second feature window | **186.277 ms** | 145.727 MiB | **2,014/2,014** |
| Decoder, one autoregressive token call | **3.932 ms** | 146.398 MiB | **2,853/2,853** |

There are zero CPU fallback and zero unsupported or unknown operations. First
and warm context-load times were 301.884/306.598 ms for the encoder and
143.951/134.014 ms for the decoder. Keep both contexts resident; load time is
not part of the steady-state component arithmetic below.

The machine-readable result is in
[`aihub_exact_qrd_profiles.json`](aihub_exact_qrd_profiles.json).

## Latency interpretation

Qualcomm describes encoder latency as time to first token and decoder latency
as the cost of each additional token. A conservative planning calculation that
also adds one decoder call is **190.209 ms**. Encoder plus 20 decoder calls is
**264.917 ms**; 50 calls is **382.877 ms**; the 200-call benchmark ceiling is
**972.677 ms**.

These are neural-component sums, not VAD-close-to-final-transcript latency.
They exclude 16 kHz audio capture, log-mel extraction, padding, transfers,
prompt/language tokens, the host decode loop, token suppression and EOS logic,
detokenization, application scheduling, thermals, and phone runtime behavior.

The fresh encoder is 11.194 ms (6.39%) slower than Qualcomm's published
Samsung Galaxy S26 value of 175.083 ms. The fresh decoder is 0.029 ms (0.73%)
faster than the published 3.961 ms per call. The exact-QRD 20-call component
sum is 264.917 ms versus the published 254.303 ms, a 10.614 ms (4.17%)
difference.

## What this changes

Whisper Small no longer has only host-CPU and neighboring-device performance
evidence. Its memory-practical W8A16 release contexts now have an account-owned
exact-QRD compatibility, latency, memory, and placement pass. For the
ES/DE/FR/IT/RU modular route, the neural ASR budget is compatible with the
project's 2-3 second first-audio target if the phone's preprocessing, decode
loop, LLM, and TTS stages remain bounded.

This does not yet displace SenseVoice for EN/JA/KO/ZH, where the existing small
quality fixture favors SenseVoice. Qualcomm publishes no numerics record for
this W8A16 package, and the archive contains no sample inputs or golden
outputs. The quantized QNN transcript path therefore still needs fixed-locale
WER/CER acceptance on real multilingual audio.

The model is also explicitly file-based, not incremental streaming ASR. Its
encoder consumes a fixed `[1, 80, 3000]` feature tensor representing up to 30
seconds. It fits the current VAD-gated interaction design, but it cannot emit
true encoder partials while the user is still speaking.

## Evidence boundary

This run proves:

- `sm8850-ad` to exact `sm8850` context portability for the two pinned files;
- exact-QRD latency and estimated per-component inference peak memory;
- strict Hexagon/HTP placement under QAIRT 2.45.

It does not prove:

- numerical or transcript equivalence to the source model;
- multilingual WER/CER after W8A16 quantization;
- end-to-end VAD-close-to-transcript latency;
- Nubia or glasses deployment/runtime compatibility;
- always-on or incremental streaming behavior.

No model was loaded, exported, compiled, trained, or run for inference on the
Mac. No AI Hub inference job was submitted because the release package has no
safe matching sample inputs.

## Reproduction

Fetch the public `qai-hub-models 0.58.0` release asset in detached tmux:

```bash
qai-hub-models fetch whisper_small_quantized \
  -r qnn_context_binary \
  -p w8a16 \
  -d "Samsung Galaxy S26"
```

Verify the ZIP SHA-256 is
`f798189b2e98c6ec139a3ad26ddc6cd2d41ee19c0ef87fab99f8d5fdc456c79e`,
extract it, and run encoder followed by decoder with
[`profile_whisper_release_context.py`](../../../../scripts/profile_whisper_release_context.py)
from a detached tmux session. The script validates metadata and component
hashes, refuses non-green macOS memory pressure, persists every model/job ID
before polling, and prevents decoder submission until the encoder passes
strict NPU placement.

Cloud jobs:

- encoder model `mq3lzg43q`, profile `jp8497lx5`;
- decoder model `mqpre1ljm`, profile `jgdz9rnr5`.
