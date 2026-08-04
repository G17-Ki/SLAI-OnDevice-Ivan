# X-ASR 480 ms EN/ZH CPU quality gate

Date: 2026-07-23 CST<br>
Decision: **retain only as an English streaming challenger; reject for Mandarin
and reject as the shared EN/ZH replacement**

Machine-readable evidence:
[`xasr_480ms_en_zh_gate_result.json`](xasr_480ms_en_zh_gate_result.json).

## What was tested

- Model: X-ASR EN/ZH 480 ms streaming Zipformer transducer, Sherpa-ONNX INT8
  encoder and joiner plus FP32 decoder.
- Release asset:
  [`sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2`](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2),
  GitHub asset 460926925, 133,895,136 bytes, SHA-256
  `fa5f63d618e5a01526e275a358bb7772e403f84808a4769fba52cffd8160bf74`.
- Upstream project: [`Gilgamesh-J/X-ASR`](https://github.com/Gilgamesh-J/X-ASR),
  Apache-2.0.
- Runtime: Sherpa-ONNX 1.13.4 on an Apple M2 CPU, exactly two inference
  threads, greedy search.
- Fixture: the same three English and three Mandarin Common Voice clips used
  by the existing SenseVoice and public Zipformer gates.
- Protocol: one untimed warm-up, three timed final passes per clip, a 660 ms
  zero tail, and one 100 ms-increment streaming replay per clip. Audio decode
  is outside model timing. See the JSON for every run and audio/model hash.

The reproducible invocation is:

```bash
python scripts/zipformer_streaming_asr/benchmark.py \
  --model-dir "$XASR_MODEL_DIR" \
  --dataset-dir "$COMMON_VOICE_FIXTURE_DIR" \
  --manifest "$COMMON_VOICE_MANIFEST" \
  --output reports/android_phone/streaming_response/zipformer_v81/xasr_480ms_en_zh_gate_result.json \
  --allow-unpinned-model
```

`--allow-unpinned-model` is required because this is intentionally a different
checkpoint from the harness's pinned Qualcomm-family Zipformer reference. The
result records all actual and expected hashes and does not assign the pinned
checkpoint's provenance to X-ASR.

Model checksums:

| Component | Bytes | SHA-256 |
|---|---:|---|
| INT8 encoder | 155,278,641 | `908596dcc137a73b95be908ca55e88caa1b3dbbe8027c171615f4b0609c5eb1e` |
| FP32 decoder | 11,309,084 | `a1cbc9eac2d5e3fb6617a218c67ad6daaa7f4e0fd225f08b2c22ab0413c8c257` |
| INT8 joiner | 2,581,422 | `aedb7fa697b2ab43f20499826fff7c997eea7d67db77be97769aeeeb726e63b3` |
| Tokens | 58,806 | `b818a60878b9aae978cbb8ad594acbd403d76d1af2e31ef4197c84e2dbdba27c` |

## Result and routing decision

| Host diagnostic | EN WER | ZH CER | Same-six final median/P95 | Incremental peak RSS | Streaming partials |
|---|---:|---:|---:|---:|---|
| SenseVoiceSmall INT8 | 15.8% (3/19) | **33.3% (11/33)** | **125/177 ms** | 621 MiB maximum across its language workers | No |
| **X-ASR 480 ms INT8** | **15.8% (3/19)** | 51.5% (17/33) | 288/374 ms | **428 MiB** | Yes |

The latency and memory figures are useful host diagnostics, not an exact
apples-to-apples runtime comparison: SenseVoice is an offline CTC model and its
benchmark used per-language workers, while X-ASR is one streaming transducer
with a separate incremental replay. They do use the same six audio clips and
two CPU threads.

- **English — retain for continued streaming evaluation, not as the default
  yet.** X-ASR exactly tied SenseVoice's aggregate WER, emitted nonempty output
  for all three English clips, and was deterministic across all repetitions.
  It also exposes partials, which SenseVoice does not. The first English
  partial appeared after 1.6, 2.1, or 3.1 seconds of supplied audio, so it does
  not yet pass the stricter 1.5-second partial target. Its final host path was
  also slower than SenseVoice. The six-clip tie is enough to keep X-ASR as an
  English streaming challenger, but not enough to replace the current route.
- **Mandarin — reject.** CER was 51.5%, substantially worse than SenseVoice's
  33.3%, and the one-character `否` clip decoded empty. This fails both the
  quality and no-empty-output gates.
- **Shared bilingual route — reject.** The Mandarin regression rules out X-ASR
  as the common EN/ZH replacement even though its English result is promising.

Per-clip evidence:

| Language | Reference | X-ASR output | First nonempty partial |
|---|---|---|---:|
| EN | `six` | `Six` | 3.1 s |
| EN | Joe Keaton sentence | `Joe Keaton disapproved of films and Buster also had reservations about the median` | 1.6 s |
| EN | `She'll be all right.` | `Should be all right` | 2.1 s |
| ZH | `宋朝末年年间定居粉岭围。` | `宋朝末年间定居粉令为` | 2.1 s |
| ZH | `否` | empty | none |
| ZH | `黑身准裂腹鱼为辐鳍鱼纲鲤形目鲤科的其中一种。` | `非真准列赋予为负几于刚离行木理科的其中一种` | 2.1 s |

## Reproduction against the exploratory run

The canonical harness exactly reproduced the exploratory run's six final
transcripts, first-partial observations, EN WER, and ZH CER. Model load was also
effectively unchanged: 1,839.4 ms versus 1,838.4 ms.

Host-only measurements varied: canonical final latency was 288/374 ms versus
254/351 ms median/P95, while canonical peak incremental RSS was 428 MiB versus
530 MiB. The exploratory run used Python 3.13.1 and NumPy 2.4.6; the canonical
run used the existing pinned gate environment with Python 3.11.11 and NumPy
1.26.4. Because model hashes and transcripts are identical, these differences
should be treated as environment and run variance, not as a model change.

## Limits

This is a six-clip Apple M2 CPU diagnostic, not a statistically powered quality
study and not phone/QNN evidence. Before promotion, the English route still
needs a larger noisy and accented set, endpointed live-microphone testing,
phone RSS/thermal measurements, and QNN operator-placement plus transcript
equivalence checks.
