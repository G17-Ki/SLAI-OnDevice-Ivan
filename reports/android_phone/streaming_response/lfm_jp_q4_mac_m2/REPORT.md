# LFM2.5-Audio-1.5B-JP Q4 feasibility on Mac M2

## Decision

The official Japanese checkpoint is a real Japanese-capable candidate, not a
cosmetic language variant. On Liquid AI's pinned 14.13-second Japanese sample,
it produced the correct normalized transcript in every measured run. The
English checkpoint control produced insertion-heavy Latin-script gibberish on
the same file.

For a **single, freshly initialized interleaved turn**, the Japanese checkpoint
also clears the latency goal comfortably: first PCM arrived in **287.5 ms
median / 307.1 ms P95** after the complete audio file was submitted. Adding the
Android UI's 140 ms startup cushion gives a simple scheduled-playback estimate
of **427.5 / 447.1 ms**. This is not an acoustic phone measurement and excludes
the time spent recording the user's utterance.

There is one release-blocking runner defect: the second identical interleaved
request to the same persistent server is corrupted even though the request has
`reset_context=true`. It returned mixed Chinese/Japanese nonsense and nearly
clipped PCM. Therefore:

- do not ship the current persistent Japanese interleaved server as a multi-turn
  assistant;
- the Japanese checkpoint is approved for continued ASR and standalone-TTS
  work;
- use the modular `Japanese ASR -> text LLM -> standalone TTS` path for the next
  phone experiment, or fix and validate the runner's audio-state reset;
- restarting the server for every voice turn is only a diagnostic workaround,
  not a product design.

## Reproducibility envelope

| Item | Value |
|---|---|
| Host | Apple M2, 8 GiB unified memory |
| OS | macOS 15.4.1 (24E263) |
| Runner | llama.cpp build 8256, commit `22e8cd6f6` |
| Runner build | AppleClang 17.0.0.17000013, Darwin arm64 |
| Model repository | `LiquidAI/LFM2.5-Audio-1.5B-JP-GGUF` |
| Pinned model revision | `c8410f9538b2e5695cf574dd5d3be54a4a3aed72` |
| Official sample repository | `Liquid4All/liquid-audio` |
| Pinned sample revision | `19e65845923a7f136442c95137884ec61eb386aa` |
| Server arguments | `-t 2 -tb 2 -c 4096` plus Metal auto-offload |
| Main-model context | 4,096 tokens, 48 MiB Metal KV buffer |
| Audio detokenizer context | runner still allocates 128,000 tokens and a 375 MiB Metal KV buffer |
| Observed ready-server RSS | approximately 1.44 GiB; point observation, not a peak-memory profile |

The four Q4 model files total 1,072,421,248 bytes. The download and validation
script refuses to publish a completed filename until its checksum matches.

| File | Bytes | SHA-256 |
|---|---:|---|
| `LFM2.5-Audio-1.5B-JP-Q4_0.gguf` | 695,750,848 | `d74e971956010953f5fbdf46ea8c4db6873b1b13a7893c4830f39f2ce7bba313` |
| `mmproj-LFM2.5-Audio-1.5B-JP-Q4_0.gguf` | 219,511,040 | `5730a8407b06c04efa55049a48709818fa8c5bf66921a07eb9eb94c34bd62cc4` |
| `vocoder-LFM2.5-Audio-1.5B-JP-Q4_0.gguf` | 108,986,656 | `d00983edd937e3dc53727319f5c5cf7f53c278545a3935b4affff6a230947398` |
| `tokenizer-LFM2.5-Audio-1.5B-JP-Q4_0.gguf` | 48,172,704 | `d776c8b615971ee21a2465ad573fcd4c348b568d16b83e6a7ce70ead74a889ec` |
| macOS arm64 runner archive | 7,280,738 | `e59b7be254b6386960fc1d3cf51851709220202a4fdf7f83eb29953987992c9b` |

The exact full manifest, including Android runner, README, license, and samples,
is in `scripts/lfm_jp_q4.sha256`.

## Exact request protocol

All requests used streaming SSE, temperature 0, and `reset_context=true`.

| Mode | System prompt | Modalities | Input |
|---|---|---|---|
| ASR | `Perform ASR in japanese.` | `text` | official `asr_jp.wav` |
| Interleaved | `Respond with interleaved text and audio.` | `text,audio` | official `question_jp.wav` |
| TTS | `Perform TTS in japanese.` | `audio` | `こんにちは、お元気ですか？` |

Pinned input hashes:

- `asr_jp.wav`: `48d3b3b0fba92aa0d95d26ac61e32b149de701ff8aef78bbd08996bebc2fd92a`
- `question_jp.wav`: `4ce958e1ddb6729a6be9479b4d1f003ac97e94c7a0abc28fa354b501380c3c18`

The checkpoint transcribed the question sample as:

> こんにちは。あなたのことを教えてください。

## Results

### Japanese ASR

One warm-up and three measured requests were run on the same loaded server.
Punctuation and spacing are removed by the normalized Japanese CER calculation.

| Metric | Median | P95 |
|---|---:|---:|
| First text | 401.0 ms | 410.5 ms |
| Complete transcript | 909.2 ms | 919.7 ms |
| Normalized CER | 0.0% | 0.0% |

Reference:

> この度は弊社の確認不足により多大なご迷惑をおかけしましたことを深くお詫び申し上げます。今後はこのようなことが二度と起こらないよう社内のチェック体制を徹底してまいります。

Output in all measured runs differed only by harmless commas:

> この度は、弊社の確認不足により多大なご迷惑をおかけしましたことを深くお詫び申し上げます。今後はこのようなことが二度と起こらないよう、社内のチェック体制を徹底してまいります。

### English-checkpoint Japanese control

The existing control used the English Q4 main model (SHA-256
`3583bee853be20331ca342b0593fefd8acc43fb61a41ec6f1a1dc7465823e0d8`)
and its documented `Perform ASR.` prompt on the same official Japanese sample.
It is a product control, not a strict prompt-isolation ablation.

| Metric | English checkpoint | Japanese checkpoint |
|---|---:|---:|
| First text median | 423.3 ms | 401.0 ms |
| Complete median | 2,904.7 ms | 909.2 ms |
| Normalized CER | 504.9% | 0.0% |

CER can exceed 100% when there are many insertions. The English checkpoint
repeated a long Latin-script fragment beginning `Hono ta viva...`; the Japanese
checkpoint was 3.19x faster to completion because it terminated with the
correct transcript instead of running to the 256-token cap.

### First interleaved turn after server initialization

Because of the repeated-turn defect below, these statistics combine one request
from each of three independently initialized servers. Model loading completed
before timing began.

| Metric | Median | P95 |
|---|---:|---:|
| First text | 214.0 ms | 223.3 ms |
| First PCM | 287.5 ms | 307.1 ms |
| First PCM + 140 ms UI cushion | 427.5 ms | 447.1 ms |
| Response generation complete | 5,421.6 ms | 5,509.8 ms |
| Generated audio duration | 14.06 s | 14.06 s |
| Stream generation RTF | 0.366 | 0.371 |
| Largest PCM arrival gap | 88.2 ms | 93.4 ms |

All three fresh-server runs produced identical text and identical PCM:

- PCM SHA-256: `c5317f13156ba309aff4dafc9e5bb3317d1d221e801b6478c85c8c8ef5795538`
- samples: 337,440 at 24 kHz (14.06 s)
- normalized peak: 0.5713
- normalized RMS: 0.07372

The 140 ms figure is a queue-start estimate, not a microphone-to-speaker or
acoustic measurement. The largest observed arrival gap is below the cushion,
and generation is substantially faster than real time, but the Android playback
queue still needs a physical-phone underflow test.

### Standalone Japanese TTS

One warm-up and three measured requests were run on the same server. All three
measured runs produced bit-identical PCM.

| Metric | Median | P95 |
|---|---:|---:|
| First PCM | 61.8 ms | 62.3 ms |
| First PCM + 140 ms UI cushion | 201.8 ms | 202.3 ms |
| Generation complete | 720.4 ms | 721.3 ms |
| Generated audio duration | 2.14 s | 2.14 s |
| Stream generation RTF | 0.308 | 0.309 |
| Largest PCM arrival gap | 25.1 ms | 25.8 ms |

Numerical output facts:

- PCM SHA-256: `2fc05cae419ca564ccba4f922d70021442ecba019ab71bcdba6b30763781f23f`
- samples: 51,360 at 24 kHz
- normalized peak: 0.4343
- normalized RMS: 0.06210

This establishes a fast Japanese standalone-TTS path on Mac. It does not yet
establish phone or HTP-v81 performance, and loading the complete LFM bundle only
for TTS is much heavier than Supertonic 3.

## Persistent interleaved-server defect

Two identical interleaved requests were sent sequentially to a freshly loaded
server with `reset_context=true` in both payloads.

| Observation | Turn 1 | Turn 2 |
|---|---|---|
| Text | coherent Japanese assistant introduction | mixed Chinese/Japanese nonsense |
| Audio duration | 14.06 s | 17.10 s |
| PCM peak | 0.5713 | 0.99997 |
| PCM RMS | 0.07372 | 0.47960 |
| PCM SHA-256 | `c5317f...95538` | `9a2e9f...fe059` |

Turn 2 text was:

> 紧密地下室に庵を構えた、専属の猫です。专门的食物和护专门的洗手场紧密地下室，場合があれば教えてください。

The second waveform is about 6.5x higher in RMS and reaches full scale. The
cause is not proven. The evidence is consistent with interleaved audio state
not being fully reset after a generation that reaches the 256-token limit.
ASR repeated correctly, and standalone TTS repeated with bit-identical output,
so the failure is currently scoped to the interleaved persistent-turn path.

Required gate before using this mode:

1. identify and reset the main decoder, audio-token decoder, and vocoder state;
2. run at least ten identical and ten varied turns without a process restart;
3. reject any non-Japanese script leakage, PCM clipping, or output-hash drift on
   deterministic fixtures;
4. repeat the gate on the Nubia phone and record playback-head start, gaps, RSS,
   and thermals.

## License and attribution boundary

The model weights and associated Liquid audio runtime work use **LFM Open
License v1.0**, not an unrestricted OSI-style open-source license. The agreement
defines its commercial-use threshold as annual revenue of USD 10,000,000 or
more and does not license commercial use by a legal entity above that
threshold.

Redistribution requires providing the license, marking modified files, and
retaining applicable copyright, patent, trademark, attribution, and NOTICE
material. The upstream model documentation also identifies audio components
derived from Nvidia NeMo/Canary and Kyutai Mimi; Canary and the redistributed
Mimi weights are identified there as CC-BY-4.0 material, so applicable
attribution must be retained. This repository stores hashes and download
instructions rather than republishing the model bundle.

## Reproduce

From the repository root:

```bash
export BUNDLE_DIR="$PWD/work/models/lfm25-audio-jp-q4"
./scripts/download_lfm_jp_q4.sh "$BUNDLE_DIR"

./scripts/run_lfm_jp_server.sh "$BUNDLE_DIR" 18081 \
  -t 2 -tb 2 -c 4096
```

In another terminal, run a bounded profile:

```bash
python3 scripts/profile_lfm_jp_server.py \
  --base-url http://127.0.0.1:18081/v1/chat/completions \
  --mode asr \
  --input-audio "$BUNDLE_DIR/assets/asr_jp.wav" \
  --reference-text 'この度は弊社の確認不足により多大なご迷惑をおかけしましたことを深くお詫び申し上げます。今後はこのようなことが二度と起こらないよう社内のチェック体制を徹底してまいります。' \
  --warmups 1 --runs 3 --max-tokens 256 \
  --output reports/android_phone/streaming_response/lfm_jp_q4_mac_m2/asr.json
```

The profiler writes each generated waveform below an output-specific directory,
records PCM statistics and SHA-256, and includes the exact prompt and request
parameters. Use `scripts/summarize_lfm_jp_profiles.py` to combine independently
initialized interleaved runs.

## Canonical evidence

- `official_asr_jp.json`
- `official_asr_summary.json`
- `official_interleaved_cold_01.json`
- `official_interleaved_cold_02.json`
- `official_interleaved_cold_03.json`
- `official_interleaved_cold_summary.json`
- `official_tts_jp.json`
- `official_tts_summary.json`
- `repeated_interleaved_same_server.json`
- `evidence.sha256`
- English checkpoint control: `../lfm_en_checkpoint_official_ja_asr_mac.json`

Server logs remain available in the local experiment directory for debugging,
but repository policy ignores `*.log`; the public checksum manifest covers only
the structured JSON and report files listed above.

No local Japanese model server was left running after measurement.
