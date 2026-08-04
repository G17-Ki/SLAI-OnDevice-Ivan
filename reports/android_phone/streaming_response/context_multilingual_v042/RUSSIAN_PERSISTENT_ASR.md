# Russian persistent ASR investigation

Date: 2026-07-24
Scope: read-only model/source investigation plus a separate reproducible server
patch; no UI or current companion-service files changed; no ADB used.

## Decision

A persistent Russian CPU server is feasible and is the correct next latency
experiment. A client-only trick is not deterministic. Use the patched
Sherpa-ONNX WebSocket server with `--duplicate-each-stream=true`, keep one
connection open across turns, and return only the first result from each
real/shadow pair.

The Android binary compiled successfully for arm64-v8a. Physical latency and
quality remain unclaimed until it is deployed to the phone.

## Facts established

1. The Russian model exposes dynamic batch `N` in all three ONNX graphs:
   encoder `[N,T,80]`, decoder `[N,2]`, and joiner inputs `[N,512]`.
   Therefore the model format does not declare a batch-two requirement.
2. With Sherpa-ONNX 1.13.4 and ONNX Runtime 1.27.0 on the Apple M2, the exact
   INT8 model produced correct text for batch one and duplicate batch two:

   | Input | Batch | Result | Decode time |
   |---|---:|---|---:|
   | upstream `test_wavs/0.wav` | 1 | `я тебя люблю` | 0.107 s |
   | same WAV duplicated | 2 | two identical `я тебя люблю` results | 0.187 s |
   | upstream `test_wavs/1.wav` | 1 | coherent Russian sentence | 0.075 s |

3. The observed Android CLI behavior is different: a one-item batch returns an
   empty transcript, while two identical items return two identical correct
   transcripts. The current HTTP companion consequently invokes the CLI with
   the WAV twice.
4. The current physical companion smoke pays cold-process cost on every
   request. The recorded explicit Russian request took 4.331287 seconds for
   `test_wavs/0.wav`; explicit English/Mandarin QNN requests took about
   1.13--1.16 seconds. This makes process persistence material.
5. Upstream `OfflineWebsocketDecoder::Decode()` drains whatever is already in
   its queue and immediately calls `DecodeStreams(..., size)`. It does not wait
   to reach `--max-batch-size`. Sending duplicate requests from a client can
   therefore race with the worker and still execute as two batch-one calls.
6. The server protocol already supports multiple utterances sequentially on a
   single connection. Model construction occurs once in the decoder
   constructor, so keeping that connection/server alive removes repeated model
   initialization.

## Cause boundary

The evidence rules out a static model-shape limitation and a generic
Sherpa-ONNX batch-one bug. It localizes the fault to the Android arm64 inference
path used by this build.

The leading hypothesis is a batch-one quantized kernel issue. The encoder
contains 264 `DynamicQuantizeLinear` and 290 `MatMulInteger` operations; the
INT8 joiner contains one of each. During greedy transducer decoding the joiner
operates at the active packed batch size. Duplicating an equal-length stream
keeps that active size even and never below two. This hypothesis is strong but
not proven without an Android operator-level A/B, such as a float joiner or a
different ONNX Runtime build.

## Implemented experiment

Artifacts:

- `scripts/russian_asr_websocket/sherpa_onnx_v1.13.4_duplicate_stream.patch`
- `scripts/russian_asr_websocket/build_android_server.sh`
- `scripts/russian_asr_websocket/persistent_client.py`
- `scripts/test_russian_asr_websocket.py`
- `scripts/russian_asr_websocket/README.md`

The opt-in server flag duplicates every real stream internally. Duplicating
all streams, rather than only a sole queued item, matters: two different-length
real streams can still leave an active packed batch of one at the longer
stream's tail. Real/shadow pairs remain equal length and prevent that state.

Compile evidence:

```text
[1/3] offline-websocket-server.cc.o
[2/3] offline-websocket-server-impl.cc.o
[3/3] sherpa-onnx-offline-websocket-server
```

Validated Android ELF before stripping:

```text
arm64 aarch64, Android linker /system/bin/linker64
embedded option: duplicate-each-stream
embedded log: real size: %d, decode size: %d
```

Staged external diagnostic artifact:

```text
~/.cache/lmf-demo/russian-asr-websocket/output/
  sherpa-onnx-1.13.4-android-arm64/
```

Hashes from this build:

| File | SHA-256 |
|---|---|
| `sherpa-onnx-offline-websocket-server` | `5c5941c0b8dc0aa4e2198d0ff50df4dde3b7cd41f0c3b1c07d5e0772cb6d6464` |
| `libonnxruntime.so` | `994848008526a934dfb579ac773b00e5867929234852b061005d45aacaee9533` |

## Physical acceptance gate

Deploy only as a separately named diagnostic route, then require:

1. stock server batch one reproduces empty text on the same fixture;
2. patched server returns expected Cyrillic text on that fixture;
3. five sequential utterances use one connection and all remain non-empty;
4. record first-turn and warm-turn median/P95 latency separately;
5. compare warm latency against the 4.331-second cold companion result;
6. record RSS before first decode and after five turns;
7. verify the server remains available after USB removal;
8. fail closed and retain the current CLI duplicate route if any transcript is
   empty, non-Cyrillic for the Russian fixture, or unstable across turns.

## Follow-up optimization

If the persistent duplicate route is correct but too expensive, test a
dequantized/float joiner or a different Android ONNX Runtime release. A float
joiner A/B is particularly diagnostic because the archived joiner has only one
`DynamicQuantizeLinear` + `MatMulInteger` pair. Passing batch one with that
single change would remove the need to duplicate the much larger encoder
workload.
