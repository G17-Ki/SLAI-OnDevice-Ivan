# Failures and recoveries

## FLEURS dataset-server probe

The pinned upstream revision was
`google/fleurs@70bb2e84b976b7e960aa89f1c648e09c59f894dd` (CC-BY-4.0).
The Hugging Face dataset-server request failed verbatim with:

```text
Parquet error: Scan size limit exceeded: attempted to read 401715275 bytes, limit is 300000000 bytes

Make sure that

1. the Parquet files contain a page index to enable random access without loading entire row groups2. otherwise use smaller row-group sizes when serializing the Parquet files
```

The 64,825,504-byte
`albertvillanova/dummy_fleurs@ebde5a321e153f84d7e56001e5a3f95faf9e938e`
archive was then downloaded. It contained 409 entries, all directories or TSV
metadata; it contained no audio. It was therefore unsuitable for ASR inference.

Recovery: use the pinned 6,880,938-byte Common Voice 17 debug archive. It has
three real clips for each of 73 languages and explicitly declares CC0-1.0.

## Initial Faster-Whisper Base download

The first single-stream download failed verbatim with:

```text
Read error at byte 1332763/145217532 (Operation timed out). Retrying.
Unable to establish SSL connection.
```

Recovery: resumable HTTP range downloads assembled the exact model payload.
The completed `model.bin` SHA-256 is
`d01c3014881c9c6f3133c182f3d2887eb6ca1c789a7538c5c007196857a0a6a9`,
matching the pinned Hub artifact.

## Scope limitation

No phone or AI Hub inference was run by this benchmark. Mac timings are host
evidence only. The v81 numbers in `qai_hub_v81_reference.json` are published
Qualcomm scorecard values extracted from `qai-hub-models==0.58.0`.
