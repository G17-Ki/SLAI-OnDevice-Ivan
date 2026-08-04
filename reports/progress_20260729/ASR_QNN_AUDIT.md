# App-owned ASR / QNN execution audit — 2026-07-29

## Result

The runtime-companion APK now executes the pinned SenseVoice context through
QNN HTP v81 from Android app UID `10514`. English, Mandarin, Japanese, and
Korean exact fixture parity plus SenseVoice language-tag checks pass, and the
Russian Zipformer CPU fallback passes. No adb/shell-owned inference process is
required.

The original EN/ZH root-cause validation completed at
`2026-07-29 09:45 CST`; the same-day final installer and full-turn runs extended
physical acceptance to JA/KO on Nubia `NX809J`, `SM8850`, Android 16.

## Root causes resolved

1. Android had to expose the public vendor dependency `libcdsprpc.so` through
   `<uses-native-library>`.
2. `libQnnHtpV81Skel.so` had to be an immutable extracted APK native library.
   Keeping the skeleton only in mutable app-private storage caused
   `QnnDevice_create` error `14001` / `QNN_DEVICE_ERROR_INVALID_CONFIG`.
3. SenseVoice's compiled context requires an exact ten-second input. The first
   app-owned success used the original short WAV and reproduced the known
   unpadded CPU variant (`...pieces of code`). `AsrHttpServer` now zero-pads
   canonical 16 kHz mono PCM input to exactly `320044` WAV bytes before QNN.
   This restored fixed-shape parity (`...pieces of gold`).

An explicit signed/unsigned process-domain patch was therefore not required.

## Physical-phone acceptance

The deployment command was:

```sh
scripts/install_app_owned_runtime.sh --serial "${ADB_SERIAL:?set ADB_SERIAL}"
```

The build was bounded to two Gradle workers and reused checksum-pinned local
caches. It installed the companion without clearing its app-private data.

Verified outputs:

| Route | Provider | Transcript | Result |
|---|---|---|---|
| English | QNN / HTP v81 | `The tribal chieftain called for the boy and presented him with 50 pieces of gold.` | exact fixed-10s parity pass |
| Mandarin | QNN / HTP v81 | `开饭时间早上9点至下午5点。` | exact fixed-10s parity pass |
| Japanese | QNN / HTP v81 | `フランス の 首都 は 田クスカ。` | exact fixture parity and `<|ja|>` tag pass |
| Korean | QNN / HTP v81 | `프랑스의의 수도는 어디인가요?` | exact fixture parity and `<|ko|>` tag pass |
| Russian | CPU Zipformer | `расскажи коротко о москве` | native-script pass |

The QNN log includes `Initializing HtpProvider`, loads
`libQnnHtp.so` plus the SM8850 context, and reports an inference RTF near
`0.009` on the English fixture. The QNN model/context and runtime hashes remain
the pinned values recorded in the existing NPU acceptance artifacts.

## Execution-aware health

`GET /health` no longer treats files on disk as proof that inference works.
Each backend now reports both artifact availability and process-lifetime
execution state. After the installer runs real fixtures, the observed state is:

```json
{
  "complete": true,
  "artifacts_complete": true,
  "app_owned": true,
  "service_uid": 10514,
  "sensevoice_qnn": {
    "ready": true,
    "execution_verified": true,
    "processor": "htp_v81",
    "input_contract": "fixed_10_seconds_zero_padded",
    "last_runner_uid": 10514,
    "runner_owned_by_service": true
  },
  "zipformer_ru_cpu": {
    "ready": true,
    "execution_verified": true,
    "last_runner_uid": 10514,
    "runner_owned_by_service": true
  }
}
```

After a service restart, `artifacts_ready` remains true but `ready` remains
false until that service process produces a non-empty transcript. This prevents
an artifact-only NPU claim while still allowing a request to perform the first
verification.

## Remaining ASR scope

- SenseVoice HTP is production-validated here for EN/ZH/JA/KO routing and the
  exact checked fixtures. The Japanese and Korean transcripts are useful route
  evidence but still contain visible quality defects, so broader accent/noise
  robustness remains open.
- Cantonese is supported by the model but remains optional until a physical
  parity/quality fixture is added.
- Russian remains a validated CPU fallback because SenseVoice does not support
  it.
- French, German, Spanish, and Italian still require their CPU multilingual
  routes; they must not be presented as SenseVoice/NPU languages.
