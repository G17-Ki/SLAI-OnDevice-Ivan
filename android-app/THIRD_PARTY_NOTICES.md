# Third-party notices

The Android application code and UI in this directory are part of this
feasibility repository. The packaged native ARM64 runner libraries are derived
without source modification from the official archive:

- Repository: `LiquidAI/LFM2.5-Audio-1.5B-GGUF`
- Revision: `7d525f883a077e20afb782f2ff618edcae0e39e4`
- Archive: `runners/llama-liquid-audio-android-arm64.zip`

`llama-liquid-audio-server` is packaged under the Android-compatible filename
`liblfmserver.so` so Android extracts it with the other native libraries. Its
contents are unchanged; `runner-files.sha256` records the packaged hashes.

The upstream runner and dependency license texts are under
`third_party/llama-liquid-audio/`. The LFM model license is copied to
`third_party/LFM_OPEN_LICENSE.txt`. Model weights are not committed to this
repository and must be downloaded separately after reviewing that license.

## Whisper frontend FFT dependency

The Whisper probe vendors KISS FFT at revision
`febd4caeed32e33ad8b2e0bb5ea77542c40f18ec`.

- KISS FFT — BSD-3-Clause — Copyright (c) 2003-2010 Mark Borgerding
- Vendored source: `whisper-probe/src/main/cpp/third_party/kissfft/`
- Full license text:
  [`whisper-probe/src/main/cpp/third_party/kissfft/LICENSES/BSD-3-Clause`](whisper-probe/src/main/cpp/third_party/kissfft/LICENSES/BSD-3-Clause)

## Experimental offline-TTS backend

The checked-in `0.5.0-tts-lab` source can build against the externally fetched
`sherpa-onnx-1.13.4.aar`. The AAR is not committed to this repository.

- sherpa-onnx 1.13.4 — Apache License 2.0:
  https://github.com/k2-fsa/sherpa-onnx
- ONNX Runtime, bundled inside the sherpa-onnx AAR — MIT License:
  https://github.com/microsoft/onnxruntime

Anyone redistributing an APK built with that AAR must preserve the upstream
copyright and license notices. TTS model files are staged separately and remain
subject to each model and training corpus license.
