# Third-party notices

The standalone benchmark builds against an external
`sherpa-onnx-1.13.4.aar`; the AAR is hash-pinned by
`tools/fetch_sherpa_aar.sh` and is not committed to this repository.

- sherpa-onnx 1.13.4 — Apache License 2.0:
  https://github.com/k2-fsa/sherpa-onnx
- ONNX Runtime, bundled inside the sherpa-onnx AAR — MIT License:
  https://github.com/microsoft/onnxruntime

Candidate model files are staged separately and remain subject to each model
and training corpus license. The repository intentionally excludes the blinded
listening WAVs because several evaluated voices have restricted or unclear
output-redistribution terms.
