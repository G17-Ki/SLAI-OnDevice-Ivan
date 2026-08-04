# Warnings, failures, and limitations

## Critical Mandarin warm-up numerical failure

The measured process first warmed English with `Hello.` and then Mandarin with
`你好。`. The Mandarin callback arrived at 628.6 ms and returned a nominal
26,400-sample / 1.10-second buffer, but **all 26,400 samples were non-finite**:

```json
{
  "case_id": "warmup_zh",
  "text": "你好。",
  "audio_status": "all_nonfinite",
  "num_samples": 26400,
  "nonfinite_count": 26400,
  "pcm_f32_sha256": "d2e2237c131b9da23336193cd9b445b05c1ba1c80648f8899be0dd810f668cb1"
}
```

The initial serializer represented the undefined peak/RMS/DC statistics as
bare `NaN`, which is not strict JSON. The evidence was mechanically repaired
without rerunning synthesis: undefined scalar statistics are now `null`, the
failure is explicit under `warmup_failures`, and serialization uses
`allow_nan=False`. The repair script is
`scripts/kokoro_int8_mac_m2/repair_benchmark_json.py`.

All 30 longer measured case outputs were finite and unclipped. That does not
make the warm-up failure safe. Any integration must reject non-finite PCM and
merge or retry an overly short chunk before demo use.

## Follow-up Mandarin boundary result

The failure is not limited to the first Mandarin call. A fresh-process and
post-warm-up sweep produced all-non-finite PCM on 10/31 Mandarin calls. The
one-character `你。` failed every time; the two-character `你好。` and `欢迎。`
were inconsistent. The tested phrases containing at least three Han characters
were finite, but the finite-PCM gate remains mandatory because the sweep is too
small to prove a universally safe minimum. Exact text/punctuation and the
unseeded ONNX random path remain plausible factors.

Evidence: `mandarin_initialization_gate.json`.

## Runtime warning retained verbatim

Sherpa emitted the following line once during the measured process. Synthesis
continued and all arrays in the six-case measurement matrix were finite and
unclipped; the separate Mandarin warm-up failure is documented above.

```text
/Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/lexicon.cc:ConvertTokensToIds:93 Unknown token: ❓
```

The `/Users/runner/...` prefix is compiled source-location metadata inside the
upstream wheel, not a path on the benchmark host.

## Rejected preliminary lexicon configuration

A pre-measurement smoke test passed both US and GB English lexicons together.
They contain overlapping entries, producing warnings beginning with:

```text
/Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/kokoro-multi-lang-lexicon.cc:InitLexicon:515 Duplicated word: kokoro at line 1:kokoro k ˈ O k ə ɹ O. Ignore it.
/Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/kokoro-multi-lang-lexicon.cc:InitLexicon:515 Duplicated word: misaki at line 2:misaki m i s ˈ ɑ k i. Ignore it.
```

The measured configuration uses only `lexicon-us-en.txt` plus
`lexicon-zh.txt`; those duplicate-word warnings did not occur in the measured
run.

## Host ONNX Runtime inventory failure

An optional metadata inspection through the pre-existing Anaconda
`onnxruntime` failed because that installation was built against NumPy 1.x but
the environment contained NumPy 2.2.6. The relevant output was:

```text
A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.2.6 as it may crash.
AttributeError: _ARRAY_API not found
ImportError
```

No inference depended on that environment. Metadata was read with the pure
`onnx==1.18.0` parser in the isolated Kokoro environment; synthesis used
Sherpa-ONNX 1.13.4.

## Unsupported languages

Spanish and Japanese were not benchmarked as Kokoro quality cases. The model
card and embedded ONNX metadata declare only English and Chinese. Bundled
generic espeak dictionaries do not expand the trained checkpoint's language
coverage.

## Third-party license inventory gap

The source model card declares Apache-2.0 and the archive's exact top-level
`LICENSE` is Apache-2.0. The archive also contains `espeak-ng-data` and
CppJieba-style dictionary assets but no complete third-party notice manifest.
The public repository therefore does not store or redistribute the model
archive, and publication materials must not describe every extracted file as
proven Apache-2.0 until a dependency-license audit is complete.
