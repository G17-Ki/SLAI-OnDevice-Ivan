# Qualcomm AI Hub runner readiness

Validated offline against installed `qai-hub==0.52.0`; no credentials or jobs
were used.

The Python client forwards compile/profile option strings to the service and
does not parse them locally. The call signatures, result schema, and dry-run
plans are validated here; final option acceptance still requires the first
authenticated remote compile/profile job.

## Runtime flows

| Flow | Compile options | Profile options | Interpretation |
|---|---|---|---|
| Strict NPU | `--target_runtime qnn_context_binary --qnn_options default_graph_htp_precision=FLOAT16` | `--max_profiler_iterations 20` | HTP context-binary compilation must succeed; runtime placement must contain NPU layers and no CPU/unknown layers. |
| Fallback diagnostic | `--target_runtime onnx` | `--onnx_execution_providers qnn --max_profiler_iterations 20` | QNN EP claims supported nodes and ONNX Runtime retains CPU fallback for the rest. |

`--compute_unit` was removed from compile options: the installed client only
documents it as a profile option, while the QNN context-binary target already
selects the HTP backend. `--compute_unit all` was also removed from the ONNX
diagnostic because the QNN provider plus ORT CPU fallback is the intended
partitioning mechanism.

## Result classification

- Compile failure: component is `unsupported_or_compile_failure`; inspect the
  compile-job error details for the source operation.
- Successful profile: `execution_detail[*].compute_unit` is persisted and
  counted as Hexagon NPU, CPU fallback, or other/unspecified.
- Placement counts are compiled runtime-layer counts. Fusion means they are not
  necessarily one-to-one with `torch.export` nodes.
- Remote inference: every output has an explicit index/name mapping to a golden
  tensor, then shape, finiteness, and numerical tolerance are checked. Integer
  token outputs require exact equality.

## Supported fixed graphs

| Component | Inputs | Explicit output mapping |
|---|---|---|
| FastConformer + adapter | `mel` | `#0=adapted` |
| LFM conv layer | `hidden_states` | `#0=output` |
| LFM attention layer | `hidden_states` | `#0=output` |
| LFM conv cached decode | `hidden_states`, `conv_cache` | `#0=output`, `#1=updated_conv_cache` |
| LFM attention cached decode | `hidden_states`, `key_cache`, `value_cache` | `#0=output`, `#1=updated_key_cache`, `#2=updated_value_cache` |
| RQ/depth decoder | `hidden` | `#0=tokens`, `#1=next_audio_embedding` |
| Detokenizer T=4/T=8 | `codes` | `#0=log_abs`, `#1=angle` |

The exact target device name, OS, and attributes should be copied from
`list_aihub_devices.py`. Its JSON now includes OS, QNN support, and an explicit
selector to avoid silently choosing a different OS image.

Cached-decode invocation uses the same generic runner with repeated mappings,
for example `--input hidden_states --input conv_cache --output '#0=output'
--output '#1=updated_conv_cache'`. The attention case similarly repeats
`--input` for `key_cache` and `value_cache`, then maps output indices 1 and 2
to their updated cache goldens.
