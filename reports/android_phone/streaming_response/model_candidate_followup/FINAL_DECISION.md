# Multilingual response-model decision

Updated: 2026-07-24 (Asia/Shanghai)

## Decision

Use **Qwen2.5-1.5B-Instruct Q4_K_M** as the preferred direct-generation
candidate for Mandarin and short context experiments. Do not call it a robust
Russian solution. Keep the currently deployed Qwen3-1.7B only until the phone
deployment is deliberately migrated and reprofiled.

None of the tested small direct models meets the requested Chinese-and-Russian
quality floor. A real offline Russian translation route is justified; the
prompt-only English-pivot control is rejected. Translation integration is a
separate workstream and was not started here.

All latency numbers below are same-environment Colab x86 CPU measurements.
They are useful only for relative comparison and are not Nubia timings.

## Corrected comparison

| Candidate | Priority quality | Mandarin / Russian | Context | Median / P95 first speakable content | Model bytes | Process RSS after evaluation | License and public-release decision |
|---|---:|---:|---:|---:|---:|---:|---|
| Qwen3-1.7B Q4_0, thinking off | 37/63 (58.7%) | 22/33 / 15/30 | 6/12 | 2.27 / 5.60 s | 1,056,782,912 | 2,126,404 KiB | Apache-2.0; redistributable in principle, but reject as the quality baseline |
| Qwen2.5-1.5B Q4_K_M | **50/63 (79.4%)** | **32/33** / 18/30 | **12/12** | **1.90 / 4.83 s** | 1,117,320,736 | **1,840,856 KiB** | Apache-2.0; best public direct candidate, but Russian remains below the acceptance floor |
| Qwen3-1.7B Q4_0, reasoning budget 16 | 31/63 (49.2%) | 21/33 / 10/30 | 6/12 | 5.19 / 8.93 s | 1,056,782,912 | 2,163,676 KiB | Apache-2.0; reject for lower quality, much higher latency, and reasoning content in 11/63 runs |
| Qwen2.5-3B Q4_K_M, one-seed core gate | 15/19 (78.9%) | 9/10 / 6/9 | 4/4 | 3.25 / 9.97 s | 2,104,932,768 | 3,480,560 KiB | Qwen Research License, non-commercial only; reject for public baseline, size, latency, and failed Russian core facts |

The Qwen2.5-1.5B score is the manually corrected **50/63**, not the earlier
false-positive result. The scorer was tightened so answers such as `-18°C
(0°F)` no longer pass the freezing-point case merely because `0` appears in
the text.

The 3B run intentionally stopped after its 19-case one-seed core gate. It
returned `Россия` instead of Moscow, mixed-script `Токyo`, and Vancouver for
Canada, so it failed the expansion rule. Its result is therefore not directly
comparable to the three-seed 63-run rows.

## Other controls

- Qwen3 prompt-only scoped English pivot: 19/39 (48.7%); reject.
- LFM2.5-1.2B Q4_0: 31/63 (49.2%); reject.
- Gemma-3-1B Q4_K_M: 37/63 (58.7%) with one collapse; reject.
- Qwen3 reasoning did not repair the known Sydney/Beijing/Kyiv/Vancouver and
  Russian wrong-script failures; it made first speakable output substantially
  slower.

## Reproducible artifacts

Primary Qwen3/Qwen2.5-1.5B run:

- `raw/colab-20260724t090221z/extracted/model_candidate_followup/aggregate.json`
- `raw/colab-20260724t090221z/model_candidate_followup.zip`
  (`sha256 446c86db85b81024894e5097ca81ab47e38c385ba0d257e414f4113c1d060103`)
- `raw/colab-20260724t090221z/orchestrator.log`

Pivot/LFM/Gemma controls:

- `raw/colab-followup-20260724t091846z/extracted/model_candidate_followup_stage2/aggregate_stage2.json`
- `raw/colab-followup-20260724t091846z/model_candidate_followup_stage2.zip`
  (`sha256 c7cc1c32b1eac8e66258b161272134647bc9e62294f605787e6c561738da3e6d`)
- `raw/colab-followup-20260724t091846z/orchestrator.log`

Reasoning-budget and 3B controls:

- `raw/colab-stage3-20260724t094448z/extracted/model_candidate_followup_stage3/aggregate_stage3.json`
- `raw/colab-stage3-20260724t094448z/model_candidate_followup_stage3.zip`
  (`sha256 0c3584651e9bf59bc023a58949103a4d57136f0d9cee9758532cd95877d629db`)
- `raw/colab-stage3-20260724t094448z/orchestrator.log`

Inputs and orchestration are pinned in `candidate_manifest.json`,
`priority_zh_ru_suite.json`, `remote_benchmark.py`, `remote_followup.py`, and
`remote_stage3.py` in this directory.

The detached stage-three job completed with return code zero, downloaded its
archive, and terminated the Colab session. No model weights are stored in the
repository and nothing was pushed to GitHub.
