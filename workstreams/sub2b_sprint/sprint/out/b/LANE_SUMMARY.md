# Lane B summary — backbone decision

## Headline

**MOVE TO Q — the sub-1B branch fails and Qwen2.5-1.5B leads Falcon-H1 under the frozen tie-break.**

## Confidence

LIMITED: deterministic HOST screen at n=120/language with only 40 machine-scored native prompts and no native-speaker validation.

The decision is mechanically faithful to the frozen B1 point-estimate rule and every selected-model score is `HOST` at n=120/language. Confidence is limited because only 40 prompts/language exercise native generation, their rubrics are mechanical, and no native speaker validated fluency, grammar or cultural appropriateness.

## Decision cells

| Lang | Qwen3.5 HOST score /100 | n | Ratio to EN |
|---|---:|---:|---:|
| EN | 43.30 | 120 | 1.000x |
| ZH | 47.50 | 120 | 1.097x |
| ES | 31.47 | 120 | 0.727x |
| FR | 34.17 | 120 | 0.789x |
| RU | 34.00 | 120 | 0.785x |

## Failure modes

- qwen35_08b: reasoning leakage 1/200 across five languages; ranking void = no.
- qwen25_15b: reasoning leakage 0/200 across five languages; ranking void = no.
- qwen3_06b: reasoning leakage 2/200 across five languages; ranking void = yes.
- falcon_h1_15b: reasoning leakage 0/200 across five languages; ranking void = no.

Wrong-language denominators exclude outputs with fewer than eight Unicode letters; B5 therefore reports eligible n explicitly. Traditional-Han leakage is measured on raw ZH output before normalization.

## Completeness and limits

- B1: COMPLETE
- B2: COMPLETE_WITH_LIVE_CELL_DEPENDENCY_UNMET
- B3: COMPLETE
- B4: COMPLETE
- B5: COMPLETE
- B6: COMPLETE

No training, fine-tuning, conversion or local >1B inference was performed. Lane B does not establish phone latency, PSS, thermal behaviour, accelerator compatibility, or native-speaker acceptance; those remain `UNKNOWN` here.
