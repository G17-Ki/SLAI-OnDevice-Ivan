# Final APK strict acceptance re-audit — 2026-07-29

> Historical scope: this re-audit applies to main hash `a9a3ce...` and the
> superseded Japanese fixture. The rebuilt `a24e...` / `33bf...` pair preserves
> the unchanged `どこ` oracle and passes strict Japanese acceptance; see
> `final_apk_a24e_33bf_cold_reboot_provenance_20260729.md`.

A final read-only review found that the physical harness could report success
without failing its process, could accept missing negative-NPU telemetry, and
did not require one neural-TTS metric record per utterance. The JA/KO semantic
oracles were also too permissive for a robustness claim.

The source harness is now stricter:

- a failed case or cleanup makes the command return nonzero;
- expected `npu_executed=false` / `npu_selected=false` requires actual boolean
  telemetry rather than treating a missing field as false;
- neural TTS requires matching start/metric/done counts and the expected engine
  ID on every metric;
- direct RU/JA/KO TTS requires first chunk <= 1200 ms, RTF <= 0.8, and zero
  dropped chunks;
- JA/KO answers require positive `パリです` / `파리입니다` forms and reject
  common negations;
- the Japanese transcript must retain the question meaning marker `どこ`, and
  Korean must retain `어디`.

All focused tests pass after this hardening.

## Re-audit of recorded physical evidence

The raw final reports contain enough evidence to apply the new non-semantic
checks without rerunning the phone:

- direct RU/JA/KO first chunks are 688/505/479 ms, RTF is
  0.220/0.243/0.263, and dropped chunks are 0/0/0;
- warm and post-reboot JA each have 2 starts, 2 engine metrics, and 2 done
  events; KO has 1/1/1;
- every JA/KO engine metric reports the expected engine, generation, and exact
  generation language;
- EN/ZH/JA/KO and RU reports contain real boolean NPU execution/selection
  fields, so the stricter positive and negative checks are evaluable.

The strengthened semantic result is:

| Route | Recorded transcript | Positive answer | Strict semantic verdict |
|---|---|---|---|
| Japanese | `フランス の 首都 は 田クスカ。` | yes, contains `パリです` and no negation | **fail**: the expected question marker `どこ` was lost |
| Korean | `프랑스의의 수도는 어디인가요?` | yes, `파리입니다.` and no negation | pass, with a visible duplicated particle quality defect |

Therefore the earlier JA/KO **2/2** values remain valid only as
route/configuration/response/playback completion evidence under the embedded
plan that produced those files. They are not a robust Japanese ASR quality
pass. The current `qwen2_5_ja_ko_current.json` plan will reject the recorded
Japanese transcript on the next physical run.

## Current honest classification

- EN/ZH/RU scripted end-to-end gates pass on the final APK, within the narrow
  prerecorded explicit-language fixture scope.
- KO passes the strengthened semantic gate but needs broader natural-speech,
  accent, and noise coverage.
- JA proves app-owned HTP execution, Japanese tag/routing, Qwen response,
  Japanese Supertonic generation configuration, and playback lifecycle, but
  its current fixture does not pass the strict semantic ASR gate.
- Audible JA/KO language quality and acoustic onset remain unmeasured.

The next Japanese step is not to loosen the oracle. Use a clearer native
fixture or improve the ASR route, then rerun the strict plan on the phone.
