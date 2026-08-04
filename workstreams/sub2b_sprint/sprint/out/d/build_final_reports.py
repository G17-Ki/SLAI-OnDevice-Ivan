#!/usr/bin/env python3
"""Build Lane D's compact D1, D3, and D5 deliverables.

The large synthesis, ASR, and waveform artifacts are intentionally git-ignored.
This script reads their frozen summaries, asserts the Lane D invariants, and
emits the small committed reports plus the row-aligned listening guide.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


OUT_D = Path(__file__).resolve().parent
ARTIFACTS = OUT_D / "artifacts"
LANGUAGE_ORDER = ("en", "zh", "es", "fr", "ru")
LANGUAGE_NAMES = {
    "en": "English",
    "zh": "Mandarin Chinese",
    "es": "Spanish",
    "fr": "French",
    "ru": "Russian",
}
TH13_MARGIN_PP = 5.0
TH13_MIN_N = 60


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def escape_md(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def fmt_pp(value: float) -> str:
    return f"{value:+.3f}"


def rel(path: Path) -> str:
    return str(path.relative_to(OUT_D))


def build_d1() -> tuple[dict, str]:
    corpus_path = OUT_D / "D1_corpus.json"
    scoring_path = ARTIFACTS / "d1_scoring" / "d1_scoring_results.json"
    receipt_path = ARTIFACTS / "d1_scoring" / "verification_receipt.json"
    kokoro_zh_path = ARTIFACTS / "kokoro_v1_1_zh" / "logs" / "execution_summary.json"
    matcha_frontend_path = ARTIFACTS / "matcha" / "frontend_oov_probe.json"
    corpus = load_json(corpus_path)
    scoring = load_json(scoring_path)
    receipt = load_json(receipt_path)
    kokoro_zh = load_json(kokoro_zh_path)
    matcha_frontend = load_json(matcha_frontend_path)

    assert receipt["status"] == "PASS"
    assert not receipt["failed_assertions"]
    assert sha256_file(corpus_path) == scoring["corpus"]["sha256"]
    assert len(scoring["summary_rows"]) == 12
    assert all(row["n"] == 40 for row in scoring["summary_rows"])
    assert kokoro_zh["d1"]["attempted"] == 40
    assert kokoro_zh["d1"]["passed"] == 33
    assert kokoro_zh["d1"]["nonfinite_outputs"] == 7
    assert len(matcha_frontend["cases"]) == 14
    assert all(row["passed_waveform_assertions"] for row in matcha_frontend["cases"])

    comparisons = []
    for row in scoring["summary_rows"]:
        comparison = scoring["languages"][row["language"]]["comparisons"][row["engine"]]
        basic_delta = comparison["controls_and_required_alternates"]["basic"]["delta"]
        primary = comparison["primary"]
        enriched = {
            "language": row["language"],
            "engine_id": row["engine"],
            "metric": row["metric"],
            "n": row["n"],
            "matched_human_rate_percent": row["human_rate_pp"],
            "synth_rate_percent": row["synth_rate_pp"],
            "delta_percent_points": row["matched_delta_pp"],
            "delta_ci95_percent_points": [
                row["matched_delta_ci95_lo_pp"],
                row["matched_delta_ci95_hi_pp"],
            ],
            "basic_control_delta_percent_points": basic_delta["point_pp"],
            "basic_control_delta_ci95_percent_points": [
                basic_delta["ci95_lo_pp"],
                basic_delta["ci95_hi_pp"],
            ],
            "normalizer_control_status": row["normalizer_control_status"],
            "ranking_inadmissible": row["ranking_inadmissible"],
            "point_estimate_at_or_below_th13_plus_5pp_ceiling": (
                row["matched_delta_pp"] <= TH13_MARGIN_PP
            ),
            "th13_gate_verdict": "INSUFFICIENT_N_40_LT_60",
            "evidence_class": "HOST",
            "edit_counts": {
                "matched_human": {
                    key: primary["human"][key]
                    for key in (
                        "substitutions",
                        "deletions",
                        "insertions",
                        "errors",
                        "reference_units",
                    )
                },
                "synth": {
                    key: primary["synth"][key]
                    for key in (
                        "substitutions",
                        "deletions",
                        "insertions",
                        "errors",
                        "reference_units",
                    )
                },
            },
        }
        if row["language"] == "ru":
            enriched["yo_preserved_delta_percent_points"] = row[
                "yo_preserved_matched_delta_pp"
            ]
        comparisons.append(enriched)

    kokoro_missing = scoring["skipped_incomplete_transcript_groups"][0]
    assert kokoro_missing["language"] == "zh"
    assert kokoro_missing["present_clip_count"] == 33
    assert len(kokoro_missing["missing_item_ids"]) == 7

    synthesis_inventory = {
        "attempted": 520,
        "valid_wavs_written": 513,
        "hard_failures": 7,
        "hard_failure_route": "zh/kokoro_v1_1",
        "derivation": (
            "Supertonic 160 + Piper 160 + Kokoro v1.0 EN/ES/FR 120 + "
            "Matcha ZH 40 + Kokoro v1.1 ZH 40"
        ),
        "kokoro_zh": {
            "attempted": 40,
            "valid_wavs_written": 33,
            "nonfinite_outputs": 7,
            "missing_item_ids": kokoro_missing["missing_item_ids"],
            "score_disposition": "NOT_SCORED_NO_IMPUTATION",
        },
    }

    mixed_script_items = [
        item["item_id"]
        for item in corpus["items"]
        if any("A" <= ch <= "Z" or "a" <= ch <= "z" for ch in item["texts"]["zh"])
    ]
    assert mixed_script_items == [
        "fleurs_03",
        "fleurs_06",
        "fleurs_10",
        "fleurs_13",
        "fleurs_21",
        "fleurs_28",
        "fleurs_37",
        "fleurs_39",
    ]

    payload = {
        "schema_version": 1,
        "sprint_id": "D1",
        "lane": "D",
        "status": "COMPLETE_WITH_KOKORO_ZH_HARD_FAILURE_AND_SPRINT_LIMITS",
        "evidence_class": "HOST",
        "headline": {
            "complete_engine_language_comparisons": 12,
            "all_complete_point_estimates_at_or_below_plus_5pp_ceiling": True,
            "th13_gate_cleared_by_any_cell": False,
            "th13_gate_blocker": "n=40 per cell; TH-13 requires n>=60",
            "hard_failure": (
                "Kokoro v1.1 ZH emitted non-finite PCM for 7/40 D1 items; "
                "the incomplete arm was not scored or imputed."
            ),
        },
        "protocol": {
            "corpus": scoring["corpus"],
            "coverage_targets": corpus["coverage_targets"],
            "coverage_achieved": corpus["coverage_achieved"],
            "matched_human_recordings": 200,
            "reference_asr": {
                "en": "faster-whisper-tiny (OpenAI Whisper family)",
                "es": "faster-whisper-tiny (OpenAI Whisper family)",
                "fr": "faster-whisper-tiny (OpenAI Whisper family)",
                "zh": "sherpa-onnx Paraformer ZH small INT8",
                "ru": "sherpa-onnx GigaAM-v2 CTC INT8; Whisper forbidden and not used",
            },
            "metric": "WER except CER for ZH",
            "delta_definition": (
                "reference-ASR error rate on synthesized audio minus the same "
                "reference-ASR error rate on the matched human recording"
            ),
            "statistics": scoring["statistics"],
            "normalization": {
                "source_sha256": scoring["normalization"]["source_sha256"],
                "golden_fixture_count": scoring["normalization"]["golden_fixtures"][
                    "fixture_count"
                ],
                "golden_fixture_status": scoring["normalization"]["golden_fixtures"][
                    "status"
                ],
                "fixture_file_sha256": scoring["normalization"]["golden_fixtures"][
                    "fixture_file_sha256"
                ],
                "basic_control_threshold_percent_points": scoring["normalization"][
                    "basic_control_threshold_pp"
                ],
            },
            "th13_context": {
                "margin_percent_points": TH13_MARGIN_PP,
                "minimum_n": TH13_MIN_N,
                "sprint_n": 40,
                "interpretation": (
                    "Point-estimate screen only. A value within the margin at n=40 "
                    "must not be reported as clearing TH-13."
                ),
            },
        },
        "synthesis_inventory": synthesis_inventory,
        "comparisons": comparisons,
        "frontend_findings": [
            {
                "route": "matcha_zh_baker_vocos",
                "finding": "SILENT_LATIN_OOV_OMISSION",
                "affected_d1_item_ids": mixed_script_items,
                "direct_host_probe": (
                    "The sherpa-onnx frontend logged 'Ignore OOV' for Wild, Card, "
                    "SANParks, Eee, PC, IT, CPU, RAM, GB, API, JSON, Wi, Fi, and VPN."
                ),
                "probe_artifact": {
                    "path": rel(matcha_frontend_path),
                    "bytes": matcha_frontend_path.stat().st_size,
                    "sha256": sha256_file(matcha_frontend_path),
                    "case_count": len(matcha_frontend["cases"]),
                },
                "metric_consequence": (
                    "Mandarin CER can look good while unsupported Latin spans are "
                    "omitted; the D1 point estimate is not evidence of mixed-script coverage."
                ),
            }
        ],
        "unscored_or_unsupported": [
            {
                "language": "zh",
                "engine": "kokoro_v1_1",
                "status": "UNSAFE_INCOMPLETE_NOT_SCORED",
                "detail": synthesis_inventory["kokoro_zh"],
            },
            {"language": "zh", "engine": "supertonic", "status": "UNSUPPORTED"},
            {"language": "ru", "engine": "kokoro", "status": "UNSUPPORTED"},
            {
                "language": "ru",
                "engine": "silero_v5_cis_base",
                "status": "NO_D1_BACKCHECK_IN_THIS_SPRINT",
            },
        ],
        "artifacts": {
            "scoring_result": {
                "path": rel(scoring_path),
                "bytes": scoring_path.stat().st_size,
                "sha256": sha256_file(scoring_path),
            },
            "verification_receipt": {
                "path": rel(receipt_path),
                "bytes": receipt_path.stat().st_size,
                "sha256": sha256_file(receipt_path),
                "status": receipt["status"],
            },
            "kokoro_zh_execution": {
                "path": rel(kokoro_zh_path),
                "sha256": sha256_file(kokoro_zh_path),
            },
            "matcha_frontend_probe": {
                "path": rel(matcha_frontend_path),
                "bytes": matcha_frontend_path.stat().st_size,
                "sha256": sha256_file(matcha_frontend_path),
                "case_count": len(matcha_frontend["cases"]),
            },
        },
        "limits": [
            "The FLEURS Parquet fetch used the mutable refs/convert/parquet ref; its resolved source revision and Parquet blob hashes were not retained. The committed manifest and each retained human WAV remain hash-pinned locally.",
            "Only one different-family reference ASR was run per language; the full parent task's dual-reference stability requirement was not attempted.",
            "n=40 is below TH-13's n>=60 screen requirement.",
            "No wrong-language LID test, speaker-consistency score, phone run, or listening panel was performed.",
            "Negative delta values do not establish naturalness or superiority to human speech; they only show how this reference ASR scored this matched corpus.",
            "Rows marked normalizer-unstable are inadmissible for engine ranking.",
        ],
    }

    table_lines = []
    for row in comparisons:
        stability = "stable" if not row["ranking_inadmissible"] else "UNSTABLE — no rank"
        ci = row["delta_ci95_percent_points"]
        table_lines.append(
            "| {lang} | `{engine}` | {human:.3f} | {synth:.3f} | {delta} "
            "[{lo}, {hi}] | {basic} | {stability} |".format(
                lang=row["language"].upper(),
                engine=row["engine_id"],
                human=row["matched_human_rate_percent"],
                synth=row["synth_rate_percent"],
                delta=fmt_pp(row["delta_percent_points"]),
                lo=fmt_pp(ci[0]),
                hi=fmt_pp(ci[1]),
                basic=fmt_pp(row["basic_control_delta_percent_points"]),
                stability=stability,
            )
        )

    markdown = f"""# D1 — Five-language TTS backcheck

Status: **complete as a sprint point-estimate screen, with one hard route failure**. Twelve complete engine-language arms were scored at `n=40` against matched FLEURS human recordings. Every complete arm's primary ΔWER/ΔCER point estimate is at or below TH-13's **+5.0 pp ceiling**, but **none clears TH-13** because the registered screen requires `n>=60`.

Kokoro v1.1 ZH is the decisive failure: **7/40 outputs contained non-finite PCM**, so only 33 valid WAVs exist and no ΔCER was calculated or imputed.

## Matched-human results (`HOST`)

Δ is synth error rate minus matched-human error rate, in percentage points. ZH uses CER; the other languages use WER. CIs are paired utterance bootstraps (`B=10,000`, seed `20260801`).

| Language | Engine | human % | synth % | primary Δ pp [95% CI] | basic-control Δ pp | normalizer control |
|---|---|---:|---:|---:|---:|---|
{chr(10).join(table_lines)}

The negative rows do not mean synthetic speech is “better than human.” They mean the evaluation ASR found the clean synthetic audio easier than these matched human recordings. ASR backcheck is an intelligibility floor, never a naturalness verdict.

## Protocol and integrity

- Corpus: 40 shared semantic FLEURS test IDs in EN/ZH/ES/FR/RU, 200 matched human WAVs; manifest SHA-256 `{scoring['corpus']['sha256']}`. Coverage achieved: 13 number, 9 date, 35 name, 9 acronym, 1 question, and 10 long-clause cases.
- Source provenance limit: the FLEURS fetch used mutable `refs/convert/parquet`; its resolved revision and source-Parquet hashes were not retained. The committed corpus manifest and every retained human WAV are SHA-256 pinned, so the local evaluated evidence is frozen even though the upstream fetch revision is not.
- Reference ASR: faster-whisper-tiny for EN/ES/FR; Paraformer for ZH; GigaAM-v2 CTC for RU. Russian never used Whisper.
- Frozen normalizer SHA-256 `{scoring['normalization']['source_sha256']}`; **40/40 golden fixtures passed**. The final verification receipt passed every assertion, including corpus/transcript hashes, complete 40-item groups, exclusion of partial groups, and the GigaAM route.
- Synthesis: **520 attempted / 513 valid WAVs / 7 hard failures**, all seven from Kokoro ZH.

## Important frontend finding

Matcha ZH passed waveform checks and posted primary ΔCER {fmt_pp(next(r['delta_percent_points'] for r in comparisons if r['language']=='zh'))} pp, but its frontend explicitly logged `Ignore OOV` for Latin spans such as `Wild Card`, `SANParks`, `Eee PC`, `IT`, `CPU`, `RAM`, `GB`, `API`, `JSON`, `Wi-Fi`, and `VPN`. The eight mixed-script D1 items are `fleurs_03`, `06`, `10`, `13`, `21`, `28`, `37`, and `39`. **The CER is therefore a metric blind spot for mixed-script coverage**, not evidence that those spans were spoken. The 14-case direct probe is preserved at `artifacts/matcha/frontend_oov_probe.json` (SHA-256 `{sha256_file(matcha_frontend_path)}`).

## Limits

This bounded sprint used one different-family reference ASR per language and `n=40`; the parent screen's dual-reference and `n>=60` requirements remain unmet. No wrong-language LID test, speaker-consistency score, phone run, or panel was performed. Rows marked normalizer-unstable carry no ranking weight. Full utterance-level S/I/D counts, transcripts, hashes, controls, and the verification receipt are under `artifacts/d1_scoring/`.
"""
    return payload, markdown


def build_d3() -> tuple[dict, str]:
    audit_path = ARTIFACTS / "d3_audit" / "audit.json"
    audit = load_json(audit_path)
    total = audit["reconciled_total"]
    assert total == {"attempted": 96, "passed": 90, "failed": 6}

    routes = []
    failures = []
    for engine in audit["engines"]:
        if "routes" in engine:
            for route in engine["routes"]:
                if route["attempted"] == 0:
                    continue
                row = {
                    "engine": route.get("engine", engine["engine"]),
                    "variant": engine.get("variant"),
                    "voice": route.get("voice"),
                    "language": route["language"],
                    "attempted": route["attempted"],
                    "passed": route["passed"],
                    "failed": route["attempted"] - route["passed"],
                    "verdict": route["label"],
                    "evidence": engine["evidence"],
                }
                routes.append(row)
        else:
            routes.append(
                {
                    "engine": engine["engine"],
                    "variant": engine.get("variant"),
                    "voice": None,
                    "language": engine["language"],
                    "attempted": engine["attempted"],
                    "passed": engine["passed"],
                    "failed": engine["failed"],
                    "verdict": engine["label"],
                    "evidence": engine["evidence"],
                }
            )
        failures.extend(engine.get("failures", []))
        if engine.get("label") == "UNSAFE_HARD_FAIL_NONFINITE_PCM":
            failures.extend(
                [
                    {
                        "engine": "kokoro",
                        "variant": engine["variant"],
                        "language": "zh",
                        "text": "你",
                        "failed_runs": 1,
                        "reason": "all samples non-finite in one 15,000-sample output",
                    },
                    {
                        "engine": "kokoro",
                        "variant": engine["variant"],
                        "language": "zh",
                        "text": "你好",
                        "failed_runs": 3,
                        "reason": "all samples non-finite in three 23,400-sample outputs",
                    },
                ]
            )

    assert len(routes) == 15
    payload = {
        "schema_version": 1,
        "sprint_id": "D3",
        "lane": "D",
        "status": "COMPLETE_WITH_THREE_UNSAFE_PRIMARY_ROUTES",
        "evidence_class": "HOST",
        "headline": {
            "attempted": 96,
            "passed": 90,
            "failed": 6,
            "unsafe_primary_routes": [
                "piper_davefx/es",
                "piper_dmitri/ru",
                "kokoro_v1_1/zh",
            ],
            "note": "Piper RU is voice-specific: Dmitri failed while Irina passed 6/6.",
        },
        "gate": {
            "finite": "all PCM samples finite",
            "nonempty": "sample_count > 0",
            "minimum_rms_dbfs_exclusive": -50,
            "minimum_rms_linear_exclusive": 0.0031622776601683794,
            "duration_seconds_inclusive": [0.15, 30.0],
            "sustained_clipping_dbfs": -0.1,
            "sustained_clipping_linear": 0.9885530946569389,
            "failure_rule": "fail if a contiguous run above the clipping threshold exceeds 5 ms",
        },
        "routes": routes,
        "named_failures": failures,
        "threshold_reconciliation": audit["threshold_reconciliation_notes"],
        "scope_warning": (
            "A D3 pass is a short-input waveform result only. It does not establish "
            "intelligibility, language correctness, naturalness, frontend coverage, or licensing."
        ),
        "artifact": {
            "path": rel(audit_path),
            "sha256": sha256_file(audit_path),
        },
    }

    rows = []
    for route in routes:
        label = route["verdict"]
        detail = "—"
        if route["language"] == "es" and route.get("voice") == "davefx":
            detail = "`A` -> 0.081270 s"
        elif route["language"] == "ru" and route.get("voice") == "dmitri":
            detail = "`Да` -> 0.116100 s"
        elif label == "UNSAFE_HARD_FAIL_NONFINITE_PCM":
            detail = "4 all-non-finite outputs (`你` once; `你好` 3/3)"
        rows.append(
            f"| `{route['engine']}` | `{route.get('voice') or '—'}` | "
            f"{route['language'].upper()} | {route['passed']}/{route['attempted']} | "
            f"`{label}` | {detail} |"
        )

    markdown = f"""# D3 — Short-input safety

Status: **complete**. Evidence class for all measured counts, durations, and waveform verdicts below: **`[HOST]`**. Across 15 route-language probes, **90/96 outputs passed** the exact Lane D waveform gate. Three primary routes are unsafe: Piper DaveFX ES, Piper Dmitri RU, and Kokoro v1.1 ZH.

| Engine | voice/style | lang | pass | verdict | named failure |
|---|---|---:|---:|---|---|
{chr(10).join(rows)}

## What failed

- Piper DaveFX ES: one-character `A` produced **0.081270 s**, below the 0.15 s duration floor.
- Piper Dmitri RU: `Да` produced **0.116100 s**, below the duration floor. Piper RU is **voice-specific**, not blanket-safe: Irina passed 6/6, but it lacks D1/D5 evidence and has a stronger licence gap.
- Kokoro v1.1 ZH: **4/9 hard failures**. `你好` was all-NaN in 3/3 runs; `你` was all-NaN in 1/3. `你好呀` passed 3/3. Any single non-finite sample is a hard fail.

Supertonic passed 24/24 across EN/ES/FR/RU; Kokoro v1.0 passed 18/18 across EN/ES/FR; Matcha passed 9/9 Han-only ZH runs; Silero `v5_cis_base` RU passed 6/6.

## Exact gate

Every output must be non-empty and finite, RMS greater than -50 dBFS, duration in [0.15, 30] seconds, and have no contiguous run above -0.1 dBFS lasting more than 5 ms. Source harnesses sometimes used stricter RMS or peak-only checks; reconciliation to this exact rule changed no verdict.

A D3 pass is only a short-input waveform catastrophe screen. It does not establish intelligibility, language correctness, naturalness, licensing, or mixed-script frontend coverage. In particular, Matcha's Han-only pass does not repair its Latin/OOV omission finding.
"""
    return payload, markdown


ENGINE_CATALOG = {
    "supertonic": {
        "display_name": "Supertonic-3",
        "model": "Supertone/supertonic-3",
        "revision": "3cadd1ee6394adea1bd021217a0e650ede09a323",
        "voice_or_style": "M1",
        "model_license": "BigScience OpenRAIL-M",
        "license_disposition": "COMMERCIAL_USE_CONDITIONAL_ON_OPENRAIL_M_RESTRICTIONS_AND_NOTICES",
        "license_sha256": "0d944a9110fed9a9602d60e0423a272903e7bd21ab060490774efc77c2275e9f",
    },
    "kokoro": {
        "display_name": "Kokoro-82M v1.0",
        "model": "onnx-community/Kokoro-82M-v1.0-ONNX q8f16",
        "revision": "1939ad2a8e416c0acfeecc08a694d14ef25f2231",
        "voice_or_style": {"en": "af_sarah", "es": "ef_dora", "fr": "ff_siwis"},
        "model_license": "Apache-2.0",
        "license_disposition": "NO_LANE_IDENTIFIED_MODEL_LICENSE_BLOCKER; TRAINING_DATA_NOT_LEGALLY_AUDITED",
    },
    "kokoro_zh": {
        "display_name": "Kokoro-82M v1.1 ZH",
        "model": "hexgrad/Kokoro-82M-v1.1-zh via sherpa-onnx dynamic INT8",
        "revision": "01e7505bd6a7a2ac4975463114c3a7650a9f7218",
        "voice_or_style": "zf_001 / sid 3",
        "model_license": "Apache-2.0",
        "license_disposition": "NO_LANE_IDENTIFIED_MODEL_LICENSE_BLOCKER; ROUTE_UNSAFE",
    },
    "piper": {
        "display_name": "Piper medium",
        "model": "rhasspy/piper-voices",
        "revision": "9f967d15e9ccdf43078586d1476ee70f314401bd",
        "voice_or_style": {
            "en": "en_US-lessac-medium",
            "es": "es_ES-davefx-medium",
            "fr": "fr_FR-siwis-medium",
            "ru": "ru_RU-dmitri-medium",
        },
        "model_license": "repo tag MIT; voice-specific provenance controls",
        "license_disposition": "BLOCKED_ALL_TESTED_VOICES_LESSAC_DERIVED_RESEARCH_ONLY_CHAIN",
    },
    "matcha": {
        "display_name": "Matcha Baker + Vocos",
        "model": "csukuangfj/matcha-icefall-zh-baker + vocos-22khz-univ",
        "revision": "75e64a57a80fb370abb18a4e4d86bb090ed46e93",
        "voice_or_style": "DataBaker female speaker",
        "model_license": "checkpoint licence absent; Baker corpus non-commercial; Vocos Apache-2.0",
        "license_disposition": "BLOCKED_PENDING_WRITTEN_RIGHTS_OR_REPLACEMENT",
    },
    "silero": {
        "display_name": "Silero v5_cis_base",
        "model": "snakers4/silero-models v5_cis_base",
        "revision": "d9355348e2781dc8fa25a135d1602c530afae24c",
        "voice_or_style": "ru_zhadyra",
        "model_license": "MIT for v5_cis_base only",
        "license_disposition": "MIT_NOTICE_REQUIRED; EXTERNAL_RU_STRESS_FRONTEND_REQUIRED",
    },
}


ROUTE_STATUS = {
    ("en", "kokoro"): {
        "technical": "SPRINT_CANDIDATE_NOT_GATE_CLEARED",
        "commercial": "NO_LANE_IDENTIFIED_MODEL_LICENSE_BLOCKER",
        "overall": "CANDIDATE_PENDING_FULL_GATE_AND_LISTENING",
    },
    ("en", "piper"): {
        "technical": "SPRINT_CANDIDATE_NOT_GATE_CLEARED",
        "commercial": "BLOCKED_LESSAC_PROVENANCE",
        "overall": "TECHNICAL_ONLY_NOT_SHIPPABLE_ON_PINNED_EVIDENCE",
    },
    ("en", "supertonic"): {
        "technical": "SPRINT_CANDIDATE_NOT_GATE_CLEARED",
        "commercial": "OPENRAIL_M_CONDITIONAL",
        "overall": "CANDIDATE_PENDING_FULL_GATE_AND_LISTENING",
    },
    ("es", "kokoro"): {
        "technical": "SPRINT_CANDIDATE_NOT_GATE_CLEARED",
        "commercial": "NO_LANE_IDENTIFIED_MODEL_LICENSE_BLOCKER",
        "overall": "CANDIDATE_PENDING_FULL_GATE_AND_LISTENING",
    },
    ("es", "piper"): {
        "technical": "UNSAFE_D3_DURATION_FAILURE",
        "commercial": "BLOCKED_LESSAC_PROVENANCE",
        "overall": "REJECT_UNSAFE_DIAGNOSTIC_ONLY",
    },
    ("es", "supertonic"): {
        "technical": "SPRINT_CANDIDATE_NOT_GATE_CLEARED",
        "commercial": "OPENRAIL_M_CONDITIONAL",
        "overall": "CANDIDATE_PENDING_FULL_GATE_AND_LISTENING",
    },
    ("fr", "kokoro"): {
        "technical": "SPRINT_CANDIDATE_NOT_GATE_CLEARED",
        "commercial": "NO_LANE_IDENTIFIED_MODEL_LICENSE_BLOCKER",
        "overall": "CANDIDATE_PENDING_FULL_GATE_AND_LISTENING",
    },
    ("fr", "piper"): {
        "technical": "SPRINT_CANDIDATE_NOT_GATE_CLEARED",
        "commercial": "BLOCKED_LESSAC_PROVENANCE",
        "overall": "TECHNICAL_ONLY_NOT_SHIPPABLE_ON_PINNED_EVIDENCE",
    },
    ("fr", "supertonic"): {
        "technical": "SPRINT_CANDIDATE_NOT_GATE_CLEARED",
        "commercial": "OPENRAIL_M_CONDITIONAL",
        "overall": "CANDIDATE_PENDING_FULL_GATE_AND_LISTENING",
    },
    ("ru", "piper"): {
        "technical": "UNSAFE_D3_DURATION_FAILURE_DMITRI",
        "commercial": "BLOCKED_LESSAC_PROVENANCE",
        "overall": "REJECT_UNSAFE_DIAGNOSTIC_ONLY",
    },
    ("ru", "silero"): {
        "technical": "D3_SAFE_BUT_NO_D1_BACKCHECK_AND_STRESS_UNKNOWN",
        "commercial": "MIT_WITH_NOTICE",
        "overall": "EVIDENCE_INCOMPLETE_LISTEN_FOR_STRESS",
    },
    ("ru", "supertonic"): {
        "technical": "SPRINT_CANDIDATE_NOT_GATE_CLEARED",
        "commercial": "OPENRAIL_M_CONDITIONAL",
        "overall": "CANDIDATE_PENDING_FULL_GATE_AND_LISTENING",
    },
    ("zh", "kokoro"): {
        "technical": "UNSAFE_HARD_FAIL_NONFINITE_PCM",
        "commercial": "NO_LANE_IDENTIFIED_MODEL_LICENSE_BLOCKER",
        "overall": "REJECT_UNSAFE_DIAGNOSTIC_ONLY",
    },
    ("zh", "matcha"): {
        "technical": "HAN_SAFE_BUT_MIXED_SCRIPT_OOV_OMISSION",
        "commercial": "BLOCKED_BAKER_AND_ABSENT_CHECKPOINT_LICENSE",
        "overall": "LIMITED_DIAGNOSTIC_NOT_GENERAL_SHIP_ROUTE",
    },
}


NO_EVIDENCE = {
    "en": ["matcha", "silero"],
    "zh": ["supertonic", "piper", "silero"],
    "es": ["matcha", "silero"],
    "fr": ["matcha", "silero"],
    "ru": ["kokoro", "matcha"],
}


def make_route_row(
    language: str,
    engine: str,
    d1_lookup: dict[tuple[str, str], dict],
    d3_lookup: dict[tuple[str, str], dict],
    d5_lookup: dict[tuple[str, str], dict],
) -> dict:
    catalog_key = "kokoro_zh" if (language, engine) == ("zh", "kokoro") else engine
    catalog = copy.deepcopy(ENGINE_CATALOG[catalog_key])
    voice = catalog["voice_or_style"]
    if isinstance(voice, dict):
        voice = voice[language]
    row = {
        "engine_id": engine,
        "display_name": catalog["display_name"],
        "model": catalog["model"],
        "revision": catalog["revision"],
        "voice_or_style": voice,
        "model_license": catalog["model_license"],
        "license_disposition": catalog["license_disposition"],
        **ROUTE_STATUS[(language, engine)],
    }
    score_key = (language, "matcha_zh_baker_vocos") if engine == "matcha" else (language, engine)
    if score_key in d1_lookup:
        score = d1_lookup[score_key]
        row["d1"] = {
            "status": "SCORED_POINT_ESTIMATE_ONLY",
            "metric": score["metric"],
            "n": score["n"],
            "delta_percent_points": score["delta_percent_points"],
            "delta_ci95_percent_points": score["delta_ci95_percent_points"],
            "normalizer_control_status": score["normalizer_control_status"],
            "ranking_inadmissible": score["ranking_inadmissible"],
            "th13_gate_verdict": score["th13_gate_verdict"],
        }
    elif (language, engine) == ("zh", "kokoro"):
        row["d1"] = {
            "status": "UNSAFE_INCOMPLETE_NOT_SCORED",
            "attempted": 40,
            "valid": 33,
            "nonfinite": 7,
        }
    else:
        row["d1"] = {"status": "NO_D1_BACKCHECK"}

    d3_key = (language, engine)
    if d3_key in d3_lookup:
        route = d3_lookup[d3_key]
        row["d3"] = {
            "attempted": route["attempted"],
            "passed": route["passed"],
            "failed": route["failed"],
            "verdict": route["verdict"],
        }
    else:
        row["d3"] = {"status": "NO_D3_EVIDENCE"}

    d5 = d5_lookup[(language, engine)]
    row["d5"] = {
        "expected": d5["expected_count"],
        "present": d5["existing_count"],
        "missing": d5["missing_files"],
        "listening_disposition": d5["listening_disposition"],
    }

    notes = list(d5["notes"])
    if (language, engine) == ("ru", "silero"):
        notes.append(
            "D5 used exact unmarked prompts; stress correctness is UNKNOWN and is a listening target."
        )
    if (language, engine) == ("ru", "piper"):
        notes.append(
            "Alternate Irina passed D3 6/6 but has no D1/D5 evidence and an unknown immediate dataset licence."
        )
    row["notes"] = notes
    return row


def build_d5_and_table(d1: dict, d3: dict) -> tuple[dict, dict, str, str]:
    prompts_path = OUT_D / "D5_prompts.json"
    audit_path = ARTIFACTS / "d5_audit" / "audit_summary.json"
    prompts = load_json(prompts_path)
    audit = load_json(audit_path)
    inventory = audit["inventory"]
    assert sha256_file(prompts_path) == audit["prompt_manifest"]["file_sha256"]
    assert inventory["expected_wav_count"] == 280
    assert inventory["existing_wav_count"] == 278
    assert inventory["missing_expected_count"] == 2
    assert inventory["extra_unexpected_count"] == 0
    assert inventory["invalid_filename_count"] == 0
    assert inventory["all_existing_wavs_readable"]
    assert inventory["all_existing_wavs_pass_waveform_gate"]
    assert [row["filename"] for row in audit["missing_expected"]] == [
        "zh_kokoro_09.wav",
        "zh_kokoro_14.wav",
    ]

    d1_lookup = {
        (row["language"], row["engine_id"]): row for row in d1["comparisons"]
    }
    d3_lookup = {}
    for row in d3["routes"]:
        engine = row["engine"].lower()
        if "supertonic" in engine:
            engine = "supertonic"
        elif "matcha" in engine:
            engine = "matcha"
        elif engine == "kokoro":
            engine = "kokoro"
        elif engine == "silero_v5_cis_base":
            engine = "silero"
        elif engine == "piper":
            engine = "piper"
        key = (row["language"], engine)
        if key == ("ru", "piper") and row.get("voice") == "irina":
            continue
        d3_lookup[key] = row
    d5_lookup = {
        (row["language"], row["engine"]): row for row in audit["route_summary"]
    }

    languages = {}
    for language in LANGUAGE_ORDER:
        route_rows = [
            make_route_row(language, engine, d1_lookup, d3_lookup, d5_lookup)
            for engine in audit["expected_engine_matrix"][language]
        ]
        technical = [
            row["engine_id"]
            for row in route_rows
            if row["technical"] == "SPRINT_CANDIDATE_NOT_GATE_CLEARED"
        ]
        commercial = [
            row["engine_id"]
            for row in route_rows
            if row["overall"] == "CANDIDATE_PENDING_FULL_GATE_AND_LISTENING"
        ]
        unsafe = [
            row["engine_id"]
            for row in route_rows
            if row["overall"].startswith("REJECT_UNSAFE")
        ]
        limited = [
            row["engine_id"]
            for row in route_rows
            if row["engine_id"] not in set(technical + unsafe)
        ]
        languages[language] = {
            "technical_candidates_on_sprint_evidence": technical,
            "candidate_routes_without_an_identified_lane_commercial_blocker": commercial,
            "unsafe_routes": unsafe,
            "limited_or_evidence_incomplete_routes": limited,
            "no_lane_evidence_or_unsupported": NO_EVIDENCE[language],
            "routes": route_rows,
        }

    table = {
        "schema_version": 1,
        "task": "Lane D five-language TTS route table",
        "status": "NO_SINGLE_SAFE_COMMERCIALLY_UNBLOCKED_FIVE_LANGUAGE_ENGINE",
        "evidence_classes": ["HOST", "EXACT", "PRIOR", "PROJECTED", "UNKNOWN"],
        "evidence_note": (
            "Technical measurements are HOST; hashes/counts and pinned source text are "
            "EXACT; earlier phone anchors are PRIOR; replacement arithmetic is PROJECTED; "
            "unmeasured or unpriced quantities remain UNKNOWN."
        ),
        "headline": {
            "single_engine_safe_all_five": False,
            "single_engine_commercially_unblocked_all_five": False,
            "general_zh_ship_route_from_tested_set": False,
            "reason": (
                "Kokoro ZH is unsafe; Matcha ZH omits Latin/OOV spans and is "
                "commercially blocked by Baker/checkpoint provenance."
            ),
            "best_supported_sprint_routes": {
                "en": ["kokoro", "supertonic"],
                "es": ["kokoro", "supertonic"],
                "fr": ["kokoro", "supertonic"],
                "ru": ["supertonic"],
                "zh": [],
            },
        },
        "interpretation": {
            "technical_candidate": (
                "Completed D1 point-estimate arm, passed D3, and complete D5 bundle. "
                "This is not full TH-13 or product acceptance."
            ),
            "commercial_candidate": (
                "No lane-identified model licence blocker, or OpenRAIL-M conditional use. "
                "This is not legal advice or a complete training-data audit."
            ),
            "naturalness": "USER_LISTENING_REQUIRED; no panel was run",
        },
        "languages": languages,
        "cross_cutting_findings": [
            "All complete D1 point estimates are <=5 pp, but n=40 is below TH-13's n>=60 requirement.",
            "All tested Piper voices are directly or indirectly Lessac-derived and commercially blocked on pinned evidence.",
            "Silero v5_cis_base is 20,926,952 learned parameters [EXACT], MIT, and D3-safe, but has no D1 backcheck and needs an external Russian stress frontend.",
            "Matcha ZH is Han-short-input safe but silently ignores Latin/OOV tokens and is commercially blocked.",
            "Kokoro v1.1 ZH reproduced non-finite PCM on short and long inputs and is rejected.",
        ],
        "evidence": {
            "D1": "D1.json",
            "D2": "D2.json",
            "D3": "D3.json",
            "D4": "D4.json",
            "D5": "D5.json",
            "listening_guide": "LISTEN.md",
        },
    }

    d5 = {
        "schema_version": 1,
        "sprint_id": "D5",
        "lane": "D",
        "status": "COMPLETE_WITH_TWO_SAFETY_WITHHELD_KOKORO_ZH_CLIPS",
        "evidence_class": "HOST",
        "prompt_manifest": audit["prompt_manifest"],
        "filename_contract": "<lang>_<engine>_<idx>.wav with two-digit 01..20 indexes",
        "inventory": inventory,
        "formats": audit["format_counts"],
        "waveform_gate": audit["lane_waveform_gate"],
        "missing_expected": audit["missing_expected"],
        "route_summary": audit["route_summary"],
        "diagnostic_classification": audit["diagnostic_classification"],
        "listening_workflow": audit["recommended_listen_md"],
        "naturalness_disposition": "USER_JUDGMENT_REQUIRED_NO_PANEL",
        "artifacts": {
            "listening_directory": "artifacts/listening/",
            "audit": {
                "path": rel(audit_path),
                "sha256": sha256_file(audit_path),
            },
            "guide": "LISTEN.md",
        },
    }

    route_lines = []
    for row in audit["route_summary"]:
        role = row["listening_disposition"]
        route_lines.append(
            f"| {row['language'].upper()} | `{row['engine']}` | "
            f"{row['existing_count']}/{row['expected_count']} | `{role}` |"
        )
    d5_md = f"""# D5 — Listening bundle

Status: **complete with two correctly withheld safety failures**. Evidence class for all bundle counts, byte totals, and waveform verdicts below: **`[HOST]`**. The bundle contains **278/280 expected WAVs** ({inventory['total_wav_bytes']:,} bytes). Every existing file is a readable mono PCM16 WAV and passes the exact Lane D finite/non-silent/duration/sustained-clipping gate.

The only missing files are `zh_kokoro_09.wav` and `zh_kokoro_14.wav`. Both Kokoro ZH generations produced non-finite PCM, so no invalid WAV was serialized. The other 18 Kokoro ZH clips remain **diagnostic material for an unsafe route**.

| Language | engine | clips | listening role |
|---|---|---:|---|
{chr(10).join(route_lines)}

Use [`LISTEN.md`](LISTEN.md) row by row: compare every engine for index 01 before moving to 02. The same semantic case has the same two-digit index in every language, and every link shows the exact filename.

Important labels in the guide:

- Piper ES and RU, plus Kokoro ZH, are D3-unsafe diagnostic routes. Listening cannot override those failures.
- All Piper routes remain commercially blocked by Lessac-derived provenance; Matcha remains blocked by Baker/checkpoint licensing.
- Matcha ZH indices 02, 09, 11, 12, 13, and 14 are explicit omission checks. Its frontend ignores unsupported Latin/OOV spans.
- Silero RU uses exact unmarked prompts. Listen especially for Russian stress; the package has no bundled accentor and D1 was not run for Silero.

Clip-manifest SHA-256: `{inventory['clip_hash_manifest_sha256']}`. Prompt-manifest SHA-256: `{audit['prompt_manifest']['file_sha256']}`. No listening panel or naturalness acceptance is claimed; the user's personal judgment is the intended D5 result.
"""

    listen_md = build_listen_md(prompts, audit)
    return d5, table, d5_md, listen_md


def build_listen_md(prompts: dict, audit: dict) -> str:
    clips = {(row["language"], row["engine"], row["listen_index"]): row for row in audit["clips"]}
    missing = {(row["language"], row["engine"], row["listen_index"]): row for row in audit["missing_expected"]}
    risk_indices = {
        row["listen_index"]: row["reason"]
        for row in audit["diagnostic_classification"]["matcha_zh_text_coverage_risk"]
    }
    engine_status = {
        ("en", "kokoro"): "✅ candidate",
        ("en", "piper"): "⚠ licence-blocked",
        ("en", "supertonic"): "✅ candidate",
        ("es", "kokoro"): "✅ candidate",
        ("es", "piper"): "⛔ D3-unsafe + blocked",
        ("es", "supertonic"): "✅ candidate",
        ("fr", "kokoro"): "✅ candidate",
        ("fr", "piper"): "⚠ licence-blocked",
        ("fr", "supertonic"): "✅ candidate",
        ("ru", "piper"): "⛔ D3-unsafe + blocked",
        ("ru", "silero"): "⚠ stress / no D1",
        ("ru", "supertonic"): "✅ candidate",
        ("zh", "kokoro"): "⛔ non-finite route",
        ("zh", "matcha"): "⚠ OOV + licence",
    }
    marker = {
        ("es", "piper"): "⛔ ",
        ("ru", "piper"): "⛔ ",
        ("zh", "kokoro"): "⛔ ",
        ("en", "piper"): "⚠ ",
        ("fr", "piper"): "⚠ ",
        ("ru", "silero"): "⚠ ",
        ("zh", "matcha"): "⚠ ",
    }
    lines = [
        "# Lane D listening guide",
        "",
        "This is the naturalness decision that could not fit a listening panel. Use headphones, keep volume fixed within a language, and **finish one row across all engines before moving to the next row**. Change `☐` to `☑` as you finish each comparison.",
        "",
        "Safety and licence labels are already decided evidence. Listening can select among candidates or describe a failure, but it cannot make a D3-unsafe or commercially blocked route shippable.",
        "",
        "Bundle counts, byte totals, and waveform safety results are `[HOST]` measurements.",
        "",
        "Suggested order: (1) EN/ES/FR row by row, (2) ZH indices 12–14 first to catch dropped acronyms, then the other ZH rows, (3) RU Supertonic vs Silero with special attention to stress; Piper RU is diagnostic only.",
        "",
        "If your Markdown viewer does not play a link, open `artifacts/listening/` and sort by filename. The exact filename is always the link label.",
        "",
        "## Quick verdict sheet",
        "",
        "| Language | preferred engine | reject / concern | notes |",
        "|---|---|---|---|",
    ]
    for language in LANGUAGE_ORDER:
        lines.append(f"| {LANGUAGE_NAMES[language]} |  |  |  |")

    for language in LANGUAGE_ORDER:
        engines = audit["expected_engine_matrix"][language]
        lines.extend(
            [
                "",
                f"## {LANGUAGE_NAMES[language]} (`{language}`)",
                "",
                "Engine columns: "
                + " · ".join(
                    f"`{engine}` {engine_status[(language, engine)]}" for engine in engines
                ),
                "",
            ]
        )
        header = "| Done | # | focus | source text | " + " | ".join(engines) + " |"
        separator = "|---|---:|---|---|" + "---|" * len(engines)
        lines.extend([header, separator])
        for prompt in prompts["languages"][language]:
            idx = int(prompt["listen_index"])
            focus = prompt["primary_category"].replace("_", " ")
            if language == "zh" and idx in risk_indices:
                focus += f"; **omission check:** {risk_indices[idx]}"
            cells = []
            for engine in engines:
                key = (language, engine, idx)
                if key in missing:
                    cells.append("⛔ non-finite output; no WAV")
                    continue
                clip = clips[key]
                filename = clip["filename"]
                prefix = marker.get((language, engine), "")
                if (language, engine) == ("zh", "matcha") and idx not in risk_indices:
                    prefix = "⚠ "
                cells.append(
                    f"{prefix}[{filename}]({clip['relative_path']})"
                )
            lines.append(
                "| ☐ | {idx:02d} | {focus} | {text} | {cells} |".format(
                    idx=idx,
                    focus=escape_md(focus),
                    text=escape_md(prompt["text"]),
                    cells=" | ".join(cells),
                )
            )

    lines.extend(
        [
            "",
            "## Bundle integrity",
            "",
            f"- Existing WAVs: **{audit['inventory']['existing_wav_count']}/{audit['inventory']['expected_wav_count']}**; the two missing files are known Kokoro ZH non-finite outputs.",
            f"- Existing WAV bytes: **{audit['inventory']['total_wav_bytes']:,}**.",
            f"- Ordered clip-hash manifest SHA-256: `{audit['inventory']['clip_hash_manifest_sha256']}`.",
            f"- Prompt manifest SHA-256: `{audit['prompt_manifest']['file_sha256']}`.",
            "- All 278 existing files are readable, finite, non-silent, 0.15–30 s, and free of sustained clipping above -0.1 dBFS for more than 5 ms.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    d1, d1_md = build_d1()
    d3, d3_md = build_d3()
    d5, table, d5_md, listen_md = build_d5_and_table(d1, d3)
    write_json(OUT_D / "D1.json", d1)
    (OUT_D / "D1.md").write_text(d1_md, encoding="utf-8")
    write_json(OUT_D / "D3.json", d3)
    (OUT_D / "D3.md").write_text(d3_md, encoding="utf-8")
    write_json(OUT_D / "D5.json", d5)
    (OUT_D / "D5.md").write_text(d5_md, encoding="utf-8")
    write_json(OUT_D / "tts_table.json", table)
    (OUT_D / "LISTEN.md").write_text(listen_md, encoding="utf-8")
    print("wrote D1, D3, D5, tts_table.json, and LISTEN.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
