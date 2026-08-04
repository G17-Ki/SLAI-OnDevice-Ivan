#!/usr/bin/env python3
"""Transcribe and score Lane G G2 MOSS-TTS Mandarin evidence.

Run the two phases with the already-retained Lane D environments:

  artifacts/matcha/venv/bin/python run_g2_moss_scoring.py transcribe
  artifacts/main/venv/bin/python run_g2_moss_scoring.py score

The first phase uses the exact pinned Paraformer backcheck from D1.  The second
phase imports the frozen D1 normalizer and score functions, then writes G2.json
and G2.md.  No training, quantization, or model mutation occurs here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


OUT_D = Path(__file__).resolve().parent
ARTIFACTS = OUT_D / "artifacts"
MOSS_RUNNER_PATH = OUT_D / "run_moss_g.py"
MOSS_ROOT = ARTIFACTS / "moss"
MOSS_RESULTS = MOSS_ROOT / "results"
CORPUS_PATH = OUT_D / "D1_corpus.json"
D1_SYNTH_PATH = MOSS_RESULTS / "g2_d1_zh_synthesis.json"
MIXED_SYNTH_PATH = MOSS_RESULTS / "g2_mixed_synthesis.json"
TRANSCRIPT_PATH = MOSS_RESULTS / "g2_paraformer.json"
HUMAN_TRANSCRIPT_PATH = ARTIFACTS / "asr_zh" / "paraformer_results.json"
ASR_ROOT = ARTIFACTS / "asr_zh" / "sherpa-onnx-paraformer-zh-small-2024-03-09"
ASR_MODEL_PATH = ASR_ROOT / "model.int8.onnx"
ASR_TOKENS_PATH = ASR_ROOT / "tokens.txt"
ASR_MVN_PATH = ASR_ROOT / "am.mvn"
SCORING_DIR = ARTIFACTS / "d1_scoring"
NORMALIZER_PATH = SCORING_DIR / "normalization.py"
NORMALIZER_FIXTURES_PATH = SCORING_DIR / "normalization_fixtures.json"
RESULT_JSON_PATH = OUT_D / "G2.json"
RESULT_MD_PATH = OUT_D / "G2.md"

THREADS = 2
EXPECTED_IDS = tuple(f"fleurs_{index:02d}" for index in range(1, 41))
ENGINE_ID = "moss_tts_nano_100m"
GROUP_ID = "moss_tts_nano_100m_zh"
BOOTSTRAP_B = 10_000
BOOTSTRAP_SEED = 20_260_801
CONTROL_THRESHOLD_PP = 2.0
TH13_MARGIN_PP = 5.0
TH13_MINIMUM_N = 60
STORAGE_CAP_BYTES = 3_000_000_000

# Frozen inputs from the completed Lane D harness and the retained G2 synthesis.
EXPECTED_HASHES = {
    "corpus": "b4c838a70db4a2d9955f5c58c4470b91d708ea08fcfd23f0ae0d6fbc0d0496a2",
    "corpus_items": "fac305ffe07d7fa9a508111001aa114e610f5541a24e0a9c74ab27600e5844c5",
    "human_transcripts": "d49a1b291294d2068100edd21c0acd1032634607dd68ae21a18d538b0fa19cd8",
    "asr_model": "3ef6c19369b912f7caf3cef8e545c5ccd1a33d9d7ec792a46668dc41c4b229ec",
    "asr_tokens": "4b2d964e18b9cf139b473003b6698fb2ed9a2a5ec55b93daa677b28f578897aa",
    "asr_mvn": "29b3c740a2c0cfc6b308126d31d7f265fa2be74f3bb095cd2f143ea970896ae5",
    "normalizer": "d754293b250e6532b1254f46afbd3a4b7fc36383e64cd44ac3ef1d79493bc989",
    "normalizer_fixtures": "2727aa790225ba7a2e4852f73793e941d962662e36f1855a807980dfc75f9835",
    "d1_synthesis": "717d6f54af29db771c2ed60eab3a87f6aa7feedf1e4c71998f936373206b09ad",
    "mixed_synthesis": "34d383641a6464a19861dbae9ea2842ccd9ebc3cc1fb59e66bdb7e5974a3b1e8",
}

EXPECTED_MIXED_CASES = (
    "wifi_full",
    "wifi_ablated",
    "5g_full",
    "5g_ablated",
    "digits_full",
    "digits_ablated",
    "cpu_full",
    "cpu_ablated",
    "api_json_full",
    "api_json_ablated",
    "vpn_full",
    "vpn_ablated",
)

# Text removed from the ablated member of each pair. These are normalized with
# the exact D1 normalizer before checking whether the reference ASR emitted the
# literal expected unit(s). A miss is diagnostic, not proof of silence: the ZH
# Paraformer itself is not a validated Latin-acronym recognizer.
PROBE_SPANS = {
    "wifi": ("WiFi",),
    "5g": ("5G",),
    "digits": ("5072",),
    "cpu": ("CPU",),
    "api_json": ("API", "JSON"),
    "vpn": ("VPN",),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        stream.write(text)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: dict) -> None:
    atomic_write_text(
        path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )


def relative(path: Path) -> str:
    return str(path.relative_to(OUT_D))


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    import numpy as np

    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def assert_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise SystemExit(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")


def directory_bytes(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        if path.is_file() and not path.is_symlink():
            total += path.stat().st_size
    return total


def validate_frozen_files(*, include_human: bool) -> None:
    required = {
        "corpus": CORPUS_PATH,
        "asr_model": ASR_MODEL_PATH,
        "asr_tokens": ASR_TOKENS_PATH,
        "asr_mvn": ASR_MVN_PATH,
        "normalizer": NORMALIZER_PATH,
        "normalizer_fixtures": NORMALIZER_FIXTURES_PATH,
        "d1_synthesis": D1_SYNTH_PATH,
        "mixed_synthesis": MIXED_SYNTH_PATH,
    }
    if include_human:
        required["human_transcripts"] = HUMAN_TRANSCRIPT_PATH
    for label, path in required.items():
        if not path.is_file():
            raise SystemExit(f"required input missing: {path}")
        assert_hash(path, EXPECTED_HASHES[label], label)


def load_and_validate_inputs(*, include_human: bool) -> tuple[dict, dict, dict]:
    validate_frozen_files(include_human=include_human)
    corpus = read_json(CORPUS_PATH)
    items = corpus.get("items", [])
    if tuple(row.get("item_id") for row in items) != EXPECTED_IDS:
        raise SystemExit("D1 corpus is not exactly fleurs_01..fleurs_40")
    if sha256_json(items) != EXPECTED_HASHES["corpus_items"]:
        raise SystemExit("D1 canonical item hash mismatch")

    d1_synth = read_json(D1_SYNTH_PATH)
    d1_records = d1_synth.get("records", [])
    if tuple(row.get("case_id") for row in d1_records) != EXPECTED_IDS:
        raise SystemExit("G2 D1 synthesis does not contain the exact 40 item IDs")
    if d1_synth.get("attempted") != 40 or d1_synth.get("passed") != 40:
        raise SystemExit("G2 D1 synthesis is not a complete 40/40 safe arm")
    for corpus_row, synth_row in zip(items, d1_records):
        if synth_row.get("raw_text") != corpus_row["texts"]["zh"]:
            raise SystemExit(f"frozen reference mismatch: {synth_row.get('case_id')}")
        validate_synthesis_wav(synth_row)

    mixed_synth = read_json(MIXED_SYNTH_PATH)
    mixed_records = mixed_synth.get("records", [])
    if tuple(row.get("case_id") for row in mixed_records) != EXPECTED_MIXED_CASES:
        raise SystemExit("mixed-script case inventory or order changed")
    if mixed_synth.get("attempted") != 12 or mixed_synth.get("passed") != 12:
        raise SystemExit("mixed-script synthesis is not the complete 12-WAV inventory")
    for synth_row in mixed_records:
        validate_synthesis_wav(synth_row)
    return corpus, d1_synth, mixed_synth


def validate_synthesis_wav(row: dict) -> Path:
    if not row.get("passed"):
        raise SystemExit(f"unsafe synthesis record: {row.get('case_id')}")
    output = row.get("output", {})
    raw_gate = row.get("raw_waveform_gate", {})
    if not output.get("passed") or not raw_gate.get("passed"):
        raise SystemExit(f"waveform gate failed: {row.get('case_id')}")
    path = OUT_D / output.get("path", "")
    if not path.is_file():
        raise SystemExit(f"synthesis WAV missing: {path}")
    if path.stat().st_size != output.get("bytes"):
        raise SystemExit(f"synthesis WAV size mismatch: {path}")
    if sha256_file(path) != output.get("sha256"):
        raise SystemExit(f"synthesis WAV hash mismatch: {path}")
    return path


def host_record(runtime_versions: dict) -> dict:
    cpu = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": cpu,
        "python": platform.python_version(),
        "threads": THREADS,
        "provider": "cpu",
        "runtime_versions": runtime_versions,
    }


def transcribe_one(recognizer, path: Path, metadata: dict) -> dict:
    import numpy as np
    import soundfile as sf

    read_start = time.perf_counter_ns()
    samples_2d, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    read_s = (time.perf_counter_ns() - read_start) / 1e9
    samples = (
        samples_2d[:, 0]
        if samples_2d.shape[1] == 1
        else samples_2d.mean(axis=1, dtype=np.float32)
    )
    if samples.size == 0 or not np.isfinite(samples).all():
        raise SystemExit(f"invalid source waveform: {path}")

    stream = recognizer.create_stream()
    accept_start = time.perf_counter_ns()
    stream.accept_waveform(int(sample_rate), samples)
    accept_s = (time.perf_counter_ns() - accept_start) / 1e9
    decode_start = time.perf_counter_ns()
    recognizer.decode_stream(stream)
    decode_s = (time.perf_counter_ns() - decode_start) / 1e9
    result = stream.result
    duration_s = float(samples.size / sample_rate)
    return {
        **metadata,
        "asr_text_raw": result.text,
        "asr_tokens_raw": list(result.tokens),
        "wav": {
            "path": relative(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "sample_rate_hz": int(sample_rate),
            "channels_in_file": int(samples_2d.shape[1]),
            "frames": int(samples_2d.shape[0]),
            "duration_s": duration_s,
            "peak_abs_after_mono_mix": float(np.max(np.abs(samples))),
        },
        "timing": {
            "read_s": read_s,
            "accept_waveform_and_resample_s": accept_s,
            "decode_s": decode_s,
            "decode_rtf": decode_s / duration_s,
            "read_accept_decode_s": read_s + accept_s + decode_s,
        },
    }


def transcription_group_summary(records: list[dict]) -> dict:
    decode = [row["timing"]["decode_s"] for row in records]
    rtf = [row["timing"]["decode_rtf"] for row in records]
    durations = [row["wav"]["duration_s"] for row in records]
    return {
        "clip_count": len(records),
        "empty_transcript_count": sum(not row["asr_text_raw"] for row in records),
        "input_duration_total_s": float(sum(durations)),
        "decode_total_s": float(sum(decode)),
        "decode_s_median": float(statistics.median(decode)),
        "decode_s_p95_numpy_linear": percentile(decode, 95),
        "decode_rtf_median": float(statistics.median(rtf)),
        "decode_rtf_p95_numpy_linear": percentile(rtf, 95),
    }


def transcribe_phase() -> int:
    try:
        import numpy as np
        import sherpa_onnx
        import soundfile as sf
    except ImportError as error:
        raise SystemExit(
            "transcribe requires the retained Matcha environment with sherpa_onnx"
        ) from error

    corpus, d1_synth, mixed_synth = load_and_validate_inputs(include_human=False)
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    load_start = time.perf_counter_ns()
    recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
        paraformer=str(ASR_MODEL_PATH),
        tokens=str(ASR_TOKENS_PATH),
        num_threads=THREADS,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
        debug=False,
        provider="cpu",
    )
    model_load_s = (time.perf_counter_ns() - load_start) / 1e9

    corpus_by_id = {row["item_id"]: row for row in corpus["items"]}
    d1_records = []
    for synth_row in d1_synth["records"]:
        item_id = synth_row["case_id"]
        corpus_row = corpus_by_id[item_id]
        record = transcribe_one(
            recognizer,
            validate_synthesis_wav(synth_row),
            {
                "language": "zh",
                "item_id": item_id,
                "sentence_id": corpus_row.get("sentence_id"),
                "categories": corpus_row.get("categories", []),
                "group_id": GROUP_ID,
                "source_kind": "synthesized",
                "engine_id": ENGINE_ID,
                "reference_text": corpus_row["texts"]["zh"],
                "synthesis_seed": synth_row.get("seed"),
                "synthesis_generated_frames": synth_row.get("generated_frames"),
                "synthesis_max_new_frames": synth_row.get("max_new_frames"),
                "synthesis_max_new_frames_hit": synth_row.get("max_new_frames_hit"),
            },
        )
        d1_records.append(record)
        print(f"D1 {item_id}: {record['asr_text_raw']}", flush=True)

    mixed_records = []
    for synth_row in mixed_synth["records"]:
        record = transcribe_one(
            recognizer,
            validate_synthesis_wav(synth_row),
            {
                "language": "zh",
                "case_id": synth_row["case_id"],
                "pair_id": synth_row["pair_id"],
                "pair_role": synth_row["pair_role"],
                "group_id": f"{ENGINE_ID}_mixed_script",
                "source_kind": "mixed_script_probe",
                "engine_id": ENGINE_ID,
                "reference_text": synth_row["raw_text"],
                "synthesis_seed": synth_row.get("seed"),
                "synthesis_generated_frames": synth_row.get("generated_frames"),
                "synthesis_max_new_frames": synth_row.get("max_new_frames"),
                "synthesis_max_new_frames_hit": synth_row.get("max_new_frames_hit"),
            },
        )
        mixed_records.append(record)
        print(f"mixed {synth_row['case_id']}: {record['asr_text_raw']}", flush=True)

    payload = {
        "schema_version": 1,
        "task": "Lane G G2 MOSS Mandarin Paraformer transcription",
        "evidence_class": "HOST",
        "scoring_status": "raw_pairs_only_score_phase_uses_frozen_d1_harness",
        "language": "zh",
        "reference_asr": {
            "family": "FunASR Paraformer",
            "artifact_name": "sherpa-onnx-paraformer-zh-small-2024-03-09",
            "upstream_model": "crazyant/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-onnx",
            "model_path": relative(ASR_MODEL_PATH),
            "model_bytes": ASR_MODEL_PATH.stat().st_size,
            "model_sha256": sha256_file(ASR_MODEL_PATH),
            "tokens_sha256": sha256_file(ASR_TOKENS_PATH),
            "am_mvn_sha256": sha256_file(ASR_MVN_PATH),
            "quantization": "prebuilt INT8 artifact; no local quantization performed",
            "model_sample_rate_hz": 16000,
            "feature_dim": 80,
            "decoding_method": "greedy_search",
            "inverse_text_normalization": "none",
            "model_load_s": model_load_s,
        },
        "frozen_inputs": {
            "corpus": {
                "path": relative(CORPUS_PATH),
                "sha256": sha256_file(CORPUS_PATH),
                "items_canonical_sha256": sha256_json(corpus["items"]),
            },
            "d1_synthesis": {
                "path": relative(D1_SYNTH_PATH),
                "sha256": sha256_file(D1_SYNTH_PATH),
            },
            "mixed_synthesis": {
                "path": relative(MIXED_SYNTH_PATH),
                "sha256": sha256_file(MIXED_SYNTH_PATH),
            },
        },
        "host": host_record(
            {
                "sherpa_onnx": sherpa_onnx.__version__,
                "soundfile": sf.__version__,
                "numpy": np.__version__,
            }
        ),
        "run_started_utc": started_utc,
        "run_finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": {
            "d1": transcription_group_summary(d1_records),
            "mixed": transcription_group_summary(mixed_records),
            "total_clip_count": len(d1_records) + len(mixed_records),
        },
        "d1_records": d1_records,
        "mixed_records": mixed_records,
    }
    atomic_write_json(TRANSCRIPT_PATH, payload)
    print(f"wrote {TRANSCRIPT_PATH}")
    return 0


def load_scoring_modules():
    sys.path.insert(0, str(SCORING_DIR))
    import normalization
    import score_d1
    import transcript_common

    return normalization, score_d1, transcript_common


def normalizer_control(primary: dict, basic: dict) -> dict:
    control = {
        "threshold_pp": CONTROL_THRESHOLD_PP,
        "primary_human_rate_pp": primary["human"]["point_pp"],
        "basic_human_rate_pp": basic["human"]["point_pp"],
        "human_abs_delta_pp": abs(
            primary["human"]["point_pp"] - basic["human"]["point_pp"]
        ),
        "primary_synth_rate_pp": primary["synth"]["point_pp"],
        "basic_synth_rate_pp": basic["synth"]["point_pp"],
        "synth_abs_delta_pp": abs(
            primary["synth"]["point_pp"] - basic["synth"]["point_pp"]
        ),
        "primary_matched_delta_pp": primary["delta"]["point_pp"],
        "basic_matched_delta_pp": basic["delta"]["point_pp"],
        "matched_delta_abs_difference_pp": abs(
            primary["delta"]["point_pp"] - basic["delta"]["point_pp"]
        ),
    }
    control["cells"] = {
        "human_error_rate": {
            "abs_difference_pp": control["human_abs_delta_pp"],
        },
        "synth_error_rate": {
            "abs_difference_pp": control["synth_abs_delta_pp"],
        },
        "matched_human_delta": {
            "abs_difference_pp": control["matched_delta_abs_difference_pp"],
        },
    }
    for cell in control["cells"].values():
        cell["status"] = (
            "UNSTABLE"
            if cell["abs_difference_pp"] > CONTROL_THRESHOLD_PP
            else "STABLE"
        )
    control["ordering_or_threshold_interpretation"] = (
        "UNSTABLE"
        if any(cell["status"] == "UNSTABLE" for cell in control["cells"].values())
        else "STABLE"
    )
    control["ranking_inadmissible"] = (
        control["ordering_or_threshold_interpretation"] == "UNSTABLE"
    )
    return control


def score_counts(score_d1, reference: dict, hypothesis: dict) -> dict:
    counts = score_d1.edit_counts(reference["tokens"], hypothesis["tokens"])
    return {
        **counts,
        "error_rate": counts["errors"] / counts["reference_units"],
        "error_rate_pp": 100.0 * counts["errors"] / counts["reference_units"],
        "reference_normalized": reference["normalized_text"],
        "hypothesis_normalized": hypothesis["normalized_text"],
        "reference_tokens": reference["tokens"],
        "hypothesis_tokens": hypothesis["tokens"],
    }


def contains_contiguous(tokens: list[str], needle: list[str]) -> bool:
    if not needle:
        return True
    width = len(needle)
    return any(tokens[index : index + width] == needle for index in range(len(tokens) - width + 1))


def mixed_pair_results(mixed_synth: dict, transcript_payload: dict, normalization, score_d1) -> dict:
    synth_by_case = {row["case_id"]: row for row in mixed_synth["records"]}
    asr_by_case = {row["case_id"]: row for row in transcript_payload["mixed_records"]}
    pairs = []
    for pair_id in PROBE_SPANS:
        full_synth = synth_by_case[f"{pair_id}_full"]
        ablated_synth = synth_by_case[f"{pair_id}_ablated"]
        full_asr = asr_by_case[f"{pair_id}_full"]
        ablated_asr = asr_by_case[f"{pair_id}_ablated"]

        full_ref = normalization.normalize("zh", full_synth["raw_text"], "primary")
        ablated_ref = normalization.normalize("zh", ablated_synth["raw_text"], "primary")
        full_hyp = normalization.normalize("zh", full_asr["asr_text_raw"], "primary")
        ablated_hyp = normalization.normalize("zh", ablated_asr["asr_text_raw"], "primary")
        span_checks = []
        for span in PROBE_SPANS[pair_id]:
            normalized_span = normalization.normalize("zh", span, "primary")
            span_tokens = normalized_span["tokens"]
            span_checks.append(
                {
                    "raw_span": span,
                    "normalized_span": normalized_span["normalized_text"],
                    "tokens": span_tokens,
                    "literal_tokens_detected_in_full_asr": contains_contiguous(
                        full_hyp["tokens"], span_tokens
                    ),
                    "literal_tokens_detected_in_ablated_asr": contains_contiguous(
                        ablated_hyp["tokens"], span_tokens
                    ),
                }
            )

        cap_cases = [
            row["case_id"]
            for row in (full_synth, ablated_synth)
            if row.get("max_new_frames_hit")
        ]
        literal_positive = all(
            check["literal_tokens_detected_in_full_asr"]
            and not check["literal_tokens_detected_in_ablated_asr"]
            for check in span_checks
        )
        if cap_cases:
            disposition = "INCONCLUSIVE_CAP_HIT_NONTERMINATED"
            establishes_span_coverage = False
        elif literal_positive:
            disposition = "POSITIVE_LITERAL_REFERENCE_ASR_DETECTION"
            establishes_span_coverage = False
        elif full_hyp["tokens"] == ablated_hyp["tokens"]:
            disposition = "NO_LITERAL_ASR_EVIDENCE_FULL_AND_ABLATED_TRANSCRIPTS_EQUAL"
            establishes_span_coverage = False
        else:
            disposition = "INCONCLUSIVE_REFERENCE_ASR_DID_NOT_EMIT_LITERAL_SPAN"
            establishes_span_coverage = False

        pairs.append(
            {
                "pair_id": pair_id,
                "expected_removed_spans": list(PROBE_SPANS[pair_id]),
                "full": {
                    "case_id": full_synth["case_id"],
                    "input_text": full_synth["raw_text"],
                    "asr_text_raw": full_asr["asr_text_raw"],
                    "asr_tokens_raw": full_asr["asr_tokens_raw"],
                    "normalized_asr": full_hyp["normalized_text"],
                    "generated_frames": full_synth["generated_frames"],
                    "max_new_frames": full_synth["max_new_frames"],
                    "max_new_frames_hit": full_synth["max_new_frames_hit"],
                    "duration_s": full_synth["output"]["duration_s"],
                    "wav": {
                        "path": full_synth["output"]["path"],
                        "bytes": full_synth["output"]["bytes"],
                        "sha256": full_synth["output"]["sha256"],
                    },
                    "tokenizer_input_evidence": full_synth["text_tokenization"],
                },
                "ablated": {
                    "case_id": ablated_synth["case_id"],
                    "input_text": ablated_synth["raw_text"],
                    "asr_text_raw": ablated_asr["asr_text_raw"],
                    "asr_tokens_raw": ablated_asr["asr_tokens_raw"],
                    "normalized_asr": ablated_hyp["normalized_text"],
                    "generated_frames": ablated_synth["generated_frames"],
                    "max_new_frames": ablated_synth["max_new_frames"],
                    "max_new_frames_hit": ablated_synth["max_new_frames_hit"],
                    "duration_s": ablated_synth["output"]["duration_s"],
                    "wav": {
                        "path": ablated_synth["output"]["path"],
                        "bytes": ablated_synth["output"]["bytes"],
                        "sha256": ablated_synth["output"]["sha256"],
                    },
                    "tokenizer_input_evidence": ablated_synth["text_tokenization"],
                },
                "primary_normalizer_diagnostics": {
                    "full_audio_vs_full_reference": score_counts(
                        score_d1, full_ref, full_hyp
                    ),
                    "full_audio_vs_ablated_reference": score_counts(
                        score_d1, ablated_ref, full_hyp
                    ),
                    "ablated_audio_vs_ablated_reference": score_counts(
                        score_d1, ablated_ref, ablated_hyp
                    ),
                    "full_asr_vs_ablated_asr": score_counts(
                        score_d1, ablated_hyp, full_hyp
                    )
                    if ablated_hyp["tokens"]
                    else None,
                    "literal_span_checks": span_checks,
                },
                "cap_hit_cases": cap_cases,
                "termination_status": (
                    "NONTERMINATED_CAP_HIT_PRESENT"
                    if cap_cases
                    else "BOTH_ARMS_TERMINATED_BELOW_CAP"
                ),
                "reference_asr_disposition": disposition,
                "establishes_spoken_span_coverage": establishes_span_coverage,
                "interpretation": (
                    "Tokenizer roundtrip and zero UNKs establish only that the text front end accepted the input. "
                    "The Mandarin Paraformer diagnostic can show literal recognition or an audio/text difference, "
                    "but it cannot by itself certify correct Latin-span pronunciation."
                ),
            }
        )

    cap_hit_cases = [
        row["case_id"]
        for row in mixed_synth["records"]
        if row.get("max_new_frames_hit")
    ]
    return {
        "status": "UNRESOLVED",
        "status_reason": (
            "Four of twelve arms hit the 160-frame generation cap, so two-arm termination is absent for three pairs; "
            "the ZH reference ASR is also not sufficient to certify Latin acronym pronunciation."
        ),
        "pair_count": len(pairs),
        "wav_count": len(mixed_synth["records"]),
        "safe_wav_count": sum(row.get("passed") is True for row in mixed_synth["records"]),
        "cap_hit_count": len(cap_hit_cases),
        "cap_hit_cases": cap_hit_cases,
        "near_cap_non_hit_cases": [
            row["case_id"]
            for row in mixed_synth["records"]
            if not row.get("max_new_frames_hit")
            and row.get("generated_frames", 0) >= row.get("max_new_frames", 0) - 4
        ],
        "tokenizer_evidence_scope": (
            "Input-front-end evidence only. Exact SentencePiece roundtrip and zero UNKs do not establish that a span was spoken."
        ),
        "reference_asr_scope": (
            "Raw Paraformer transcripts are retained for every full/ablated WAV. Literal detection is positive diagnostic evidence only; "
            "lack of literal Latin output cannot distinguish TTS span dropping from ASR language limitations."
        ),
        "pairs": pairs,
    }


def safety_inventory(d1_synth: dict) -> dict:
    items = []
    for row in d1_synth["records"]:
        output = row["output"]
        path = OUT_D / output["path"]
        items.append(
            {
                "item_id": row["case_id"],
                "seed": row["seed"],
                "generated_frames": row["generated_frames"],
                "max_new_frames": row["max_new_frames"],
                "max_new_frames_hit": row["max_new_frames_hit"],
                "raw_waveform_gate": row["raw_waveform_gate"],
                "raw_waveform_gate_passed": row["raw_waveform_gate"]["passed"],
                "raw_waveform_assertions": row["raw_waveform_gate"]["assertions"],
                "serialized_waveform_gate_passed": output["passed"],
                "serialized_waveform_assertions": output["assertions"],
                "wav": {
                    "path": output["path"],
                    "bytes": output["bytes"],
                    "sha256": output["sha256"],
                    "rehash_matches": sha256_file(path) == output["sha256"],
                    "sample_rate_hz": output["sample_rate_hz"],
                    "channels": output["channels"],
                    "duration_s": output["duration_s"],
                    "rms": output["rms"],
                    "peak_abs": output["peak_abs"],
                    "longest_contiguous_clip_run_s": output[
                        "longest_contiguous_clip_run_s"
                    ],
                },
                "passed": row["passed"],
            }
        )
    return {
        "attempted": d1_synth["attempted"],
        "passed": d1_synth["passed"],
        "failed": d1_synth["failed"],
        "max_new_frames_hit_count": d1_synth["max_new_frames_hit_count"],
        "all_raw_waveform_gates_passed": all(
            row["raw_waveform_gate_passed"] for row in items
        ),
        "all_serialized_waveform_gates_passed": all(
            row["serialized_waveform_gate_passed"] for row in items
        ),
        "all_retained_hashes_match": all(row["wav"]["rehash_matches"] for row in items),
        "score_disposition": "SCORE_ALL_40_COMPLETE_SAFE_FILES_NO_IMPUTATION",
        "items": items,
    }


def markdown_escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(payload: dict) -> str:
    comparison = payload["d1_mandarin"]["comparison"]
    primary = comparison["primary"]
    basic = comparison["controls_and_required_alternates"]["basic"]
    control = comparison["normalizer_delta_control"]
    screen = payload["final_verdict"]["zh_point_estimate_technical_screen"]
    mixed_status = payload["final_verdict"]["mixed_script_coverage"]
    if screen == "FAIL":
        opening = (
            "MOSS-TTS-Nano fails the Mandarin D1 point-estimate technical screen on this sampled HOST run."
        )
    elif screen == "PASS":
        opening = (
            "MOSS-TTS-Nano is within the Mandarin D1 +5 pp point-estimate screen on this sampled HOST run, "
            "but n=40 cannot clear TH-13."
        )
    else:
        opening = "MOSS-TTS-Nano Mandarin D1 remains unresolved."

    lines = [
        "# G2 — MOSS-TTS-Nano Mandarin first",
        "",
        opening,
        "",
        f"**Verdict:** D1 screen `{screen}`; mixed-script coverage `{mixed_status}`; TH-13 `INSUFFICIENT_N_40_LT_60`. "
        "Every number below is `HOST`; file/model hashes are `EXACT`.",
        "",
        "## D1 matched-human result",
        "",
        "| Normalizer | Human CER | MOSS CER | ΔCER (MOSS − human) | Paired 95% CI |",
        "|---|---:|---:|---:|---:|",
        f"| Primary | {primary['human']['point_pp']:.3f}% | {primary['synth']['point_pp']:.3f}% | "
        f"{primary['delta']['point_pp']:+.3f} pp | [{primary['delta']['ci95_lo_pp']:+.3f}, {primary['delta']['ci95_hi_pp']:+.3f}] pp |",
        f"| Basic control | {basic['human']['point_pp']:.3f}% | {basic['synth']['point_pp']:.3f}% | "
        f"{basic['delta']['point_pp']:+.3f} pp | [{basic['delta']['ci95_lo_pp']:+.3f}, {basic['delta']['ci95_hi_pp']:+.3f}] pp |",
        "",
        f"The arm is complete and safety-gated: **40/40** retained mono PCM16 WAVs, all raw and serialized gates passed, "
        f"zero generation-cap hits, and no imputation. Generation used sampling (`do_sample=true`), one deterministic-seed draw per item. "
        f"Primary/basic differences are {control['human_abs_delta_pp']:.3f} pp for human CER, "
        f"{control['synth_abs_delta_pp']:.3f} pp for MOSS CER, and {control['matched_delta_abs_difference_pp']:.3f} pp for the matched delta. "
        f"Because the arm-rate cells exceed the frozen 2 pp threshold, the overall control is `{control['ordering_or_threshold_interpretation']}` and ranking is "
        f"`{'INADMISSIBLE' if control['ranking_inadmissible'] else 'ADMISSIBLE'}`.",
        "",
        "TH-13 requires n≥60. This sprint has n=40, so even a point estimate within +5 pp is only a technical screen, never a gate clearance.",
        "",
        "## Mixed-script probe",
        "",
        "| Pair | Full input → raw Paraformer ASR | Ablated input → raw Paraformer ASR | Termination | Diagnostic |",
        "|---|---|---|---|---|",
    ]
    for pair in payload["mixed_script_probe"]["pairs"]:
        full = pair["full"]
        ablated = pair["ablated"]
        lines.append(
            f"| {markdown_escape(pair['pair_id'])} | "
            f"{markdown_escape(full['input_text'])} → {markdown_escape(full['asr_text_raw'])} | "
            f"{markdown_escape(ablated['input_text'])} → {markdown_escape(ablated['asr_text_raw'])} | "
            f"{markdown_escape(pair['termination_status'])} | "
            f"{markdown_escape(pair['reference_asr_disposition'])} |"
        )
    lines.extend(
        [
            "",
            f"Four arms hit the 160-frame/12.8 s cap: `{', '.join(payload['mixed_script_probe']['cap_hit_cases'])}`. "
            "Those outputs are nonterminated and cannot establish span coverage. `vpn_ablated` reached 156/160 frames but did not technically hit the cap.",
            "",
            "All 12 WAVs passed waveform safety. All tokenizer records round-tripped exactly with zero UNKs, but that is **input-front-end evidence only**: "
            "it does not prove that WiFi, 5G, digits, CPU, API/JSON, or VPN was spoken. The raw Paraformer pairs are retained as diagnostics; "
            "this Mandarin ASR cannot alone certify Latin-acronym pronunciation. Mixed-script coverage therefore remains `UNRESOLVED`.",
            "",
            "## Frozen protocol and provenance",
            "",
            f"- Corpus: `D1_corpus.json`, SHA-256 `{payload['protocol']['corpus']['sha256']}`, canonical items `{payload['protocol']['corpus']['items_canonical_sha256']}`.",
            f"- Reference ASR: Paraformer INT8 model SHA-256 `{payload['protocol']['reference_asr']['model_sha256']}`; greedy CPU decode, automatic 48→16 kHz resampling.",
            f"- Normalizer: SHA-256 `{payload['protocol']['normalization']['source_sha256']}`; golden fixtures `{payload['protocol']['normalization']['golden_fixtures']['status']}`.",
            f"- Statistics: ratio of sums; paired utterance bootstrap B={BOOTSTRAP_B:,}, seed={BOOTSTRAP_SEED}; no imputation.",
            f"- MOSS TTS weights: `{payload['model']['tts_weight_sha256']}` at revision `{payload['model']['tts_revision']}`; "
            f"codec weights: `{payload['model']['codec_weight_sha256']}` at revision `{payload['model']['codec_revision']}`.",
            "- The exact execution-time `run_moss_g.py` source hash was not retained for the G2 synthesis receipts (`UNKNOWN`). The current runner is recorded separately as a reproduction candidate and is not misattributed as the executing file.",
            "",
            "Artifacts: `artifacts/moss/results/g2_d1_zh_synthesis.json`, `g2_mixed_synthesis.json`, and `g2_paraformer.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def score_phase() -> int:
    import numpy as np

    validate_frozen_files(include_human=True)
    if not TRANSCRIPT_PATH.is_file():
        raise SystemExit("run the transcribe phase first")
    corpus, d1_synth, mixed_synth = load_and_validate_inputs(include_human=True)
    transcript_payload = read_json(TRANSCRIPT_PATH)
    if transcript_payload.get("reference_asr", {}).get("model_sha256") != EXPECTED_HASHES[
        "asr_model"
    ]:
        raise SystemExit("G2 transcript artifact used a different ASR model")
    if transcript_payload.get("frozen_inputs", {}).get("corpus", {}).get(
        "items_canonical_sha256"
    ) != EXPECTED_HASHES["corpus_items"]:
        raise SystemExit("G2 transcript artifact used a different corpus")

    normalization, score_d1, transcript_common = load_scoring_modules()
    if normalization.source_sha256() != EXPECTED_HASHES["normalizer"]:
        raise SystemExit("imported normalizer hash changed")
    fixture_receipt = normalization.validate_golden_fixtures(NORMALIZER_FIXTURES_PATH)

    human_payload = read_json(HUMAN_TRANSCRIPT_PATH)
    human_records = [
        {**row, "transcript_artifact_id": "paraformer_zh_d1_human"}
        for row in human_payload.get("records", [])
        if row.get("group_id") == "human_fleurs_zh"
    ]
    moss_records = [
        {**row, "transcript_artifact_id": "g2_moss_paraformer"}
        for row in transcript_payload.get("d1_records", [])
    ]
    if tuple(row.get("item_id") for row in human_records) != EXPECTED_IDS:
        raise SystemExit("human baseline is not the exact frozen 40-item group")
    if tuple(row.get("item_id") for row in moss_records) != EXPECTED_IDS:
        raise SystemExit("MOSS transcript group is not the exact 40-item arm")
    for corpus_row, human_row, moss_row in zip(
        corpus["items"], human_records, moss_records
    ):
        expected_text = corpus_row["texts"]["zh"]
        if human_row["reference_text"] != expected_text:
            raise SystemExit(f"human reference mismatch: {corpus_row['item_id']}")
        if moss_row["reference_text"] != expected_text:
            raise SystemExit(f"MOSS reference mismatch: {corpus_row['item_id']}")

    variant_results = {}
    utterance_scores = []
    normalizer_registry = {}
    for variant in normalization.scoring_variants("zh"):
        human_scored, human_registry = score_d1.score_group(
            "zh", "human_fleurs_zh", human_records, variant
        )
        moss_scored, moss_registry = score_d1.score_group(
            "zh", GROUP_ID, moss_records, variant
        )
        if human_registry != moss_registry:
            raise SystemExit(f"normalizer registry mismatch for {variant}")
        normalizer_registry[human_registry["normalizer_id"]] = human_registry
        variant_results[variant] = score_d1.paired_summary(
            human_scored, moss_scored
        )
        utterance_scores.extend(human_scored)
        utterance_scores.extend(moss_scored)

    primary = variant_results["primary"]
    basic = variant_results["basic"]
    control = normalizer_control(primary, basic)
    comparison = {
        "source_kind": "synthesized",
        "metric": "CER",
        "human_group_id": "human_fleurs_zh",
        "engine_group_id": GROUP_ID,
        "primary": primary,
        "controls_and_required_alternates": {"basic": basic},
        "normalizer_delta_control": control,
        "ranking_admissibility": (
            "INADMISSIBLE_NORMALIZER_SENSITIVITY"
            if control["ranking_inadmissible"]
            else "ADMISSIBLE_FOR_POINT_ESTIMATE_SCREEN_ONLY"
        ),
    }

    d1_complete = len(moss_records) == 40 and d1_synth["passed"] == 40
    if not d1_complete:
        point_screen = "UNRESOLVED"
    elif primary["delta"]["point_pp"] <= TH13_MARGIN_PP:
        point_screen = "PASS"
    else:
        point_screen = "FAIL"
    mixed = mixed_pair_results(
        mixed_synth, transcript_payload, normalization, score_d1
    )

    inventory = safety_inventory(d1_synth)
    storage_bytes_before_outputs = directory_bytes(OUT_D)
    payload = {
        "schema_version": 1,
        "sprint_id": "G2",
        "lane": "G",
        "status": "COMPLETE_WITH_MIXED_SCRIPT_COVERAGE_UNRESOLVED",
        "evidence_class": "HOST",
        "exact_hash_evidence_class": "EXACT",
        "final_verdict": {
            "d1_complete_and_scored": d1_complete,
            "zh_point_estimate_technical_screen": point_screen,
            "zh_point_estimate_explanation": (
                f"Primary matched-human ΔCER is {primary['delta']['point_pp']:+.3f} pp against the +5 pp screen. "
                "This is a point-estimate screen only; TH-13 remains insufficient at n=40."
            ),
            "mixed_script_coverage": mixed["status"],
            "mixed_script_explanation": mixed["status_reason"],
            "th13_gate": "INSUFFICIENT_N_40_LT_60",
            "ranking_admissible": not control["ranking_inadmissible"],
        },
        "headline": {
            "d1_attempted": d1_synth["attempted"],
            "d1_safe_and_scored": d1_synth["passed"],
            "human_cer_percent": primary["human"]["point_pp"],
            "moss_cer_percent": primary["synth"]["point_pp"],
            "delta_cer_percent_points": primary["delta"]["point_pp"],
            "delta_ci95_percent_points": [
                primary["delta"]["ci95_lo_pp"],
                primary["delta"]["ci95_hi_pp"],
            ],
            "normalizer_control_status": control[
                "ordering_or_threshold_interpretation"
            ],
            "mixed_safe_wavs": mixed["safe_wav_count"],
            "mixed_cap_hit_count": mixed["cap_hit_count"],
        },
        "model": d1_synth["model"],
        "protocol": {
            "corpus": transcript_common.corpus_provenance(corpus),
            "reference_asr": {
                **transcript_payload["reference_asr"],
                "transcript_artifact_path": relative(TRANSCRIPT_PATH),
                "transcript_artifact_sha256": sha256_file(TRANSCRIPT_PATH),
                "frozen_human_transcript_path": relative(HUMAN_TRANSCRIPT_PATH),
                "frozen_human_transcript_sha256": sha256_file(
                    HUMAN_TRANSCRIPT_PATH
                ),
            },
            "metric": "CER with Han/digit characters and embedded Latin runs as units",
            "delta_definition": "reference-ASR CER on MOSS audio minus reference-ASR CER on the matched human recording",
            "statistics": {
                "point_estimator": "ratio_of_sums_sum_errors_div_sum_reference_units",
                "paired_delta": "synth_error_rate_minus_matched_human_error_rate",
                "bootstrap_B": BOOTSTRAP_B,
                "seed": BOOTSTRAP_SEED,
                "rng": "numpy.random.PCG64",
                "ci_method": "paired_utterance_bootstrap_percentile",
                "percentiles": [2.5, 97.5],
                "cluster_unit": "utterance",
                "imputation": "none",
            },
            "normalization": {
                "source_path": relative(NORMALIZER_PATH),
                "source_sha256": normalization.source_sha256(),
                "package_versions": normalization.package_versions(),
                "golden_fixtures": fixture_receipt,
                "normalizers": normalizer_registry,
                "mandarin_declared_substitution": (
                    "cn2an 0.5.24 numeric canonicalization replaces WeTextProcessing for this bounded sprint; "
                    "BasicTextNormalizer control remains mandatory"
                ),
                "basic_control_threshold_pp": CONTROL_THRESHOLD_PP,
            },
            "th13_context": {
                "margin_percent_points": TH13_MARGIN_PP,
                "minimum_n": TH13_MINIMUM_N,
                "sprint_n": 40,
                "verdict": "INSUFFICIENT_N_40_LT_60",
                "interpretation": (
                    "Point-estimate screen only. A value within the margin at n=40 must not be reported as clearing TH-13."
                ),
            },
            "generation": {
                "sampling": d1_synth["runtime"]["sampling"],
                "sampling_disposition": (
                    "do_sample=true; one deterministic-seed sampled draw per item, not greedy decoding and not a repeated-draw stability estimate"
                ),
                "external_text_preprocessing": d1_synth["runtime"][
                    "text_preprocessing"
                ],
                "retained_wav_format": "48 kHz mono PCM16 downmix of raw stereo decoder tensor",
            },
            "input_artifacts": [
                {
                    "path": relative(D1_SYNTH_PATH),
                    "bytes": D1_SYNTH_PATH.stat().st_size,
                    "sha256": sha256_file(D1_SYNTH_PATH),
                },
                {
                    "path": relative(MIXED_SYNTH_PATH),
                    "bytes": MIXED_SYNTH_PATH.stat().st_size,
                    "sha256": sha256_file(MIXED_SYNTH_PATH),
                },
            ],
        },
        "d1_mandarin": {
            "status": "SCORED_COMPLETE_40_NO_IMPUTATION",
            "safety_inventory": inventory,
            "comparison": comparison,
            "utterance_scores": utterance_scores,
        },
        "mixed_script_probe": mixed,
        "reproducibility": {
            "helper_path": relative(Path(__file__).resolve()),
            "helper_sha256": sha256_file(Path(__file__).resolve()),
            "runner_execution": {
                "path": relative(MOSS_RUNNER_PATH),
                "sha256": None,
                "evidence_class": "UNKNOWN",
                "scope": "g2_d1_zh and g2_mixed synthesis receipts",
                "note": (
                    "The exact execution-time untracked runner content was not "
                    "retained. The receipts pin model revisions, weights, "
                    "prompts, seeds, sampling, generated tokens, and WAV "
                    "hashes, but not the Python source hash."
                ),
            },
            "runner_current_reproduction_version": {
                "path": relative(MOSS_RUNNER_PATH),
                "bytes": MOSS_RUNNER_PATH.stat().st_size,
                "sha256": sha256_file(MOSS_RUNNER_PATH),
                "evidence_class": "EXACT",
                "note": (
                    "Current committed-candidate runner; not misattributed as "
                    "the G2 execution-time file."
                ),
            },
            "commands": [
                "artifacts/matcha/venv/bin/python run_g2_moss_scoring.py transcribe",
                "artifacts/main/venv/bin/python run_g2_moss_scoring.py score",
            ],
            "host": host_record({"numpy": np.__version__}),
            "storage_cap_bytes": STORAGE_CAP_BYTES,
            "out_d_bytes_before_G2_json_and_md": storage_bytes_before_outputs,
        },
    }
    atomic_write_json(RESULT_JSON_PATH, payload)
    atomic_write_text(RESULT_MD_PATH, render_markdown(payload))
    final_bytes = directory_bytes(OUT_D)
    if final_bytes >= STORAGE_CAP_BYTES:
        raise SystemExit(
            f"storage cap exceeded after report write: {final_bytes} >= {STORAGE_CAP_BYTES}"
        )
    print(f"wrote {RESULT_JSON_PATH}")
    print(f"wrote {RESULT_MD_PATH}")
    print(
        f"primary ΔCER={primary['delta']['point_pp']:+.3f} pp "
        f"CI95=[{primary['delta']['ci95_lo_pp']:+.3f}, {primary['delta']['ci95_hi_pp']:+.3f}] "
        f"screen={point_screen} mixed={mixed['status']} storage={final_bytes}",
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("transcribe", "score"))
    args = parser.parse_args()
    if args.phase == "transcribe":
        return transcribe_phase()
    return score_phase()


if __name__ == "__main__":
    raise SystemExit(main())
