#!/usr/bin/env python3
"""Validate Lane G reports against retained MOSS and Lane D artifacts."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import update_lane_g_reports as lane_g


OUT_D = Path(__file__).resolve().parent
REPO = OUT_D.parents[4]
MAX_BYTES = 3_000_000_000

HISTORICAL_D_HASHES = {
    "D1.json": "b4c838a70db4a2d9955f5c58c4470b91d708ea08fcfd23f0ae0d6fbc0d0496a2",
    "D1.md": "94b9463bff9043503a6f90212dc2bc49a2140bc9c7f95a4346089062c9f0557c",
    "D2.json": "d642d9e485e0caecf81a83367e64497707e2cd9fe187d9581c6552b92fa0933e",
    "D2.md": "2ff8f903908295b7bd82cba7f2861f12cfb179e45166314a602d4c5990eb8b14",
    "D3.json": "c68e1d37be7bacb362de0ca1563d5831ad80f86f283db2a03f93f672a020f5f0",
    "D3.md": "eec28de69608b77a128b5bc394d5cd97fd06d28b3e8f2eb3f9b5499e2cbf122c",
    "D4.json": "3208a9d41ff7e261e8ae863ce97ab2df5e5fb49e3fec29eff1358c37c0d03fbd",
    "D4.md": "5f4b1de6d327cee43cb527e5e350ca6c516a513254919417cada94bdede2485e",
    "D5.json": "dac26b1fb522c6c730435054c33be122f56bcf78b2b0813464cab4b193dcaed6",
    "D5.md": "e5a0f62b94e5e5cb2421f038812521dd800516d3a2b67e99816d51979467a757",
    "D5_prompts.json": "5bfb957efc25482f6b8d9c8f90e5af8e55b99d11fc982c83815c5ea2fd782b20",
    "D1_corpus.json": "b4c838a70db4a2d9955f5c58c4470b91d708ea08fcfd23f0ae0d6fbc0d0496a2",
}


def load(name: str) -> dict[str, Any]:
    return json.loads((OUT_D / name).read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def subprocess_text(args: list[str]) -> str:
    return subprocess.run(
        args,
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> int:
    assert subprocess_text(["git", "branch", "--show-current"]) == "sprint/d-tts"
    for name, expected_hash in HISTORICAL_D_HASHES.items():
        assert sha256_file(OUT_D / name) == expected_hash, name

    reports = {name: load(f"{name}.json") for name in ("G1", "G2", "G3", "G4", "G5")}
    table = load("tts_table.json")
    listen = (OUT_D / "LISTEN.md").read_text(encoding="utf-8")
    g5_md = (OUT_D / "G5.md").read_text(encoding="utf-8")
    summary = (OUT_D / "LANE_SUMMARY.md").read_text(encoding="utf-8")
    for payload in (*reports.values(), table):
        lane_g.assert_evidence_classes(payload)

    g1, g2, g3, g4, g5 = (reports[name] for name in ("G1", "G2", "G3", "G4", "G5"))
    assert g1["headline"]["tts_unique_learned_parameters"]["value"] == 117_311_232
    assert g1["headline"]["audio_tokenizer_unique_learned_parameters"]["value"] == 21_969_664
    assert g1["headline"]["pair_unique_learned_parameters"]["value"] == 139_280_896
    assert g1["headline"]["lane_a_count_reconciliation"] == "PASS_EXACT"
    assert "REDUNDANT" in g1["headline"]["onnx_tied_weight_trap"]
    g1_trap = g1["official_onnx_export"]["counting_trap"]
    audio_materialization = g1_trap[
        "audio_codebook_table_materialization_observation"
    ]
    assert audio_materialization["group_count"] == 16
    assert audio_materialization["physical_bytes"] == 150_994_944
    assert audio_materialization["redundant_materialization_bytes"] == 100_663_296
    residual = g1_trap["gross_duplicate_vs_net_export_excess_reconciliation"]
    assert residual["gross_tied_table_duplicate_elements"] == 50_331_648
    assert residual["net_unique_offset_excess_over_pytorch_elements"] == 50_311_680
    assert residual["difference_elements"] == 19_968
    licence = g1["licence"]["weight_and_export_licence_scope"]
    assert licence["commercially_unblocked"] is False
    assert not g1["licence"]["immutable_snapshot_manifest_check"]["license_file_present_in_any_snapshot"]

    assert g2["final_verdict"]["d1_complete_and_scored"] is True
    assert g2["headline"]["d1_safe_and_scored"] == 40
    lane_g.require_enum(
        g2["final_verdict"]["zh_point_estimate_technical_screen"], "G2 point screen"
    )
    lane_g.require_enum(g2["final_verdict"]["mixed_script_coverage"], "G2 mixed")
    assert g2["headline"]["mixed_safe_wavs"] == 12
    g2_runner_execution = g2["reproducibility"]["runner_execution"]
    assert g2_runner_execution["sha256"] is None
    assert g2_runner_execution["evidence_class"] == "UNKNOWN"
    g2_current_runner = g2["reproducibility"][
        "runner_current_reproduction_version"
    ]
    assert g2_current_runner["sha256"] == sha256_file(OUT_D / "run_moss_g.py")
    assert g2_current_runner["bytes"] == (OUT_D / "run_moss_g.py").stat().st_size

    assert g3["headline"]["attempted"] == 42
    assert g3["headline"]["effective_safe"] == 0
    assert g3["headline"]["effective_unresolved"] == 19
    assert g3["headline"]["effective_unsafe"] == 23
    assert g3["final_verdict"]["effective_safe_attempts"] == 0
    assert g3["final_verdict"]["effective_unresolved_attempts"] == 19
    assert g3["final_verdict"]["effective_unsafe_attempts"] == 23
    expected_g3_effective_counts = {
        "en": (0, 3, 3),
        "es": (0, 4, 2),
        "fr": (0, 3, 3),
        "ru": (0, 6, 0),
        "zh": (0, 3, 15),
    }
    assert {
        row["language"]: (
            row["effective_safe_attempts"],
            row["effective_unresolved_attempts"],
            row["effective_unsafe_attempts"],
        )
        for row in g3["language_results"]
    } == expected_g3_effective_counts
    assert {
        row["effective_safety_verdict"] for row in g3["attempts"]
    } == {
        "UNSAFE_PROBABLE_NONTERMINATION_OR_TRUNCATION",
        "UNRESOLVED_PLAUSIBLE_DURATION",
    }
    assert sum(
        row["effective_safety_verdict"]
        == "UNSAFE_PROBABLE_NONTERMINATION_OR_TRUNCATION"
        for row in g3["attempts"]
    ) == 23
    assert sum(
        row["effective_safety_verdict"] == "UNRESOLVED_PLAUSIBLE_DURATION"
        for row in g3["attempts"]
    ) == 19
    assert len(g3["routes"]) == 5
    for language in lane_g.LANGUAGE_ORDER:
        verdict = lane_g.require_enum(
            g3["final_verdict"]["by_language"][language], f"G3 {language}"
        )
        route, attempted, passed, cap_hits = lane_g.g3_route_stats(g3, language)
        assert attempted == (18 if language == "zh" else 6)
        assert route["verdict"] == verdict
        if verdict == "PASS":
            assert passed == attempted and cap_hits == 0
        elif verdict == "FAIL":
            assert passed < attempted or cap_hits > 0
        else:
            assert passed == attempted and cap_hits == 0
            assert route.get("warning", {}).get(
                "route_clearance_disqualifying"
            ) is True
    assert g3["final_verdict"]["by_language"]["ru"] == "UNRESOLVED"
    assert g3["final_verdict"]["unresolved_languages"] == ["ru"]
    assert g3["final_verdict"]["overall"] in {"PASS", "FAIL"}

    assert g4["final_verdict"]["reduced_n_exploratory"] is True
    assert g4["final_verdict"]["single_engine_safe_all_five"] is False
    assert set(g4["languages"]) == {"en", "es", "fr", "ru"}
    assert g4["headline"]["n_per_language"] == 10
    assert g4["headline"]["total_synth_clips"] == 40
    g4_receipt = g4["synthesis_receipt"]
    assert g4_receipt["record_count"] == g4_receipt["expected_record_count"] == 40
    assert g4_receipt["reported_passed"] == 36
    assert g4_receipt["reported_failed"] == 4
    assert g4_receipt["reported_max_new_frames_hit_count"] == 0
    g4_rewrite = g4["synthesis_runner_provenance"]["receipt_rewrite"]
    assert g4_rewrite["records_with_case_config_sha256"] == 40
    assert g4_rewrite["records_marked_checkpoint_migrated_from_legacy"] == 40
    assert g4_rewrite["case_timeout_s_values"] == [300]
    assert "must not upgrade" in g4_rewrite["provenance_interpretation"]
    assert [row["case_id"] for row in g4_receipt["named_failures"]] == [
        "es_fleurs_02",
        "fr_fleurs_02",
        "fr_fleurs_03",
        "fr_fleurs_04",
    ]
    g4_provenance = g4["synthesis_runner_provenance"]
    assert g4_provenance["scope"] == "PARTIAL_ROW_BINDING"
    assert g4_provenance["bound_row_count"] == 28
    assert g4_provenance["unknown_execution_runner_row_count"] == 12
    assert (
        g4_provenance["observed_runner_for_bound_rows"]["sha256"]
        == lane_g.G4_G5_EXECUTION_RUNNER_SHA256
    )
    assert (
        g4_provenance["observed_runner_for_bound_rows"]["bytes"]
        == lane_g.G4_G5_EXECUTION_RUNNER_BYTES
    )
    for language in ("en", "es", "fr", "ru"):
        verdict = lane_g.require_enum(
            g4["final_verdict"]["language_coverage"][language], f"G4 {language}"
        )
        language_result = g4["languages"][language]
        expected_pass = bool(
            language_result["waveform_and_cap_safety"]["waveform_and_cap_safe"]
            and language_result["reduced_n_point_screen"]["passed"]
            and language_result["comparison"]["status"]
            == "SCORED_COMPLETE_10_OF_10"
        )
        assert (verdict == "PASS") == expected_pass
        assert language_result["coverage_verdict"] == verdict
    assert g4["final_verdict"]["language_coverage"].get("zh", "UNRESOLVED") == "UNRESOLVED"
    assert all(
        g4["languages"][language]["reduced_n_exploratory"] is True
        and g4["languages"][language]["ranking_inadmissible"] is True
        for language in ("en", "es", "fr", "ru")
    )

    inventory = lane_g.scan_listening_bundle()
    reported_inventory = g5["inventory"]
    for key in (
        "expected_wav_count",
        "existing_wav_count",
        "missing_expected_count",
        "extra_unexpected_count",
        "missing_expected_filenames",
        "extra_unexpected_filenames",
        "total_wav_bytes",
        "clip_hash_manifest_sha256",
        "clip_hash_manifest_definition",
        "lane_d_frozen_identity",
        "all_existing_wavs_readable",
        "all_existing_wavs_pass_waveform_gate",
    ):
        assert reported_inventory[key] == inventory[key], key
    assert reported_inventory["expected_wav_count"] == 300
    lane_d_identity = reported_inventory["lane_d_frozen_identity"]
    assert lane_d_identity["preserved"] is True
    assert lane_d_identity["existing_non_moss_wav_count"] == 278
    assert lane_d_identity["clip_hash_manifest_sha256"] == (
        lane_g.D5_LISTENING_MANIFEST_SHA256
    )
    assert lane_d_identity["expected_clip_hash_manifest_sha256"] == (
        lane_g.D5_LISTENING_MANIFEST_SHA256
    )
    assert reported_inventory["extra_unexpected_filenames"] == []
    assert reported_inventory["all_existing_wavs_pass_waveform_gate"]
    assert g5["formats"] == inventory["formats"]

    synthesis_receipt = lane_g.load_g5_synthesis_receipt(g1=g1, inventory=inventory)
    assert g5["synthesis_receipt"] == synthesis_receipt
    provenance = synthesis_receipt["execution_provenance"]
    executed = provenance["executed_runner"]
    assert executed["sha256"] == lane_g.G4_G5_EXECUTION_RUNNER_SHA256
    assert executed["bytes"] == lane_g.G4_G5_EXECUTION_RUNNER_BYTES
    assert executed["scope"] == "ALL_20_G5_RECORDS"
    assert executed["receipt_embeds_runner_identity"] is False
    assert provenance["receipt_rewrite"]["records_with_case_config_sha256"] == 20
    assert provenance["receipt_rewrite"][
        "records_marked_checkpoint_migrated_from_legacy"
    ] == 20

    moss_names = [f"zh_moss_{index:02d}.wav" for index in range(1, 21)]
    by_name = {clip["filename"]: clip for clip in inventory["clips"]}
    cases = synthesis_receipt["records"]
    assert g5["moss_zh"]["cases"] == cases
    present_names = [row["filename"] for row in cases if row["listening_available"]]
    waveform_safe_names = [row["filename"] for row in cases if row["waveform_safe"]]
    safe_names = [row["filename"] for row in cases if row["safe"]]
    missing_names = [row["filename"] for row in cases if not row["listening_available"]]
    cap_names = [row["filename"] for row in cases if row["max_new_frames_hit"]]
    assert g5["moss_zh"]["expected"] == 20
    assert g5["moss_zh"]["present"] == len(present_names)
    assert g5["moss_zh"]["waveform_safe"] == len(waveform_safe_names)
    assert g5["moss_zh"]["safe"] == len(safe_names)
    assert g5["moss_zh"]["missing"] == missing_names
    assert g5["moss_zh"]["frame_cap_hit"] == cap_names
    assert [row["filename"] for row in g5["moss_zh"]["failures"]] == missing_names
    assert reported_inventory["existing_wav_count"] == 278 + len(present_names)
    assert reported_inventory["missing_expected_count"] == 2 + len(missing_names)
    assert set(reported_inventory["missing_expected_filenames"]) == {
        "zh_kokoro_09.wav",
        "zh_kokoro_14.wav",
        *missing_names,
    }
    complete_safe = len(safe_names) == 20
    assert g5["final_verdict"][
        "moss_zh_listening_clips_complete_and_safe"
    ] is complete_safe
    expected_naturalness = (
        "PENDING_USER_LISTENING"
        if present_names
        else "UNRESOLVED_NO_RETAINED_G5_CLIPS"
    )
    assert g5["final_verdict"]["naturalness"] == expected_naturalness
    expected_status = (
        "COMPLETE_PENDING_USER_LISTENING"
        if complete_safe
        else (
            "PARTIAL_PENDING_USER_LISTENING"
            if present_names
            else "INCOMPLETE_NO_MOSS_LISTENING_CLIPS"
        )
    )
    assert g5["status"] == expected_status
    assert len(g5["moss_zh"]["clips"]) == len(present_names)
    assert {clip["filename"] for clip in g5["moss_zh"]["clips"]} == set(
        present_names
    )
    for clip in g5["moss_zh"]["clips"]:
        current = by_name[clip["filename"]]
        assert clip["bytes"] == current["bytes"]
        assert clip["sha256"] == current["sha256"]
        assert clip["passed"]

    mixed = lane_g.load_mixed_pairs()
    assert len(g5["mixed_script_pairs"]) == len(mixed) == 6
    assert [row["pair_id"] for row in g5["mixed_script_pairs"]] == list(
        lane_g.MIXED_PAIR_ORDER
    )
    for reported, current in zip(g5["mixed_script_pairs"], mixed):
        assert reported["pair_id"] == current["pair_id"]
        for role in ("full", "ablated"):
            assert reported[role]["relative_path"] == current[role]["relative_path"]
            assert reported[role]["bytes"] == current[role]["bytes"]
            assert reported[role]["sha256"] == current[role]["sha256"]
            assert (OUT_D / reported[role]["relative_path"]).is_file()

    assert "# Lane D + G listening guide" in listen
    assert listen.count("<!-- LANE_G_INTRO_START -->") == 1
    assert listen.count("<!-- LANE_G_MIXED_START -->") == 1
    linked_listening = re.findall(r"\]\((artifacts/listening/[^)]+\.wav)\)", listen)
    assert len(linked_listening) == len(set(linked_listening)) == len(by_name)
    assert {Path(link).name for link in linked_listening} == set(by_name)
    assert "zh_kokoro_09.wav" not in linked_listening
    assert "zh_kokoro_14.wav" not in linked_listening
    assert listen.count("non-finite output; no WAV") == 2
    zh_section = listen[
        listen.index("## Mandarin Chinese (`zh`)") : listen.index("## Spanish (`es`)")
    ]
    moss_listen_marker = "⛔" if table["lane_g_moss"]["zh_bucket"] == "unsafe" else "⚠"
    cases_by_name = {row["filename"]: row for row in cases}
    for index, name in enumerate(moss_names, 1):
        case = cases_by_name[name]
        zh_row = next(
            line
            for line in zh_section.splitlines()
            if line.startswith(f"| ☐ | {index:02d} |")
        )
        if case["listening_available"]:
            assert listen.count(f"[{name}](artifacts/listening/{name})") == 1
            assert zh_row.index(f"zh_matcha_{index:02d}.wav") < zh_row.index(name)
            assert f"{moss_listen_marker} [{name}]" in zh_row
            assert ("**FRAME CAP**" in zh_row) == case["max_new_frames_hit"]
        else:
            assert f"[{name}](artifacts/listening/{name})" not in listen
            label = case["failure_label"] or "synthesis failure"
            assert f"no WAV — {label}" in zh_row
    assert "decision (yes / no / unclear)" in listen
    linked_mixed = re.findall(r"\]\((artifacts/moss/audio/g2_mixed/[^)]+\.wav)\)", listen)
    assert len(linked_mixed) == len(set(linked_mixed)) == 12
    expected_mixed = {
        f"artifacts/moss/audio/g2_mixed/{pair}_{role}.wav"
        for pair in lane_g.MIXED_PAIR_ORDER
        for role in ("full", "ablated")
    }
    assert set(linked_mixed) == expected_mixed
    assert all((OUT_D / path).is_file() for path in linked_mixed)

    assert f"{reported_inventory['existing_wav_count']}/{reported_inventory['expected_wav_count']}" in g5_md
    assert f"{reported_inventory['total_wav_bytes']:,}" in g5_md
    assert reported_inventory["clip_hash_manifest_sha256"] in g5_md
    assert lane_g.D5_LISTENING_MANIFEST_SHA256 in g5_md
    assert synthesis_receipt["sha256"] in g5_md
    assert lane_g.G4_G5_EXECUTION_RUNNER_SHA256 in g5_md
    assert expected_naturalness in g5_md

    historical = lane_g.lane_d_route_rows(table)
    assert lane_g.canonical_sha256(historical) == lane_g.LANE_D_ROUTE_ROWS_SHA256
    assert table["lane_g_moss"]["historical_lane_d_route_rows_sha256"] == lane_g.LANE_D_ROUTE_ROWS_SHA256
    assert not table["headline"]["single_engine_safe_all_five"]
    assert not table["headline"]["single_engine_commercially_unblocked_all_five"]
    assert not table["headline"]["general_zh_ship_route_from_tested_set"]
    assert not table["lane_g_moss"]["commercially_unblocked"]
    for language in lane_g.LANGUAGE_ORDER:
        language_row = table["languages"][language]
        moss_rows = [row for row in language_row["routes"] if row["engine_id"] == "moss"]
        assert len(moss_rows) == 1
        moss = moss_rows[0]
        technical = language_row["technical_candidates_on_sprint_evidence"]
        commercial = language_row[
            "candidate_routes_without_an_identified_lane_commercial_blocker"
        ]
        unsafe = language_row["unsafe_routes"]
        limited = language_row["limited_or_evidence_incomplete_routes"]
        no_evidence = language_row["no_lane_evidence_or_unsupported"]
        memberships = sum("moss" in values for values in (technical, unsafe, limited))
        assert memberships == 1, (language, technical, unsafe, limited)
        assert (moss["technical"] == "SPRINT_CANDIDATE_NOT_GATE_CLEARED") == (
            "moss" in technical
        )
        assert moss["overall"].startswith("REJECT_UNSAFE") == ("moss" in unsafe)
        assert "moss" not in commercial
        assert "moss" not in no_evidence
        assert moss["commercial"] == "BLOCKED_PINNED_WEIGHT_REDISTRIBUTION_SCOPE_UNLICENSED"
        assert moss["license_disposition"] == licence["disposition"]
    assert "moss" not in table["headline"]["best_supported_sprint_routes"]["zh"]
    assert table["evidence"]["G5"] == "G5.json"

    assert summary.count("<!-- LANE_G_ADDENDUM_START -->") == 1
    assert summary.count("<!-- LANE_G_ADDENDUM_END -->") == 1
    assert "## Lane G addendum — MOSS-TTS-Nano" in summary
    assert "MOSS is not a shippable Mandarin route" in summary
    assert expected_naturalness in summary

    tracked = subprocess_text(["git", "ls-files", "workstreams/sub2b_sprint/sprint/out/d"])
    forbidden_suffixes = (".wav", ".bin", ".safetensors", ".onnx", ".pt", ".tar.bz2")
    assert not any(line.endswith(forbidden_suffixes) for line in tracked.splitlines())

    total_bytes = lane_g.tree_logical_bytes(OUT_D)
    assert total_bytes < MAX_BYTES, (total_bytes, MAX_BYTES)
    result = {
        "status": "PASS",
        "g2_delta_cer_percent_points": g2["headline"]["delta_cer_percent_points"],
        "g2_mixed_script_coverage": g2["final_verdict"]["mixed_script_coverage"],
        "g3_overall": g3["final_verdict"]["overall"],
        "g4_language_coverage": g4["final_verdict"]["language_coverage"],
        "g5_listening_wavs": reported_inventory["existing_wav_count"],
        "g5_expected_wavs": reported_inventory["expected_wav_count"],
        "g5_mixed_probe_wavs": 12,
        "historical_lane_d_route_rows_sha256": lane_g.LANE_D_ROUTE_ROWS_SHA256,
        "out_d_total_bytes": total_bytes,
        "storage_cap_bytes": MAX_BYTES,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
