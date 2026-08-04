#!/usr/bin/env python3
"""Lightweight invariants for the generated phone neural-TTS catalog."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/tts_research_20260726/stage_device_bundle.py"
SPEC = importlib.util.spec_from_file_location("stage_device_bundle", SOURCE)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def main() -> None:
    candidates = MODULE.CANDIDATES
    by_id = {candidate["id"]: candidate for candidate in candidates}
    assert len(by_id) == len(candidates), "candidate ids must be unique"

    expected = {
        "supertonic3_en_s5": "en",
        "supertonic3_s5": "ru",
        "supertonic3_ja_s5": "ja",
        "supertonic3_ko_s5": "ko",
        "supertonic3_es_s5": "es",
        "supertonic3_fr_s5": "fr",
    }
    shared_files = None
    for candidate_id, language in expected.items():
        candidate = by_id[candidate_id]
        assert candidate["language"] == language
        assert candidate["extra"]["lang"] == language
        assert candidate["auto_priority"] > 0
        assert candidate["num_steps"] == 5
        if shared_files is None:
            shared_files = candidate["files"]
        else:
            assert candidate["files"] == shared_files, "multilingual routes must reuse graphs"

    assert {"ja", "ko", "es", "fr"}.issubset(MODULE.PHRASES)

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        resident = root / "resident.json"
        output = root / "metadata"
        output.mkdir()
        shared_path = next(iter(shared_files.values()))
        resident.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "missing_sources": [],
                    "files": {
                        path: {"bytes": index + 1, "sha256": hashlib.sha256(path.encode()).hexdigest()}
                        for index, path in enumerate(shared_files.values())
                    },
                }
            ),
            encoding="utf-8",
        )
        selected = [by_id[candidate_id].copy() for candidate_id in expected]
        manifest = MODULE.stage_resident_metadata(resident, output, selected)
        assert manifest["candidate_count"] == len(expected)
        assert manifest["files"][shared_path]["bytes"] > 0
        staged = json.loads((output / "candidates.json").read_text(encoding="utf-8"))
        assert [candidate["id"] for candidate in staged["candidates"]] == list(expected)

        hashless = root / "resident-hashless.json"
        bad = json.loads(resident.read_text(encoding="utf-8"))
        del next(iter(bad["files"].values()))["sha256"]
        hashless.write_text(json.dumps(bad), encoding="utf-8")
        try:
            MODULE.stage_resident_metadata(hashless, output, selected)
        except ValueError as error:
            assert "requires SHA-256" in str(error)
        else:
            raise AssertionError("hashless resident assets must be rejected")

    print("TTS device bundle metadata checks passed")


if __name__ == "__main__":
    main()
