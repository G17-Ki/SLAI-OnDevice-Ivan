#!/usr/bin/env python3
"""Common quality/latency harness for the 2026-07-26 multilingual TTS run.

One engine per process. Keeping a single model resident is a hard requirement
of this host (8 GiB RAM, 2.5 GiB process-tree cap), not a stylistic choice, so
the runner deliberately refuses to load a second candidate.

Outputs, WAVs, and hashes go to ``--output-dir``, which must be outside the
repository.

    python bench.py --engine melotts_zh --suite zh \
        --model-root "$TTS_RESEARCH_MODEL_ROOT" \
        --output-dir "$RUN_ROOT/bench/melotts_zh"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))

import engines as engines_mod  # noqa: E402
import gates as gates_mod  # noqa: E402


RSS_CAP_BYTES = int(2.5 * 1024**3)


class PeakRss:
    def __init__(self, interval_s: float = 0.01):
        self.process = psutil.Process()
        self.interval_s = interval_s
        self.peak = self._tree_rss()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.cap_exceeded = False

    def _tree_rss(self) -> int:
        total = self.process.memory_info().rss
        for child in self.process.children(recursive=True):
            try:
                total += child.memory_info().rss
            except psutil.Error:
                pass
        return total

    def _sample(self) -> None:
        while not self.stop.is_set():
            current = self._tree_rss()
            if current > self.peak:
                self.peak = current
            if current > RSS_CAP_BYTES:
                self.cap_exceeded = True
            self.stop.wait(self.interval_s)

    def __enter__(self) -> "PeakRss":
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop.set()
        self.thread.join()
        self.peak = max(self.peak, self._tree_rss())


def distribution(values: list) -> dict:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return {"n": 0, "median": None, "p95": None, "min": None, "max": None}
    return {
        "n": len(clean),
        "median": float(statistics.median(clean)),
        "p95": float(np.percentile(np.asarray(clean, dtype=np.float64), 95)),
        "min": float(min(clean)),
        "max": float(max(clean)),
    }


def simulate_playback(chunk_times_s, chunk_sizes, sample_rate: int, slowdown: float) -> dict:
    """Queue-supply simulation for clause-streamed playback.

    Chunk *k* is produced at ``chunk_times_s[k] * slowdown``. Playback starts
    when the first chunk lands plus a startup cushion, then advances in real
    time. A gap is any interval where the player has drained the queue but the
    next chunk has not been produced yet.
    """
    if not chunk_times_s or not chunk_sizes:
        return {"simulated": False, "reason": "no_chunk_callbacks"}

    cushion_s = 0.140  # matches the shipped Android WebAudio startup cushion
    produced = [t * slowdown for t in chunk_times_s]
    durations = [n / float(sample_rate) for n in chunk_sizes]

    playback_start = produced[0] + cushion_s
    cursor = playback_start  # wall-clock time at which the player needs data
    gaps = []
    for produced_at, dur in zip(produced, durations):
        if produced_at > cursor:
            gaps.append(produced_at - cursor)
            cursor = produced_at
        cursor += dur

    return {
        "simulated": True,
        "slowdown_x": slowdown,
        "startup_cushion_s": cushion_s,
        "first_audible_s": playback_start,
        "gap_count": len(gaps),
        "total_gap_s": float(sum(gaps)),
        "max_gap_s": float(max(gaps)) if gaps else 0.0,
        "gap_free": not gaps,
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_case(engine, case: dict, language: str, repeat: int, audio_dir: Path, keep_wav: bool) -> dict:
    with PeakRss() as rss:
        result = engine.synth(case["text"], language)

    gate = gates_mod.evaluate(result.samples, result.sample_rate)
    duration_s = gate.metrics.get("audio_duration_s") or 0.0

    record = {
        "case_id": case["id"],
        "kind": case.get("kind"),
        "language": language,
        "text": case["text"],
        "repeat_index": repeat,
        "first_chunk_s": result.first_chunk_s,
        "chunk_times_s": result.chunk_times_s,
        "chunk_sizes": result.chunk_sizes,
        "chunk_count": len(result.chunk_times_s),
        "total_synthesis_s": result.total_s,
        "rtf": (result.total_s / duration_s) if duration_s > 0 else None,
        "peak_rss_bytes": rss.peak,
        "rss_cap_exceeded": rss.cap_exceeded,
        "gate": gate.to_dict(),
        "playback_simulation": {
            f"{s:.1f}x": simulate_playback(result.chunk_times_s, result.chunk_sizes, result.sample_rate, s)
            for s in (1.0, 1.5, 2.0)
        },
        "wav_path": None,
        "wav_sha256": None,
        "first_chunk_wav_path": None,
    }

    # A waveform that fails the hard gates is never serialized as if it were
    # playable audio. The rejected sample statistics are already captured above.
    if keep_wav and repeat == 0:
        if gate.passed:
            audio_dir.mkdir(parents=True, exist_ok=True)
            wav = audio_dir / f"{engine.candidate.id}__{case['id']}.wav"
            sf.write(wav, result.samples, result.sample_rate, subtype="PCM_16")
            record["wav_path"] = wav.name
            record["wav_sha256"] = sha256_file(wav)
            if result.first_chunk_samples is not None and result.first_chunk_samples.size:
                fc_gate = gates_mod.evaluate(result.first_chunk_samples, result.sample_rate)
                if fc_gate.passed:
                    fc = audio_dir / f"{engine.candidate.id}__{case['id']}.first_chunk.wav"
                    sf.write(fc, result.first_chunk_samples, result.sample_rate, subtype="PCM_16")
                    record["first_chunk_wav_path"] = fc.name
                record["first_chunk_gate"] = fc_gate.to_dict()
        else:
            record["wav_withheld_reason"] = "hard_gate_failure"
    return record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=engines_mod.ALL_ENGINE_IDS)
    ap.add_argument("--suite", required=True, help="fixture suite key, e.g. zh / ru / en")
    ap.add_argument("--model-root", type=Path)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--fixtures", type=Path, default=Path(__file__).resolve().parent / "fixtures.json")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--num-steps", type=int, default=None, help="denoising steps (Supertonic)")
    ap.add_argument("--sid", type=int, default=0)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--tag", default="", help="suffix distinguishing configurations of one engine")
    ap.add_argument("--chunk-floor-sweep", action="store_true")
    ap.add_argument("--flush-sweep", action="store_true",
                    help="measure first-chunk latency against first-flush length")
    ap.add_argument("--no-wav", action="store_true")
    args = ap.parse_args()

    root = args.model_root.expanduser().resolve() if args.model_root else engines_mod.model_root()
    fixtures = json.loads(args.fixtures.read_text(encoding="utf-8"))
    suite = fixtures["suites"][args.suite]
    language = suite["language"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = args.output_dir / "audio"

    kwargs = {"sid": args.sid, "speed": args.speed}
    if args.num_steps is not None:
        kwargs["num_steps"] = args.num_steps
    engine = engines_mod.build(args.engine, root, args.threads, **kwargs)

    proc = psutil.Process()
    rss_before = proc.memory_info().rss
    load_start = time.perf_counter()
    with PeakRss() as load_rss:
        engine.load()
    load_s = time.perf_counter() - load_start
    rss_after = proc.memory_info().rss

    for _ in range(args.warmup):
        try:
            engine.synth("Hello.", "en" if "en" in engine.candidate.languages else language)
        except Exception:  # noqa: BLE001 - a failing warmup must not abort the run
            break

    runs = []
    for case in suite["cases"]:
        for repeat in range(args.repeats):
            rec = run_case(engine, case, language, repeat, audio_dir, keep_wav=not args.no_wav)
            runs.append(rec)
            g = rec["gate"]
            print(
                f"{engine.candidate.id}{args.tag} {case['id']} run={repeat + 1}/{args.repeats} "
                f"first={rec['first_chunk_s']:.3f}s total={rec['total_synthesis_s']:.3f}s "
                f"rtf={rec['rtf'] if rec['rtf'] is None else round(rec['rtf'], 3)} "
                f"gate={'PASS' if g['passed'] else 'FAIL ' + ','.join(g['hard_failures'])}",
                flush=True,
            )

    flush_runs = []
    if args.flush_sweep:
        # Single-segment texts of graded length: how long a first flush of N
        # characters costs is what an orchestrator can actually control.
        for text in suite.get("flush_sweep", []):
            for repeat in range(max(3, args.repeats)):
                case = {"id": f"flush_{len(text)}c", "kind": "flush_sweep", "text": text}
                rec = run_case(engine, case, language, repeat, audio_dir, keep_wav=False)
                rec["flush_text"] = text
                rec["flush_len"] = len(text)
                flush_runs.append(rec)
                print(
                    f"  flush len={len(text):>3} run={repeat + 1} "
                    f"first={rec['first_chunk_s']:.3f}s "
                    f"gate={'PASS' if rec['gate']['passed'] else 'FAIL'}",
                    flush=True,
                )

    floor_runs = []
    if args.chunk_floor_sweep:
        for text in suite.get("chunk_floor_sweep", []):
            for repeat in range(max(3, args.repeats)):
                case = {"id": f"floor_{len(text)}c_{abs(hash(text)) % 9973}", "kind": "chunk_floor", "text": text}
                rec = run_case(engine, case, language, repeat, audio_dir, keep_wav=False)
                rec["floor_text"] = text
                rec["floor_len"] = len(text)
                floor_runs.append(rec)
                print(
                    f"  floor len={len(text)} run={repeat + 1} "
                    f"gate={'PASS' if rec['gate']['passed'] else 'FAIL ' + ','.join(rec['gate']['hard_failures'])}",
                    flush=True,
                )

    summary = {}
    for case in suite["cases"]:
        sel = [r for r in runs if r["case_id"] == case["id"]]
        summary[case["id"]] = {
            "kind": case.get("kind"),
            "text": case["text"],
            "reference": case.get("reference"),
            "gate_pass_count": sum(1 for r in sel if r["gate"]["passed"]),
            "gate_run_count": len(sel),
            "hard_failures": sorted({f for r in sel for f in r["gate"]["hard_failures"]}),
            "first_chunk_s": distribution([r["first_chunk_s"] for r in sel]),
            "total_synthesis_s": distribution([r["total_synthesis_s"] for r in sel]),
            "rtf": distribution([r["rtf"] for r in sel]),
            "audio_duration_s": distribution([r["gate"]["metrics"].get("audio_duration_s") for r in sel]),
            "peak_rss_bytes": distribution([r["peak_rss_bytes"] for r in sel]),
            "chunk_count": distribution([r["chunk_count"] for r in sel]),
            "gap_free_1_5x": all(
                r["playback_simulation"]["1.5x"].get("gap_free", False) for r in sel
            ),
            "max_gap_s_1_5x": max(
                (r["playback_simulation"]["1.5x"].get("max_gap_s", 0.0) for r in sel), default=0.0
            ),
            "wav_path": next((r["wav_path"] for r in sel if r["wav_path"]), None),
            "wav_sha256": next((r["wav_sha256"] for r in sel if r["wav_sha256"]), None),
        }

    floor_summary = {}
    for rec in floor_runs:
        key = rec["floor_text"]
        entry = floor_summary.setdefault(
            key, {"chars": rec["floor_len"], "runs": 0, "gate_pass": 0, "failures": []}
        )
        entry["runs"] += 1
        entry["gate_pass"] += int(rec["gate"]["passed"])
        entry["failures"].extend(rec["gate"]["hard_failures"])
    for entry in floor_summary.values():
        entry["failures"] = sorted(set(entry["failures"]))

    flush_summary = {}
    for rec in flush_runs:
        entry = flush_summary.setdefault(
            rec["flush_text"],
            {"chars": rec["flush_len"], "first_chunk_s": [], "gate_pass": 0, "runs": 0},
        )
        entry["runs"] += 1
        entry["gate_pass"] += int(rec["gate"]["passed"])
        if rec["first_chunk_s"] is not None:
            entry["first_chunk_s"].append(rec["first_chunk_s"])
    for entry in flush_summary.values():
        entry["first_chunk"] = distribution(entry.pop("first_chunk_s"))

    payload = {
        "schema_version": 1,
        "engine_id": engine.candidate.id,
        "tag": args.tag,
        "family": engine.candidate.family,
        "suite": args.suite,
        "language": language,
        "license_weights": engine.candidate.license_weights,
        "license_risk": engine.candidate.license_risk,
        "notes": engine.candidate.notes,
        "footprint": engine.candidate.footprint_bytes(root),
        "config": engine.config_record(),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu": subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
            ).stdout.strip(),
            "python": sys.version.split()[0],
        },
        "load": {
            "load_s": load_s,
            "rss_before_bytes": rss_before,
            "rss_after_bytes": rss_after,
            "incremental_rss_bytes": rss_after - rss_before,
            "peak_rss_during_load_bytes": load_rss.peak,
        },
        "rss_cap_bytes": RSS_CAP_BYTES,
        "rss_cap_exceeded_any_run": any(
            r["rss_cap_exceeded"] for r in runs + floor_runs + flush_runs
        ),
        "summary": summary,
        "chunk_floor_summary": floor_summary,
        "flush_sweep_summary": flush_summary,
        "runs": runs,
        "chunk_floor_runs": floor_runs,
        "flush_sweep_runs": flush_runs,
    }

    name = f"bench_{engine.candidate.id}{args.tag}_{args.suite}.json"
    out = args.output_dir / name
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")
    print(
        f"gate pass: {sum(1 for r in runs if r['gate']['passed'])}/{len(runs)} runs; "
        f"peak RSS {max((r['peak_rss_bytes'] for r in runs), default=0) / 1048576:.0f} MiB; "
        f"cap exceeded: {payload['rss_cap_exceeded_any_run']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
