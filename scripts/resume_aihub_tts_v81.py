#!/usr/bin/env python3
"""Resume an interrupted AI Hub Piper v81 run from persisted job IDs."""

from __future__ import annotations

import argparse
import atexit
import json
from pathlib import Path
import time
from typing import Any

from run_aihub_tts_v81 import (
    ResourceGuard,
    compare_outputs,
    download_job_logs,
    poll_jobs,
    submitted_job_record,
    write_json,
)

import numpy as np
import qai_hub as hub
from submit_encoder_aihub import classify_profile, profile_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--item", required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--ceiling-seconds", type=int, default=10_800)
    parser.add_argument("--atol", type=float, default=3e-2)
    parser.add_argument("--rtol", type=float, default=3e-2)
    return parser.parse_args()


def target_model(job: Any, stage: str) -> Any:
    model = job.get_target_model()
    if model is None:
        raise RuntimeError(f"Successful {stage} job returned no target model")
    return model


def load_io(path: Path) -> tuple[dict[str, list[np.ndarray]], dict[str, np.ndarray]]:
    with np.load(path) as archive:
        inputs = {
            name.removeprefix("input__"): [np.array(archive[name])]
            for name in archive.files
            if name.startswith("input__")
        }
        goldens = {
            name.removeprefix("golden__"): np.array(archive[name])
            for name in archive.files
            if name.startswith("golden__")
        }
    if not inputs or not goldens:
        raise ValueError(f"Golden I/O archive is incomplete: {path}")
    return inputs, goldens


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    state_path = run_dir / "state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"Persistent AI Hub state is absent: {state_path}")
    resource_guard = ResourceGuard(run_dir / "resource_guard_resume.json")
    resource_guard.start()
    atexit.register(resource_guard.stop)
    state = json.loads(state_path.read_text())
    if args.item not in state.get("items", {}):
        raise KeyError(f"Unknown item {args.item!r}; available: {list(state.get('items', {}))}")
    item = state["items"][args.item]
    item_dir = run_dir / args.item
    inputs, goldens = load_io(Path(item["local"]["golden_io"]))

    client = hub.Client()
    target = state["target"]
    device = hub.Device(target["device"], target["os"], ["framework:qnn"])
    item.setdefault("resume_attempts", []).append(
        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )

    compile_record = item.get("jobs", {}).get("compile", {})
    compile_job_id = compile_record.get("job_id")
    if not compile_job_id:
        raise ValueError("No compile job ID is recorded; use a fresh output directory")
    compile_job = client.get_job(compile_job_id)
    compile_snapshot = poll_jobs(
        {args.item: compile_job},
        state,
        state_path,
        f"{args.item}:compile_resume",
        args.poll_seconds,
        args.ceiling_seconds,
    )[args.item]
    item["jobs"]["compile"] = compile_snapshot
    if not compile_snapshot["success"]:
        item["status"] = (
            "compile_pending" if not compile_snapshot["finished"] else "compile_failed"
        )
        write_json(state_path, state)
        raise SystemExit(2)
    dlc_model = target_model(compile_job, "compile")

    link_record = item.get("jobs", {}).get("link")
    if link_record and link_record.get("job_id"):
        link_job = client.get_job(link_record["job_id"])
    else:
        resource_guard.check_now(persist=True)
        link_job = client.submit_link_job(
            [dlc_model],
            device=device,
            options=item["options"]["link"],
            name=f"lfm-demo-v81-{args.item}-context-resume",
        )
        item["jobs"]["link"] = submitted_job_record(link_job)
        item["status"] = "link_submitted"
        write_json(state_path, state)
    link_snapshot = poll_jobs(
        {args.item: link_job},
        state,
        state_path,
        f"{args.item}:link_resume",
        args.poll_seconds,
        args.ceiling_seconds,
    )[args.item]
    item["jobs"]["link"] = link_snapshot
    if not link_snapshot["success"]:
        item["status"] = "link_pending" if not link_snapshot["finished"] else "link_failed"
        write_json(state_path, state)
        raise SystemExit(2)
    context_model = target_model(link_job, "link")

    profile_record = item.get("jobs", {}).get("profile")
    inference_record = item.get("jobs", {}).get("inference")
    if profile_record and profile_record.get("job_id"):
        profile_job = client.get_job(profile_record["job_id"])
    else:
        resource_guard.check_now(persist=True)
        profile_job = client.submit_profile_job(
            model=context_model,
            device=device,
            options=item["options"]["profile"],
            name=f"lfm-demo-v81-{args.item}-profile-resume",
            retry=False,
        )
        item["jobs"]["profile"] = submitted_job_record(profile_job)
        item["status"] = "profile_submitted"
        write_json(state_path, state)
    if inference_record and inference_record.get("job_id"):
        inference_job = client.get_job(inference_record["job_id"])
    else:
        resource_guard.check_now(persist=True)
        inference_job = client.submit_inference_job(
            model=context_model,
            device=device,
            inputs=inputs,
            options=item["options"]["inference"],
            name=f"lfm-demo-v81-{args.item}-inference-resume",
            retry=False,
        )
        item["jobs"]["inference"] = submitted_job_record(inference_job)
        item["status"] = "inference_submitted"
        write_json(state_path, state)
    item["status"] = "profile_and_inference_submitted"
    write_json(state_path, state)

    dependent = poll_jobs(
        {"profile": profile_job, "inference": inference_job},
        state,
        state_path,
        f"{args.item}:profile_and_inference_resume",
        args.poll_seconds,
        args.ceiling_seconds,
    )
    item["jobs"].update(dependent)
    if not all(entry["success"] for entry in dependent.values()):
        item["status"] = (
            "remote_jobs_pending"
            if any(not entry["finished"] for entry in dependent.values())
            else "profile_or_inference_failed"
        )
        write_json(state_path, state)
        raise SystemExit(3)

    profile = profile_job.download_profile()
    remote = inference_job.download_output_data()
    if not isinstance(profile, dict) or not isinstance(remote, dict):
        raise TypeError("AI Hub returned an unexpected profile or inference payload")
    write_json(item_dir / "profile.json", profile)
    placement = classify_profile(profile)
    write_json(item_dir / "placement.json", placement)
    comparisons, saved = compare_outputs(
        remote, goldens, context_model, args.atol, args.rtol
    )
    np.savez(item_dir / "remote_outputs.npz", **saved)
    all_numerics_pass = bool(comparisons) and all(
        result.get("passed", False) for result in comparisons.values()
    )
    strict_npu = (
        placement["npu_runtime_layers"] > 0
        and placement["cpu_fallback_runtime_layers"] == 0
        and placement["other_runtime_layers"] == 0
    )
    item["result"] = {
        "metrics": profile_metrics(profile),
        "placement": placement,
        "strict_npu_pass": strict_npu,
        "comparisons": comparisons,
        "all_numerics_pass": all_numerics_pass,
    }
    item["status"] = (
        "passed"
        if strict_npu and all_numerics_pass
        else "completed_with_numerical_or_placement_warning"
    )
    item["job_logs"] = {
        "compile": download_job_logs(compile_job, item_dir / "job_logs" / "compile"),
        "link": download_job_logs(link_job, item_dir / "job_logs" / "link"),
        "profile": download_job_logs(profile_job, item_dir / "job_logs" / "profile"),
        "inference": download_job_logs(
            inference_job, item_dir / "job_logs" / "inference"
        ),
    }
    state["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json(state_path, state)
    print(json.dumps({"item": args.item, "status": item["status"], "jobs": item["jobs"]}, indent=2))
    resource_guard.stop()
    atexit.unregister(resource_guard.stop)


if __name__ == "__main__":
    main()
