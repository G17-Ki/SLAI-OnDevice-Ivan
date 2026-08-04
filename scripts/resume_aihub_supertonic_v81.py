#!/usr/bin/env python3
"""Resume one interrupted Supertonic v81 item from its persisted job IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import qai_hub as hub

from run_aihub_supertonic_v81 import (
    classify_profile,
    compare,
    download_logs,
    load_bundle,
    poll,
    profile_metrics,
    sha256,
    status_record,
    upload_with_retry,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--item", required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--ceiling-seconds", type=int, default=10_800)
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument(
        "--retry-missing-or-failed-compile",
        action="store_true",
        help=(
            "Submit a compile job only when no compile ID exists or the recorded "
            "compile finished unsuccessfully. Successful compile IDs are always reused."
        ),
    )
    return parser.parse_args()


def get_job(client: hub.Client, record: dict[str, Any], stage: str) -> Any:
    job_id = record.get("job_id")
    if not job_id:
        raise ValueError(f"No {stage} job ID is recorded")
    return client.get_job(job_id)


def sanitize_duplicate_io_value_info(source: Path, destination: Path) -> dict[str, Any]:
    """Remove graph I/O names duplicated in ONNX value_info.

    QAIRT rejects this redundant metadata even though ONNX Runtime accepts it.
    The original model is never modified; the repaired copy and its hash remain
    beside the persisted run state.
    """

    import onnx

    model = onnx.load(source, load_external_data=True)
    io_names = {value.name for value in model.graph.input}
    io_names.update(value.name for value in model.graph.output)
    retained = [value for value in model.graph.value_info if value.name not in io_names]
    removed = [value.name for value in model.graph.value_info if value.name in io_names]
    if not removed:
        return {
            "applied": False,
            "source": str(source),
            "source_sha256": sha256(source),
            "reason": "No graph I/O names occurred in value_info.",
        }
    del model.graph.value_info[:]
    model.graph.value_info.extend(retained)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, destination)
    onnx.checker.check_model(destination)
    return {
        "applied": True,
        "source": str(source),
        "source_sha256": sha256(source),
        "destination": str(destination),
        "destination_sha256": sha256(destination),
        "removed_value_info_names": removed,
    }


def compile_input_specs(item: dict[str, Any]) -> dict[str, tuple[tuple[int, ...], str]]:
    specs = item.get("input_specs")
    if not isinstance(specs, dict) or not specs:
        raise ValueError("Persisted item has no compile input specs")
    result: dict[str, tuple[tuple[int, ...], str]] = {}
    for name, value in specs.items():
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"Malformed persisted input spec for {name}: {value!r}")
        result[name] = (tuple(int(dim) for dim in value[0]), str(value[1]))
    return result


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    state_path = run_dir / "state.json"
    state = json.loads(state_path.read_text())
    if args.item not in state.get("items", {}):
        raise KeyError(f"Unknown item {args.item!r}")
    item = state["items"][args.item]
    if not item.get("bundle"):
        raise ValueError(f"Item {args.item} has no persisted I/O bundle")
    remote_inputs, goldens = load_bundle(Path(item["bundle"]))
    remote_inputs = {
        key: value.astype(np.int32, copy=False) if value.dtype == np.int64 else value
        for key, value in remote_inputs.items()
    }
    client = hub.Client()
    target = state["target"]
    device = hub.Device(target["device"], target["os"], ["framework:qnn"])
    item.setdefault("resume_attempts", []).append(
        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    write_json(state_path, state)

    compile_record = item.get("jobs", {}).get("compile")
    compile_job = None
    compile_snapshot: dict[str, Any] = {}
    if compile_record and compile_record.get("job_id"):
        compile_job = get_job(client, compile_record, "compile")
        compile_snapshot = status_record(compile_job)

    needs_compile = compile_job is None or (
        compile_snapshot.get("finished") and not compile_snapshot.get("success")
    )
    if needs_compile:
        if not args.retry_missing_or_failed_compile:
            if compile_job is None:
                raise ValueError(
                    "No compile job is recorded; pass --retry-missing-or-failed-compile "
                    "to submit the missing stage."
                )
            raise RuntimeError(
                "Recorded compile failed; pass --retry-missing-or-failed-compile "
                "to submit a corrected/retried graph."
            )
        if compile_record:
            item.setdefault("job_attempts", {}).setdefault("compile", []).append(
                dict(compile_record)
            )
        source_path = Path(item["model"])
        prior_message = str((compile_record or {}).get("message", ""))
        if "occur in value_info but also in model IO" in prior_message:
            repaired_path = run_dir / args.item / "source" / f"{source_path.stem}.io_sanitized.onnx"
            sanitization = sanitize_duplicate_io_value_info(source_path, repaired_path)
            item.setdefault("source_repairs", []).append(sanitization)
            if sanitization.get("applied"):
                source_path = repaired_path
        source_model = upload_with_retry(
            source_path,
            name=f"lfm-demo-supertonic-{args.item}-resume",
        )
        item.setdefault("source_model_attempts", []).append(
            {
                "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "model_id": source_model.model_id,
                "path": str(source_path),
                "sha256": sha256(source_path),
            }
        )
        item["source_model_id"] = source_model.model_id
        compile_job = client.submit_compile_job(
            model=source_model,
            device=device,
            input_specs=compile_input_specs(item),
            options=item["compile_options"],
            name=f"lfm-demo-v81-supertonic-{args.item}-resume",
            retry=False,
        )
        item.setdefault("jobs", {})["compile"] = status_record(compile_job)
        item["status"] = "compile_submitted"
        write_json(state_path, state)

    assert compile_job is not None
    compile_snapshot = poll(
        {args.item: compile_job},
        state,
        state_path,
        f"{args.item}:compile_resume",
        args.poll_seconds,
        args.ceiling_seconds,
    ).get(args.item, {})
    item["jobs"]["compile"] = compile_snapshot or status_record(compile_job)
    if not compile_snapshot.get("success"):
        item["status"] = (
            "compile_pending" if not compile_snapshot.get("finished") else "compile_failed"
        )
        item.setdefault("job_logs", {})["compile"] = download_logs(
            compile_job, run_dir / args.item / "job_logs" / "compile"
        )
        write_json(state_path, state)
        return 2
    target_model = compile_job.get_target_model()
    if target_model is None:
        raise RuntimeError("Successful compile job returned no target model")

    profile_record = item.get("jobs", {}).get("profile")
    inference_record = item.get("jobs", {}).get("inference")
    if profile_record and profile_record.get("job_id"):
        profile_job = get_job(client, profile_record, "profile")
    else:
        profile_job = client.submit_profile_job(
            model=target_model,
            device=device,
            options=item["profile_options"],
            name=f"lfm-demo-v81-supertonic-{args.item}-profile-resume",
            retry=False,
        )
        item.setdefault("jobs", {})["profile"] = status_record(profile_job)
    if inference_record and inference_record.get("job_id"):
        inference_job = get_job(client, inference_record, "inference")
    else:
        inference_job = client.submit_inference_job(
            model=target_model,
            device=device,
            inputs={key: [value] for key, value in remote_inputs.items()},
            options=f"--qairt_version {target['qairt_version']}",
            name=f"lfm-demo-v81-supertonic-{args.item}-inference-resume",
            retry=False,
        )
        item.setdefault("jobs", {})["inference"] = status_record(inference_job)
    item["status"] = "profile_and_inference_submitted"
    write_json(state_path, state)

    dependent = poll(
        {"profile": profile_job, "inference": inference_job},
        state,
        state_path,
        f"{args.item}:profile_and_inference_resume",
        args.poll_seconds,
        args.ceiling_seconds,
    )
    item["jobs"].update(dependent)
    if not all(value.get("success") for value in dependent.values()):
        item["status"] = (
            "remote_jobs_pending"
            if any(not value.get("finished") for value in dependent.values())
            else "profile_or_inference_failed"
        )
        write_json(state_path, state)
        return 3

    profile = profile_job.download_profile()
    remote = inference_job.download_output_data()
    if not isinstance(profile, dict) or not isinstance(remote, dict):
        raise TypeError("AI Hub returned an unexpected profile or inference payload")
    item_dir = run_dir / args.item
    write_json(item_dir / "profile.json", profile)
    placement = classify_profile(profile)
    write_json(item_dir / "placement.json", placement)
    golden_name, golden = next(iter(goldens.items()))
    remote_name, samples = next(iter(remote.items()))
    if not isinstance(samples, list) or len(samples) != 1:
        raise ValueError(f"Unexpected remote output structure for {remote_name}")
    actual = np.asarray(samples[0])
    comparison = compare(actual, golden, args.atol, args.rtol)
    np.savez(
        item_dir / "remote_outputs.npz",
        **{f"actual__{golden_name}": actual, f"golden__{golden_name}": golden},
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
        "remote_output_name": remote_name,
        "golden_output_name": golden_name,
        "comparison": comparison,
    }
    item["status"] = (
        "passed" if strict_npu and comparison["passed"] else "completed_with_warning"
    )
    item["job_logs"] = {
        "compile": download_logs(compile_job, item_dir / "job_logs" / "compile"),
        "profile": download_logs(profile_job, item_dir / "job_logs" / "profile"),
        "inference": download_logs(inference_job, item_dir / "job_logs" / "inference"),
    }
    write_json(state_path, state)
    print(
        json.dumps(
            {"item": args.item, "status": item["status"], "jobs": item["jobs"]},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
