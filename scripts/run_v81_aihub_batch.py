#!/usr/bin/env python3
"""Re-run the fixed-shape FP16 matrix using models already stored in AI Hub.

The batch submits every compile job before polling. Successful compiles then
have every profile and inference job submitted before the second polling
stage. Existing Hub datasets are reused so this run does not upload local
workspace tensors again.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import qai_hub as hub

from path_config import WORK_ROOT
from submit_component_aihub import (
    compare_output,
    compiled_output_names,
    ordered_remote_outputs,
    select_remote_output,
)
from submit_encoder_aihub import classify_profile, profile_metrics


EXPORTS = Path(
    os.environ.get(
        "LFM_EXPORTS_DIR",
        str(WORK_ROOT / "lfm-feasibility/exports"),
    )
).expanduser()


COMPONENTS: list[dict[str, Any]] = [
    {
        "name": "fastconformer",
        "source_model_id": "mmxzgkxen",
        "source_compile_job_id": "jp84j2xq5",
        "input_dataset_id": "d7jg8zyw7",
        "source_inference_job_id": "jprnx6re5",
        "io": EXPORTS / "fastconformer_adapter_mel80.npz",
        "inputs": {"mel": "mel"},
        "outputs": [("#0", "adapted")],
    },
    {
        "name": "backbone-conv-prefill",
        "source_model_id": "mmr28090n",
        "source_compile_job_id": "j5w1l9y4g",
        "input_dataset_id": "d7lvvjgn2",
        "source_inference_job_id": "jpxdl1vjg",
        "io": EXPORTS / "lfm2_layer0_conv_seq16.npz",
        "inputs": {"hidden_states": "hidden_states"},
        "outputs": [("#0", "output")],
    },
    {
        "name": "backbone-attention-prefill",
        "source_model_id": "mnjk65y9m",
        "source_compile_job_id": "jpv9l1vj5",
        "input_dataset_id": "d9155mxe7",
        "source_inference_job_id": "jp84e3eo5",
        "io": EXPORTS / "lfm2_layer2_attention_seq16.npz",
        "inputs": {"hidden_states": "hidden_states"},
        "outputs": [("#0", "output")],
    },
    {
        "name": "backbone-conv-cached-decode",
        "source_model_id": "mn09kezxn",
        "source_compile_job_id": "jp1vndrkp",
        "input_dataset_id": "d9kvvjj52",
        "source_inference_job_id": "jpel74o0g",
        "io": EXPORTS / "lfm2_layer0_conv_decode_cache.npz",
        "inputs": {"hidden_states": "hidden_states", "conv_cache": "conv_cache"},
        "outputs": [("#0", "output"), ("#1", "updated_conv_cache")],
    },
    {
        "name": "backbone-attention-cached-decode",
        "source_model_id": "mqpy88xjq",
        "source_compile_job_id": "jp3wovwn5",
        "input_dataset_id": "d9pnlndd7",
        "source_inference_job_id": "jgn71lqvp",
        "io": EXPORTS / "lfm2_layer2_attention_decode_kv16_past8.npz",
        "inputs": {
            "hidden_states": "hidden_states",
            "key_cache": "key_cache",
            "value_cache": "value_cache",
        },
        "outputs": [
            ("#0", "output"),
            ("#1", "updated_key_cache"),
            ("#2", "updated_value_cache"),
        ],
    },
    {
        "name": "depth-decoder",
        "source_model_id": "mqkxyyzxn",
        "source_compile_job_id": "jp49ek2q5",
        "input_dataset_id": "d9emwmov2",
        "source_inference_job_id": "jpxd06k1g",
        "io": EXPORTS / "depth_decoder_hidden1x2048.npz",
        "inputs": {"hidden": "hidden"},
        "outputs": [("#0", "tokens"), ("#1", "next_audio_embedding")],
        "truncate_64bit_io": True,
    },
    {
        "name": "detokenizer-t4",
        "source_model_id": "mq2zg7zlq",
        "source_compile_job_id": "j5w1l116g",
        "input_dataset_id": "d26qq6zg7",
        "source_inference_job_id": "j5qmlm4ng",
        "io": EXPORTS / "detok_neural_probe_codes_t4.npz",
        "inputs": {"codes": "codes"},
        "outputs": [("#0", "log_abs"), ("#1", "angle")],
        "truncate_64bit_io": True,
    },
    {
        "name": "detokenizer-t8",
        "source_model_id": "mm5wrk1km",
        "source_compile_job_id": "jp84jm2x5",
        "input_dataset_id": "d26qmqeg7",
        "source_inference_job_id": "jgdz6qqe5",
        "io": EXPORTS / "detok_neural_probe_codes_t8.npz",
        "inputs": {"codes": "codes"},
        "outputs": [("#0", "log_abs"), ("#1", "angle")],
        "truncate_64bit_io": True,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="Snapdragon 8 Elite Gen 5 QRD")
    parser.add_argument("--device-os", default="16")
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--ceiling-seconds", type=int, default=10_800)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Optional component name to run; repeat for more than one.",
    )
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def snapshot_job(job: Any) -> dict[str, Any]:
    status = job.get_status()
    return {
        "job_id": job.job_id,
        "url": job.url,
        "status": status.code,
        "message": status.message,
    }


def submitted_job(job: Any) -> dict[str, Any]:
    return {"job_id": job.job_id, "url": job.url, "status": "SUBMITTED"}


def download_logs(job: Any, output_dir: Path) -> dict[str, Any]:
    try:
        return {"status": "downloaded", "files": job.download_job_logs(str(output_dir))}
    except Exception as error:
        return {
            "status": "failed",
            "error_type": type(error).__name__,
            "error_verbatim": str(error),
        }


def input_specs(component: dict[str, Any]) -> dict[str, tuple[tuple[int, ...], str]]:
    with np.load(component["io"]) as archive:
        result = {}
        for graph_name, npz_name in component["inputs"].items():
            value = archive[npz_name]
            dtype = str(value.dtype)
            if component.get("truncate_64bit_io") and dtype == "int64":
                dtype = "int64"
            result[graph_name] = (tuple(value.shape), dtype)
        return result


def poll_stage(
    stage: str,
    jobs: dict[str, Any],
    state: dict[str, Any],
    state_path: Path,
    start: float,
    poll_seconds: int,
    ceiling_seconds: int,
) -> None:
    while True:
        terminal = True
        now = time.time()
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))
        poll = {name: snapshot_job(job) for name, job in jobs.items()}
        state["polls"].append({"stage": stage, "time_utc": stamp, "jobs": poll})
        state["jobs"][stage] = poll
        write_json(state_path, state)
        print(json.dumps({"stage": stage, "time_utc": stamp, "jobs": poll}, indent=2))
        for item in poll.values():
            terminal = terminal and item["status"] in {"SUCCESS", "FAILED"}
        if terminal:
            return
        if time.monotonic() - start >= ceiling_seconds:
            state.setdefault("pending_at_publication_time", {})[stage] = {
                name: item
                for name, item in poll.items()
                if item["status"] not in {"SUCCESS", "FAILED"}
            }
            write_json(state_path, state)
            return
        time.sleep(poll_seconds)


def compare_component(
    component: dict[str, Any],
    target_model: Any,
    profile: dict[str, Any],
    remote: dict[str, Any],
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    with np.load(component["io"]) as archive:
        goldens = {npz_name: np.array(archive[npz_name]) for _, npz_name in component["outputs"]}
    target_names = compiled_output_names(target_model)
    ordered = ordered_remote_outputs(remote)
    comparisons: dict[str, Any] = {}
    resolved: dict[str, str] = {}
    saved: dict[str, np.ndarray] = {}
    for selector, golden_name in component["outputs"]:
        resolved_selector = selector
        if selector.startswith("#"):
            index = int(selector[1:])
            if index < len(target_names) and target_names[index] in remote:
                resolved_selector = target_names[index]
        remote_name, actual = select_remote_output(resolved_selector, remote, ordered)
        golden = goldens[golden_name]
        resolved[remote_name] = golden_name
        comparisons[golden_name] = compare_output(actual, golden, atol, rtol)
        saved[f"actual__{golden_name}"] = actual
        saved[f"golden__{golden_name}"] = golden
    placement = classify_profile(profile)
    return {
        "status": "passed" if all(v["passed"] for v in comparisons.values()) else "numerical_mismatch",
        "resolved_output_mapping": resolved,
        "comparisons": comparisons,
        "placement": placement,
        "metrics": profile_metrics(profile),
        "strict_npu_pass": (
            placement["npu_runtime_layers"] > 0
            and placement["cpu_fallback_runtime_layers"] == 0
            and placement["other_runtime_layers"] == 0
        ),
        "arrays": saved,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / "batch-state.json"
    selected = [
        component
        for component in COMPONENTS
        if not args.only or component["name"] in set(args.only)
    ]
    if not selected:
        raise ValueError(f"No components matched --only values: {args.only}")
    start = time.monotonic()
    client = hub.Client()
    device = hub.Device(args.device, args.device_os, ["framework:qnn"])
    state: dict[str, Any] = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": {"name": args.device, "os": args.device_os, "chipset": "sm8850", "htp": "v81"},
        "runtime": "strict-npu",
        "compile_options": "--target_runtime qnn_context_binary --qnn_options default_graph_htp_precision=FLOAT16",
        "profile_options": "--max_profiler_iterations 20",
        "poll_interval_seconds": args.poll_seconds,
        "ceiling_seconds": args.ceiling_seconds,
        "source_reuse_note": "Existing AI Hub source models and input datasets are reused; no local model or input tensor is uploaded by this round.",
        "components": {},
        "jobs": {},
        "polls": [],
    }

    compile_jobs: dict[str, Any] = {}
    for component in selected:
        name = component["name"]
        options = state["compile_options"]
        if component.get("truncate_64bit_io"):
            options += " --truncate_64bit_io"
        state["components"][name] = {
            "source_model_id": component["source_model_id"],
            "source_compile_job_id": component["source_compile_job_id"],
            "input_dataset_id": component["input_dataset_id"],
            "source_inference_job_id": component["source_inference_job_id"],
            "io": str(component["io"]),
            "input_mapping": component["inputs"],
            "output_mapping": component["outputs"],
            "input_specs": input_specs(component),
            "compile_options": options,
        }
        job = client.submit_compile_job(
            model=client.get_model(component["source_model_id"]),
            device=device,
            input_specs=input_specs(component),
            options=options,
            name=f"lfm-v81-round-{name}-strict-npu",
            retry=False,
        )
        compile_jobs[name] = job
        state.setdefault("submitted", {}).setdefault("compile", {})[name] = submitted_job(job)
        write_json(state_path, state)
        print(f"submitted compile {name}: {job.job_id}", flush=True)

    print("all compile jobs submitted; beginning 5-minute polling", flush=True)
    time.sleep(args.poll_seconds)
    poll_stage("compile", compile_jobs, state, state_path, start, args.poll_seconds, args.ceiling_seconds)

    profile_jobs: dict[str, Any] = {}
    inference_jobs: dict[str, Any] = {}
    targets: dict[str, Any] = {}
    for component in selected:
        name = component["name"]
        compile_status = compile_jobs[name].get_status()
        if not compile_status.finished:
            state["components"][name]["result"] = {
                "status": "pending_at_publication_time",
                "job_id": compile_jobs[name].job_id,
                "state": compile_status.code,
            }
            continue
        if not compile_status.success:
            state["components"][name]["result"] = {
                "status": "unsupported_or_compile_failure",
                "error_verbatim": compile_status.message,
            }
            continue
        target = compile_jobs[name].get_target_model()
        if target is None:
            state["components"][name]["result"] = {
                "status": "unsupported_or_compile_failure",
                "error_verbatim": "Compile status was SUCCESS but get_target_model() returned None.",
            }
            continue
        targets[name] = target
        profile_jobs[name] = client.submit_profile_job(
            model=target,
            device=device,
            options=state["profile_options"],
            name=f"lfm-v81-round-{name}-profile-strict-npu",
            retry=False,
        )
        state.setdefault("submitted", {}).setdefault("profile", {})[name] = submitted_job(profile_jobs[name])
        write_json(state_path, state)

    for component in selected:
        name = component["name"]
        if name not in targets:
            continue
        inference_jobs[name] = client.submit_inference_job(
            model=targets[name],
            device=device,
            inputs=client.get_dataset(component["input_dataset_id"]),
            name=f"lfm-v81-round-{name}-inference-strict-npu",
            retry=False,
        )
        state.setdefault("submitted", {}).setdefault("inference", {})[name] = submitted_job(inference_jobs[name])
        write_json(state_path, state)

    print("all dependent profile/inference jobs submitted; beginning 5-minute polling", flush=True)
    if profile_jobs or inference_jobs:
        time.sleep(args.poll_seconds)
    if profile_jobs:
        poll_stage("profile", profile_jobs, state, state_path, start, args.poll_seconds, args.ceiling_seconds)
    if inference_jobs:
        poll_stage("inference", inference_jobs, state, state_path, start, args.poll_seconds, args.ceiling_seconds)

    for component in selected:
        name = component["name"]
        run_dir = args.output_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)
        logs: dict[str, Any] = {}
        if compile_jobs[name].get_status().finished:
            logs["compile"] = download_logs(compile_jobs[name], run_dir / "server-logs" / "compile")
        if name in profile_jobs and profile_jobs[name].get_status().finished:
            logs["profile"] = download_logs(profile_jobs[name], run_dir / "server-logs" / "profile")
        if name in inference_jobs and inference_jobs[name].get_status().finished:
            logs["inference"] = download_logs(inference_jobs[name], run_dir / "server-logs" / "inference")
        state["components"][name]["server_logs"] = logs
        write_json(state_path, state)

    for component in selected:
        name = component["name"]
        run_dir = args.output_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)
        result = state["components"][name].get("result")
        if result is not None:
            write_json(run_dir / "summary-strict-npu.json", {"plan": state["components"][name], "jobs": state["jobs"], **result})
            continue
        pstatus = profile_jobs[name].get_status()
        istatus = inference_jobs[name].get_status()
        if not pstatus.finished or not istatus.finished:
            result = {
                "status": "pending_at_publication_time",
                "profile_job_id": profile_jobs[name].job_id,
                "profile_state": pstatus.code,
                "inference_job_id": inference_jobs[name].job_id,
                "inference_state": istatus.code,
            }
            state["components"][name]["result"] = result
            write_json(run_dir / "summary-strict-npu.json", {"plan": state["components"][name], "jobs": state["jobs"], **result})
            continue
        if not pstatus.success or not istatus.success:
            result = {
                "status": "profile_or_inference_failure",
                "profile_error_verbatim": pstatus.message,
                "inference_error_verbatim": istatus.message,
            }
            state["components"][name]["result"] = result
            write_json(run_dir / "summary-strict-npu.json", {"plan": state["components"][name], "jobs": state["jobs"], **result})
            continue
        profile = profile_jobs[name].download_profile()
        remote = inference_jobs[name].download_output_data()
        if not isinstance(profile, dict) or not isinstance(remote, dict):
            result = {
                "status": "artifact_download_failure",
                "error_verbatim": f"profile_type={type(profile).__name__}, inference_type={type(remote).__name__}",
            }
            state["components"][name]["result"] = result
            write_json(run_dir / "summary-strict-npu.json", {"plan": state["components"][name], "jobs": state["jobs"], **result})
            continue
        comparison = compare_component(component, targets[name], profile, remote, args.atol, args.rtol)
        arrays = comparison.pop("arrays")
        np.savez(run_dir / "outputs-strict-npu.npz", **arrays)
        write_json(run_dir / "profile-strict-npu.json", profile)
        write_json(run_dir / "placement-strict-npu.json", comparison["placement"])
        result = {
            **comparison,
            "jobs": {
                "compile": snapshot_job(compile_jobs[name]),
                "profile": snapshot_job(profile_jobs[name]),
                "inference": snapshot_job(inference_jobs[name]),
            },
        }
        state["components"][name]["result"] = result
        write_json(run_dir / "summary-strict-npu.json", {"plan": state["components"][name], **result})
        write_json(state_path, state)

    state["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json(state_path, state)
    print(json.dumps(state, indent=2, default=str))


if __name__ == "__main__":
    main()
