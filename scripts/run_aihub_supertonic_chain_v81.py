#!/usr/bin/env python3
"""Run a real Supertonic chain through existing SM8850 AI Hub builds.

The four compile jobs are immutable inputs. This runner submits inference jobs
only, persists every job ID before polling, and resumes from saved outputs.
Launch it inside detached tmux; it refuses direct cloud execution otherwise.
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

from supertonic_chain_common import (
    audio_metrics,
    compare_arrays,
    describe_array,
    sha256_array,
    sha256_file,
    write_json,
    write_pcm16_wav,
)


DEFAULT_COMPILE_JOBS = {
    "duration_predictor": "jp2v287x5",
    "text_encoder": "j5mdj79d5",
    "vector_estimator": "j5w1o2k6g",
    "vocoder": "jgo4ykxx5",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="Snapdragon 8 Elite Gen 5 QRD")
    parser.add_argument("--device-os", default="16")
    parser.add_argument("--qairt-version", default="2.45")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--ceiling-seconds", type=int, default=10_800)
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-no-tmux",
        action="store_true",
        help="Permit cloud submission outside tmux. Not for unattended runs.",
    )
    for component, default in DEFAULT_COMPILE_JOBS.items():
        parser.add_argument(
            f"--{component.replace('_', '-')}-compile-job", default=default
        )
    args = parser.parse_args()
    if args.poll_seconds < 1 or args.ceiling_seconds < 1:
        parser.error("poll and ceiling values must be positive")
    if args.atol < 0 or args.rtol < 0:
        parser.error("tolerances must be non-negative")
    if not args.dry_run and not args.allow_no_tmux and not os.environ.get("TMUX"):
        parser.error("cloud execution must run inside detached tmux")
    return args


def job_snapshot(job: Any) -> dict[str, Any]:
    status = job.get_status()
    return {
        "job_id": getattr(job, "job_id", None),
        "url": getattr(job, "url", None),
        "status": status.code,
        "message": status.message,
        "finished": bool(status.finished),
        "success": bool(status.success),
    }


def poll_job(
    job: Any,
    *,
    stage: str,
    state: dict[str, Any],
    state_path: Path,
    poll_seconds: int,
    ceiling_seconds: int,
) -> dict[str, Any]:
    started = time.monotonic()
    consecutive_errors = 0
    while True:
        try:
            snapshot = job_snapshot(job)
            consecutive_errors = 0
        except Exception as error:
            consecutive_errors += 1
            state.setdefault("poll_errors", []).append(
                {
                    "stage": stage,
                    "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "type": type(error).__name__,
                    "message": str(error),
                    "consecutive": consecutive_errors,
                }
            )
            write_json(state_path, state)
            if time.monotonic() - started >= ceiling_seconds:
                raise TimeoutError(f"Polling ceiling reached for {stage}") from error
            time.sleep(min(60, poll_seconds * consecutive_errors))
            continue

        state["stages"][stage]["job"] = snapshot
        state.setdefault("polls", []).append(
            {
                "stage": stage,
                "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "job": snapshot,
            }
        )
        write_json(state_path, state)
        print(json.dumps({"stage": stage, "job": snapshot}, indent=2), flush=True)
        if snapshot["finished"]:
            return snapshot
        if time.monotonic() - started >= ceiling_seconds:
            raise TimeoutError(f"Polling ceiling reached for {stage}")
        time.sleep(poll_seconds)


def save_array(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as destination:
        np.savez(destination, output=np.asarray(value))
    temporary.replace(path)


def load_array(path: Path) -> np.ndarray:
    with np.load(path) as archive:
        if list(archive.files) != ["output"]:
            raise ValueError(f"Unexpected saved output keys in {path}: {archive.files}")
        return np.array(archive["output"])


def extract_single_output(payload: Any) -> tuple[str, np.ndarray]:
    if not isinstance(payload, dict) or len(payload) != 1:
        raise TypeError(
            f"Expected one output tensor, received {type(payload).__name__}"
        )
    name, samples = next(iter(payload.items()))
    if not isinstance(samples, list) or len(samples) != 1:
        raise ValueError(f"Expected one sample for output {name!r}")
    return str(name), np.asarray(samples[0])


def successful_target(client: hub.Client, job_id: str, component: str) -> Any:
    job = client.get_job(job_id)
    status = job.get_status()
    if not status.finished or not status.success:
        raise RuntimeError(
            f"Compile job {job_id} for {component} is not successful: "
            f"{status.code} {status.message}"
        )
    target = job.get_target_model()
    if target is None:
        raise RuntimeError(f"Compile job {job_id} returned no target model")
    return target


def stage_output(
    *,
    client: hub.Client,
    device: hub.Device,
    model: Any,
    inputs: dict[str, np.ndarray],
    stage: str,
    output_dir: Path,
    state: dict[str, Any],
    state_path: Path,
    options: str,
    poll_seconds: int,
    ceiling_seconds: int,
) -> np.ndarray:
    stage_record = state["stages"].setdefault(stage, {})
    output_path = output_dir / "outputs" / f"{stage}.npz"
    expected_hash = stage_record.get("output_sha256")
    if output_path.is_file() and expected_hash:
        saved = load_array(output_path)
        if sha256_array(saved) != expected_hash:
            raise RuntimeError(f"Saved output hash changed for {stage}")
        return saved

    job_record = stage_record.get("job", {})
    job_id = job_record.get("job_id") if isinstance(job_record, dict) else None
    if job_id:
        job = client.get_job(job_id)
    else:
        job = client.submit_inference_job(
            model=model,
            device=device,
            inputs={name: [value] for name, value in inputs.items()},
            options=options,
            name=f"lfm-demo-v81-supertonic-chain-{stage}",
            retry=False,
        )
        stage_record.update(
            {
                "status": "submitted",
                "input_arrays": {
                    name: describe_array(value) for name, value in inputs.items()
                },
                "job": job_snapshot(job),
            }
        )
        write_json(state_path, state)

    snapshot = poll_job(
        job,
        stage=stage,
        state=state,
        state_path=state_path,
        poll_seconds=poll_seconds,
        ceiling_seconds=ceiling_seconds,
    )
    if not snapshot["success"]:
        stage_record["status"] = "failed"
        write_json(state_path, state)
        raise RuntimeError(
            f"Inference job {snapshot['job_id']} failed for {stage}: "
            f"{snapshot['message']}"
        )

    output_name, output = extract_single_output(job.download_output_data())
    save_array(output_path, output)
    stage_record.update(
        {
            "status": "output_saved",
            "remote_output_name": output_name,
            "output_file": str(output_path.relative_to(output_dir)),
            "output": describe_array(output),
            "output_sha256": sha256_array(output),
        }
    )
    write_json(state_path, state)
    return output


def load_fixture(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {name: np.array(archive[name]) for name in archive.files}


def main() -> int:
    args = parse_args()
    fixture_path = args.fixture.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "state.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_fixture_hash = manifest["fixture_file"]["sha256"]
    actual_fixture_hash = sha256_file(fixture_path)
    if actual_fixture_hash != expected_fixture_hash:
        raise RuntimeError(
            f"Fixture hash mismatch: {actual_fixture_hash} != {expected_fixture_hash}"
        )
    arrays = load_fixture(fixture_path)
    steps = int(manifest["fixture"]["steps"])
    wav_length = int(manifest["fixture"]["wav_length"])
    sample_rate = int(manifest["fixture"]["sample_rate"])

    compile_jobs = {
        component: getattr(args, f"{component}_compile_job")
        for component in DEFAULT_COMPILE_JOBS
    }
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("fixture_sha256") != actual_fixture_hash:
            raise RuntimeError("Existing state belongs to a different fixture")
        if state.get("compile_jobs") != compile_jobs:
            raise RuntimeError("Existing state uses different compile jobs")
    else:
        state = {
            "schema_version": 1,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "fixture_sha256": actual_fixture_hash,
            "fixture_manifest_sha256": sha256_file(manifest_path),
            "compile_jobs": compile_jobs,
            "target": {
                "device": args.device,
                "os": args.device_os,
                "chipset": "sm8850",
                "htp": "v81",
                "qairt_version": args.qairt_version,
            },
            "steps": steps,
            "stages": {},
            "polls": [],
        }
        write_json(state_path, state)

    plan = (
        ["duration_predictor", "text_encoder"]
        + [f"vector_step_{step}" for step in range(steps)]
        + ["vocoder"]
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "fixture_sha256": actual_fixture_hash,
                    "compile_jobs": compile_jobs,
                    "stages": plan,
                    "existing_stages": state.get("stages", {}),
                },
                indent=2,
            )
        )
        return 0

    client = hub.Client()
    device = hub.Device(args.device, args.device_os, ["framework:qnn"])
    models = {
        component: successful_target(client, job_id, component)
        for component, job_id in compile_jobs.items()
    }
    options = f"--qairt_version {args.qairt_version}"

    duration_remote = stage_output(
        client=client,
        device=device,
        model=models["duration_predictor"],
        inputs={
            "text_ids": arrays["text_ids_remote"],
            "style_dp": arrays["style_dp"],
            "text_mask": arrays["text_mask"],
        },
        stage="duration_predictor",
        output_dir=output_dir,
        state=state,
        state_path=state_path,
        options=options,
        poll_seconds=args.poll_seconds,
        ceiling_seconds=args.ceiling_seconds,
    )
    state["stages"]["duration_predictor"]["comparison"] = compare_arrays(
        duration_remote,
        arrays["duration_raw_local"],
        atol=args.atol,
        rtol=args.rtol,
    )
    write_json(state_path, state)

    text_remote = stage_output(
        client=client,
        device=device,
        model=models["text_encoder"],
        inputs={
            "text_ids": arrays["text_ids_remote"],
            "style_ttl": arrays["style_ttl"],
            "text_mask": arrays["text_mask"],
        },
        stage="text_encoder",
        output_dir=output_dir,
        state=state,
        state_path=state_path,
        options=options,
        poll_seconds=args.poll_seconds,
        ceiling_seconds=args.ceiling_seconds,
    )
    state["stages"]["text_encoder"]["comparison"] = compare_arrays(
        text_remote,
        arrays["text_embedding_local"],
        atol=args.atol,
        rtol=args.rtol,
    )
    write_json(state_path, state)

    latent = arrays["latent_initial"].astype(np.float32, copy=True)
    for step in range(steps):
        stage = f"vector_step_{step}"
        latent_remote = stage_output(
            client=client,
            device=device,
            model=models["vector_estimator"],
            inputs={
                "noisy_latent": latent,
                "text_emb": text_remote,
                "style_ttl": arrays["style_ttl"],
                "latent_mask": arrays["latent_mask"],
                "text_mask": arrays["text_mask"],
                "current_step": np.asarray([step], dtype=np.float32),
                "total_step": np.asarray([steps], dtype=np.float32),
            },
            stage=stage,
            output_dir=output_dir,
            state=state,
            state_path=state_path,
            options=options,
            poll_seconds=args.poll_seconds,
            ceiling_seconds=args.ceiling_seconds,
        )
        state["stages"][stage]["comparison"] = compare_arrays(
            latent_remote,
            arrays[f"latent_step_{step}_raw_local"],
            atol=args.atol,
            rtol=args.rtol,
        )
        latent = latent_remote.astype(np.float32, copy=False) * arrays["latent_mask"]
        state["stages"][stage]["masked_output"] = describe_array(latent)
        write_json(state_path, state)

    waveform_remote = stage_output(
        client=client,
        device=device,
        model=models["vocoder"],
        inputs={"latent": latent},
        stage="vocoder",
        output_dir=output_dir,
        state=state,
        state_path=state_path,
        options=options,
        poll_seconds=args.poll_seconds,
        ceiling_seconds=args.ceiling_seconds,
    )
    trimmed_remote = waveform_remote.reshape(1, -1)[:, :wav_length]
    write_pcm16_wav(
        output_dir / "remote_fixed_trimmed.wav", trimmed_remote, sample_rate
    )
    comparisons = {
        "fixed_local": audio_metrics(
            trimmed_remote, arrays["waveform_trimmed_local"], sample_rate=sample_rate
        ),
        "dynamic_local": audio_metrics(
            trimmed_remote,
            arrays["waveform_dynamic_trimmed_local"],
            sample_rate=sample_rate,
        ),
    }
    state["stages"]["vocoder"]["comparison"] = compare_arrays(
        waveform_remote,
        arrays["waveform_padded_local"],
        atol=args.atol,
        rtol=args.rtol,
    )
    state["result"] = {
        "status": "completed",
        "remote_trimmed": describe_array(trimmed_remote),
        "audio_comparisons": comparisons,
        "remote_wav": "remote_fixed_trimmed.wav",
    }
    state["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json(state_path, state)
    write_json(output_dir / "result.json", state["result"])
    print(json.dumps(state["result"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
