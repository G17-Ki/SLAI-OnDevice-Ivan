#!/usr/bin/env python3
"""Run one semantically structured Piper flow fixture on an existing context.

The generic random sample produced by ``BaseModel.sample_inputs`` is not valid
for Piper's flow boundary: dense random attention makes every projected
``logs_p`` value overflow ``exp`` in Float32.  This runner instead feeds a
fixed phoneme-ID sequence through the already serialized official encoder and
SDP, constructs the official monotonic attention path, evaluates the exact
serialized flow locally, and submits only an inference job against an existing
linked SM8850 context binary.

Run cloud work from detached tmux.  The inference job ID is written to
``state.json`` immediately after AI Hub accepts it.
"""

from __future__ import annotations

import argparse
import atexit
import gc
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any


for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = str(min(max(int(os.environ.get(_name, "2")), 1), 2))

import numpy as np
import qai_hub as hub
import torch
from torch.nn import functional as F

from run_aihub_tts_v81 import (
    ResourceGuard,
    compare_outputs,
    download_job_logs,
    poll_jobs,
    submitted_job_record,
    write_json,
)


MAX_SEQ_LEN = 512
UPSAMPLED_MAX_SEQ_LEN = 1536
DEFAULT_PHONEME_IDS = (
    1,
    25,
    21,
    3,
    19,
    14,
    3,
    28,
    21,
    14,
    16,
    18,
    30,
    18,
    3,
    23,
    18,
    3,
    24,
    18,
    21,
    3,
    31,
    21,
    14,
    3,
    34,
    18,
    26,
    33,
    32,
    27,
    10,
    2,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--sdp", type=Path, required=True)
    parser.add_argument("--flow", type=Path, required=True)
    parser.add_argument("--link-job", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="Snapdragon 8 Elite Gen 5 QRD")
    parser.add_argument("--device-os", default="16")
    parser.add_argument("--qairt-version", default="2.45")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--ceiling-seconds", type=int, default=10_800)
    parser.add_argument("--atol", type=float, default=3e-2)
    parser.add_argument("--rtol", type=float, default=3e-2)
    parser.add_argument("--length-scale", type=float, default=1.0)
    parser.add_argument("--noise-scale-w", type=float, default=0.8)
    parser.add_argument("--noise-scale", type=float, default=0.8)
    parser.add_argument(
        "--phoneme-ids",
        default=",".join(str(value) for value in DEFAULT_PHONEME_IDS),
        help="Comma-separated Piper symbol IDs; defaults to a fixed Italian-like sequence.",
    )
    args = parser.parse_args()
    if args.poll_seconds < 1 or args.ceiling_seconds < 1:
        parser.error("poll and ceiling values must be positive")
    if args.atol < 0 or args.rtol < 0:
        parser.error("tolerances must be non-negative")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_phoneme_ids(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values or len(values) > MAX_SEQ_LEN:
        raise ValueError(f"phoneme ID count must be in [1, {MAX_SEQ_LEN}]")
    if min(values) < 0 or max(values) > 255:
        raise ValueError("Piper phoneme IDs must be in [0, 255]")
    return values


def monotonic_attention(duration: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Equivalent to the official Voice AI ``generate_path`` helper."""

    batch, _, target_len, source_len = mask.shape
    cumulative = torch.cumsum(duration, -1).view(batch * source_len)
    positions = torch.arange(
        target_len, dtype=cumulative.dtype, device=cumulative.device
    )
    path = (positions.unsqueeze(0) < cumulative.unsqueeze(1)).to(mask.dtype)
    path = path.view(batch, source_len, target_len)
    path = path - F.pad(path, [0, 0, 1, 0, 0, 0])[:, :-1]
    return path.unsqueeze(1).transpose(2, 3) * mask


def tensor_array(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def prepare_fixture(
    encoder_path: Path,
    sdp_path: Path,
    flow_path: Path,
    phoneme_ids: list[int],
    length_scale: float,
    noise_scale_w: float,
    noise_scale: float,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    x = torch.zeros((1, MAX_SEQ_LEN), dtype=torch.int32)
    x[0, : len(phoneme_ids)] = torch.tensor(phoneme_ids, dtype=torch.int32)
    x_lengths = torch.tensor([len(phoneme_ids)], dtype=torch.int32)

    encoder = torch.jit.load(str(encoder_path), map_location="cpu").eval()
    with torch.no_grad():
        x_encoded, m_p, logs_p, x_mask = encoder(x, x_lengths)
    del encoder
    gc.collect()

    sdp = torch.jit.load(str(sdp_path), map_location="cpu").eval()
    with torch.no_grad():
        y_lengths, w_ceil = sdp(
            x_encoded,
            x_mask,
            torch.tensor([length_scale], dtype=torch.float32),
            torch.tensor([noise_scale_w], dtype=torch.float32),
        )
    del sdp, x_encoded
    gc.collect()

    y_length = int(y_lengths[0].item())
    if not 1 <= y_length <= UPSAMPLED_MAX_SEQ_LEN:
        raise ValueError(f"SDP produced unsupported y_length={y_length}")
    y_mask = (
        torch.arange(UPSAMPLED_MAX_SEQ_LEN) < y_lengths.unsqueeze(dim=-1)
    ).unsqueeze(1).to(torch.float32)
    attention_mask = x_mask.unsqueeze(dim=2) * y_mask.unsqueeze(dim=-1)
    attention = monotonic_attention(w_ceil, attention_mask).squeeze(1).to(torch.float32)

    flow = torch.jit.load(str(flow_path), map_location="cpu").eval()
    with torch.no_grad():
        golden = flow(
            m_p.to(torch.float32),
            logs_p.to(torch.float32),
            y_mask,
            attention,
            torch.tensor([noise_scale], dtype=torch.float32),
        )
    del flow
    gc.collect()

    tensors = {
        "m_p": m_p.to(torch.float32),
        "logs_p": logs_p.to(torch.float32),
        "y_mask": y_mask,
        "attn_squeezed": attention,
        "noise_scale": torch.tensor([noise_scale], dtype=torch.float32),
    }
    if not all(bool(torch.isfinite(value).all()) for value in tensors.values()):
        raise ValueError("Fixture input contains NaN or infinity")
    if not bool(torch.isfinite(golden).all()):
        raise ValueError("Structured local flow output contains NaN or infinity")
    if not bool(torch.all((y_mask == 0) | (y_mask == 1))):
        raise ValueError("y_mask is not binary")
    if not bool(torch.all((attention == 0) | (attention == 1))):
        raise ValueError("attention path is not binary")
    valid_row_sums = attention[0, :y_length].sum(dim=-1)
    if not bool(torch.all(valid_row_sums == 1)):
        raise ValueError("valid attention rows are not one-hot")
    padded_tail_max_abs = float(golden[:, :, y_length:].abs().max().item())
    if padded_tail_max_abs != 0.0:
        raise ValueError("flow output is non-zero outside y_mask")

    arrays = {name: tensor_array(value) for name, value in tensors.items()}
    diagnostics = {
        "phoneme_id_count": len(phoneme_ids),
        "phoneme_ids": phoneme_ids,
        "y_length": y_length,
        "valid_mask_positions": int(y_mask.sum().item()),
        "attention_nonzero": int(attention.sum().item()),
        "golden_finite_values": int(torch.isfinite(golden).sum().item()),
        "golden_total_values": golden.numel(),
        "golden_min": float(golden.min().item()),
        "golden_max": float(golden.max().item()),
        "padded_tail_max_abs": padded_tail_max_abs,
    }
    return arrays, tensor_array(golden), diagnostics


def main() -> None:
    args = parse_args()
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "state.json"
    if state_path.exists():
        raise FileExistsError(f"Refusing to overwrite persistent state: {state_path}")
    for path in (args.encoder, args.sdp, args.flow):
        if not path.expanduser().is_file():
            raise FileNotFoundError(path)

    guard = ResourceGuard(output_dir / "resource_guard.json")
    guard.start()
    atexit.register(guard.stop)
    guard.check_now(persist=True)

    phoneme_ids = parse_phoneme_ids(args.phoneme_ids)
    inputs, golden, diagnostics = prepare_fixture(
        args.encoder.expanduser().resolve(),
        args.sdp.expanduser().resolve(),
        args.flow.expanduser().resolve(),
        phoneme_ids,
        args.length_scale,
        args.noise_scale_w,
        args.noise_scale,
    )
    fixture_path = output_dir / "fixture.npz"
    np.savez(
        fixture_path,
        **{f"input__{name}": value for name, value in inputs.items()},
        golden__z=golden,
    )

    state: dict[str, Any] = {
        "schema_version": 1,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Piper flow inference-only valid-fixture recheck",
        "target": {
            "device": args.device,
            "os": args.device_os,
            "chipset": "sm8850",
            "htp": "v81",
            "qairt_version": args.qairt_version,
        },
        "reused_link_job": args.link_job,
        "sources": {
            "encoder_sha256": sha256(args.encoder.expanduser().resolve()),
            "sdp_sha256": sha256(args.sdp.expanduser().resolve()),
            "flow_sha256": sha256(args.flow.expanduser().resolve()),
        },
        "fixture": {
            "path": str(fixture_path),
            "sha256": sha256(fixture_path),
            "diagnostics": diagnostics,
        },
        "jobs": {},
        "polls": [],
        "status": "fixture_ready",
    }
    write_json(state_path, state)

    client = hub.Client()
    link_job = client.get_job(args.link_job)
    link_status = link_job.get_status()
    if not link_status.success:
        raise RuntimeError(
            f"Link job {args.link_job} is not successful: {link_status.code}"
        )
    context_model = link_job.get_target_model()
    if context_model is None:
        raise RuntimeError(f"Link job {args.link_job} returned no context model")

    guard.check_now(persist=True)
    inference_job = client.submit_inference_job(
        model=context_model,
        device=hub.Device(args.device, args.device_os, ["framework:qnn"]),
        inputs={name: [value] for name, value in inputs.items()},
        options=f"--qairt_version {args.qairt_version}",
        name="lfm-demo-v81-piper-it-flow-valid-fixture",
        retry=False,
    )
    state["jobs"]["inference"] = submitted_job_record(inference_job)
    state["status"] = "inference_submitted"
    write_json(state_path, state)

    snapshot = poll_jobs(
        {"inference": inference_job},
        state,
        state_path,
        "flow_valid_fixture:inference",
        args.poll_seconds,
        args.ceiling_seconds,
    )["inference"]
    state["jobs"]["inference"] = snapshot
    if not snapshot["success"]:
        state["status"] = (
            "inference_pending" if not snapshot["finished"] else "inference_failed"
        )
        write_json(state_path, state)
        raise SystemExit(2)

    remote = inference_job.download_output_data()
    if not isinstance(remote, dict):
        raise TypeError(f"Inference output is {type(remote).__name__}")
    comparisons, saved = compare_outputs(
        remote, {"z": golden}, context_model, args.atol, args.rtol
    )
    np.savez(output_dir / "remote_outputs.npz", **saved)
    state["result"] = {
        "comparisons": comparisons,
        "all_numerics_pass": bool(comparisons)
        and all(value.get("passed", False) for value in comparisons.values()),
        "remote_outputs_sha256": sha256(output_dir / "remote_outputs.npz"),
    }
    state["job_logs"] = download_job_logs(
        inference_job, output_dir / "job_logs" / "inference"
    )
    state["status"] = (
        "passed" if state["result"]["all_numerics_pass"] else "numerical_warning"
    )
    state["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json(state_path, state)
    print(
        json.dumps(
            {
                "status": state["status"],
                "jobs": state["jobs"],
                "result": state["result"],
            },
            indent=2,
        )
    )
    guard.stop()
    atexit.unregister(guard.stop)


if __name__ == "__main__":
    main()
