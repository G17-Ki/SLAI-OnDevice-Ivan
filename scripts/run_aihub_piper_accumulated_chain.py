#!/usr/bin/env python3
"""Run a resume-safe accumulated Piper chain on existing SM8850 contexts.

No graph is compiled, linked, or profiled. The runner prepares a local
reference with the exact serialized official components, then submits encoder,
SDP, flow, and decoder inference jobs sequentially. Remote outputs are fed
forward through host-side monotonic attention and the official decoder overlap
slicing rule. Every accepted job ID is persisted before polling.
"""

from __future__ import annotations

import argparse
import atexit
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
import wave


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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_aihub_piper_flow_valid_fixture import (
    MAX_SEQ_LEN,
    UPSAMPLED_MAX_SEQ_LEN,
    monotonic_attention,
    parse_phoneme_ids,
)
from run_aihub_tts_v81 import (
    ResourceGuard,
    download_job_logs,
    poll_jobs,
    submitted_job_record,
    write_json,
)
from submit_component_aihub import compare_output


MAX_DEC_SEQ_LEN = 40
DEC_SEQ_OVERLAP = 12
UPSAMPLE_FACTOR = 256
SAMPLE_RATE = 22_050


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--sdp", type=Path, required=True)
    parser.add_argument("--flow", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--piper-config", type=Path, required=True)
    parser.add_argument("--encoder-link-job", required=True)
    parser.add_argument("--sdp-link-job", required=True)
    parser.add_argument("--flow-link-job", required=True)
    parser.add_argument("--decoder-link-job", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-text", default="Mi fa piacere che lei sia venuto.")
    parser.add_argument(
        "--phoneme-ids",
        required=True,
        help=(
            "Explicit Piper phoneme-ID sequence. The runner intentionally has no "
            "default because production fixtures must include the phonemizer's "
            "exact BOS/pad/EOS policy."
        ),
    )
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
    args = parser.parse_args()
    if args.poll_seconds < 1 or args.ceiling_seconds < 1:
        parser.error("poll and ceiling values must be positive")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def as_array(value: Any) -> np.ndarray:
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"Expected one inference sample, got {len(value)}")
        value = value[0]
    return np.asarray(value)


def remote_array(remote: dict[str, Any], name: str) -> np.ndarray:
    if name not in remote:
        raise KeyError(f"Remote output {name!r} absent; available: {sorted(remote)}")
    return as_array(remote[name])


def comparison_rows(value: Any) -> list[dict[str, Any]]:
    """Return every leaf comparison row produced by ``compare_output``."""

    if isinstance(value, dict) and "passed" in value:
        return [value]
    if isinstance(value, dict):
        rows: list[dict[str, Any]] = []
        for child in value.values():
            rows.extend(comparison_rows(child))
        return rows
    return []


def comparisons_pass(value: Any) -> bool:
    """Require at least one comparison and require every comparison to pass."""

    rows = comparison_rows(value)
    return bool(rows) and all(row.get("passed") is True for row in rows)


def save_wav(path: Path, samples: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mono = np.asarray(samples, dtype=np.float32).reshape(-1)
    pcm16 = np.round(np.clip(mono, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(SAMPLE_RATE)
        target.writeframes(pcm16.tobytes())


def decoded_phoneme_string(config_path: Path, phoneme_ids: list[int]) -> str:
    config = json.loads(config_path.read_text())
    reverse: dict[int, str] = {}
    for symbol, ids in config["phoneme_id_map"].items():
        for value in ids:
            reverse[int(value)] = symbol
    missing = [value for value in phoneme_ids if value not in reverse]
    if missing:
        raise KeyError(f"Phoneme IDs absent from config reverse map: {missing}")
    return "".join(reverse[value] for value in phoneme_ids)


def make_x(phoneme_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.zeros((1, MAX_SEQ_LEN), dtype=torch.int32)
    x[0, : len(phoneme_ids)] = torch.tensor(phoneme_ids, dtype=torch.int32)
    return x, torch.tensor([len(phoneme_ids)], dtype=torch.int32)


def host_attention(
    x_mask: torch.Tensor, y_lengths: torch.Tensor, w_ceil: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    y_mask = (
        torch.arange(UPSAMPLED_MAX_SEQ_LEN) < y_lengths.unsqueeze(dim=-1)
    ).unsqueeze(1).to(torch.float32)
    attention_mask = x_mask.unsqueeze(dim=2) * y_mask.unsqueeze(dim=-1)
    attention = monotonic_attention(w_ceil, attention_mask).squeeze(1).to(torch.float32)
    return y_mask, attention


def decoder_inputs(z: np.ndarray, y_length: int) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    first = np.zeros(
        (z.shape[0], z.shape[1], MAX_DEC_SEQ_LEN + 2 * DEC_SEQ_OVERLAP),
        dtype=np.float32,
    )
    first[:, :, : MAX_DEC_SEQ_LEN + DEC_SEQ_OVERLAP] = z[
        :, :, : MAX_DEC_SEQ_LEN + DEC_SEQ_OVERLAP
    ]
    result.append(first)
    total = MAX_DEC_SEQ_LEN
    while total < min(
        y_length, z.shape[2] - MAX_DEC_SEQ_LEN - DEC_SEQ_OVERLAP
    ):
        result.append(
            np.asarray(
                z[
                    :,
                    :,
                    total - DEC_SEQ_OVERLAP : total
                    + MAX_DEC_SEQ_LEN
                    + DEC_SEQ_OVERLAP,
                ],
                dtype=np.float32,
            )
        )
        total += MAX_DEC_SEQ_LEN
    return result


def stitch_decoder_outputs(outputs: list[np.ndarray], y_length: int) -> np.ndarray:
    chunks = [np.asarray(outputs[0]).squeeze()[: MAX_DEC_SEQ_LEN * UPSAMPLE_FACTOR]]
    for output in outputs[1:]:
        chunks.append(
            np.asarray(output).squeeze()[
                DEC_SEQ_OVERLAP
                * UPSAMPLE_FACTOR : (MAX_DEC_SEQ_LEN + DEC_SEQ_OVERLAP)
                * UPSAMPLE_FACTOR
            ]
        )
    return np.concatenate(chunks)[: y_length * UPSAMPLE_FACTOR].astype(np.float32)


def prepare_local_chain(
    paths: dict[str, Path], phoneme_ids: list[int], args: argparse.Namespace
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    x, x_lengths = make_x(phoneme_ids)

    encoder = torch.jit.load(str(paths["encoder"]), map_location="cpu").eval()
    with torch.no_grad():
        x_encoded, m_p, logs_p, x_mask = encoder(x, x_lengths)
    del encoder
    gc.collect()

    sdp = torch.jit.load(str(paths["sdp"]), map_location="cpu").eval()
    with torch.no_grad():
        y_lengths, w_ceil = sdp(
            x_encoded,
            x_mask,
            torch.tensor([args.length_scale], dtype=torch.float32),
            torch.tensor([args.noise_scale_w], dtype=torch.float32),
        )
    del sdp
    gc.collect()

    y_length = int(y_lengths[0].item())
    if not 1 <= y_length <= UPSAMPLED_MAX_SEQ_LEN:
        raise ValueError(f"Local SDP produced unsupported y_length={y_length}")
    y_mask, attention = host_attention(x_mask, y_lengths, w_ceil)

    flow = torch.jit.load(str(paths["flow"]), map_location="cpu").eval()
    with torch.no_grad():
        z = flow(
            m_p.float(),
            logs_p.float(),
            y_mask,
            attention,
            torch.tensor([args.noise_scale], dtype=torch.float32),
        )
    del flow
    gc.collect()

    z_np = z.detach().cpu().numpy().astype(np.float32)
    local_decoder_inputs = decoder_inputs(z_np, y_length)
    decoder = torch.jit.load(str(paths["decoder"]), map_location="cpu").eval()
    local_decoder_outputs = []
    with torch.no_grad():
        for value in local_decoder_inputs:
            local_decoder_outputs.append(
                decoder(torch.from_numpy(value)).detach().cpu().numpy()
            )
    del decoder
    gc.collect()
    audio = stitch_decoder_outputs(local_decoder_outputs, y_length)

    arrays = {
        "x": x.numpy(),
        "x_lengths": x_lengths.numpy(),
        "x_encoded": x_encoded.detach().cpu().numpy(),
        "m_p": m_p.detach().cpu().numpy(),
        "logs_p": logs_p.detach().cpu().numpy(),
        "x_mask": x_mask.detach().cpu().numpy(),
        "y_lengths": y_lengths.detach().cpu().numpy(),
        "w_ceil": w_ceil.detach().cpu().numpy(),
        "y_mask": y_mask.detach().cpu().numpy(),
        "attn_squeezed": attention.detach().cpu().numpy(),
        "z": z_np,
        "audio": audio,
    }
    for index, value in enumerate(local_decoder_inputs):
        arrays[f"decoder_input_{index}"] = value
        arrays[f"decoder_output_{index}"] = local_decoder_outputs[index]
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Local accumulated chain contains NaN or infinity")
    metadata = {
        "y_length": y_length,
        "decoder_calls": len(local_decoder_inputs),
        "audio_samples": int(audio.size),
        "audio_seconds": audio.size / SAMPLE_RATE,
        "audio_peak": float(np.max(np.abs(audio))),
        "audio_rms": float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))),
        "padded_flow_tail_max_abs": float(np.max(np.abs(z_np[:, :, y_length:]))),
    }
    return arrays, metadata


def context_from_link(client: hub.Client, job_id: str) -> Any:
    job = client.get_job(job_id)
    status = job.get_status()
    if not status.success:
        raise RuntimeError(f"Link job {job_id} is not successful: {status.code}")
    model = job.get_target_model()
    if model is None:
        raise RuntimeError(f"Link job {job_id} returned no context model")
    return model


def run_stage(
    *,
    name: str,
    model: Any,
    inputs: dict[str, np.ndarray],
    client: hub.Client,
    device: hub.Device,
    args: argparse.Namespace,
    state: dict[str, Any],
    state_path: Path,
    output_dir: Path,
    guard: ResourceGuard,
) -> dict[str, np.ndarray]:
    stage = state.setdefault("stages", {}).setdefault(name, {})
    record = stage.get("job", {})
    if record.get("job_id"):
        job = client.get_job(record["job_id"])
    else:
        guard.check_now(persist=True)
        job = client.submit_inference_job(
            model=model,
            device=device,
            inputs={key: [value] for key, value in inputs.items()},
            options=f"--qairt_version {args.qairt_version}",
            name=f"lfm-demo-v81-piper-it-chain-{name}",
            retry=False,
        )
        stage["job"] = submitted_job_record(job)
        stage["status"] = "submitted"
        write_json(state_path, state)

    snapshot = poll_jobs(
        {name: job},
        state,
        state_path,
        f"piper_chain:{name}",
        args.poll_seconds,
        args.ceiling_seconds,
    )[name]
    stage["job"] = snapshot
    if not snapshot["success"]:
        stage["status"] = "pending" if not snapshot["finished"] else "failed"
        write_json(state_path, state)
        raise SystemExit(2)
    remote = job.download_output_data()
    if not isinstance(remote, dict):
        raise TypeError(f"Stage {name} output is {type(remote).__name__}")
    arrays = {key: as_array(value) for key, value in remote.items()}
    path = output_dir / "remote" / f"{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    stage["status"] = "passed"
    stage["output_sha256"] = sha256(path)
    stage["job_logs"] = download_job_logs(
        job, output_dir / "job_logs" / name
    )
    write_json(state_path, state)
    return arrays


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

    paths = {
        name: getattr(args, name).expanduser().resolve()
        for name in ("encoder", "sdp", "flow", "decoder")
    }
    config_path = args.piper_config.expanduser().resolve()
    for path in (*paths.values(), config_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    link_jobs = {
        "encoder": args.encoder_link_job,
        "sdp": args.sdp_link_job,
        "flow": args.flow_link_job,
        "decoder": args.decoder_link_job,
    }

    guard = ResourceGuard(output_dir / "resource_guard.json")
    guard.start()
    atexit.register(guard.stop)
    guard.check_now(persist=True)
    phoneme_ids = parse_phoneme_ids(args.phoneme_ids)
    token_string = decoded_phoneme_string(config_path, phoneme_ids)
    local_path = output_dir / "local_chain.npz"
    local_wav = output_dir / "local_reference.wav"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("reused_link_jobs") != link_jobs:
            raise ValueError("Resume link-job IDs differ from persisted state")
        if state.get("fixture", {}).get("phoneme_ids") != phoneme_ids:
            raise ValueError("Resume phoneme IDs differ from persisted state")
        if not local_path.is_file() or not local_wav.is_file():
            raise FileNotFoundError("Resume state exists but local artifacts are absent")
        with np.load(local_path) as archive:
            local = {name: np.array(archive[name]) for name in archive.files}
        state.setdefault("resume_attempts", []).append(
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        )
        write_json(state_path, state)
    else:
        local, local_meta = prepare_local_chain(paths, phoneme_ids, args)
        np.savez(local_path, **local)
        save_wav(local_wav, local["audio"])
        state = {
            "schema_version": 1,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "local_reference_ready",
            "target": {
                "device": args.device,
                "os": args.device_os,
                "chipset": "sm8850",
                "htp": "v81",
                "qairt_version": args.qairt_version,
            },
            "fixture": {
                "expected_text": args.expected_text,
                "decoded_phoneme_string": token_string,
                "phoneme_ids": phoneme_ids,
                "piper_config_sha256": sha256(config_path),
                "local_chain_sha256": sha256(local_path),
                "local_wav_sha256": sha256(local_wav),
                **local_meta,
            },
            "sources": {name: sha256(path) for name, path in paths.items()},
            "reused_link_jobs": link_jobs,
            "stages": {},
            "jobs": {},
            "polls": [],
        }
        write_json(state_path, state)

    client = hub.Client()
    device = hub.Device(args.device, args.device_os, ["framework:qnn"])
    contexts = {
        name: context_from_link(client, job_id) for name, job_id in link_jobs.items()
    }

    encoder_remote = run_stage(
        name="encoder",
        model=contexts["encoder"],
        inputs={"x": local["x"], "x_lengths": local["x_lengths"]},
        client=client,
        device=device,
        args=args,
        state=state,
        state_path=state_path,
        output_dir=output_dir,
        guard=guard,
    )
    encoder_values = {
        name: remote_array(encoder_remote, name)
        for name in ("x_encoded", "m_p", "logs_p", "x_mask")
    }

    sdp_remote = run_stage(
        name="sdp",
        model=contexts["sdp"],
        inputs={
            "x_encoded": encoder_values["x_encoded"],
            "x_mask": encoder_values["x_mask"],
            "length_scale": np.array([args.length_scale], dtype=np.float32),
            "noise_scale_w": np.array([args.noise_scale_w], dtype=np.float32),
        },
        client=client,
        device=device,
        args=args,
        state=state,
        state_path=state_path,
        output_dir=output_dir,
        guard=guard,
    )
    remote_y_lengths = remote_array(sdp_remote, "y_lengths")
    remote_w_ceil = remote_array(sdp_remote, "w_ceil")
    remote_y_length = int(round(float(remote_y_lengths.reshape(-1)[0])))
    if not 1 <= remote_y_length <= UPSAMPLED_MAX_SEQ_LEN:
        raise ValueError(f"Remote SDP produced unsupported y_length={remote_y_length}")
    y_lengths_t = torch.from_numpy(remote_y_lengths.astype(np.float32))
    y_mask_t, attention_t = host_attention(
        torch.from_numpy(encoder_values["x_mask"].astype(np.float32)),
        y_lengths_t,
        torch.from_numpy(remote_w_ceil.astype(np.float32)),
    )

    flow_remote = run_stage(
        name="flow",
        model=contexts["flow"],
        inputs={
            "m_p": encoder_values["m_p"].astype(np.float32),
            "logs_p": encoder_values["logs_p"].astype(np.float32),
            "y_mask": y_mask_t.numpy(),
            "attn_squeezed": attention_t.numpy(),
            "noise_scale": np.array([args.noise_scale], dtype=np.float32),
        },
        client=client,
        device=device,
        args=args,
        state=state,
        state_path=state_path,
        output_dir=output_dir,
        guard=guard,
    )
    remote_z = remote_array(flow_remote, "z").astype(np.float32)

    remote_decoder_inputs = decoder_inputs(remote_z, remote_y_length)
    remote_decoder_outputs = []
    for index, value in enumerate(remote_decoder_inputs):
        result = run_stage(
            name=f"decoder_{index}",
            model=contexts["decoder"],
            inputs={"z": value},
            client=client,
            device=device,
            args=args,
            state=state,
            state_path=state_path,
            output_dir=output_dir,
            guard=guard,
        )
        remote_decoder_outputs.append(remote_array(result, "audio"))

    remote_audio = stitch_decoder_outputs(remote_decoder_outputs, remote_y_length)
    remote_wav = output_dir / "remote_chain.wav"
    save_wav(remote_wav, remote_audio)
    np.savez(
        output_dir / "remote_chain.npz",
        y_lengths=remote_y_lengths,
        y_mask=y_mask_t.numpy(),
        attn_squeezed=attention_t.numpy(),
        z=remote_z,
        audio=remote_audio,
    )

    comparisons = {
        "encoder": {
            name: compare_output(value, local[name], args.atol, args.rtol)
            for name, value in encoder_values.items()
        },
        "sdp": {
            "y_lengths": compare_output(
                remote_y_lengths, local["y_lengths"], args.atol, args.rtol
            ),
            "w_ceil": compare_output(
                remote_w_ceil, local["w_ceil"], args.atol, args.rtol
            ),
        },
        "flow": compare_output(remote_z, local["z"], args.atol, args.rtol),
        "audio": compare_output(remote_audio, local["audio"], args.atol, args.rtol),
    }
    all_arrays_finite = all(
        np.isfinite(value).all()
        for value in (
            *encoder_values.values(),
            remote_y_lengths,
            remote_w_ceil,
            y_mask_t.numpy(),
            attention_t.numpy(),
            remote_z,
            *remote_decoder_outputs,
            remote_audio,
        )
    )
    strict_numerical_pass = comparisons_pass(comparisons) and all_arrays_finite
    state["result"] = {
        "execution_complete": True,
        "all_remote_arrays_finite": bool(all_arrays_finite),
        "strict_numerical_pass": bool(strict_numerical_pass),
        "remote_y_length": remote_y_length,
        "decoder_calls": len(remote_decoder_inputs),
        "remote_audio_samples": int(remote_audio.size),
        "remote_audio_seconds": remote_audio.size / SAMPLE_RATE,
        "remote_audio_peak": float(np.max(np.abs(remote_audio))),
        "remote_audio_rms": float(
            np.sqrt(np.mean(remote_audio.astype(np.float64) ** 2))
        ),
        "remote_wav_sha256": sha256(remote_wav),
        "remote_chain_npz_sha256": sha256(output_dir / "remote_chain.npz"),
        "comparisons": comparisons,
    }
    state["status"] = (
        "execution_complete_numerical_pass"
        if strict_numerical_pass
        else "execution_complete_numerical_warning"
    )
    state["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json(state_path, state)
    print(json.dumps({"status": state["status"], "result": state["result"]}, indent=2))
    guard.stop()
    atexit.unregister(guard.stop)


if __name__ == "__main__":
    main()
