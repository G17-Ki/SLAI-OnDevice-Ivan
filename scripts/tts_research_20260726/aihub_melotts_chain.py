#!/usr/bin/env python3
"""Real-Mandarin MeloTTS-ZH chain on the exact SM8850 / HTP v81 QRD.

Runs the local fixture produced by ``scripts/melotts_zh_fixture/prepare_fixture.py``
through the four already-uploaded Qualcomm Voice AI contexts and compares the
remote tensors with the local golden.

Two variants are submitted because they answer different questions:

``fixed_local``
    Every remote stage receives the *local* golden inputs. This isolates the
    numerical error of each graph.
``accumulated``
    Each stage receives the *remote* output of the previous stage, which is
    what a phone would actually experience once error starts propagating.

Resume safety: job IDs are written to ``state.json`` immediately after each
submission, before any polling, so an interrupted local poller never orphans a
server-side job.

Cloud work requires ``--confirm-cloud`` and a detached tmux session.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "melotts_zh_fixture"))
from fixture_common import stitch_decoder_audio  # noqa: E402

TARGET_DEVICE = "Snapdragon 8 Elite Gen 5 QRD"
TARGET_OS = "16"
DEFAULT_MODEL_IDS = {
    "bert_wrapper": "mngxvev5q",
    "encoder": "mqk5d8d7q",
    "flow": "mmxkydy2n",
    "decoder": "mmr8ygo6n",
}
# Encoder outputs, in the order the Voice AI metadata declares them.
ENCODER_OUTPUTS = ("m_p", "logs_p", "w_ceil", "y_lengths", "x_mask", "g")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict:
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
    if ref.shape != cand.shape:
        return {"error": f"shape mismatch {ref.shape} vs {cand.shape}"}
    delta = cand - ref
    denom = max(float(np.sqrt(np.mean(ref * ref))), 1e-12)
    cos_denom = max(float(np.linalg.norm(ref) * np.linalg.norm(cand)), 1e-12)
    return {
        "finite": bool(np.isfinite(cand).all()),
        "nonfinite_count": int(np.count_nonzero(~np.isfinite(cand))),
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "nrmse": float(np.sqrt(np.mean(delta * delta)) / denom),
        "cosine": float(np.dot(ref, cand) / cos_denom),
    }


def load_npz(path: Path) -> dict:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def first(value):
    """AI Hub returns each output as a list of per-input arrays."""
    return value[0] if isinstance(value, list) else value


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--variant", choices=("fixed_local", "accumulated"), required=True)
    ap.add_argument("--device", default=TARGET_DEVICE)
    ap.add_argument("--device-os", default=TARGET_OS)
    ap.add_argument("--qairt-version", default="2.45")
    ap.add_argument("--model-ids", type=Path, help="JSON overriding the pinned hub model ids")
    ap.add_argument("--poll-seconds", type=int, default=30)
    ap.add_argument("--wait-timeout", type=int, default=10800)
    ap.add_argument("--confirm-cloud", action="store_true")
    args = ap.parse_args()

    if not args.confirm_cloud:
        raise SystemExit("cloud work requires --confirm-cloud")
    if not os.environ.get("TMUX"):
        raise SystemExit("cloud work must run inside a detached tmux session")

    import qai_hub as hub

    fixture = args.fixture_dir.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "state.json"

    manifest = json.loads((fixture / "manifest.json").read_text(encoding="utf-8"))
    model_ids = dict(DEFAULT_MODEL_IDS)
    if args.model_ids:
        model_ids.update(json.loads(args.model_ids.read_text(encoding="utf-8")))

    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {
        "schema_version": 1,
        "variant": args.variant,
        "target": {
            "device": args.device,
            "os": args.device_os,
            "chipset": "sm8850",
            "htp": "v81",
            "requested_qairt": args.qairt_version,
        },
        "fixture": {
            "dir_name": fixture.name,
            "text": manifest["text"],
            "normalized_text": manifest["normalized_text"],
            "phone_count": manifest["phone_count"],
            "y_length_frames": manifest["y_length_frames"],
            "decoder_call_count": len(manifest["decoder_schedule"]),
        },
        "model_ids": model_ids,
        "stages": {},
    }
    write_json(state_path, state)

    device = hub.Device(name=args.device, os=args.device_os)
    options = f"--qairt_version {args.qairt_version}"

    bert_inputs = load_npz(fixture / "bert_inputs.npz")
    encoder_inputs = load_npz(fixture / "encoder_inputs.npz")
    flow_inputs = load_npz(fixture / "flow_inputs.npz")
    decoder_inputs = load_npz(fixture / "decoder_inputs.npz")
    local_bert = load_npz(fixture / "bert_local_outputs.npz")
    local_encoder = load_npz(fixture / "encoder_local_outputs.npz")
    local_flow = load_npz(fixture / "flow_local_outputs.npz")
    local_decoder = load_npz(fixture / "decoder_local_outputs.npz")

    remote: dict[str, dict] = {}

    def run_stage(name: str, model_key: str, inputs: dict, output_names) -> dict:
        """Submit (or resume) one inference job and return its downloaded outputs."""
        stage = state["stages"].setdefault(name, {"status": "pending", "job_id": None})
        npz_path = out / f"remote_{name}.npz"

        if stage.get("status") == "downloaded" and npz_path.is_file():
            print(f"[{name}] cached", flush=True)
            return load_npz(npz_path)

        if not stage.get("job_id"):
            # A bare id string is not accepted; the uploaded context has to be
            # resolved to a Model first.
            job = hub.submit_inference_job(
                model=hub.get_model(model_ids[model_key]),
                device=device,
                inputs={k: (v if isinstance(v, list) else [v]) for k, v in inputs.items()},
                options=options,
                name=f"lfm-tts-2026-07-26-melotts-zh-{args.variant}-{name}",
                retry=False,
            )
            # Persist before polling so a killed poller never orphans the job.
            stage["job_id"] = job.job_id
            stage["submitted_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            stage["status"] = "submitted"
            write_json(state_path, state)
            print(f"[{name}] submitted {job.job_id}", flush=True)
        else:
            job = hub.get_job(stage["job_id"])
            print(f"[{name}] resuming {stage['job_id']}", flush=True)

        deadline = time.time() + args.wait_timeout
        while True:
            status = job.get_status()
            if status.finished:
                break
            if time.time() > deadline:
                stage["status"] = "timeout"
                write_json(state_path, state)
                raise SystemExit(f"[{name}] timed out waiting for {stage['job_id']}")
            time.sleep(args.poll_seconds)
            job = hub.get_job(stage["job_id"])

        if not status.success:
            stage["status"] = "failed"
            stage["failure"] = str(status.message)[:500]
            write_json(state_path, state)
            raise SystemExit(f"[{name}] job {stage['job_id']} failed: {status.message}")

        downloaded = job.download_output_data()
        arrays = {}
        for key in output_names:
            if key not in downloaded:
                raise SystemExit(f"[{name}] output {key!r} missing; got {list(downloaded)}")
            entries = downloaded[key]
            arrays[key] = np.asarray(entries[0]) if len(entries) == 1 else np.stack(
                [np.asarray(e) for e in entries]
            )
        np.savez(npz_path, **arrays)
        stage["status"] = "downloaded"
        stage["output_shapes"] = {k: list(v.shape) for k, v in arrays.items()}
        write_json(state_path, state)
        return arrays

    use_remote = args.variant == "accumulated"

    # ---- BERT wrapper ----
    remote["bert"] = run_stage(
        "bert",
        "bert_wrapper",
        {k: bert_inputs[k].astype(np.int32) for k in ("input_ids", "attention_mask", "token_type_ids")},
        ("hidden_states",),
    )

    # ---- Encoder ----
    enc_in = {k: v for k, v in encoder_inputs.items()}
    if use_remote:
        # Rebuild the phone-level ja_bert from the remote hidden states so the
        # accumulated variant really carries the NPU BERT error forward.
        hidden = remote["bert"]["hidden_states"]
        word2ph = manifest.get("word2ph")
        if word2ph:
            phone_level = np.concatenate(
                [np.repeat(hidden[0, i][None, :], r, axis=0) for i, r in enumerate(word2ph)], axis=0
            ).T
            ja = np.zeros_like(encoder_inputs["ja_bert"])
            ja[0, :, : phone_level.shape[1]] = phone_level
            enc_in["ja_bert"] = ja
        else:
            state["stages"].setdefault("encoder", {})["ja_bert_source"] = (
                "local_fixture (manifest has no word2ph; remote BERT could not be re-expanded)"
            )
    remote["encoder"] = run_stage("encoder", "encoder", enc_in, ENCODER_OUTPUTS)

    # ---- Flow ----
    flow_in = {k: v for k, v in flow_inputs.items()}
    if use_remote:
        for key in ("m_p", "logs_p", "g"):
            flow_in[key] = remote["encoder"][key].astype(np.float32)
    remote["flow"] = run_stage("flow", "flow", flow_in, ("z",))

    # ---- Decoder (all overlap calls batched into one job) ----
    z_keys = sorted(k for k in decoder_inputs if k.startswith("z_"))
    if use_remote:
        from fixture_common import decoder_windows

        windows, _schedule = decoder_windows(
            np.asarray(remote["flow"]["z"], dtype=np.float32), int(manifest["y_length_frames"])
        )
        g_used = remote["encoder"]["g"].astype(np.float32)
    else:
        windows = [decoder_inputs[k] for k in z_keys]
        g_used = decoder_inputs["g"]
    dec = run_stage(
        "decoder",
        "decoder",
        {"z": [np.asarray(w, dtype=np.float32) for w in windows],
         "g": [np.asarray(g_used, dtype=np.float32) for _ in windows]},
        ("audio",),
    )

    # Each decoder call returns [1, 1, 32768]; N calls come back stacked. Flatten
    # and re-split rather than guessing at the stacking rank.
    flat = np.asarray(dec["audio"], dtype=np.float32).reshape(-1)
    expected = len(windows) * 64 * 512
    if flat.size != expected:
        raise SystemExit(f"decoder returned {flat.size} samples, expected {expected}")
    chunks = list(flat.reshape(len(windows), 64 * 512))
    remote_audio = stitch_decoder_audio(chunks, int(manifest["y_length_frames"]))

    # ---- Comparison ----
    comparisons = {
        "bert.hidden_states": metrics(local_bert["hidden_states"], remote["bert"]["hidden_states"]),
        "encoder": {
            name: metrics(local_encoder[name], remote["encoder"][name]) for name in ENCODER_OUTPUTS
        },
        "flow.z": metrics(local_flow["z"], remote["flow"]["z"]),
        "decoder.stitched_audio": metrics(local_decoder["stitched_audio"], remote_audio),
    }

    import soundfile as sf

    wav_path = out / f"remote_{args.variant}.wav"
    finite = bool(np.isfinite(remote_audio).all())
    if finite:
        sf.write(wav_path, remote_audio.astype(np.float32), 44_100, subtype="PCM_16")
    else:
        wav_path = None

    result = {
        "schema_version": 1,
        "variant": args.variant,
        "target": state["target"],
        "model_ids": model_ids,
        "job_ids": {k: v.get("job_id") for k, v in state["stages"].items()},
        "fixture": state["fixture"],
        "remote_audio": {
            "samples": int(remote_audio.size),
            "duration_s": float(remote_audio.size / 44_100),
            "finite": finite,
            "peak_abs": float(np.max(np.abs(remote_audio[np.isfinite(remote_audio)]))) if finite else None,
            "wav": wav_path.name if wav_path else None,
        },
        "comparisons": comparisons,
    }
    write_json(out / f"chain_result_{args.variant}.json", result)
    state["status"] = "complete"
    write_json(state_path, state)
    print(json.dumps(result, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
