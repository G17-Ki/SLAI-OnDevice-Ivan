#!/usr/bin/env python3
"""Replay a real pinned BF16 turn and capture depth-decoder input embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

from path_config import MODEL_REVISION, REPO_ROOT, WORK_ROOT, model_snapshot_for

LIQUID_AUDIO_COMMIT = "19e65845923a7f136442c95137884ec61eb386aa"
DEFAULT_REPO = REPO_ROOT
LEGACY_WORKSPACE = os.environ.get("LFM_WORKSPACE")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(LEGACY_WORKSPACE).expanduser() if LEGACY_WORKSPACE else None,
        help="Legacy checkout-root override; maps mutable state to WORKSPACE/work.",
    )
    parser.add_argument("--work-root", type=Path, default=WORK_ROOT)
    parser.add_argument("--model-snapshot", type=Path)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=80)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if not 32 <= args.samples <= 128:
        raise ValueError("--samples must be between 32 and 128")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source = args.repo / "vendor/liquid-audio/src"
    if args.model_snapshot:
        snapshot = args.model_snapshot.expanduser().resolve()
    elif args.workspace:
        snapshot = (
            args.workspace.expanduser().resolve()
            / "work/cache/huggingface/hub/models--LiquidAI--LFM2.5-Audio-1.5B/snapshots"
            / MODEL_REVISION
        )
    else:
        snapshot = model_snapshot_for(work_root=args.work_root)
    question = args.repo / "vendor/liquid-audio/assets/question.wav"
    golden = args.repo / "reports/colab_bf16/golden_tokens.npz"
    for required in (source, snapshot, question, golden):
        if not required.exists():
            raise FileNotFoundError(required)

    sys.path.insert(0, str(source))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import numpy as np
    import soundfile as sf
    import torch
    from liquid_audio import ChatState, LFM2AudioModel, LFM2AudioProcessor

    with np.load(golden) as archive:
        recorded_text = torch.from_numpy(archive["turn1_text_tokens"]).to(torch.int64)
        recorded_audio = torch.from_numpy(archive["turn1_audio_tokens"]).to(torch.int64)
        recorded_modality = torch.from_numpy(archive["turn1_modality"]).to(torch.int64)

    processor = LFM2AudioProcessor.from_pretrained(snapshot, device="cpu").eval()
    model = LFM2AudioModel.from_pretrained(
        snapshot, device="cpu", dtype=torch.bfloat16
    ).eval()

    wave, rate = sf.read(question, dtype="float32")
    if wave.ndim == 2:
        wave = wave.mean(axis=1)
    chat = ChatState(processor, dtype=torch.bfloat16)
    chat.new_turn("system")
    chat.add_text("Respond with interleaved text and audio.")
    chat.end_turn()
    chat.new_turn("user")
    chat.add_audio(torch.from_numpy(wave).unsqueeze(0), int(rate))
    chat.end_turn()
    chat.new_turn("assistant")

    hidden: list[torch.Tensor] = []
    tokens: list[torch.Tensor] = []
    next_embeddings: list[torch.Tensor] = []
    generation_steps: list[int] = []
    text_index = 0
    audio_index = 0

    def replay_text(logits: torch.Tensor, *, temperature=None, top_k=None):
        nonlocal text_index
        if text_index >= recorded_text.numel():
            raise RuntimeError("Recorded text stream exhausted during replay")
        value = recorded_text[text_index].reshape(1).to(logits.device)
        text_index += 1
        return value

    def replay_audio(embedding: torch.Tensor, *, temperature=None, top_k=None):
        nonlocal audio_index
        if audio_index >= recorded_audio.shape[1]:
            raise RuntimeError("Recorded audio stream exhausted during replay")
        value = recorded_audio[:, audio_index].to(embedding.device)
        audio_index += 1
        if len(hidden) < args.samples:
            hidden.append(embedding.detach().float().cpu().reshape(1, 2048))
            tokens.append(value.detach().cpu().reshape(1, 8))
            offsets = value + model.codebook_offsets
            next_embedding = model.audio_embedding(offsets).sum(0)
            next_embeddings.append(next_embedding.detach().float().cpu().reshape(1, 2048))
        return value

    model._sample_text_token = replay_text
    model._sample_audio_frame = replay_audio

    yielded_modalities: list[int] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for step, value in enumerate(
            model.generate_interleaved(
                **chat,
                max_new_tokens=int(recorded_modality.numel()),
                audio_temperature=0.0,
                audio_top_k=1,
            )
        ):
            modality = 1 if value.numel() == 1 else 3 if value.numel() == 8 else -1
            yielded_modalities.append(modality)
            expected = int(recorded_modality[step])
            if modality != expected:
                raise RuntimeError(
                    f"Replay modality mismatch at step {step}: yielded={modality}, recorded={expected}"
                )
            if modality == 3 and len(generation_steps) < len(hidden):
                generation_steps.append(step)
            if len(hidden) >= args.samples:
                break
    elapsed = time.perf_counter() - started
    if len(hidden) < args.samples:
        raise RuntimeError(f"Captured {len(hidden)} samples, expected {args.samples}")

    npz_path = args.output_dir / "depth_real_embeddings.npz"
    np.savez_compressed(
        npz_path,
        hidden=np.stack([value.numpy() for value in hidden]),
        source_bf16_tokens=np.stack([value.numpy() for value in tokens]),
        source_bf16_next_audio_embedding=np.stack(
            [value.numpy() for value in next_embeddings]
        ),
        generation_step=np.asarray(generation_steps, dtype=np.int64),
        audio_frame_index=np.arange(len(hidden), dtype=np.int64),
    )
    manifest = {
        "status": "passed",
        "capture_method": "CPU replay of the pinned real Colab BF16 turn",
        "sample_count": len(hidden),
        "sample_shape": [1, 2048],
        "sample_inputs": [
            {
                "sample_index": index,
                "turn": 1,
                "generation_step": generation_steps[index],
                "audio_frame_index": index,
            }
            for index in range(len(hidden))
        ],
        "source": {
            "model_revision": MODEL_REVISION,
            "liquid_audio_commit": LIQUID_AUDIO_COMMIT,
            "question_wav_sha256": sha256(question),
            "golden_tokens_sha256": sha256(golden),
            "system_prompt": "Respond with interleaved text and audio.",
        },
        "runtime": {
            "device": "cpu",
            "dtype": "torch.bfloat16",
            "torch": torch.__version__,
            "elapsed_seconds": elapsed,
        },
        "artifact": {
            "file": npz_path.name,
            "sha256": sha256(npz_path),
            "bytes": npz_path.stat().st_size,
        },
        "reference_note": (
            "The NPZ tokens are the recorded BF16 turn tokens used for deterministic replay. "
            "FP32 depth-probe reference tokens are computed separately before quantization acceptance."
        ),
    }
    (args.output_dir / "calibration-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        output = None
        try:
            output = parse_args().output_dir
            output.mkdir(parents=True, exist_ok=True)
            failure = {
                "status": "failed",
                "exception_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
            (output / "capture-failure.json").write_text(
                json.dumps(failure, indent=2) + "\n"
            )
        finally:
            raise
