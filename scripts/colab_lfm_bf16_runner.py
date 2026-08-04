#!/usr/bin/env python3
"""Pinned Colab runner for LFM2.5-Audio BF16 golden inference.

This file is executed on a fresh Colab L4 by run_colab_bf16_golden.sh.  It
installs the pinned official implementation, runs sequential ASR and the
official two-turn interleaved chat flow, and writes only compact outputs.  It
never packages model weights, caches, input audio, or credentials.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CONFIG_PATH = Path("/content/colab_lfm_bf16_config.json")
DEFAULT_OUTPUT_DIR = Path("/content/lfm_bf16_result")
DEFAULT_ZIP_PATH = Path("/content/lfm_bf16_result.zip")
SOURCE_DIR = Path("/content/liquid-audio-pinned")
TEXT_MODALITY = 1
AUDIO_OUT_MODALITY = 3


def run(command: list[str], *, cwd: Path | None = None) -> None:
    """Run a public, non-secret-bearing setup command."""
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def install_dependencies(config: dict[str, Any]) -> None:
    if sys.version_info < (3, 12):
        raise RuntimeError(
            "The pinned Liquid Audio source requires Python >=3.12; "
            f"this runtime has {platform.python_version()}."
        )

    deps = config["dependencies"]
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

    torch_pins = {"torch": deps["torch"], "torchaudio": deps["torchaudio"]}
    if any(installed_version(name) != version for name, version in torch_pins.items()):
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "--index-url",
                deps["pytorch_cuda_index"],
                *[f"{name}=={version}" for name, version in torch_pins.items()],
            ]
        )

    # Colab may retain a torchvision wheel built for its original, newer
    # PyTorch. Transformers probes torchvision even for this audio-only model,
    # and an ABI mismatch then fails at import time. Liquid Audio does not use
    # torchvision, matching the working local environment where it is absent.
    if installed_version("torchvision") is not None:
        run([sys.executable, "-m", "pip", "uninstall", "-y", "torchvision"])

    python_pins = [
        "accelerate",
        "transformers",
        "datasets",
        "einops",
        "librosa",
        "numpy",
        "safetensors",
        "sentencepiece",
        "soundfile",
        "psutil",
        "scipy",
        "scikit-learn",
    ]
    mismatched = [name for name in python_pins if installed_version(name) != deps[name]]
    if mismatched:
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                *[f"{name}=={deps[name]}" for name in python_pins],
            ]
        )


def install_pinned_source(config: dict[str, Any]) -> str:
    source = config["source"]
    expected_commit = source["commit"]
    if SOURCE_DIR.exists():
        actual = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=SOURCE_DIR,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout.strip()
        if actual != expected_commit:
            shutil.rmtree(SOURCE_DIR)

    if not SOURCE_DIR.exists():
        run(["git", "init", str(SOURCE_DIR)])
        run(["git", "remote", "add", "origin", source["repository"]], cwd=SOURCE_DIR)
        run(["git", "fetch", "--depth", "1", "origin", expected_commit], cwd=SOURCE_DIR)
        run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=SOURCE_DIR)

    actual_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=SOURCE_DIR,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    if actual_commit != expected_commit:
        raise RuntimeError(f"Liquid Audio source mismatch: {actual_commit} != {expected_commit}")

    run([sys.executable, "-m", "pip", "install", "--no-deps", "-e", str(SOURCE_DIR)])
    return actual_commit


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def synchronize(torch: Any) -> None:
    torch.cuda.synchronize()


def cuda_memory(torch: Any) -> dict[str, int]:
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "device_free_bytes": int(free),
        "device_total_bytes": int(total),
    }


@dataclass
class GeneratedTurn:
    text_tokens: Any
    audio_tokens: Any
    modality: Any
    decoded_text: str
    generation_seconds: float
    first_text_seconds: float | None
    first_audio_seconds: float | None
    memory: dict[str, int]


def generate_turn(
    *,
    torch: Any,
    model: Any,
    processor: Any,
    chat: Any,
    max_new_tokens: int,
    audio_temperature: float,
    audio_top_k: int,
) -> GeneratedTurn:
    text_parts: list[Any] = []
    audio_parts: list[Any] = []
    modality: list[int] = []
    first_text: float | None = None
    first_audio: float | None = None

    torch.cuda.reset_peak_memory_stats()
    synchronize(torch)
    started = time.perf_counter()
    with torch.inference_mode():
        for token in model.generate_interleaved(
            **chat,
            max_new_tokens=max_new_tokens,
            audio_temperature=audio_temperature,
            audio_top_k=audio_top_k,
        ):
            elapsed = time.perf_counter() - started
            if token.numel() == 1:
                if first_text is None:
                    first_text = elapsed
                text_parts.append(token.detach().cpu())
                modality.append(TEXT_MODALITY)
            elif token.numel() == 8:
                if first_audio is None:
                    first_audio = elapsed
                audio_parts.append(token.detach().cpu())
                modality.append(AUDIO_OUT_MODALITY)
            else:
                raise RuntimeError(f"Unexpected interleaved token shape: {tuple(token.shape)}")
    synchronize(torch)
    generation_seconds = time.perf_counter() - started

    text_tokens = (
        torch.cat([part.reshape(-1) for part in text_parts]).to(torch.int64)
        if text_parts
        else torch.empty((0,), dtype=torch.int64)
    )
    audio_tokens = (
        torch.stack(audio_parts, dim=1).to(torch.int64)
        if audio_parts
        else torch.empty((8, 0), dtype=torch.int64)
    )
    modality_tensor = torch.tensor(modality, dtype=torch.int8)
    decoded_text = processor.text.decode(text_tokens).removesuffix("<|text_end|>")
    return GeneratedTurn(
        text_tokens=text_tokens,
        audio_tokens=audio_tokens,
        modality=modality_tensor,
        decoded_text=decoded_text,
        generation_seconds=generation_seconds,
        first_text_seconds=first_text,
        first_audio_seconds=first_audio,
        memory=cuda_memory(torch),
    )


def valid_audio_codes(torch: Any, audio_tokens: Any) -> Any:
    """Drop the EOAudio frame and any incomplete/invalid suffix."""
    if audio_tokens.shape[1] == 0:
        return audio_tokens
    invalid = (audio_tokens >= 2048).any(dim=0)
    indices = torch.nonzero(invalid, as_tuple=False)
    stop = int(indices[0, 0]) if indices.numel() else int(audio_tokens.shape[1])
    return audio_tokens[:, :stop]


def turn_summary(turn: GeneratedTurn) -> dict[str, Any]:
    total = int(turn.modality.numel())
    return {
        "decoded_text": turn.decoded_text,
        "text_token_count": int(turn.text_tokens.numel()),
        "audio_frame_count_including_eos": int(turn.audio_tokens.shape[1]),
        "generated_step_count": total,
        "generation_seconds": turn.generation_seconds,
        "generated_steps_per_second": total / turn.generation_seconds if turn.generation_seconds else None,
        "first_text_seconds": turn.first_text_seconds,
        "first_audio_seconds": turn.first_audio_seconds,
        "memory": turn.memory,
    }


def append_turn(torch: Any, chat: Any, turn: GeneratedTurn) -> None:
    chat.append(
        text=turn.text_tokens.unsqueeze(0),
        audio_out=turn.audio_tokens,
        modality_flag=turn.modality,
    )
    chat.end_turn()


def decode_turn(
    *,
    torch: Any,
    soundfile: Any,
    processor: Any,
    turn: GeneratedTurn,
    path: Path,
) -> dict[str, Any]:
    codes = valid_audio_codes(torch, turn.audio_tokens)
    if codes.shape[1] == 0:
        return {"status": "no_audio_frames"}
    torch.cuda.reset_peak_memory_stats()
    synchronize(torch)
    started = time.perf_counter()
    with torch.inference_mode():
        waveform = processor.decode(codes.unsqueeze(0).to("cuda"))
    synchronize(torch)
    seconds = time.perf_counter() - started
    soundfile.write(path, waveform.float().cpu()[0].numpy(), 24_000, subtype="PCM_16")
    detokenizer_param = next(processor.audio_detokenizer.parameters())
    return {
        "status": "decoded",
        "valid_audio_frames": int(codes.shape[1]),
        "waveform_samples": int(waveform.shape[-1]),
        "waveform_seconds": float(waveform.shape[-1] / 24_000),
        "decode_seconds": seconds,
        "detokenizer_dtype": str(detokenizer_param.dtype),
        "memory": cuda_memory(torch),
    }


def write_npz(output_dir: Path, arrays: dict[str, Any]) -> None:
    import numpy as np

    np.savez_compressed(
        output_dir / "golden_tokens.npz",
        **{name: tensor.numpy() for name, tensor in arrays.items()},
    )


def package_outputs(output_dir: Path, zip_path: Path) -> None:
    allowed_suffixes = {".json", ".npz", ".txt", ".wav"}
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_dir.iterdir()):
            if path.is_file() and path.suffix in allowed_suffixes:
                archive.write(path, arcname=path.name)


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported config schema")
    if config["model"]["dtype"] != "bfloat16":
        raise ValueError("This golden runner intentionally supports BF16 only")
    for key in ("asr_audio", "interleaved_audio"):
        path = Path(config["inputs"][key])
        if not path.is_file():
            raise FileNotFoundError(f"Missing uploaded input: {path}")


def execute(config: dict[str, Any], output_dir: Path, zip_path: Path) -> None:
    install_dependencies(config)
    source_commit = install_pinned_source(config)

    # pip's editable install writes a path hook that is normally processed at
    # interpreter startup. Colab executes this file inside an already-running
    # notebook kernel, so make the pinned src-layout package importable now.
    pinned_src = SOURCE_DIR / "src"
    if not pinned_src.is_dir():
        raise FileNotFoundError(f"Pinned Liquid Audio src directory missing: {pinned_src}")
    sys.path.insert(0, str(pinned_src))

    import numpy as np
    import psutil
    import soundfile as sf
    import torch
    from liquid_audio import ChatState, LFM2AudioModel, LFM2AudioProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; an L4 GPU runtime is required")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"{torch.cuda.get_device_name(0)} does not report BF16 support")

    seed = int(config["generation"]["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_spec = config["model"]
    input_spec = config["inputs"]
    generation = config["generation"]
    dtype = torch.bfloat16

    torch.cuda.reset_peak_memory_stats()
    synchronize(torch)
    load_started = time.perf_counter()
    processor = LFM2AudioProcessor.from_pretrained(
        model_spec["id"], revision=model_spec["revision"], device="cuda"
    ).eval()
    model = LFM2AudioModel.from_pretrained(
        model_spec["id"], revision=model_spec["revision"], device="cuda", dtype=dtype
    ).eval()
    synchronize(torch)
    load_seconds = time.perf_counter() - load_started
    load_memory = cuda_memory(torch)

    asr_audio_path = Path(input_spec["asr_audio"])
    asr_wave, asr_rate = sf.read(asr_audio_path, dtype="float32")
    if asr_wave.ndim == 2:
        asr_wave = asr_wave.mean(axis=1)
    asr_chat = ChatState(processor, dtype=dtype)
    preprocess_started = time.perf_counter()
    asr_chat.new_turn("system")
    asr_chat.add_text("Perform ASR.")
    asr_chat.end_turn()
    asr_chat.new_turn("user")
    asr_chat.add_audio(torch.from_numpy(asr_wave).unsqueeze(0), int(asr_rate))
    asr_chat.end_turn()
    asr_chat.new_turn("assistant")
    synchronize(torch)
    asr_preprocess_seconds = time.perf_counter() - preprocess_started

    asr_text_parts: list[Any] = []
    asr_non_text_shapes: list[list[int]] = []
    first_asr_token: float | None = None
    torch.cuda.reset_peak_memory_stats()
    synchronize(torch)
    asr_started = time.perf_counter()
    with torch.inference_mode():
        for token in model.generate_sequential(
            **asr_chat, max_new_tokens=int(generation["asr_max_new_tokens"])
        ):
            if first_asr_token is None:
                first_asr_token = time.perf_counter() - asr_started
            if token.numel() == 1:
                asr_text_parts.append(token.detach().cpu())
            else:
                asr_non_text_shapes.append(list(token.shape))
    synchronize(torch)
    asr_generation_seconds = time.perf_counter() - asr_started
    asr_tokens = (
        torch.cat([part.reshape(-1) for part in asr_text_parts]).to(torch.int64)
        if asr_text_parts
        else torch.empty((0,), dtype=torch.int64)
    )
    asr_text = processor.text.decode(asr_tokens)
    asr_memory = cuda_memory(torch)

    chat_audio_path = Path(input_spec["interleaved_audio"])
    chat_wave, chat_rate = sf.read(chat_audio_path, dtype="float32")
    if chat_wave.ndim == 2:
        chat_wave = chat_wave.mean(axis=1)
    chat = ChatState(processor, dtype=dtype)
    chat_preprocess_started = time.perf_counter()
    chat.new_turn("system")
    chat.add_text("Respond with interleaved text and audio.")
    chat.end_turn()
    chat.new_turn("user")
    chat.add_audio(torch.from_numpy(chat_wave).unsqueeze(0), int(chat_rate))
    chat.end_turn()
    chat.new_turn("assistant")
    synchronize(torch)
    chat_preprocess_seconds = time.perf_counter() - chat_preprocess_started

    turn1 = generate_turn(
        torch=torch,
        model=model,
        processor=processor,
        chat=chat,
        max_new_tokens=int(generation["interleaved_max_new_tokens"]),
        audio_temperature=float(generation["audio_temperature"]),
        audio_top_k=int(generation["audio_top_k"]),
    )
    append_turn(torch, chat, turn1)

    followup = input_spec["interleaved_followup_text"]
    chat.new_turn("user")
    chat.add_text(followup)
    chat.end_turn()
    chat.new_turn("assistant")
    turn2 = generate_turn(
        torch=torch,
        model=model,
        processor=processor,
        chat=chat,
        max_new_tokens=int(generation["interleaved_max_new_tokens"]),
        audio_temperature=float(generation["audio_temperature"]),
        audio_top_k=int(generation["audio_top_k"]),
    )

    decode_metrics: dict[str, Any] = {}
    if config["output"]["save_decoded_wav"]:
        decode_metrics["turn1"] = decode_turn(
            torch=torch,
            soundfile=sf,
            processor=processor,
            turn=turn1,
            path=output_dir / "interleaved_turn1.wav",
        )
        decode_metrics["turn2"] = decode_turn(
            torch=torch,
            soundfile=sf,
            processor=processor,
            turn=turn2,
            path=output_dir / "interleaved_turn2.wav",
        )

    write_npz(
        output_dir,
        {
            "asr_text_tokens": asr_tokens,
            "turn1_text_tokens": turn1.text_tokens,
            "turn1_audio_tokens": turn1.audio_tokens,
            "turn1_modality": turn1.modality,
            "turn2_text_tokens": turn2.text_tokens,
            "turn2_audio_tokens": turn2.audio_tokens,
            "turn2_modality": turn2.modality,
        },
    )
    (output_dir / "asr.txt").write_text(asr_text + "\n", encoding="utf-8")
    (output_dir / "interleaved_turn1.txt").write_text(turn1.decoded_text + "\n", encoding="utf-8")
    (output_dir / "interleaved_turn2.txt").write_text(turn2.decoded_text + "\n", encoding="utf-8")

    process = psutil.Process()
    summary = {
        "status": "passed",
        "model": model_spec,
        "source": {**config["source"], "resolved_commit": source_commit},
        "generation": generation,
        "inputs": {
            "asr_audio_sha256": sha256_file(asr_audio_path),
            "asr_audio_seconds": float(len(asr_wave) / asr_rate),
            "interleaved_audio_sha256": sha256_file(chat_audio_path),
            "interleaved_audio_seconds": float(len(chat_wave) / chat_rate),
            "followup_text": followup,
        },
        "load": {"seconds": load_seconds, "memory": load_memory},
        "asr": {
            "decoded_text": asr_text,
            "text_token_count": int(asr_tokens.numel()),
            "non_text_token_shapes": asr_non_text_shapes,
            "preprocess_seconds": asr_preprocess_seconds,
            "generation_seconds": asr_generation_seconds,
            "first_token_seconds": first_asr_token,
            "tokens_per_second": (
                int(asr_tokens.numel()) / asr_generation_seconds if asr_generation_seconds else None
            ),
            "memory": asr_memory,
        },
        "interleaved": {
            "preprocess_turn1_seconds": chat_preprocess_seconds,
            "turn1": turn_summary(turn1),
            "turn2": turn_summary(turn2),
            "decode": decode_metrics,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchaudio": importlib.metadata.version("torchaudio"),
            "transformers": importlib.metadata.version("transformers"),
            "liquid_audio": importlib.metadata.version("liquid-audio"),
            "cuda_runtime": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(0),
            "cuda_capability": list(torch.cuda.get_device_capability(0)),
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            "process_rss_bytes": int(process.memory_info().rss),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    manifest = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    package_outputs(output_dir, zip_path)
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    config: dict[str, Any] = {}
    output_dir = DEFAULT_OUTPUT_DIR
    zip_path = DEFAULT_ZIP_PATH
    try:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        output_dir = Path(config.get("output", {}).get("directory", DEFAULT_OUTPUT_DIR))
        zip_path = Path(config.get("output", {}).get("zip", DEFAULT_ZIP_PATH))
        validate_config(config)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        zip_path.unlink(missing_ok=True)
        execute(config, output_dir, zip_path)
    except BaseException as exc:
        output_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "status": "failed",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "python": sys.version,
            "platform": platform.platform(),
        }
        (output_dir / "failure.json").write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        package_outputs(output_dir, zip_path)
        print(json.dumps(failure, indent=2), file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
