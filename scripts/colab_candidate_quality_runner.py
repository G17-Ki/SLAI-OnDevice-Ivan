#!/usr/bin/env python3
"""Compare LFM2.5-Audio and Whisper ASR quality on matched speech conditions."""

from __future__ import annotations

import csv
import gc
import io
import json
import math
import re
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any


OUTPUT_DIR = Path("/content/candidate_quality_result")
ZIP_PATH = Path("/content/candidate_quality_result.zip")
CONFIG_PATH = Path("/content/colab_lfm_bf16_config.json")
N_SAMPLES = 18
SEED = 20260715
WHISPER_MODELS = (
    "openai/whisper-tiny",
    "openai/whisper-base",
    "openai/whisper-small",
)


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<\|[^>]+\|>", " ", text)
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return " ".join(text.split())


def load_audio_bytes(soundfile: Any, librosa: Any, value: dict[str, Any]) -> Any:
    source: Any
    if value.get("bytes") is not None:
        source = io.BytesIO(value["bytes"])
    elif value.get("path"):
        source = value["path"]
    else:
        raise ValueError("Dataset audio sample has neither bytes nor path")
    wave, rate = soundfile.read(source, dtype="float32")
    if wave.ndim == 2:
        wave = wave.mean(axis=1)
    if rate != 16000:
        wave = librosa.resample(wave, orig_sr=rate, target_sr=16000)
    return wave.astype("float32", copy=False)


def peak_safe(wave: Any) -> Any:
    import numpy as np

    peak = float(np.max(np.abs(wave))) if wave.size else 0.0
    return wave * (0.98 / peak) if peak > 0.98 else wave


def mix_at_snr(signal: Any, interference: Any, snr_db: float) -> Any:
    import numpy as np

    if interference.size < signal.size:
        repeats = int(math.ceil(signal.size / max(1, interference.size)))
        interference = np.tile(interference, repeats)
    interference = interference[: signal.size]
    signal_rms = float(np.sqrt(np.mean(signal * signal) + 1e-12))
    noise_rms = float(np.sqrt(np.mean(interference * interference) + 1e-12))
    scale = signal_rms / (10 ** (snr_db / 20) * noise_rms)
    return peak_safe(signal + interference * scale).astype("float32")


def build_conditions(samples: list[dict[str, Any]]) -> dict[str, list[Any]]:
    import numpy as np

    rng = np.random.default_rng(SEED)
    clean = [sample["wave"] for sample in samples]
    gaussian: list[Any] = []
    competing: list[Any] = []
    for index, wave in enumerate(clean):
        gaussian.append(mix_at_snr(wave, rng.standard_normal(wave.shape).astype("float32"), 10.0))
        competing.append(mix_at_snr(wave, clean[(index + 7) % len(clean)], 5.0))
    return {
        "clean": clean,
        "gaussian_10db": gaussian,
        "competing_speech_5db": competing,
    }


def word_error_rate(jiwer: Any, references: list[str], predictions: list[str]) -> float:
    return float(jiwer.wer([normalize_text(x) for x in references], [normalize_text(x) for x in predictions]) * 100)


def evaluate_lfm(
    torch: Any,
    jiwer: Any,
    processor: Any,
    model: Any,
    references: list[str],
    conditions: dict[str, list[Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from liquid_audio import ChatState

    summaries: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for condition, waves in conditions.items():
        predictions: list[str] = []
        elapsed_total = 0.0
        first_total = 0.0
        audio_total = 0.0
        torch.cuda.reset_peak_memory_stats()
        for index, wave in enumerate(waves):
            chat = ChatState(processor, dtype=torch.bfloat16)
            chat.new_turn("system")
            chat.add_text("Perform ASR.")
            chat.end_turn()
            chat.new_turn("user")
            chat.add_audio(torch.from_numpy(wave).unsqueeze(0), 16000)
            chat.end_turn()
            chat.new_turn("assistant")
            pieces = []
            torch.cuda.synchronize()
            started = time.perf_counter()
            first = None
            with torch.inference_mode():
                for token in model.generate_sequential(**chat, max_new_tokens=160):
                    if token.numel() == 1:
                        if first is None:
                            first = time.perf_counter() - started
                        pieces.append(token.detach().cpu().reshape(-1))
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            token_tensor = torch.cat(pieces) if pieces else torch.empty((0,), dtype=torch.int64)
            prediction = processor.text.decode(token_tensor)
            predictions.append(prediction)
            elapsed_total += elapsed
            first_total += first or elapsed
            duration = float(len(wave) / 16000)
            audio_total += duration
            details.append(
                {
                    "model": "LiquidAI/LFM2.5-Audio-1.5B",
                    "condition": condition,
                    "sample_index": index,
                    "reference": references[index],
                    "prediction": prediction,
                    "audio_seconds": duration,
                    "inference_seconds": elapsed,
                    "first_token_seconds": first,
                }
            )
        summaries.append(
            {
                "model": "LiquidAI/LFM2.5-Audio-1.5B",
                "condition": condition,
                "samples": len(waves),
                "audio_seconds": audio_total,
                "inference_seconds": elapsed_total,
                "real_time_factor": elapsed_total / audio_total,
                "mean_first_token_ms": first_total / len(waves) * 1000,
                "wer_percent": word_error_rate(jiwer, references, predictions),
                "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            }
        )
    return summaries, details


def evaluate_whisper(
    torch: Any,
    jiwer: Any,
    model_id: str,
    references: list[str],
    conditions: dict[str, list[Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Any, Any]:
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to("cuda").eval()
    summaries: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for condition, waves in conditions.items():
        predictions: list[str] = []
        elapsed_total = 0.0
        audio_total = sum(len(wave) / 16000 for wave in waves)
        torch.cuda.reset_peak_memory_stats()
        for start in range(0, len(waves), 6):
            batch = waves[start : start + 6]
            inputs = processor(batch, sampling_rate=16000, return_tensors="pt", padding=True)
            features = inputs.input_features.to(device="cuda", dtype=torch.float16)
            torch.cuda.synchronize()
            began = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    features,
                    language="en",
                    task="transcribe",
                    max_new_tokens=160,
                )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - began
            texts = processor.batch_decode(generated, skip_special_tokens=True)
            elapsed_total += elapsed
            predictions.extend(texts)
            for offset, prediction in enumerate(texts):
                index = start + offset
                details.append(
                    {
                        "model": model_id,
                        "condition": condition,
                        "sample_index": index,
                        "reference": references[index],
                        "prediction": prediction,
                        "audio_seconds": len(waves[index]) / 16000,
                        "inference_seconds": None,
                        "first_token_seconds": None,
                    }
                )
        summaries.append(
            {
                "model": model_id,
                "condition": condition,
                "samples": len(waves),
                "audio_seconds": audio_total,
                "inference_seconds": elapsed_total,
                "real_time_factor": elapsed_total / audio_total,
                "mean_first_token_ms": None,
                "wer_percent": word_error_rate(jiwer, references, predictions),
                "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            }
        )
    return summaries, details, processor, model


def transcribe_files(torch: Any, jiwer: Any, processor: Any, model: Any) -> list[dict[str, Any]]:
    import librosa
    import soundfile as sf

    pairs = [
        (Path("/content/lfm_turn1.wav"), Path("/content/lfm_turn1.txt")),
        (Path("/content/lfm_turn2.wav"), Path("/content/lfm_turn2.txt")),
    ]
    output = []
    for wav_path, text_path in pairs:
        wave, rate = sf.read(wav_path, dtype="float32")
        source_seconds = len(wave) / rate
        if rate != 16000:
            wave = librosa.resample(wave, orig_sr=rate, target_sr=16000)
            rate = 16000
        inputs = processor(wave, sampling_rate=rate, return_tensors="pt")
        features = inputs.input_features.to(device="cuda", dtype=torch.float16)
        with torch.inference_mode():
            tokens = model.generate(features, language="en", task="transcribe", max_new_tokens=192)
        prediction = processor.batch_decode(tokens, skip_special_tokens=True)[0]
        reference = text_path.read_text(encoding="utf-8").strip()
        output.append(
            {
                "file": wav_path.name,
                "reference": reference,
                "whisper_small_transcript": prediction,
                "wer_percent": word_error_rate(jiwer, [reference], [prediction]),
                "audio_seconds": source_seconds,
            }
        )
    return output


def package() -> None:
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUTPUT_DIR.iterdir()):
            if path.is_file() and path.suffix in {".json", ".jsonl", ".csv", ".txt"}:
                archive.write(path, path.name)


def execute() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    sys.path.insert(0, "/content")
    import colab_lfm_bf16_runner as base

    base.install_dependencies(config)
    run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "jiwer==4.0.0"])
    source_commit = base.install_pinned_source(config)
    sys.path.insert(0, "/content/liquid-audio-pinned/src")

    import datasets
    import jiwer
    import librosa
    import numpy as np
    import soundfile as sf
    import torch
    from datasets import Audio
    from liquid_audio import LFM2AudioModel, LFM2AudioProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = datasets.load_dataset(
        "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
    ).cast_column("audio", Audio(decode=False))
    samples = []
    for row in dataset.select(range(min(N_SAMPLES, len(dataset)))):
        samples.append(
            {
                "wave": load_audio_bytes(sf, librosa, row["audio"]),
                "text": row["text"],
            }
        )
    references = [sample["text"] for sample in samples]
    conditions = build_conditions(samples)

    processor = LFM2AudioProcessor.from_pretrained(
        config["model"]["id"], revision=config["model"]["revision"], device="cuda"
    ).eval()
    model = LFM2AudioModel.from_pretrained(
        config["model"]["id"],
        revision=config["model"]["revision"],
        device="cuda",
        dtype=torch.bfloat16,
    ).eval()
    lfm_summary, lfm_details = evaluate_lfm(
        torch, jiwer, processor, model, references, conditions
    )
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()

    summary_rows = list(lfm_summary)
    detail_rows = list(lfm_details)
    whisper_small_processor = None
    whisper_small_model = None
    for model_id in WHISPER_MODELS:
        model_summary, model_details, wp, wm = evaluate_whisper(
            torch, jiwer, model_id, references, conditions
        )
        summary_rows.extend(model_summary)
        detail_rows.extend(model_details)
        if model_id == "openai/whisper-small":
            whisper_small_processor, whisper_small_model = wp, wm
        else:
            del wp, wm
            gc.collect()
            torch.cuda.empty_cache()

    assert whisper_small_processor is not None and whisper_small_model is not None
    intelligibility = transcribe_files(
        torch, jiwer, whisper_small_processor, whisper_small_model
    )

    fieldnames = list(summary_rows[0])
    with (OUTPUT_DIR / "asr_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    with (OUTPUT_DIR / "asr_details.jsonl").open("w", encoding="utf-8") as handle:
        for row in detail_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    payload = {
        "status": "passed",
        "dataset": "hf-internal-testing/librispeech_asr_dummy clean/validation",
        "sample_count": len(samples),
        "conditions": {
            "clean": "unaltered",
            "gaussian_10db": "deterministic white noise at 10 dB SNR",
            "competing_speech_5db": "another utterance mixed at 5 dB target-to-interferer ratio",
        },
        "normalization": "lowercase; strip control tokens and punctuation; collapse whitespace",
        "models": [config["model"]["id"], *WHISPER_MODELS],
        "lfm_revision": config["model"]["revision"],
        "liquid_audio_commit": source_commit,
        "summary": summary_rows,
        "lfm_output_intelligibility": intelligibility,
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "datasets": datasets.__version__,
            "cuda_device": torch.cuda.get_device_name(0),
        },
        "limitations": [
            "This 18-utterance dummy LibriSpeech subset is a controlled smoke benchmark, not a publication-scale ASR evaluation.",
            "Synthetic noise and two-speaker mixtures do not replace real glasses microphone recordings.",
            "L4 throughput is a quality-run diagnostic and is not an AR1 latency estimate.",
        ],
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    package()


def main() -> None:
    try:
        execute()
    except Exception as error:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "failure.json").write_text(
            json.dumps(
                {
                    "status": "failed",
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        package()
        raise


if __name__ == "__main__":
    main()
