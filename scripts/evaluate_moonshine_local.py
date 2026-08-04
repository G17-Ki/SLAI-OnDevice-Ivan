#!/usr/bin/env python3
"""Run Moonshine Base on the same small clean/noisy ASR diagnostic as Colab."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import datasets
import jiwer
import librosa
import numpy as np
import soundfile as sf
import torch
from datasets import Audio
from transformers import AutoProcessor, MoonshineForConditionalGeneration

from colab_candidate_quality_runner import (
    N_SAMPLES,
    build_conditions,
    load_audio_bytes,
    word_error_rate,
)


MODEL_ID = "UsefulSensors/moonshine-base"


def main() -> None:
    output_dir = Path(sys.argv[1])
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float32
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = MoonshineForConditionalGeneration.from_pretrained(MODEL_ID).to(device).to(dtype).eval()

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
    summaries = []
    details = []
    token_limit_factor = 6.5 / processor.feature_extractor.sampling_rate
    for condition, waves in conditions.items():
        predictions = []
        elapsed_total = 0.0
        audio_total = sum(len(wave) / 16000 for wave in waves)
        for index, wave in enumerate(waves):
            inputs = processor(wave, return_tensors="pt", sampling_rate=16000)
            inputs = inputs.to(device, dtype)
            seq_lens = inputs.attention_mask.sum(dim=-1)
            max_length = max(2, int((seq_lens * token_limit_factor).max().item()))
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(**inputs, max_length=max_length)
            elapsed = time.perf_counter() - started
            prediction = processor.decode(generated[0], skip_special_tokens=True)
            predictions.append(prediction)
            elapsed_total += elapsed
            details.append(
                {
                    "model": MODEL_ID,
                    "condition": condition,
                    "sample_index": index,
                    "reference": references[index],
                    "prediction": prediction,
                    "audio_seconds": len(wave) / 16000,
                    "inference_seconds": elapsed,
                }
            )
        summaries.append(
            {
                "model": MODEL_ID,
                "condition": condition,
                "samples": len(waves),
                "audio_seconds": audio_total,
                "inference_seconds": elapsed_total,
                "real_time_factor": elapsed_total / audio_total,
                "wer_percent": word_error_rate(jiwer, references, predictions),
                "device": device,
            }
        )

    with (output_dir / "asr_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    with (output_dir / "asr_details.jsonl").open("w", encoding="utf-8") as handle:
        for row in details:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "model": MODEL_ID,
                "dataset": "hf-internal-testing/librispeech_asr_dummy clean/validation",
                "sample_count": len(samples),
                "conditions": list(conditions),
                "summary": summaries,
                "environment": {
                    "device": device,
                    "torch": torch.__version__,
                    "transformers": __import__("transformers").__version__,
                    "datasets": datasets.__version__,
                },
                "limitations": [
                    "This is the same 18-utterance controlled smoke diagnostic, not a publication-scale WER benchmark.",
                    "Local runtime is reported only for reproducibility and must not be compared with L4 or Qualcomm latency.",
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
