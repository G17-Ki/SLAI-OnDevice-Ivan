"""Token-level target masks and real, differentiable audio inputs."""

from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from audio_agent.models.dialogue_prompt import encode_segments


def load_audio(path):
    """Read-only source access; normalization never writes a waveform to disk."""
    waveform, sr = sf.read(path, dtype="float32", always_2d=True)
    if not waveform.size or not np.isfinite(waveform).all():
        raise ValueError(f"Empty/non-finite audio: {path}")
    waveform = waveform.mean(axis=1)
    if sr != 16000:
        import librosa
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000, res_type="soxr_hq")
    return np.ascontiguousarray(waveform)


class SharedDecoderCollator:
    def __init__(self, processor, project_root, max_length=2048):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.project_root = Path(project_root)
        self.max_length = max_length
        self.eos = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.pad = self.tokenizer.pad_token_id
        if self.eos is None or self.pad is None:
            raise ValueError("Tokenizer needs EOS and padding token ids")

    def text(self, examples):
        rows, prefixes = [], []
        for example in examples:
            prefix = encode_segments(self.tokenizer, example["segments"])
            target = self.tokenizer.encode(example["target"], add_special_tokens=False) + [self.eos]
            rows.append(prefix + target)
            prefixes.append(len(prefix))
        return self._pad(rows, prefixes)

    def _pad(self, rows, prefixes):
        length = max(map(len, rows))
        if length > self.max_length:
            raise ValueError(f"Sequence length {length} exceeds limit {self.max_length}; target was not truncated")
        ids = torch.full((len(rows), length), self.pad, dtype=torch.long)
        mask = torch.zeros_like(ids)
        labels = torch.full_like(ids, -100)
        for i, (row, prefix) in enumerate(zip(rows, prefixes)):
            ids[i, :len(row)] = torch.tensor(row)
            mask[i, :len(row)] = 1
            labels[i, prefix:len(row)] = ids[i, prefix:len(row)]
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}

    def audio(self, examples):
        waves, prompts = [], []
        for example in examples:
            path = Path(example["audio"])
            path = path if path.is_absolute() else self.project_root / path
            waveform = load_audio(path)
            waves.append(waveform)
            prompts.append(self.processor.apply_chat_template(
                [{"role": "system", "content": ""}, {"role": "user", "content": [{"type": "audio", "audio": ""}]}],
                add_generation_prompt=True, tokenize=False,
            ))
        # Prefix masks come from unpadded ids after real placeholder expansion.
        inputs = self.processor(text=prompts, audio=waves, return_tensors="pt", padding=True)
        rows, prefixes = [], []
        for i, example in enumerate(examples):
            prefix = inputs["input_ids"][i][inputs["attention_mask"][i].bool()].tolist()
            target = self.tokenizer.encode(example["target"], add_special_tokens=False) + [self.eos]
            rows.append(prefix + target)
            prefixes.append(len(prefix))
        result = self._pad(rows, prefixes)
        for key in ("input_features", "feature_attention_mask"):
            result[key] = inputs[key]
        return result


def to_device(batch, device, dtype):
    return {key: value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
            for key, value in batch.items()}
