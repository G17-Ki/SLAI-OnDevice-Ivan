from __future__ import annotations

import json
from dataclasses import asdict

from path_config import (
    MODEL_REVISION,
    MODEL_SNAPSHOT,
    REPO_ROOT,
    VENDOR_ROOT,
    WORK_ROOT,
    configure_hf_home,
)

__all__ = [
    "MODEL_REVISION",
    "MODEL_SNAPSHOT",
    "REPO_ROOT",
    "VENDOR_ROOT",
    "WORK_ROOT",
    "WORKSPACE",
    "EncoderAdapter",
    "load_config",
    "load_encoder_adapter",
    "preprocessor_config",
]

# Backward-compatible name for scripts that imported WORKSPACE. It now means
# the repository checkout, not an ancestor containing outputs/lfm-feasibility.
WORKSPACE = REPO_ROOT

configure_hf_home()

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from torch import nn  # noqa: E402

from liquid_audio.model.conformer.encoder import ConformerEncoder, ConformerEncoderConfig  # noqa: E402
from liquid_audio.model.mlp import MLP  # noqa: E402
from liquid_audio.processor import PreprocessorConfig  # noqa: E402


class EncoderAdapter(nn.Module):
    def __init__(self, encoder_config: ConformerEncoderConfig, hidden_size: int) -> None:
        super().__init__()
        self.conformer = ConformerEncoder(**asdict(encoder_config))
        self.audio_adapter = MLP(
            self.conformer._feat_out,
            hidden_size,
            [hidden_size],
        )

    def forward(self, mel: torch.Tensor, mel_len: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded, encoded_len = self.conformer.forward_for_export(mel, mel_len)
        adapted = self.audio_adapter(encoded.mT)
        return adapted, encoded_len


def load_config() -> dict:
    if not MODEL_SNAPSHOT.is_dir():
        raise FileNotFoundError(
            f"Model snapshot not found at {MODEL_SNAPSHOT}. Run the local model download first."
        )
    return json.loads((MODEL_SNAPSHOT / "config.json").read_text())


def load_encoder_adapter(*, dtype: torch.dtype = torch.float32) -> EncoderAdapter:
    config = load_config()
    encoder_config = ConformerEncoderConfig(**config["encoder"])
    component = EncoderAdapter(encoder_config, int(config["lfm"]["hidden_size"]))

    selected: dict[str, torch.Tensor] = {}
    checkpoint = MODEL_SNAPSHOT / "model.safetensors"
    with safe_open(checkpoint, framework="pt", device="cpu") as reader:
        for key in reader.keys():
            if key.startswith("conformer.") or key.startswith("audio_adapter."):
                selected[key] = reader.get_tensor(key).to(dtype=dtype)

    missing, unexpected = component.load_state_dict(selected, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Selective state load mismatch: missing={missing}, unexpected={unexpected}")
    return component.eval().to(dtype=dtype)


def preprocessor_config(*, deterministic: bool = True) -> PreprocessorConfig:
    config = dict(load_config()["preprocessor"])
    if deterministic:
        config["dither"] = 0.0
    return PreprocessorConfig(**config)
