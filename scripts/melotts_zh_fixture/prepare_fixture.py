#!/usr/bin/env python3
"""Create a real-Mandarin MeloTTS component fixture and local golden chain.

The resource guard starts before Torch, Transformers, QAIHM, or MeloTTS are
imported. The script contains no AI Hub submission code.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile
from typing import Any

from resource_guard import ResourceGuard, preflight_snapshot, preflight_violation, write_json


DEFAULT_TEXT = "巴黎在法国。"
MELOTTS_REVISION = "af5d207a364ea4208c6f589c89f57f88414bdd16"
BERT_REVISION = "7cbf9a625e29989f6b9c6c2fa68234c304f7e38f"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--qaihm-wheel", type=Path, required=True)
    parser.add_argument("--melo-source", type=Path, required=True)
    parser.add_argument("--voiceai-package", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--noise-scale", type=float, default=0.667)
    parser.add_argument("--length-scale", type=float, default=1.0)
    parser.add_argument("--noise-scale-w", type=float, default=0.8)
    parser.add_argument("--sdp-ratio", type=float, default=0.2)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if not args.preflight_only and args.output_dir is None:
        parser.error("--output-dir is required unless --preflight-only is used")
    if not args.text.strip():
        parser.error("--text must not be empty")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sources(args: argparse.Namespace) -> dict[str, Any]:
    lock = json.loads(args.source_lock.read_text())
    wheel = args.qaihm_wheel.expanduser().resolve()
    melo = args.melo_source.expanduser().resolve()
    voiceai = args.voiceai_package.expanduser().resolve()
    expected_wheel = lock["qai_hub_models"]
    if wheel.stat().st_size != expected_wheel["wheel_bytes"] or sha256(wheel) != expected_wheel["wheel_sha256"]:
        raise RuntimeError("qai-hub-models wheel does not match source_lock.json")
    with zipfile.ZipFile(wheel) as archive:
        for member, expected in expected_wheel["source_members"].items():
            actual = hashlib.sha256(archive.read(member)).hexdigest()
            if actual != expected:
                raise RuntimeError(f"QAIHM source member mismatch: {member}")
    revision = subprocess.run(
        ["git", "-C", str(melo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    if revision != lock["melotts"]["commit"]:
        raise RuntimeError(f"MeloTTS revision is {revision}, expected {lock['melotts']['commit']}")
    for member, expected in lock["melotts"]["source_members"].items():
        if sha256(melo / member) != expected:
            raise RuntimeError(f"MeloTTS source member mismatch: {member}")
    for filename, expected in lock["voice_ai_release"]["artifacts"].items():
        path = voiceai / filename
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise RuntimeError(f"Voice AI artifact mismatch: {filename}")
    return {
        "qai_hub_models_wheel": {"bytes": wheel.stat().st_size, "sha256": sha256(wheel)},
        "melotts_commit": revision,
        "voice_ai_artifacts": lock["voice_ai_release"]["artifacts"],
    }


def save_npz(path: Path, values: dict[str, Any]) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{name: np.asarray(value) for name, value in values.items()})


def main() -> int:
    args = parse_args()
    source_state = validate_sources(args)
    snapshot = preflight_snapshot()
    reason = preflight_violation(snapshot)
    if args.preflight_only:
        print(json.dumps({"status": "ready" if reason is None else "blocked", "reason": reason, **snapshot}, indent=2))
        return 0 if reason is None else 86

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    guard = ResourceGuard(output_dir / "resource_guard.json")
    guard.start()
    atexit.register(guard.stop)

    # Heavy imports are intentionally below the fail-closed gate.
    import numpy as np
    import soundfile as sf
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)

    melo_source = args.melo_source.expanduser().resolve()
    if str(melo_source) not in sys.path:
        sys.path.insert(0, str(melo_source))

    import melo
    from melo import commons
    from melo.api import TTS
    from melo.text import cleaned_text_to_sequence
    from melo.text.cleaner import clean_text
    from qai_hub_models.datasets.common_voice import TTSLanguage
    from qai_hub_models.models._shared.common import replace_module_recursively
    from qai_hub_models.models._shared.melotts.meloTTS_encoder import FFNMod
    from qai_hub_models.models._shared.voiceai_tts.language import BERT_MODEL_IDS
    from qai_hub_models.models.melotts_zh.model import BertWrapper_ZH, Decoder_ZH, Encoder_ZH, Flow_ZH

    qhm_version = importlib.metadata.version("qai-hub-models")
    if qhm_version != "0.58.0":
        raise RuntimeError(f"qai-hub-models {qhm_version} is installed; expected 0.58.0")

    cache_dir = output_dir.parent / "model_cache"
    config_path = Path(
        hf_hub_download(
            "myshell-ai/MeloTTS-Chinese",
            "config.json",
            revision=MELOTTS_REVISION,
            cache_dir=cache_dir,
        )
    )
    checkpoint_path = Path(
        hf_hub_download(
            "myshell-ai/MeloTTS-Chinese",
            "checkpoint.pth",
            revision=MELOTTS_REVISION,
            cache_dir=cache_dir,
        )
    )
    tts = TTS("ZH", device="cpu", config_path=str(config_path), ckpt_path=str(checkpoint_path))
    replace_module_recursively(tts.model, melo.attentions.FFN, FFNMod)

    norm_text, phone_symbols, tones_raw, word2ph = clean_text(args.text, tts.language)
    phones, tones, languages = cleaned_text_to_sequence(
        phone_symbols, tones_raw, tts.language, tts.symbol_to_id
    )
    if tts.hps.data.add_blank:
        phones = commons.intersperse(phones, 0)
        tones = commons.intersperse(tones, 0)
        languages = commons.intersperse(languages, 0)
        word2ph = [value * 2 for value in word2ph]
        word2ph[0] += 1

    bert_id = BERT_MODEL_IDS[TTSLanguage.CHINESE]
    tokenizer = AutoTokenizer.from_pretrained(bert_id, revision=BERT_REVISION, cache_dir=cache_dir)
    tokenized = tokenizer(
        norm_text,
        padding="max_length",
        truncation=True,
        max_length=200,
        return_tensors="pt",
    )
    bert_model = AutoModelForMaskedLM.from_pretrained(
        bert_id, revision=BERT_REVISION, cache_dir=cache_dir
    ).cpu().eval()
    bert_wrapper = BertWrapper_ZH(bert_model).eval()
    with torch.inference_mode():
        hidden_states = bert_wrapper(
            tokenized["input_ids"].to(torch.int32),
            tokenized["attention_mask"].to(torch.int32),
            tokenized["token_type_ids"].to(torch.int32),
        )
    valid_tokens = int(tokenized["attention_mask"].sum().item())
    if len(word2ph) != valid_tokens:
        raise RuntimeError(f"word2ph has {len(word2ph)} entries but BERT has {valid_tokens} valid tokens")
    phone_level = torch.cat(
        [hidden_states[0, index].repeat(repeat, 1) for index, repeat in enumerate(word2ph)],
        dim=0,
    ).transpose(0, 1)
    if phone_level.shape[1] != len(phones):
        raise RuntimeError("phone-level BERT expansion does not match phone count")

    max_seq_len = 512
    def pad_1d(values: list[int]) -> torch.Tensor:
        result = torch.zeros(max_seq_len, dtype=torch.int32)
        result[: len(values)] = torch.tensor(values, dtype=torch.int32)
        return result

    x = pad_1d(phones).unsqueeze(0)
    tone = pad_1d(tones).unsqueeze(0)
    language = pad_1d(languages).unsqueeze(0)
    x_lengths = torch.tensor([len(phones)], dtype=torch.int32)
    sid = torch.tensor([next(iter(tts.hps.data.spk2id.values()))], dtype=torch.int32)
    bert = torch.zeros((1, 1024, max_seq_len), dtype=torch.float32)
    ja_bert = torch.zeros((1, 768, max_seq_len), dtype=torch.float32)
    ja_bert[0, :, : len(phones)] = phone_level
    sdp_ratio = torch.tensor([args.sdp_ratio], dtype=torch.float32)
    length_scale = torch.tensor([args.length_scale], dtype=torch.float32)
    noise_scale_w = torch.tensor([args.noise_scale_w], dtype=torch.float32)
    noise_scale = torch.tensor([args.noise_scale], dtype=torch.float32)

    encoder = Encoder_ZH(tts, speed_adjustment=0.85).eval()
    flow = Flow_ZH(tts).eval()
    decoder = Decoder_ZH(tts).eval()
    with torch.inference_mode():
        y_lengths, x_mask, m_p, logs_p, g, w_ceil = encoder(
            x, x_lengths, tone, sid, language, bert, ja_bert,
            sdp_ratio, length_scale, noise_scale_w,
        )
    y_length = int(round(float(y_lengths[0].item())))
    y_mask = (
        torch.arange(1536, dtype=torch.float32) < y_lengths.unsqueeze(-1)
    ).unsqueeze(1).to(torch.float32)
    attention_mask = x_mask.unsqueeze(2) * y_mask.unsqueeze(-1)

    from qai_hub_models.models._shared.voiceai_tts.app_utils import generate_path
    attention = generate_path(w_ceil, attention_mask).squeeze(1).to(torch.float32)
    with torch.inference_mode():
        z = flow(m_p.float(), logs_p.float(), y_mask, g, attention, noise_scale)

    from fixture_common import decoder_windows, stitch_decoder_audio
    windows, schedule = decoder_windows(z.detach().cpu().numpy(), y_length)
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for window in windows:
            chunks.append(decoder(torch.from_numpy(window), g).detach().cpu().numpy())
    audio = stitch_decoder_audio(chunks, y_length)

    save_npz(output_dir / "bert_inputs.npz", {name: value.cpu().numpy() for name, value in tokenized.items()})
    save_npz(output_dir / "bert_local_outputs.npz", {"hidden_states": hidden_states.cpu().numpy()})
    save_npz(output_dir / "encoder_inputs.npz", {
        "x": x, "x_lengths": x_lengths, "tone": tone, "sid": sid,
        "language": language, "bert": bert, "ja_bert": ja_bert,
        "sdp_ratio": sdp_ratio, "length_scale": length_scale,
        "noise_scale_w": noise_scale_w,
    })
    save_npz(output_dir / "encoder_local_outputs.npz", {
        "y_lengths": y_lengths, "x_mask": x_mask, "m_p": m_p,
        "logs_p": logs_p, "g": g, "w_ceil": w_ceil,
    })
    save_npz(output_dir / "flow_inputs.npz", {
        "m_p": m_p, "logs_p": logs_p, "y_mask": y_mask, "g": g,
        "attn_squeezed": attention, "noise_scale": noise_scale,
    })
    save_npz(output_dir / "flow_local_outputs.npz", {"z": z})
    save_npz(output_dir / "decoder_inputs.npz", {
        **{f"z_{index:03d}": window for index, window in enumerate(windows)},
        "g": g,
    })
    save_npz(output_dir / "decoder_local_outputs.npz", {
        **{f"audio_{index:03d}": chunk for index, chunk in enumerate(chunks)},
        "stitched_audio": audio,
    })
    sf.write(output_dir / "local_golden.wav", audio, 44_100, subtype="FLOAT")

    artifacts = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name not in {"manifest.json", "resource_guard.json"}:
            artifacts[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest = {
        "schema_version": 1,
        "status": "completed_local_golden",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "text": args.text,
        "normalized_text": norm_text,
        "phone_count": len(phones),
        "bert_valid_tokens": valid_tokens,
        # Persisted so a later remote-BERT run can re-expand hidden states to
        # phone level without re-importing MeloTTS.
        "word2ph": [int(value) for value in word2ph],
        "y_length_frames": y_length,
        "audio_samples": int(audio.size),
        "sample_rate_hz": 44_100,
        "decoder_schedule": schedule,
        "parameters": {
            "noise_scale": args.noise_scale,
            "length_scale": args.length_scale,
            "noise_scale_w": args.noise_scale_w,
            "sdp_ratio": args.sdp_ratio,
        },
        "sources": {
            **source_state,
            "melotts_config": {"sha256": sha256(config_path)},
            "melotts_checkpoint": {"sha256": sha256(checkpoint_path)},
            "bert_revision": BERT_REVISION,
        },
        "artifacts": artifacts,
    }
    write_json(output_dir / "manifest.json", manifest)
    guard.stop()
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
