#!/usr/bin/env python3
"""Run Lane G's pinned MOSS-TTS-Nano synthesis suites on local CPU.

The runner deliberately bypasses the mutable CLI defaults and WeTextProcessing.
It loads the two pinned local Hugging Face snapshots, encodes each fixed upstream
voice prompt once, and then reuses those prompt codes for every case.  Outputs
are written only after both raw-tensor and PCM16 waveform gates pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import AutoModel, AutoModelForCausalLM


OUT_D = Path(__file__).resolve().parent
ARTIFACTS = OUT_D / "artifacts" / "moss"
TTS_DIR = ARTIFACTS / "tts"
CODEC_DIR = ARTIFACTS / "audio_tokenizer"
PROMPT_DIR = ARTIFACTS / "prompts"
AUDIO_DIR = ARTIFACTS / "audio"
RESULT_DIR = ARTIFACTS / "results"
CHECKPOINT_PATH = ARTIFACTS / "synthesis_records.jsonl"
CORPUS_PATH = OUT_D / "D1_corpus.json"
D5_PROMPTS_PATH = OUT_D / "D5_prompts.json"

TTS_REPO = "OpenMOSS-Team/MOSS-TTS-Nano-100M"
TTS_REVISION = "44502f80dbf9743528fa921cc544d662c685ebec"
TTS_WEIGHT_SHA256 = "24003f2f11ac8a2cbf70514db2d8f1c02fb451aa6b3c0bffc9da09f31cd7caa5"
TTS_PARAMETERS = 117_311_232
CODEC_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano"
CODEC_REVISION = "6aa02b01e445cc585582cf0ba480bc3ea6c8dd68"
CODEC_WEIGHT_SHA256 = "34d9880d805eecb21bde975202b1c256dbd0eb98c8680b9d3aeffd2bc6ac2f67"
CODEC_PARAMETERS = 21_969_664
SOURCE_REVISION = "cc7bdf19c7639c0870dab22045a33b442760f6be"
GENERATION_PROTOCOL_VERSION = 1

THREADS = 4
SAMPLE_RATE = 48_000
CHANNELS = 2
NQ = 16
RMS_FLOOR = 0.0031622776601683794
CLIP_THRESHOLD = 0.9885530946569389
MAX_CLIP_RUN_S = 0.005
MIN_DURATION_S = 0.15
MAX_DURATION_S = 30.0

SAMPLING = {
    "do_sample": True,
    "text_temperature": 1.0,
    "text_top_p": 1.0,
    "text_top_k": 50,
    "audio_temperature": 0.8,
    "audio_top_p": 0.95,
    "audio_top_k": 25,
    "audio_repetition_penalty": 1.2,
    "use_kv_cache": True,
}

G4_ITEM_IDS = (
    "fleurs_02",
    "fleurs_03",
    "fleurs_04",
    "fleurs_08",
    "fleurs_11",
    "fleurs_17",
    "fleurs_23",
    "fleurs_32",
    "fleurs_34",
    "fleurs_40",
)

MIXED_PAIRS = (
    ("wifi", "WiFi密码", "密码"),
    ("5g", "5G网络", "网络"),
    ("digits", "订单号是5072", "订单号是"),
    ("cpu", "CPU温度达到82度", "温度达到82度"),
    ("api_json", "API返回24条JSON记录", "返回24条记录"),
    ("vpn", "VPN每10分钟断开", "每10分钟断开"),
)

D3_TEXTS = {
    "en": ("A", "OK", "Yes", "Hello", "Hello there", "Hello there friend"),
    "es": ("A", "Sí", "Sol", "Hola", "Hola mundo", "Hola mundo ahora"),
    "fr": ("À", "Ça", "Oui", "Bonjour", "Bonjour Marie", "Bonjour mon ami"),
    "ru": ("Я", "Да", "Нет", "Привет", "Привет мир", "Привет мой друг"),
    "zh": ("好", "好的", "没问题", "你", "你好", "你好呀"),
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(suite: str, case_id: str, repeat: int = 1) -> int:
    payload = f"lane-g-moss-v1:{suite}:{case_id}:{repeat}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def case_config_payload(case: dict, case_timeout_s: int) -> dict:
    return {
        "generation_protocol_version": GENERATION_PROTOCOL_VERSION,
        "tts_revision": TTS_REVISION,
        "tts_weight_sha256": TTS_WEIGHT_SHA256,
        "codec_revision": CODEC_REVISION,
        "codec_weight_sha256": CODEC_WEIGHT_SHA256,
        "source_revision": SOURCE_REVISION,
        "suite": case["suite"],
        "case_id": case["case_id"],
        "repeat": int(case["repeat"]),
        "seed": int(case["seed"]),
        "language": case["language"],
        "raw_text": case["text"],
        "prompt_name": case["prompt_name"],
        "max_new_frames": int(case["max_new_frames"]),
        "output_planned_path": str(case["output_path"].relative_to(OUT_D)),
        "sampling": SAMPLING,
        "external_text_normalization": "DISABLED_EXACT_FROZEN_INPUT",
        "case_timeout_s": int(case_timeout_s),
    }


def legacy_checkpoint_compatible(prior: dict, case: dict) -> bool:
    required_equal = {
        "suite": case["suite"],
        "case_id": case["case_id"],
        "repeat": int(case["repeat"]),
        "seed": int(case["seed"]),
        "language": case["language"],
        "raw_text": case["text"],
        "prompt_name": case["prompt_name"],
        "output_planned_path": str(case["output_path"].relative_to(OUT_D)),
    }
    if any(prior.get(key) != value for key, value in required_equal.items()):
        return False
    if prior.get("prepared_text", case["text"]) != case["text"]:
        return False
    if prior.get("max_new_frames", case["max_new_frames"]) != case["max_new_frames"]:
        return False
    if prior.get("sampling", SAMPLING) != SAMPLING:
        return False
    return prior.get(
        "external_text_normalization", "DISABLED_EXACT_FROZEN_INPUT"
    ) == "DISABLED_EXACT_FROZEN_INPUT"


def longest_true_run(mask: np.ndarray) -> int:
    longest = current = 0
    for value in mask.reshape(-1):
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def waveform_record(samples: np.ndarray, sample_rate: int) -> dict:
    values = np.asarray(samples, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    finite = bool(values.size and np.isfinite(values).all())
    duration_s = float(values.shape[-1] / sample_rate) if values.ndim == 2 else 0.0
    rms = float(np.sqrt(np.mean(values * values))) if finite else None
    peak = float(np.max(np.abs(values))) if finite else None
    longest_clip = 0
    if finite:
        for channel in values:
            longest_clip = max(longest_clip, longest_true_run(np.abs(channel) >= CLIP_THRESHOLD))
    longest_clip_s = float(longest_clip / sample_rate)
    assertions = {
        "nonempty": bool(values.size and values.shape[-1] > 0),
        "finite": finite,
        "rms_above_floor": bool(rms is not None and rms > RMS_FLOOR),
        "duration_in_range": MIN_DURATION_S <= duration_s <= MAX_DURATION_S,
        "no_sustained_clipping": longest_clip_s <= MAX_CLIP_RUN_S,
    }
    return {
        "sample_rate_hz": int(sample_rate),
        "channels": int(values.shape[0]) if values.ndim == 2 else None,
        "frames": int(values.shape[-1]) if values.ndim == 2 else 0,
        "sample_count": int(values.size),
        "duration_s": duration_s,
        "rms": rms,
        "peak_abs": peak,
        "longest_contiguous_clip_run_s": longest_clip_s,
        "assertions": assertions,
        "passed": all(assertions.values()),
    }


def file_waveform_record(path: Path) -> dict:
    values, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    record = waveform_record(values.T, int(sample_rate))
    record.update(
        {
            "path": str(path.relative_to(OUT_D)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "format": sf.info(path).format,
            "subtype": sf.info(path).subtype,
        }
    )
    return record


def read_checkpoint() -> dict[tuple[str, str, int, int], dict]:
    rows: dict[tuple[str, str, int, int], dict] = {}
    if not CHECKPOINT_PATH.is_file():
        return rows
    for line in CHECKPOINT_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row["suite"], row["case_id"], int(row["repeat"]), int(row["seed"]))
        rows[key] = row
    return rows


def append_checkpoint(row: dict) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHECKPOINT_PATH.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def soundfile_load(path: str | Path, *args, **kwargs) -> tuple[torch.Tensor, int]:
    del args, kwargs
    values, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(values.T.copy()), int(sample_rate)


class MossRuntime:
    def __init__(self) -> None:
        assert sha256_file(TTS_DIR / "pytorch_model.bin") == TTS_WEIGHT_SHA256
        assert sha256_file(CODEC_DIR / "model-00001-of-00001.safetensors") == CODEC_WEIGHT_SHA256
        torch.set_num_threads(THREADS)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        torchaudio.load = soundfile_load
        load_started = time.perf_counter()
        self.tts = AutoModelForCausalLM.from_pretrained(
            TTS_DIR,
            trust_remote_code=True,
            local_files_only=True,
        ).eval()
        self.tts._set_attention_implementation("sdpa")
        self.codec = AutoModel.from_pretrained(
            CODEC_DIR,
            trust_remote_code=True,
            local_files_only=True,
        ).eval()
        self.text_tokenizer = self.tts._load_text_tokenizer(text_tokenizer_path=str(TTS_DIR))
        self.load_s = time.perf_counter() - load_started
        assert sum(parameter.numel() for parameter in self.tts.parameters()) == TTS_PARAMETERS
        assert sum(parameter.numel() for parameter in self.codec.parameters()) == CODEC_PARAMETERS
        self.prompt_cache: dict[str, torch.LongTensor] = {}
        self.prompt_records: dict[str, dict] = {}

    def tokenize(self, text: str) -> dict:
        token_ids = list(self.text_tokenizer.encode(text, add_special_tokens=False))
        unk_id = self.text_tokenizer.unk_token_id
        decoded = self.text_tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        pieces = (
            list(self.text_tokenizer.sp_model.encode(text, out_type=str))
            if hasattr(self.text_tokenizer, "sp_model")
            else None
        )
        return {
            "token_ids": token_ids,
            "token_count": len(token_ids),
            "pieces": pieces,
            "unk_token_id": unk_id,
            "unk_count": sum(token_id == unk_id for token_id in token_ids) if unk_id is not None else 0,
            "decoded": decoded,
            "roundtrip_exact_after_strip": decoded.strip() == text.strip(),
        }

    def prompt_codes(self, prompt_name: str) -> torch.LongTensor:
        if prompt_name in self.prompt_cache:
            return self.prompt_cache[prompt_name]
        path = PROMPT_DIR / prompt_name
        started = time.perf_counter()
        waveform, sample_rate = self.tts._load_reference_audio(path, SAMPLE_RATE, CHANNELS)
        encoded = self.tts._call_audio_encode(
            audio_tokenizer=self.codec,
            waveform=waveform,
            sample_rate=sample_rate,
        )
        codes = self.tts._mask_unused_audio_channels(
            self.tts._normalize_audio_codes(encoded),
            nq=NQ,
        ).cpu()
        self.prompt_cache[prompt_name] = codes
        self.prompt_records[prompt_name] = {
            "path": str(path.relative_to(OUT_D)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "source_revision": SOURCE_REVISION,
            "encoded_frames": int(codes.shape[0]),
            "codebooks": int(codes.shape[1]),
            "encode_s": time.perf_counter() - started,
        }
        return codes

    @torch.no_grad()
    def synthesize(
        self,
        *,
        suite: str,
        case_id: str,
        language: str,
        text: str,
        prompt_name: str,
        output_path: Path,
        seed: int,
        repeat: int,
        max_new_frames: int,
        item_id: str | None = None,
        pair_id: str | None = None,
        pair_role: str | None = None,
    ) -> dict:
        started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        output_path.unlink(missing_ok=True)
        prompt_codes = self.prompt_codes(prompt_name)
        tokenization = self.tokenize(text)
        input_ids, attention_mask = self.tts.build_inference_input_ids(
            text=text,
            text_tokenizer=self.text_tokenizer,
            mode="voice_clone",
            prompt_audio_codes=prompt_codes,
            device="cpu",
        )
        torch.manual_seed(seed)
        generation_started = time.perf_counter()
        generated = self.tts._generate_audio_token_ids_with_fallback(
            prompt_input_ids=input_ids,
            attention_mask=attention_mask,
            effective_nq=NQ,
            max_new_frames=max_new_frames,
            resolved_device=torch.device("cpu"),
            **SAMPLING,
        )
        generation_s = time.perf_counter() - generation_started
        audio_token_ids = self.tts._trim_generated_audio_token_ids(
            generated[0],
            effective_nq=NQ,
        )
        generated_frames = int(audio_token_ids.shape[0])
        max_new_frames_hit = generated_frames >= max_new_frames
        audio_token_sha256 = hashlib.sha256(
            audio_token_ids.detach().cpu().numpy().astype(np.int16).tobytes()
        ).hexdigest()
        if max_new_frames_hit:
            output_path.unlink(missing_ok=True)
            return {
                "schema_version": 1,
                "suite": suite,
                "case_id": case_id,
                "repeat": repeat,
                "language": language,
                "item_id": item_id,
                "raw_text": text,
                "prepared_text": text,
                "external_text_normalization": "DISABLED_EXACT_FROZEN_INPUT",
                "pair_id": pair_id,
                "pair_role": pair_role,
                "seed": seed,
                "sampling": SAMPLING,
                "prompt_name": prompt_name,
                "prompt": self.prompt_records[prompt_name],
                "text_tokenization": tokenization,
                "max_new_frames": max_new_frames,
                "generated_frames": generated_frames,
                "max_new_frames_hit": True,
                "audio_token_shape": list(audio_token_ids.shape),
                "audio_token_sha256": audio_token_sha256,
                "timing": {
                    "generation_s": generation_s,
                    "decode_s": None,
                    "inference_s": generation_s,
                    "rtf": None,
                },
                "raw_waveform_gate": None,
                "output_planned_path": str(output_path.relative_to(OUT_D)),
                "output": None,
                "passed": False,
                "failure_reason": "MAX_NEW_FRAMES_HIT_DECODE_SKIPPED",
                "failure_scope": (
                    "Generation did not terminate before the hard frame cap; "
                    "codec decode and WAV retention were skipped because the route "
                    "is already unsafe and cap-length decode can stall this CPU host."
                ),
                "run_started_utc": started_utc,
                "run_finished_utc": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                ),
            }
        decode_started = time.perf_counter()
        waveform, sample_rate = self.tts._decode_audio_token_ids_to_waveform(
            audio_tokenizer=self.codec,
            audio_token_ids=audio_token_ids,
            target_sample_rate=SAMPLE_RATE,
            effective_nq=NQ,
            resolved_device=torch.device("cpu"),
        )
        decode_s = time.perf_counter() - decode_started
        raw_values = waveform.detach().cpu().float().numpy()
        raw_gate = waveform_record(raw_values, int(sample_rate))
        row: dict[str, Any] = {
            "schema_version": 1,
            "suite": suite,
            "case_id": case_id,
            "repeat": repeat,
            "language": language,
            "item_id": item_id,
            "raw_text": text,
            "prepared_text": text,
            "external_text_normalization": "DISABLED_EXACT_FROZEN_INPUT",
            "pair_id": pair_id,
            "pair_role": pair_role,
            "seed": seed,
            "sampling": SAMPLING,
            "prompt_name": prompt_name,
            "prompt": self.prompt_records[prompt_name],
            "text_tokenization": tokenization,
            "max_new_frames": max_new_frames,
            "generated_frames": generated_frames,
            "max_new_frames_hit": max_new_frames_hit,
            "audio_token_shape": list(audio_token_ids.shape),
            "audio_token_sha256": audio_token_sha256,
            "timing": {
                "generation_s": generation_s,
                "decode_s": decode_s,
                "inference_s": generation_s + decode_s,
                "rtf": (
                    (generation_s + decode_s) / raw_gate["duration_s"]
                    if raw_gate["duration_s"] > 0
                    else None
                ),
            },
            "raw_waveform_gate": raw_gate,
            "output_planned_path": str(output_path.relative_to(OUT_D)),
            "output": None,
            "passed": False,
            "run_started_utc": started_utc,
            "run_finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if raw_gate["passed"]:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            serialized_values = raw_values.mean(axis=0, keepdims=True, dtype=np.float32)
            sf.write(output_path, serialized_values.T, int(sample_rate), subtype="PCM_16")
            file_gate = file_waveform_record(output_path)
            if file_gate["passed"]:
                row["output"] = file_gate
                row["serialization"] = {
                    "source": "raw 48 kHz stereo decoder tensor",
                    "operation": "arithmetic mean across two channels",
                    "retained_format": "mono PCM16 WAV",
                }
                row["passed"] = True
            else:
                output_path.unlink(missing_ok=True)
                row["file_waveform_gate_rejected"] = file_gate
        return row


def cases_for_suite(suite: str) -> list[dict]:
    corpus = load_json(CORPUS_PATH)
    d5_prompts = load_json(D5_PROMPTS_PATH)
    cases: list[dict] = []
    if suite == "g2_d1_zh":
        for item in corpus["items"]:
            cases.append(
                {
                    "case_id": item["item_id"],
                    "language": "zh",
                    "text": item["texts"]["zh"],
                    "prompt_name": "zh_1.wav",
                    "output_path": AUDIO_DIR / suite / f"{item['item_id']}.wav",
                    "repeat": 1,
                    "seed": stable_seed(suite, item["item_id"]),
                    "max_new_frames": 375,
                }
            )
    elif suite == "g2_mixed":
        for pair_id, full_text, ablated_text in MIXED_PAIRS:
            seed = stable_seed(suite, pair_id)
            for role, text in (("full", full_text), ("ablated", ablated_text)):
                case_id = f"{pair_id}_{role}"
                cases.append(
                    {
                        "case_id": case_id,
                        "language": "zh",
                        "text": text,
                        "prompt_name": "zh_1.wav",
                        "output_path": AUDIO_DIR / suite / f"{case_id}.wav",
                        "repeat": 1,
                        "seed": seed,
                        "max_new_frames": 160,
                        "pair_id": pair_id,
                        "pair_role": role,
                    }
                )
    elif suite == "g3":
        for language, texts in D3_TEXTS.items():
            for index, text in enumerate(texts, 1):
                repeats = (1, 2, 3) if language == "zh" else (1,)
                for repeat in repeats:
                    case_id = f"{language}_{index:02d}"
                    cases.append(
                        {
                            "case_id": case_id,
                            "language": language,
                            "text": text,
                            "prompt_name": "zh_1.wav" if language == "zh" else "en_4.wav",
                            "output_path": AUDIO_DIR / suite / language / f"{case_id}_r{repeat}.wav",
                            "repeat": repeat,
                            "seed": stable_seed(suite, case_id, repeat),
                            "max_new_frames": 120,
                        }
                    )
    elif suite == "g4":
        by_id = {item["item_id"]: item for item in corpus["items"]}
        for language in ("en", "es", "fr", "ru"):
            for item_id in G4_ITEM_IDS:
                item = by_id[item_id]
                cases.append(
                    {
                        "case_id": f"{language}_{item_id}",
                        "item_id": item_id,
                        "language": language,
                        "text": item["texts"][language],
                        "prompt_name": "en_4.wav",
                        "output_path": AUDIO_DIR / suite / language / f"{item_id}.wav",
                        "repeat": 1,
                        "seed": stable_seed(suite, f"{language}:{item_id}"),
                        "max_new_frames": 375,
                    }
                )
    elif suite == "g5":
        for prompt in d5_prompts["languages"]["zh"]:
            index = int(prompt["listen_index"])
            case_id = f"zh_moss_{index:02d}"
            cases.append(
                {
                    "case_id": case_id,
                    "language": "zh",
                    "text": prompt["text"],
                    "prompt_name": "zh_1.wav",
                    "output_path": OUT_D / "artifacts" / "listening" / f"{case_id}.wav",
                    "repeat": 1,
                    "seed": stable_seed(suite, case_id),
                    "max_new_frames": 375,
                }
            )
    else:
        raise ValueError(f"unknown suite: {suite}")
    return cases


def host_record() -> dict:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "numpy": np.__version__,
        "soundfile": sf.__version__,
        "threads": THREADS,
        "device": "cpu",
    }


def summarize_suite(suite: str, rows: list[dict], runtime: MossRuntime) -> dict:
    passed = [row for row in rows if row["passed"]]
    durations = [row["output"]["duration_s"] for row in passed]
    rtfs = [row["timing"]["rtf"] for row in passed]
    payload = {
        "schema_version": 1,
        "suite": suite,
        "evidence_class": "HOST",
        "model": {
            "tts_repo": TTS_REPO,
            "tts_revision": TTS_REVISION,
            "tts_weight_sha256": TTS_WEIGHT_SHA256,
            "tts_parameters": TTS_PARAMETERS,
            "codec_repo": CODEC_REPO,
            "codec_revision": CODEC_REVISION,
            "codec_weight_sha256": CODEC_WEIGHT_SHA256,
            "codec_parameters": CODEC_PARAMETERS,
            "combined_parameters": TTS_PARAMETERS + CODEC_PARAMETERS,
            "source_revision": SOURCE_REVISION,
        },
        "runtime": {
            "model_load_s": runtime.load_s,
            "attention_implementation": "sdpa",
            "sample_rate_hz": SAMPLE_RATE,
            "channels": CHANNELS,
            "nq": NQ,
            "sampling": SAMPLING,
            "text_preprocessing": "none; exact frozen input passed to SentencePiece",
        },
        "host": host_record(),
        "attempted": len(rows),
        "passed": len(passed),
        "failed": len(rows) - len(passed),
        "max_new_frames_hit_count": sum(bool(row.get("max_new_frames_hit")) for row in rows),
        "duration_s": {
            "min": min(durations) if durations else None,
            "median": float(np.median(durations)) if durations else None,
            "max": max(durations) if durations else None,
        },
        "rtf": {
            "median": float(np.median(rtfs)) if rtfs else None,
            "p95": float(np.percentile(rtfs, 95)) if rtfs else None,
        },
        "prompt_records": runtime.prompt_records,
        "records": rows,
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        nargs="+",
        choices=("g2_d1_zh", "g2_mixed", "g3", "g4", "g5", "all"),
        required=True,
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--case-timeout-s",
        type=int,
        default=300,
        help="Fail one synthesis attempt after this many wall-clock seconds; 0 disables.",
    )
    args = parser.parse_args()
    suites = list(args.suite)
    if "all" in suites:
        suites = ["g2_d1_zh", "g2_mixed", "g3", "g4", "g5"]

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    existing = {} if args.force else read_checkpoint()
    runtime = MossRuntime()
    for suite in suites:
        rows = []
        for case in cases_for_suite(suite):
            case = {**case, "suite": suite}
            key = (suite, case["case_id"], int(case["repeat"]), int(case["seed"]))
            prior = existing.get(key)
            config_payload = case_config_payload(case, args.case_timeout_s)
            config_sha256 = canonical_sha256(config_payload)
            config_matches = bool(
                prior
                and (
                    prior.get("case_config_sha256") == config_sha256
                    or (
                        prior.get("case_config_sha256") is None
                        and legacy_checkpoint_compatible(prior, case)
                    )
                )
            )
            if config_matches:
                output = prior.get("output")
                if output is None or (
                    (OUT_D / output["path"]).is_file()
                    and sha256_file(OUT_D / output["path"]) == output["sha256"]
                ):
                    if prior.get("case_config_sha256") is None:
                        prior = dict(prior)
                        prior["generation_protocol_version"] = GENERATION_PROTOCOL_VERSION
                        prior["case_config"] = config_payload
                        prior["case_config_sha256"] = config_sha256
                        prior["checkpoint_migrated_from_legacy"] = True
                        append_checkpoint(prior)
                        existing[key] = prior
                    rows.append(prior)
                    print(f"reuse {suite} {case['case_id']} r{case['repeat']}", flush=True)
                    continue
            try:
                previous_alarm_handler = None
                if args.case_timeout_s:
                    def case_timeout_handler(signum, frame):
                        del signum, frame
                        raise TimeoutError(
                            f"case exceeded {args.case_timeout_s} wall-clock seconds"
                        )

                    previous_alarm_handler = signal.signal(
                        signal.SIGALRM, case_timeout_handler
                    )
                    signal.setitimer(signal.ITIMER_REAL, args.case_timeout_s)
                try:
                    row = runtime.synthesize(**case)
                finally:
                    if args.case_timeout_s:
                        signal.setitimer(signal.ITIMER_REAL, 0)
                        signal.signal(signal.SIGALRM, previous_alarm_handler)
            except Exception as exc:
                case["output_path"].unlink(missing_ok=True)
                row = {
                    "schema_version": 1,
                    "suite": suite,
                    "case_id": case["case_id"],
                    "repeat": case["repeat"],
                    "language": case["language"],
                    "item_id": case.get("item_id"),
                    "raw_text": case["text"],
                    "prepared_text": case["text"],
                    "seed": case["seed"],
                    "prompt_name": case["prompt_name"],
                    "sampling": SAMPLING,
                    "max_new_frames": case["max_new_frames"],
                    "max_new_frames_hit": None,
                    "output_planned_path": str(case["output_path"].relative_to(OUT_D)),
                    "output": None,
                    "passed": False,
                    "failure_reason": "EXCEPTION_OR_TIMEOUT",
                    "exception": {"type": type(exc).__name__, "message": str(exc)},
                    "run_finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            row["generation_protocol_version"] = GENERATION_PROTOCOL_VERSION
            row["case_config"] = config_payload
            row["case_config_sha256"] = config_sha256
            append_checkpoint(row)
            existing[key] = row
            rows.append(row)
            print(
                f"{suite} {case['case_id']} r{case['repeat']} "
                f"pass={row['passed']} duration="
                f"{None if row.get('output') is None else round(row['output']['duration_s'], 3)}",
                flush=True,
            )
        summary = summarize_suite(suite, rows, runtime)
        result_path = RESULT_DIR / f"{suite}_synthesis.json"
        result_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {result_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
