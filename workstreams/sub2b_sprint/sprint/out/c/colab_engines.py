"""Pinned Google Colab adapters for Lane C ASR evaluation.

This module deliberately imports only Python's standard library at import time.
Each engine is intended to run in a fresh Colab runtime: call
``install_engine(name)`` before ``run_engine(name, samples, output_dir)``.

The runners perform inference only.  There is no training, fine-tuning,
quantization, calibration, or export path in this file.
"""

from __future__ import annotations

import datetime as _datetime
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import unicodedata
from typing import Any, Callable, Iterable, Mapping, Sequence


__all__ = ["ENGINE_SPECS", "install_engine", "run_engine"]

_SCHEMA_VERSION = "lane-c-colab-engines-v1"
_RUN_PROTOCOL_VERSION = "lane-c-asr-inference-checkpoint-v2"
_REQUIRED_SAMPLE_KEYS = (
    "utterance_id",
    "language",
    "condition",
    "audio_path",
    "reference",
    "duration_s",
)
_MIRRORED_SAMPLE_KEYS = (
    "config",
    "audio_relpath",
    "reference_source",
    "raw_transcription",
    "transcription",
    "source_id",
    "fleurs_id",
    "source_path",
    "source_parquet_repo_path",
    "audio_pcm_sha256",
    "clean_pcm_sha256",
    "raw_audio_sha256",
    "audio_file_sha256",
    "reference_sha256",
    "num_samples",
    "sample_rate",
    "channels",
    "pcm_dtype",
    "source_sample_rate",
    "source_channels",
    "noise_repo_path",
    "noise_raw_audio_sha256",
    "noise_pcm_sha256",
    "noise_segment_pcm_sha256",
    "mix_entry_sha256",
    "mix_selector_sha256",
    "noise_offset_samples",
    "noise_wrapped",
    "target_snr_db",
    "achieved_snr_db",
)
_LANGUAGES = ("en", "zh", "es", "fr", "ru")
_SENSEVOICE_FIXED_CONTEXT_SECONDS = 10.0
_SENSEVOICE_FIXED_CONTEXT_SAMPLE_RATE = 16_000
_SENSEVOICE_FIXED_CONTEXT_NUM_SAMPLES = int(
    _SENSEVOICE_FIXED_CONTEXT_SECONDS * _SENSEVOICE_FIXED_CONTEXT_SAMPLE_RATE
)
_SENSEVOICE_FIXED_CONTEXT_PROBE_N = 30


# Keep this object JSON-serializable: notebooks and result aggregators persist it
# verbatim as part of the provenance record.
ENGINE_SPECS: dict[str, dict[str, Any]] = {
    "sensevoice": {
        "engine": "sensevoice",
        "runtime": "FunASR",
        "runtime_version": "1.4.0",
        "languages": ["en", "zh"],
        "install_commands": [
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "numpy==1.26.4",
                "funasr==1.4.0",
                "transformers==4.57.6",
                "huggingface_hub==0.36.0",
                "soundfile==0.13.1",
            ]
        ],
        "models": {
            "sensevoice": {
                "repo_id": "FunAudioLLM/SenseVoiceSmall",
                "revision": "3847d57b6bdf2dd8875cb1508d2af43d80a16bf7",
                "bank": "D-fork",
                "files": {
                    "model.pt": "833ca2dcfdf8ec91bd4f31cfac36d6124e0c459074d5e909aec9cabe6204a3ea",
                    "chn_jpn_yue_eng_ko_spectok.bpe.model": "aa87f86064c3730d799ddf7af3c04659151102cba548bce325cf06ba4da4e6a8",
                },
                "file_sizes": {
                    "model.pt": 936291369,
                    "chn_jpn_yue_eng_ko_spectok.bpe.model": 377341,
                },
                "allow_patterns": [
                    "model.pt",
                    "*.json",
                    "*.yaml",
                    "*.model",
                    "*.mvn",
                ],
                "languages": ["en", "zh"],
            }
        },
        "inference": {
            "use_itn": False,
            "vad": None,
            "device": "cuda:0",
            "context_policy": {
                "main_scoring_input": "whole utterance",
                "main_scoring_inputs_unchanged": True,
                "product_fixed_context_s": 10.0,
                "fixed_context_probe_condition": "clean",
                "fixed_context_probe_n_per_language": 30,
                "fixed_context_probe_scoring_included": False,
            },
            "partial_api": "utterance-only/no partial API",
        },
    },
    "nemo": {
        "engine": "nemo",
        "runtime": "NeMo",
        "runtime_version": "2.7.3",
        "languages": ["en", "es", "fr", "ru"],
        "install_commands": [
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "nemo_toolkit[asr]==2.7.3",
                "numpy==1.26.4",
                "transformers==4.57.6",
                "huggingface_hub==0.36.0",
                "tokenizers==0.22.1",
            ]
        ],
        "models": {
            "parakeet": {
                "repo_id": "nvidia/parakeet-tdt-0.6b-v3",
                "revision": "7c35754d166cca382ad1e53e68b01e7c575f3a1d",
                "bank": "D-fork",
                "checkpoint": "parakeet-tdt-0.6b-v3.nemo",
                "files": {
                    "parakeet-tdt-0.6b-v3.nemo": "3cbdc85877e668ca7b82d0d56770eb1fac76691f55d6b97545e8d61ca588d10d",
                    "tokenizer.json": "bd321b096832a3f270bd3b2a88823957920f1a5c5ada71114a26ea729d0cbe91",
                },
                "file_sizes": {
                    "parakeet-tdt-0.6b-v3.nemo": 2509332480,
                    "tokenizer.json": 1159960,
                },
                "allow_patterns": ["parakeet-tdt-0.6b-v3.nemo", "tokenizer.json"],
                "languages": ["es", "fr", "ru"],
                "route": "fixed-window utterance route",
                "language_argument": False,
            },
            "canary": {
                "repo_id": "nvidia/canary-180m-flash",
                "revision": "b12ab418510d093e83890178fd0e8b0d0f7918a6",
                "bank": "L-fork",
                "checkpoint": "canary-180m-flash.nemo",
                "files": {
                    "canary-180m-flash.nemo": "7b97de18b718ef01bf3398715ce8d18486c4d83961e230230b26f5712e112675"
                },
                "file_sizes": {"canary-180m-flash.nemo": 736665600},
                "allow_patterns": ["canary-180m-flash.nemo"],
                "languages": ["en", "es", "fr"],
                "source_lang_equals_target_lang": True,
                "pnc": "no",
                "timestamps": False,
            },
        },
        "inference": {
            "device": "cuda:0",
            "partial_api": "utterance-only/no partial API",
            "parakeet_streaming_script_not_used": True,
        },
    },
    "zipformer": {
        "engine": "zipformer",
        "runtime": "sherpa-onnx",
        "runtime_version": "1.13.4",
        "languages": ["zh"],
        "install_commands": [
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "sherpa-onnx==1.13.4",
                "huggingface_hub==0.36.0",
                "soundfile==0.13.1",
                "scipy==1.16.1",
            ]
        ],
        "models": {
            "zipformer": {
                "repo_id": "pfluo/k2fsa-zipformer-chinese-english-mixed",
                "revision": "6eb615ae77ecac05c5628d5c8ed7037c14a338d5",
                "bank": "L-fork",
                "files": {
                    "exp/encoder-epoch-99-avg-1.onnx": "a509c0d2797d7da93891950cefb1ab2daa33608f48895551d96a1a2fbc00d254",
                    "exp/decoder-epoch-99-avg-1.onnx": "d91b63c84f27f1d822244ced4cc2b9f7fb8a60a328f13b860408b8889dd0a0ed",
                    "exp/joiner-epoch-99-avg-1.onnx": "40f1268ddf9d65f0c2d5b8d2ba7ed6f3443d51898dfa826216afd9c53598f074",
                    "data/lang_char_bpe/tokens.txt": "a8e0e4ec53810e433789b54a5c0134a7eaa2ffca595a6334d54c00da858841d3",
                },
                "file_sizes": {
                    "exp/encoder-epoch-99-avg-1.onnx": 268919863,
                    "exp/decoder-epoch-99-avg-1.onnx": 13876468,
                    "exp/joiner-epoch-99-avg-1.onnx": 12833675,
                    "data/lang_char_bpe/tokens.txt": 56317,
                },
                "allow_patterns": [
                    "exp/encoder-epoch-99-avg-1.onnx",
                    "exp/decoder-epoch-99-avg-1.onnx",
                    "exp/joiner-epoch-99-avg-1.onnx",
                    "data/lang_char_bpe/tokens.txt",
                ],
                "languages": ["zh"],
            }
        },
        "inference": {
            "recognizer": "sherpa_onnx.OnlineRecognizer.from_transducer",
            "decoding_method": "modified_beam_search",
            "max_active_paths": 4,
            "sample_rate": 16000,
            "feature_dim": 80,
            "tail_silence_s": 0.66,
            "partial_trace_interval_s": 0.1,
            "partial_trace_clean_n": 30,
            "retained_trace_pacing": "real-time cumulative 100 ms deadlines",
            "non_retained_pacing": "accelerated",
            "provider": "cpu",
        },
        "artifact_caveats": {
            "readme_score_checkpoint": "epoch-14.pt",
            "selected_export_checkpoint": "epoch-99-avg-1",
            "checkpoint_mismatch": True,
            "readme_unit": "char+bpe",
            "selected_exact_runtime_assets": "tokens.txt only; bpe.model is outside the exact four-file set",
            "bpe_attribution_mismatch": True,
        },
    },
    "gigaam": {
        "engine": "gigaam",
        "runtime": "transformers",
        "runtime_version": "5.14.1",
        "languages": ["ru"],
        "install_commands": [
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "transformers==5.14.1",
                "hydra-core==1.3.2",
                "omegaconf==2.3.0",
                "huggingface_hub==1.26.0",
                "soundfile==0.13.1",
                "librosa==0.11.0",
                # The pinned modeling file imports pyannote.audio only under
                # TYPE_CHECKING, but Transformers' static import scan still
                # requires the top-level ``pyannote`` namespace.  core is the
                # minimal compatible provider; the unused audio stack stays
                # out of this short-form transcription route.
                "pyannote.core==5.0.0",
            ]
        ],
        "models": {
            "gigaam": {
                "repo_id": "ai-sage/GigaAM-Multilingual",
                "revision": "2f8a57144e6ec3adfd32fe0484d9ea9913305bc8",
                "bank": "L-fork",
                "files": {
                    "pytorch_model.bin": "e1db43873ec5e296f229572e06e2470fc157ac9f8d4aacabda295630b9b91728"
                },
                "file_sizes": {"pytorch_model.bin": 883170115},
                "allow_patterns": [
                    "pytorch_model.bin",
                    "config.json",
                    "modeling_gigaam.py",
                ],
                "languages": ["ru"],
                "trust_remote_code": True,
            }
        },
        "inference": {
            "call": "model.transcribe(path).text",
            "device": "cuda:0",
            "partial_api": "utterance-only/no partial API",
        },
    },
    "omni": {
        "engine": "omni",
        "runtime": "omnilingual-asr",
        "runtime_version": "0.2.0",
        "languages": list(_LANGUAGES),
        "install_commands": [
            ["apt-get", "update", "-qq"],
            ["apt-get", "install", "-y", "-qq", "libsndfile1"],
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--index-url",
                "https://download.pytorch.org/whl/cu128",
                "torch==2.8.0",
                "torchaudio==2.8.0",
            ],
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                # omnilingual-asr 0.2.0 declares ``Requires-Python <=3.12``.
                # PEP 440 interprets that as <=3.12.0, so current Colab
                # Python 3.12 patch releases are rejected despite being the
                # package's documented supported minor.  This narrowly
                # bypasses that malformed metadata; all package/model versions
                # remain exact and runtime imports are verified afterwards.
                "--ignore-requires-python",
                "fairseq2==0.6",
                "omnilingual-asr==0.2.0",
                "huggingface_hub==0.32.6",
            ],
        ],
        "models": {
            "omni": {
                "repo_id": "facebook/omniASR-CTC-300M",
                "revision": "8e35f0cc28fa6099e0c14d56db85ce0423baa691",
                "bank": "OmniASR",
                "files": {
                    "omniASR-CTC-300M.pt": "4d31c6a7725a03e90cf26080c67361928755a9f43d322dd574145fbe91c76aec",
                    "omniASR_tokenizer.model": "b954cc166b0c9e0271b953fa226fa27ca706a25b7029e84579fe2c60a2b451fe",
                },
                "file_sizes": {
                    "omniASR-CTC-300M.pt": 1302113892,
                    "omniASR_tokenizer.model": 87562,
                },
                "allow_patterns": [
                    "omniASR-CTC-300M.pt",
                    "omniASR_tokenizer.model",
                ],
                "languages": list(_LANGUAGES),
            }
        },
        "inference": {
            "package_model_card": "lane_c_omniASR_CTC_300M_v1_pinned",
            "asset_card_version": "v1",
            "fairseq2_user_asset_dir": True,
            "torch": "2.8.0+cu128",
            "fairseq2": "0.6",
            "dtype": "float16",
            "device": "cuda",
            "language_codes": {
                "en": "eng_Latn",
                "zh": "cmn_Hans",
                "es": "spa_Latn",
                "fr": "fra_Latn",
                "ru": "rus_Cyrl",
            },
            "smoke_clean_n_per_language": 30,
            "execution_order": (
                "all five language smoke gates before any remaining clean or noisy "
                "evaluation"
            ),
            "collapse_gate": {
                "empty_fraction_gt": 0.20,
                "wrong_script_fraction_gt": 0.50,
            },
            "partial_api": "utterance-only/no partial API",
        },
    },
}


def install_engine(engine: str) -> dict[str, Any]:
    """Install one engine's isolated Colab dependencies.

    Commands are executed without a shell and fail fast.  The returned mapping
    is suitable for inclusion in run provenance.  For ``omni`` this must be
    called before importing torch in the notebook, because it installs the
    Torch 2.8 / CUDA 12.8 build required by fairseq2 0.6.
    """

    name = _engine_name(engine)
    spec = ENGINE_SPECS[name]
    started = _utc_now()
    completed: list[dict[str, Any]] = []
    for command in spec["install_commands"]:
        t0 = time.perf_counter()
        subprocess.run(command, check=True)
        completed.append(
            {
                "argv": list(command),
                "wall_s": time.perf_counter() - t0,
                "returncode": 0,
            }
        )
    return {
        "schema_version": _SCHEMA_VERSION,
        "engine": name,
        "started_at_utc": started,
        "finished_at_utc": _utc_now(),
        "commands": completed,
    }


def run_engine(
    engine: str,
    samples: Sequence[Mapping[str, Any]],
    output_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    """Run a pinned engine over Lane C sample dictionaries.

    Returns ``{"records": ..., "metadata": ..., "partial_traces": ...}``.
    A resumable ``<engine>_progress.jsonl`` checkpoint is fsynced after every
    inference batch.  Existing checkpoints are reused only when their manifest
    fingerprint exactly matches this call.
    """

    name = _engine_name(engine)
    clean_samples = _validate_samples(samples)
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    run_id = _run_id(name, clean_samples)
    progress_path = out / f"{name}_progress.jsonl"
    progress = _load_progress(progress_path, run_id)

    started_at = _utc_now()
    t0 = time.perf_counter()
    if not clean_samples:
        metadata = _base_metadata(name, run_id, started_at, progress_path)
        metadata.update(
            {
                "finished_at_utc": _utc_now(),
                "wall_s": time.perf_counter() - t0,
                "sample_count": 0,
                "record_count": 0,
                "partial_trace_count": 0,
                "note": "No samples supplied; no checkpoint was downloaded.",
            }
        )
        return {"records": [], "metadata": metadata, "partial_traces": []}

    runners: dict[str, Callable[..., dict[str, Any]]] = {
        "sensevoice": _run_sensevoice,
        "nemo": _run_nemo,
        "zipformer": _run_zipformer,
        "gigaam": _run_gigaam,
        "omni": _run_omni,
    }
    result = runners[name](clean_samples, out, run_id, progress_path, progress)
    records = _dedupe_records(result["records"])
    traces = _dedupe_traces(result["partial_traces"])
    records.sort(key=_record_sort_key)
    traces.sort(key=_trace_sort_key)

    metadata = _base_metadata(name, run_id, started_at, progress_path)
    metadata.update(result["metadata"])
    metadata.update(
        {
            "finished_at_utc": _utc_now(),
            "wall_s": time.perf_counter() - t0,
            "sample_count": len(clean_samples),
            "record_count": len(records),
            "partial_trace_count": len(traces),
            "resumed_record_count": len(progress["records"]),
            "resumed_partial_trace_count": len(progress["partial_traces"]),
        }
    )
    _append_progress(
        progress_path,
        {
            "schema_version": _SCHEMA_VERSION,
            "run_protocol_version": _RUN_PROTOCOL_VERSION,
            "event": "complete",
            "run_id": run_id,
            "engine": name,
            "timestamp_utc": _utc_now(),
            "record_count": len(records),
            "partial_trace_count": len(traces),
        },
    )
    return {"records": records, "metadata": metadata, "partial_traces": traces}


def _run_sensevoice(
    samples: list[dict[str, Any]],
    output_dir: Path,
    run_id: str,
    progress_path: Path,
    progress: dict[str, Any],
) -> dict[str, Any]:
    spec = ENGINE_SPECS["sensevoice"]
    selected = [s for s in samples if s["language"] in spec["languages"]]
    metadata = _route_metadata(spec, samples, selected)
    context_policy = _sensevoice_context_policy(selected, None)
    metadata["context_policy"] = context_policy
    metadata["partial_capability"] = _utterance_only_metadata(
        "SenseVoice exposes one completed result per invocation and no "
        "incremental callback."
    )
    if not selected:
        return {
            "records": progress["records"],
            "partial_traces": progress["partial_traces"],
            "metadata": metadata,
        }

    _require_cuda("sensevoice")
    model_dir, artifact_meta = _download_model("sensevoice", "sensevoice", output_dir)

    from funasr import AutoModel  # type: ignore[import-not-found]
    from funasr.utils.postprocess_utils import (  # type: ignore[import-not-found]
        rich_transcription_postprocess,
    )
    import numpy as np  # type: ignore[import-not-found]
    import soundfile as sf  # type: ignore[import-not-found]

    try:
        model = AutoModel(
            model=str(model_dir),
            device="cuda:0",
            disable_update=True,
        )
        update_guard = "disable_update=True"
    except TypeError:
        model = AutoModel(model=str(model_dir), device="cuda:0")
        update_guard = "FunASR 1.4.0 rejected disable_update; local pinned path still used"

    records = list(progress["records"])
    traces = list(progress["partial_traces"])
    completed = _successful_record_identities(records)
    batch_index = progress["next_batch_index"]

    for language in ("en", "zh"):
        route_samples = [s for s in selected if s["language"] == language]

        def infer(batch: list[dict[str, Any]]) -> Any:
            raw = model.generate(
                input=[s["audio_path"] for s in batch],
                cache={},
                language=language,
                use_itn=False,
                batch_size=len(batch),
            )
            values = _coerce_output_sequence(raw, len(batch))
            return [rich_transcription_postprocess(_extract_text(v)) for v in values]

        new_records, batch_index = _run_batched_samples(
            engine="sensevoice",
            model_id=spec["models"]["sensevoice"]["repo_id"],
            route="utterance/no-vad/use_itn-false",
            samples=route_samples,
            batch_size=16,
            infer_batch=infer,
            completed=completed,
            run_id=run_id,
            progress_path=progress_path,
            batch_index=batch_index,
        )
        records = _merge_records(records, new_records)

    fixed_context_probe = _run_sensevoice_fixed_context_probe(
        model=model,
        samples=selected,
        output_dir=output_dir,
        postprocess=rich_transcription_postprocess,
        np=np,
        sf=sf,
    )
    context_policy = _sensevoice_context_policy(selected, fixed_context_probe)
    partial_capability = _utterance_only_metadata(
        "A separate exact 10 s clean-context probe invokes the same pinned "
        "FunASR final-only API; it exposes no incremental callback."
    )
    partial_capability.update(
        {
            "fixed_context_probe_valid": fixed_context_probe["valid"],
            "fixed_context_probe_receipt_sha256": fixed_context_probe[
                "receipt_sha256"
            ],
            "fixed_context_probe_n_per_language": (
                _SENSEVOICE_FIXED_CONTEXT_PROBE_N
            ),
            "fixed_context_probe_scoring_included": False,
        }
    )
    metadata.update(
        {
            "artifacts": [artifact_meta],
            "context_policy": context_policy,
            "fixed_context_probe": fixed_context_probe,
            "partial_capability": partial_capability,
            "settings": {
                "languages": ["en", "zh"],
                "use_itn": False,
                "vad_model": None,
                "batch_size": 16,
                "model_update_guard": update_guard,
                "context_policy": context_policy,
            },
            "runtime": _runtime_metadata(
                ["funasr", "numpy", "soundfile", "torch"]
            ),
        }
    )
    return {"records": records, "partial_traces": traces, "metadata": metadata}


def _run_nemo(
    samples: list[dict[str, Any]],
    output_dir: Path,
    run_id: str,
    progress_path: Path,
    progress: dict[str, Any],
) -> dict[str, Any]:
    spec = ENGINE_SPECS["nemo"]
    selected = [s for s in samples if s["language"] in spec["languages"]]
    metadata = _route_metadata(spec, samples, selected)
    metadata["partial_capability"] = {
        "api": "utterance-only/no partial API",
        "partial_traces_retained": False,
        "parakeet_route": "fixed-window prebuilt route selected",
        "separate_nemo_streaming_script_exists": True,
        "streaming_script_used": False,
        "canary_route": "AED over a complete utterance",
    }
    if not selected:
        return {
            "records": progress["records"],
            "partial_traces": progress["partial_traces"],
            "metadata": metadata,
        }

    _require_cuda("nemo")
    import torch  # type: ignore[import-not-found]
    import nemo.collections.asr as nemo_asr  # type: ignore[import-not-found]
    from omegaconf import OmegaConf  # type: ignore[import-not-found]

    records = list(progress["records"])
    traces = list(progress["partial_traces"])
    completed = _successful_record_identities(records)
    batch_index = progress["next_batch_index"]
    artifacts: list[dict[str, Any]] = []

    parakeet_dir, parakeet_artifact = _download_model(
        "nemo", "parakeet", output_dir
    )
    artifacts.append(parakeet_artifact)
    tokenizer_audit = _inspect_parakeet_tokenizer(parakeet_dir / "tokenizer.json")
    parakeet_spec = spec["models"]["parakeet"]
    parakeet_samples = [
        s for s in samples if s["language"] in parakeet_spec["languages"]
    ]
    if parakeet_samples:
        checkpoint = parakeet_dir / parakeet_spec["checkpoint"]
        model = nemo_asr.models.ASRModel.restore_from(
            restore_path=str(checkpoint), map_location=torch.device("cuda:0")
        )
        model.eval()

        def infer_parakeet(batch: list[dict[str, Any]]) -> Any:
            # Deliberately no language argument: this checkpoint performs LID.
            with torch.inference_mode():
                return model.transcribe(
                    [s["audio_path"] for s in batch],
                    batch_size=len(batch),
                    timestamps=False,
                )

        new_records, batch_index = _run_batched_samples(
            engine="nemo",
            model_id=parakeet_spec["repo_id"],
            route="fixed-window utterance/no-language-argument",
            samples=parakeet_samples,
            batch_size=8,
            infer_batch=infer_parakeet,
            completed=completed,
            run_id=run_id,
            progress_path=progress_path,
            batch_index=batch_index,
        )
        records = _merge_records(records, new_records)
        del model
        _empty_cuda_cache(torch)

    canary_dir, canary_artifact = _download_model("nemo", "canary", output_dir)
    artifacts.append(canary_artifact)
    canary_spec = spec["models"]["canary"]
    canary_samples = [
        s for s in samples if s["language"] in canary_spec["languages"]
    ]
    if canary_samples:
        checkpoint = canary_dir / canary_spec["checkpoint"]
        canary_cfg = nemo_asr.models.ASRModel.restore_from(
            restore_path=str(checkpoint), return_config=True
        )
        # Canary's optional timestamp helper is a separate neural model.  This
        # sprint selected timestamps=False, so prevent construction/download
        # of that unpinned helper before instantiating the checkpoint.
        OmegaConf.update(
            canary_cfg,
            "restore_timestamps_model",
            False,
            merge=False,
            force_add=True,
        )
        model = nemo_asr.models.ASRModel.restore_from(
            restore_path=str(checkpoint),
            override_config_path=canary_cfg,
            map_location=torch.device("cuda:0"),
        )
        model.eval()
        for language in ("en", "es", "fr"):
            language_samples = [
                s for s in canary_samples if s["language"] == language
            ]

            def infer_canary(
                batch: list[dict[str, Any]], language: str = language
            ) -> Any:
                with torch.inference_mode():
                    return model.transcribe(
                        [s["audio_path"] for s in batch],
                        batch_size=len(batch),
                        source_lang=language,
                        target_lang=language,
                        pnc="no",
                        timestamps=False,
                    )

            new_records, batch_index = _run_batched_samples(
                engine="nemo",
                model_id=canary_spec["repo_id"],
                route="utterance-aed/source-equals-target/pnc-no",
                samples=language_samples,
                batch_size=8,
                infer_batch=infer_canary,
                completed=completed,
                run_id=run_id,
                progress_path=progress_path,
                batch_index=batch_index,
            )
            records = _merge_records(records, new_records)
        del model
        _empty_cuda_cache(torch)

    metadata.update(
        {
            "artifacts": artifacts,
            "parakeet_tokenizer": tokenizer_audit,
            "parakeet_tokenizer_audit": tokenizer_audit,
            "settings": {
                "parakeet": {
                    "languages": ["es", "fr", "ru"],
                    "language_argument": None,
                    "timestamps": False,
                    "route": "fixed-window utterance",
                },
                "canary": {
                    "languages": ["en", "es", "fr"],
                    "source_lang_equals_target_lang": True,
                    "pnc": "no",
                    "timestamps": False,
                    "restore_timestamps_model": False,
                },
                "batch_size": 8,
            },
            "runtime": _runtime_metadata(
                ["nemo_toolkit", "torch", "tokenizers", "huggingface_hub"]
            ),
        }
    )
    return {"records": records, "partial_traces": traces, "metadata": metadata}


def _run_zipformer(
    samples: list[dict[str, Any]],
    output_dir: Path,
    run_id: str,
    progress_path: Path,
    progress: dict[str, Any],
) -> dict[str, Any]:
    spec = ENGINE_SPECS["zipformer"]
    selected = [s for s in samples if s["language"] == "zh"]
    metadata = _route_metadata(spec, samples, selected)
    if not selected:
        return {
            "records": progress["records"],
            "partial_traces": progress["partial_traces"],
            "metadata": metadata,
        }

    model_dir, artifact_meta = _download_model("zipformer", "zipformer", output_dir)
    import numpy as np  # type: ignore[import-not-found]
    import sherpa_onnx  # type: ignore[import-not-found]
    import soundfile as sf  # type: ignore[import-not-found]
    from scipy.signal import resample_poly  # type: ignore[import-not-found]

    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=str(model_dir / "data/lang_char_bpe/tokens.txt"),
        encoder=str(model_dir / "exp/encoder-epoch-99-avg-1.onnx"),
        decoder=str(model_dir / "exp/decoder-epoch-99-avg-1.onnx"),
        joiner=str(model_dir / "exp/joiner-epoch-99-avg-1.onnx"),
        num_threads=2,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="modified_beam_search",
        max_active_paths=4,
        enable_endpoint_detection=False,
        provider="cpu",
        model_type="zipformer",
    )

    records = list(progress["records"])
    traces = list(progress["partial_traces"])
    completed = _successful_record_identities(records)
    batch_index = progress["next_batch_index"]
    clean_for_trace = {
        _sample_identity(s)
        for s in [x for x in selected if x["condition"] == "clean"][:30]
    }

    for sample in selected:
        record_identity = _record_identity(
            spec["models"]["zipformer"]["repo_id"], sample
        )
        if record_identity in completed:
            continue
        retain_trace = _sample_identity(sample) in clean_for_trace
        batch_records: list[dict[str, Any]] = []
        batch_traces: list[dict[str, Any]] = []
        started = time.perf_counter()
        try:
            waveform, sample_rate = sf.read(
                sample["audio_path"], dtype="float32", always_2d=False
            )
            waveform = np.asarray(waveform, dtype=np.float32)
            if waveform.ndim == 2:
                waveform = waveform.mean(axis=1, dtype=np.float32)
            if waveform.ndim != 1:
                raise ValueError(f"Expected mono/2-D audio, got shape {waveform.shape}")
            if int(sample_rate) != 16000:
                divisor = math.gcd(int(sample_rate), 16000)
                waveform = resample_poly(
                    waveform, 16000 // divisor, int(sample_rate) // divisor
                ).astype(np.float32, copy=False)
                sample_rate = 16000
            waveform = np.ascontiguousarray(waveform, dtype=np.float32)
            stream_started = time.perf_counter()
            hypothesis, emitted_traces, first_partial_s = _decode_zipformer_stream(
                recognizer=recognizer,
                waveform=waveform,
                sample=sample,
                retain_trace=retain_trace,
                np=np,
                started=stream_started,
            )
            elapsed = time.perf_counter() - started
            record = _success_record(
                engine="zipformer",
                model_id=spec["models"]["zipformer"]["repo_id"],
                route="online-transducer/modified-beam-search",
                sample=sample,
                hypothesis=hypothesis,
                batch_wall_s=elapsed,
                batch_size=1,
            )
            record.update(
                {
                    "partial_trace_retained": retain_trace,
                    "first_partial_wall_s": first_partial_s,
                    "first_partial_wall_origin": (
                        "stream feed start after offline audio load/resample"
                    ),
                    "feed_pacing": (
                        "real_time_cumulative_100ms_deadlines"
                        if retain_trace
                        else "accelerated"
                    ),
                    "tail_silence_s": 0.66,
                }
            )
            batch_records.append(record)
            batch_traces.extend(emitted_traces)
        except Exception as exc:  # preserve progress and continue the corpus
            elapsed = time.perf_counter() - started
            batch_records.append(
                _error_record(
                    engine="zipformer",
                    model_id=spec["models"]["zipformer"]["repo_id"],
                    route="online-transducer/modified-beam-search",
                    sample=sample,
                    exc=exc,
                    batch_wall_s=elapsed,
                    batch_size=1,
                )
            )
        _checkpoint_batch(
            progress_path,
            run_id,
            "zipformer",
            batch_index,
            batch_records,
            batch_traces,
        )
        batch_index += 1
        records = _merge_records(records, batch_records)
        traces.extend(batch_traces)
        if batch_records[-1].get("status") == "ok":
            completed.add(record_identity)

    metadata.update(
        {
            "artifacts": [artifact_meta],
            "settings": dict(spec["inference"]),
            "partial_capability": {
                "api": "genuine online partial API",
                "partial_traces_retained": True,
                "trace_interval_s": 0.1,
                "scope": "first 30 clean Chinese samples",
                "tail_silence_s": 0.66,
                "input_finished_called": True,
                "retained_trace_pacing": {
                    "mode": "real_time_cumulative_deadlines",
                    "nominal_chunk_s": 0.1,
                    "scope": "audio plus synthetic tail silence",
                    "sleep_before_accept": True,
                    "deadline_basis": "cumulative samples / 16000 Hz",
                    "decoder_compute_consumes_deadline_budget": True,
                },
                "non_retained_evaluation_pacing": "accelerated/no pacing sleeps",
                "first_partial_latency_semantics": (
                    "monotonic wall time from stream feed start; retained traces "
                    "therefore include real-time audio arrival"
                ),
            },
            "artifact_caveats": dict(spec["artifact_caveats"]),
            "runtime": _runtime_metadata(
                ["sherpa-onnx", "numpy", "scipy", "soundfile"]
            ),
        }
    )
    return {"records": records, "partial_traces": traces, "metadata": metadata}


def _run_gigaam(
    samples: list[dict[str, Any]],
    output_dir: Path,
    run_id: str,
    progress_path: Path,
    progress: dict[str, Any],
) -> dict[str, Any]:
    spec = ENGINE_SPECS["gigaam"]
    selected = [s for s in samples if s["language"] == "ru"]
    metadata = _route_metadata(spec, samples, selected)
    metadata["partial_capability"] = _utterance_only_metadata(
        "GigaAM exposes model.transcribe(path), not incremental hypotheses."
    )
    if not selected:
        return {
            "records": progress["records"],
            "partial_traces": progress["partial_traces"],
            "metadata": metadata,
        }

    _require_cuda("gigaam")
    model_dir, artifact_meta = _download_model("gigaam", "gigaam", output_dir)
    import torch  # type: ignore[import-not-found]
    from transformers import AutoModel  # type: ignore[import-not-found]

    model = AutoModel.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=True,
    )
    if hasattr(model, "to"):
        model = model.to("cuda:0")
    if hasattr(model, "eval"):
        model.eval()

    def infer(batch: list[dict[str, Any]]) -> Any:
        outputs = []
        with torch.inference_mode():
            for sample in batch:
                value = model.transcribe(sample["audio_path"])
                # The pinned API returns an object with .text.  The fallback
                # keeps the evidence run usable if the remote class returns a
                # string while still recording the same transcript.
                outputs.append(value.text if hasattr(value, "text") else value)
        return outputs

    records = list(progress["records"])
    completed = _successful_record_identities(records)
    new_records, _ = _run_batched_samples(
        engine="gigaam",
        model_id=spec["models"]["gigaam"]["repo_id"],
        route="utterance/model.transcribe(path).text",
        samples=selected,
        batch_size=4,
        infer_batch=infer,
        completed=completed,
        run_id=run_id,
        progress_path=progress_path,
        batch_index=progress["next_batch_index"],
    )
    records = _merge_records(records, new_records)
    metadata.update(
        {
            "artifacts": [artifact_meta],
            "settings": {
                "language": "ru",
                "trust_remote_code": True,
                "local_files_only": True,
                "call": "model.transcribe(path).text",
                "batch_size": 4,
            },
            "runtime": _runtime_metadata(
                [
                    "transformers",
                    "torch",
                    "hydra-core",
                    "omegaconf",
                    "pyannote.core",
                ]
            ),
        }
    )
    return {
        "records": records,
        "partial_traces": progress["partial_traces"],
        "metadata": metadata,
    }


def _run_omni(
    samples: list[dict[str, Any]],
    output_dir: Path,
    run_id: str,
    progress_path: Path,
    progress: dict[str, Any],
) -> dict[str, Any]:
    spec = ENGINE_SPECS["omni"]
    selected = [s for s in samples if s["language"] in spec["languages"]]
    metadata = _route_metadata(spec, samples, selected)
    metadata["partial_capability"] = _utterance_only_metadata(
        "The omnilingual-asr 0.2.0 reference pipeline returns utterance batches."
    )
    if not selected:
        return {
            "records": progress["records"],
            "partial_traces": progress["partial_traces"],
            "metadata": metadata,
        }

    _require_cuda("omni")
    model_dir, artifact_meta = _download_model("omni", "omni", output_dir)
    asset_dir, asset_card = _write_omni_asset_card(model_dir, output_dir)
    # fairseq2 reads this while its global runtime context is initialized, so it
    # must be set before importing fairseq2 or omnilingual_asr.
    os.environ["FAIRSEQ2_USER_ASSET_DIR"] = str(asset_dir)

    import torch  # type: ignore[import-not-found]

    if torch.__version__.split("+")[0] != "2.8.0":
        raise RuntimeError(
            f"Omni requires torch 2.8.0 (cu128); found {torch.__version__}. "
            "Start a fresh Colab runtime and call install_engine('omni') first."
        )
    if getattr(torch.version, "cuda", None) != "12.8":
        raise RuntimeError(
            f"Omni requires the CUDA 12.8 Torch build; found torch CUDA "
            f"{getattr(torch.version, 'cuda', None)!r}."
        )

    from omnilingual_asr.models.inference.pipeline import (  # type: ignore[import-not-found]
        ASRInferencePipeline,
    )

    pipeline = ASRInferencePipeline(
        model_card=spec["inference"]["package_model_card"],
        device="cuda",
        dtype=torch.float16,
    )
    records = list(progress["records"])
    completed = _successful_record_identities(records)
    batch_index = progress["next_batch_index"]
    gate_results: dict[str, dict[str, Any]] = {}
    model_id = spec["models"]["omni"]["repo_id"]
    language_codes = spec["inference"]["language_codes"]

    def infer(batch: list[dict[str, Any]]) -> Any:
        paths = [s["audio_path"] for s in batch]
        langs = [language_codes[s["language"]] for s in batch]
        with torch.inference_mode():
            return pipeline.transcribe(paths, lang=langs, batch_size=len(batch))

    records, batch_index, gate_results = _run_omni_two_phase_samples(
        selected=selected,
        records=records,
        completed=completed,
        model_id=model_id,
        infer_batch=infer,
        run_id=run_id,
        progress_path=progress_path,
        batch_index=batch_index,
        smoke_n=int(spec["inference"]["smoke_clean_n_per_language"]),
    )

    metadata.update(
        {
            "artifacts": [artifact_meta],
            "fairseq2_asset_card": {
                "path": str(asset_card),
                "directory": str(asset_dir),
                "environment_variable": "FAIRSEQ2_USER_ASSET_DIR",
                "card_name": spec["inference"]["package_model_card"],
                "card_version": "v1",
                "uses_local_file_uris": True,
            },
            "settings": {
                "device": "cuda",
                "dtype": "float16",
                "batch_size": 8,
                "language_codes": dict(language_codes),
                "note": "CTC ignores language conditioning; codes are retained as route provenance.",
                "execution_order": (
                    "all five languages complete the clean smoke phase before any "
                    "non-collapsed language expands to remaining clean and noisy samples"
                ),
            },
            "smoke_execution": {
                "languages_in_order": list(_LANGUAGES),
                "clean_n_per_language": int(
                    spec["inference"]["smoke_clean_n_per_language"]
                ),
                "all_language_smokes_before_expansion": True,
                "expansion_policy": "only languages whose smoke gate did not collapse",
                "resumability": "record identity checkpoints are reused in both phases",
            },
            "smoke_gate": gate_results,
            "runtime": _runtime_metadata(
                ["omnilingual-asr", "fairseq2", "fairseq2n", "torch", "torchaudio"]
            ),
        }
    )
    return {
        "records": records,
        "partial_traces": progress["partial_traces"],
        "metadata": metadata,
    }


def _run_omni_two_phase_samples(
    *,
    selected: list[dict[str, Any]],
    records: list[dict[str, Any]],
    completed: set[tuple[str, str, str, str]],
    model_id: str,
    infer_batch: Callable[[list[dict[str, Any]]], Any],
    run_id: str,
    progress_path: Path,
    batch_index: int,
    smoke_n: int,
) -> tuple[list[dict[str, Any]], int, dict[str, dict[str, Any]]]:
    if smoke_n <= 0:
        raise ValueError("Omni smoke sample count must be positive")

    all_records = list(records)
    language_samples = {
        language: [s for s in selected if s["language"] == language]
        for language in _LANGUAGES
    }
    smoke_samples = {
        language: [
            s
            for s in language_samples[language]
            if s["condition"] == "clean"
        ][:smoke_n]
        for language in _LANGUAGES
    }

    # Phase 1 is deliberately global: no language may consume its remaining
    # clean/noisy budget before every language's collapse gate has evidence.
    for language in _LANGUAGES:
        new_records, batch_index = _run_batched_samples(
            engine="omni",
            model_id=model_id,
            route="ctc-utterance/fp16-cuda/smoke",
            samples=smoke_samples[language],
            batch_size=8,
            infer_batch=infer_batch,
            completed=completed,
            run_id=run_id,
            progress_path=progress_path,
            batch_index=batch_index,
            record_tags={"phase": "smoke", "smoke_test": True},
        )
        all_records = _merge_records(all_records, new_records)

    gate_results: dict[str, dict[str, Any]] = {}
    for language in _LANGUAGES:
        smoke_ids = {
            _record_identity(model_id, sample)
            for sample in smoke_samples[language]
        }
        observed_smoke = [
            record
            for record in all_records
            if record.get("engine") == "omni"
            and record.get("model") == model_id
            and record.get("status") == "ok"
            and _record_identity_from_record(record) in smoke_ids
        ]
        gate_results[language] = _collapse_gate(
            language,
            observed_smoke,
            expected_n=len(smoke_samples[language]),
        )

    # Phase 2 begins only after all five gates above are materialized.  Resume
    # checkpoints simply remove already-completed identities from either phase.
    for language in _LANGUAGES:
        gate = gate_results[language]
        smoke_identities = {
            _sample_identity(sample) for sample in smoke_samples[language]
        }
        remaining = [
            sample
            for sample in language_samples[language]
            if _sample_identity(sample) not in smoke_identities
        ]
        gate["remaining_samples_total"] = len(remaining)
        if gate["collapsed"]:
            gate["remaining_samples_evaluated"] = 0
            gate["remaining_samples_skipped"] = len(remaining)
            continue

        new_records, batch_index = _run_batched_samples(
            engine="omni",
            model_id=model_id,
            route="ctc-utterance/fp16-cuda/full",
            samples=remaining,
            batch_size=8,
            infer_batch=infer_batch,
            completed=completed,
            run_id=run_id,
            progress_path=progress_path,
            batch_index=batch_index,
            record_tags={"phase": "evaluation", "smoke_test": False},
        )
        all_records = _merge_records(all_records, new_records)
        remaining_ids = {_record_identity(model_id, sample) for sample in remaining}
        gate["remaining_samples_evaluated"] = sum(
            record.get("engine") == "omni"
            and record.get("model") == model_id
            and record.get("status") == "ok"
            and _record_identity_from_record(record) in remaining_ids
            for record in all_records
        )
        gate["remaining_samples_skipped"] = 0

    return all_records, batch_index, gate_results


def _run_batched_samples(
    *,
    engine: str,
    model_id: str,
    route: str,
    samples: list[dict[str, Any]],
    batch_size: int,
    infer_batch: Callable[[list[dict[str, Any]]], Any],
    completed: set[tuple[str, str, str, str]],
    run_id: str,
    progress_path: Path,
    batch_index: int,
    record_tags: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    pending = [s for s in samples if _record_identity(model_id, s) not in completed]
    records: list[dict[str, Any]] = []
    for batch in _chunks(pending, batch_size):
        outcomes = _infer_resilient(infer_batch, batch)
        batch_records: list[dict[str, Any]] = []
        for sample, outcome in zip(batch, outcomes):
            if outcome["error"] is None:
                record = _success_record(
                    engine=engine,
                    model_id=model_id,
                    route=route,
                    sample=sample,
                    hypothesis=outcome["text"],
                    batch_wall_s=outcome["batch_wall_s"],
                    batch_size=outcome["batch_size"],
                )
            else:
                record = _error_record(
                    engine=engine,
                    model_id=model_id,
                    route=route,
                    sample=sample,
                    exc=outcome["error"],
                    batch_wall_s=outcome["batch_wall_s"],
                    batch_size=outcome["batch_size"],
                )
            if record_tags:
                record.update(record_tags)
            batch_records.append(record)
            if record.get("status") == "ok":
                completed.add(_record_identity(model_id, sample))
        _checkpoint_batch(
            progress_path,
            run_id,
            engine,
            batch_index,
            batch_records,
            [],
        )
        batch_index += 1
        records.extend(batch_records)
    return records, batch_index


def _infer_resilient(
    infer_batch: Callable[[list[dict[str, Any]]], Any],
    batch: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    started = time.perf_counter()
    try:
        raw = infer_batch(batch)
        elapsed = time.perf_counter() - started
        values = _coerce_output_sequence(raw, len(batch))
        return [
            {
                "text": _extract_text(value),
                "error": None,
                "batch_wall_s": elapsed,
                "batch_size": len(batch),
            }
            for value in values
        ]
    except Exception as batch_exc:
        if len(batch) == 1:
            return [
                {
                    "text": "",
                    "error": batch_exc,
                    "batch_wall_s": time.perf_counter() - started,
                    "batch_size": 1,
                }
            ]

    # Isolate a malformed/corrupt sample rather than losing an entire batch.
    outcomes: list[dict[str, Any]] = []
    for sample in batch:
        item_started = time.perf_counter()
        try:
            raw = infer_batch([sample])
            value = _coerce_output_sequence(raw, 1)[0]
            outcomes.append(
                {
                    "text": _extract_text(value),
                    "error": None,
                    "batch_wall_s": time.perf_counter() - item_started,
                    "batch_size": 1,
                }
            )
        except Exception as exc:
            outcomes.append(
                {
                    "text": "",
                    "error": exc,
                    "batch_wall_s": time.perf_counter() - item_started,
                    "batch_size": 1,
                }
            )
    return outcomes


def _decode_zipformer_stream(
    *,
    recognizer: Any,
    waveform: Any,
    sample: Mapping[str, Any],
    retain_trace: bool,
    np: Any,
    started: float,
) -> tuple[str, list[dict[str, Any]], float | None]:
    sample_rate = 16000
    nominal_chunk_s = 0.1
    chunk_samples = int(round(nominal_chunk_s * sample_rate))
    tail_samples = int(round(0.66 * sample_rate))
    stream = recognizer.create_stream()
    traces: list[dict[str, Any]] = []
    first_partial_s: float | None = None
    stream_samples = 0
    pacing_mode = (
        "real_time_cumulative_deadlines" if retain_trace else "accelerated"
    )

    def decode_ready() -> str:
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
        return str(recognizer.get_result(stream)).strip()

    def accept_at_deadline(chunk: Any) -> tuple[str, float]:
        nonlocal stream_samples
        next_stream_samples = stream_samples + len(chunk)
        if retain_trace:
            deadline = started + (next_stream_samples / sample_rate)
            delay = deadline - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        stream.accept_waveform(sample_rate, chunk)
        stream_samples = next_stream_samples
        text = decode_ready()
        return text, time.perf_counter() - started

    for offset in range(0, len(waveform), chunk_samples):
        chunk = waveform[offset : offset + chunk_samples]
        text, wall_s = accept_at_deadline(chunk)
        if text and first_partial_s is None:
            first_partial_s = wall_s
        if retain_trace:
            traces.append(
                _partial_trace(
                    sample=sample,
                    text=text,
                    wall_s=wall_s,
                    stream_time_s=stream_samples / sample_rate,
                    audio_time_s=min(len(waveform), stream_samples) / sample_rate,
                    phase="audio",
                    is_final=False,
                    pacing_mode=pacing_mode,
                    nominal_chunk_s=nominal_chunk_s,
                    chunk_duration_s=len(chunk) / sample_rate,
                )
            )

    silence = np.zeros(tail_samples, dtype=np.float32)
    for offset in range(0, tail_samples, chunk_samples):
        chunk = silence[offset : offset + chunk_samples]
        text, wall_s = accept_at_deadline(chunk)
        if text and first_partial_s is None:
            first_partial_s = wall_s
        if retain_trace:
            traces.append(
                _partial_trace(
                    sample=sample,
                    text=text,
                    wall_s=wall_s,
                    stream_time_s=stream_samples / sample_rate,
                    audio_time_s=len(waveform) / sample_rate,
                    phase="tail_silence",
                    is_final=False,
                    pacing_mode=pacing_mode,
                    nominal_chunk_s=nominal_chunk_s,
                    chunk_duration_s=len(chunk) / sample_rate,
                )
            )
    stream.input_finished()
    final_text = decode_ready()
    input_finished_wall_s = time.perf_counter() - started
    if retain_trace:
        trace_container = {
            "schema_version": _SCHEMA_VERSION,
            "engine": "zipformer",
            "model": ENGINE_SPECS["zipformer"]["models"]["zipformer"]["repo_id"],
            "bank": "L-fork",
            "revision": ENGINE_SPECS["zipformer"]["models"]["zipformer"]["revision"],
            "route": "Zipformer online",
            "utterance_id": sample["utterance_id"],
            "language": sample["language"],
            "condition": sample["condition"],
            "audio_path": sample["audio_path"],
            "duration_s": sample["duration_s"],
            "first_partial_latency_s": first_partial_s,
            "final_hypothesis": final_text,
            "input_finished_wall_s": input_finished_wall_s,
            "input_finished_stream_time_s": stream_samples / sample_rate,
            "tail_silence_s": 0.66,
            "trace_interval_s": nominal_chunk_s,
            "feed_pacing": {
                "applied": True,
                "mode": pacing_mode,
                "nominal_chunk_s": nominal_chunk_s,
                "scope": "audio plus synthetic tail silence",
                "sleep_before_accept": True,
                "deadline_basis": "cumulative samples / 16000 Hz",
                "decoder_compute_consumes_deadline_budget": True,
            },
            "first_partial_latency_semantics": (
                "monotonic wall time from stream feed start, including real-time "
                "audio arrival but excluding offline file load/resample"
            ),
            "partials": traces,
        }
        trace_container["trace_sha256"] = hashlib.sha256(
            json.dumps(
                trace_container, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        return final_text, [trace_container], first_partial_s
    return final_text, [], first_partial_s


def _partial_trace(
    *,
    sample: Mapping[str, Any],
    text: str,
    wall_s: float,
    stream_time_s: float,
    audio_time_s: float,
    phase: str,
    is_final: bool,
    pacing_mode: str,
    nominal_chunk_s: float,
    chunk_duration_s: float,
) -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "engine": "zipformer",
        "model": ENGINE_SPECS["zipformer"]["models"]["zipformer"]["repo_id"],
        "utterance_id": sample["utterance_id"],
        "language": sample["language"],
        "condition": sample["condition"],
        "audio_path": sample["audio_path"],
        "stream_time_s": stream_time_s,
        "audio_time_s": audio_time_s,
        "wall_elapsed_s": wall_s,
        "t3p_s": wall_s,
        "source_audio_consumed_s": audio_time_s,
        "audio_consumed_s": audio_time_s,
        "text": text,
        "phase": phase,
        "is_final": is_final,
        "feed_pacing_mode": pacing_mode,
        "feed_pacing_applied": pacing_mode != "accelerated",
        "nominal_feed_chunk_s": nominal_chunk_s,
        "feed_chunk_duration_s": chunk_duration_s,
        "feed_deadline_s": stream_time_s,
        "wall_latency_origin": "stream feed start",
    }


def _download_model(
    engine: str, model_key: str, output_dir: Path
) -> tuple[Path, dict[str, Any]]:
    from huggingface_hub import snapshot_download  # type: ignore[import-not-found]

    model_spec = ENGINE_SPECS[engine]["models"][model_key]
    root = _model_cache_root(output_dir) / engine
    safe_repo = model_spec["repo_id"].replace("/", "--")
    model_dir = root / f"{safe_repo}--{model_spec['revision']}"
    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_spec["repo_id"],
        revision=model_spec["revision"],
        local_dir=str(model_dir),
        allow_patterns=list(model_spec["allow_patterns"]),
    )
    verified_files = []
    for relative_path, expected in model_spec["files"].items():
        path = model_dir / relative_path
        expected_bytes = int(model_spec["file_sizes"][relative_path])
        observed_bytes = path.stat().st_size if path.is_file() else None
        if observed_bytes != expected_bytes:
            raise RuntimeError(
                f"Byte-size mismatch for {path}: expected {expected_bytes}, "
                f"got {observed_bytes}"
            )
        observed = _sha256_file(path)
        if observed != expected:
            raise RuntimeError(
                f"SHA-256 mismatch for {path}: expected {expected}, got {observed}"
            )
        verified_files.append(
            {
                "path": str(path),
                "relative_path": relative_path,
                "sha256_expected": expected,
                "sha256_observed": observed,
                "sha256_verified": True,
                "bytes_expected": expected_bytes,
                "bytes_observed": observed_bytes,
                "bytes_verified": True,
            }
        )
    return model_dir, {
        "repo_id": model_spec["repo_id"],
        "revision": model_spec["revision"],
        "local_path": str(model_dir),
        "files": verified_files,
        "pin_class": "exact commit plus exact SHA-256",
    }


def _write_omni_asset_card(model_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    asset_dir = output_dir / "artifacts" / "omni_fairseq2_assets_v1"
    asset_dir.mkdir(parents=True, exist_ok=True)
    card = asset_dir / "lane_c_omni_v1.yaml"
    checkpoint_uri = (model_dir / "omniASR-CTC-300M.pt").resolve().as_uri()
    tokenizer_uri = (model_dir / "omniASR_tokenizer.model").resolve().as_uri()
    text = (
        "name: lane_c_omniASR_tokenizer_v1_pinned\n"
        "tokenizer_family: char_tokenizer\n"
        f"tokenizer: {tokenizer_uri}\n"
        "\n---\n\n"
        "name: lane_c_omniASR_CTC_300M_v1_pinned\n"
        "model_family: wav2vec2_asr\n"
        "model_arch: 300m\n"
        f"checkpoint: {checkpoint_uri}\n"
        "tokenizer_ref: lane_c_omniASR_tokenizer_v1_pinned\n"
    )
    if not card.exists() or card.read_text(encoding="utf-8") != text:
        card.write_text(text, encoding="utf-8")
    return asset_dir, card


def _inspect_parakeet_tokenizer(path: Path) -> dict[str, Any]:
    observed_sha = _sha256_file(path)
    expected_sha = ENGINE_SPECS["nemo"]["models"]["parakeet"]["files"][
        "tokenizer.json"
    ]
    if observed_sha != expected_sha:
        raise RuntimeError(
            f"Parakeet tokenizer SHA-256 mismatch: expected {expected_sha}, "
            f"got {observed_sha}"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    vocab_obj = data.get("model", {}).get("vocab", {})
    if isinstance(vocab_obj, dict):
        vocab = vocab_obj
    elif isinstance(vocab_obj, list):
        vocab = {str(token): idx for idx, token in enumerate(vocab_obj)}
    else:
        raise ValueError("tokenizer.json model.vocab is neither a mapping nor a list")

    token_counts = {"cjk": 0, "kana": 0, "hangul": 0}
    codepoint_counts = {"cjk": 0, "kana": 0, "hangul": 0}
    for token in vocab:
        present = {"cjk": False, "kana": False, "hangul": False}
        for char in token:
            script = _character_script(char)
            if script in present:
                present[script] = True
                codepoint_counts[script] += 1
        for script, found in present.items():
            if found:
                token_counts[script] += 1

    from tokenizers import Tokenizer  # type: ignore[import-not-found]

    tokenizer = Tokenizer.from_file(str(path))
    probe_text = "你好世界"
    encoded = tokenizer.encode(probe_text)
    decoded = tokenizer.decode(encoded.ids)
    return {
        "artifact": str(path),
        "sha256_expected": expected_sha,
        "sha256_observed": observed_sha,
        "sha256_verified": True,
        "vocab_size_expected": 8192,
        "vocab_size_observed": len(vocab),
        "vocab_size_matches": len(vocab) == 8192,
        "script_token_counts": token_counts,
        "script_codepoint_occurrences": codepoint_counts,
        "cjk_absent_from_vocabulary": token_counts["cjk"] == 0,
        "special_token_zh": {
            "token": "<|zh|>",
            "present": "<|zh|>" in vocab,
            "id": vocab.get("<|zh|>"),
            "note": "A language-control token is not a Chinese text vocabulary.",
        },
        "probe": {
            "text": probe_text,
            "ids": list(encoded.ids),
            "tokens": list(encoded.tokens),
            "decoded": decoded,
            "round_trip_equal": decoded == probe_text,
        },
    }


def _collapse_gate(
    language: str, records: list[dict[str, Any]], expected_n: int
) -> dict[str, Any]:
    hypotheses = [str(r.get("hypothesis", "")) for r in records]
    empty_count = sum(not h.strip() for h in hypotheses)
    wrong_script_count = sum(
        bool(h.strip()) and _wrong_script(h, language) for h in hypotheses
    )
    denominator = len(records)
    empty_fraction = empty_count / denominator if denominator else 1.0
    wrong_script_fraction = wrong_script_count / denominator if denominator else 1.0
    incomplete = denominator < expected_n
    collapsed = (
        incomplete or empty_fraction > 0.20 or wrong_script_fraction > 0.50
    )
    reasons = []
    if incomplete:
        reasons.append(f"smoke incomplete: {denominator}/{expected_n} records")
    if empty_fraction > 0.20:
        reasons.append(f"empty fraction {empty_fraction:.3f} > 0.20")
    if wrong_script_fraction > 0.50:
        reasons.append(f"wrong-script fraction {wrong_script_fraction:.3f} > 0.50")
    return {
        "language": language,
        "smoke_n_expected": expected_n,
        "smoke_n_observed": denominator,
        "empty_count": empty_count,
        "empty_fraction": empty_fraction,
        "wrong_script_count": wrong_script_count,
        "wrong_script_fraction": wrong_script_fraction,
        "thresholds": {
            "empty_fraction_gt": 0.20,
            "wrong_script_fraction_gt": 0.50,
        },
        "collapsed": collapsed,
        "reasons": reasons,
    }


def _wrong_script(text: str, language: str) -> bool:
    counts = {"latin": 0, "cyrillic": 0, "cjk": 0, "kana": 0, "hangul": 0}
    for char in text:
        script = _character_script(char)
        if script in counts:
            counts[script] += 1
    if language == "zh":
        expected = counts["cjk"]
        competing = counts["latin"] + counts["cyrillic"] + counts["kana"] + counts["hangul"]
    elif language == "ru":
        expected = counts["cyrillic"]
        competing = counts["latin"] + counts["cjk"] + counts["kana"] + counts["hangul"]
    else:
        expected = counts["latin"]
        competing = counts["cyrillic"] + counts["cjk"] + counts["kana"] + counts["hangul"]
    return expected == 0 or expected < competing


def _character_script(char: str) -> str:
    cp = ord(char)
    if (
        0x3400 <= cp <= 0x4DBF
        or 0x4E00 <= cp <= 0x9FFF
        or 0xF900 <= cp <= 0xFAFF
        or 0x20000 <= cp <= 0x323AF
    ):
        return "cjk"
    if 0x3040 <= cp <= 0x30FF or 0x31F0 <= cp <= 0x31FF:
        return "kana"
    if 0x1100 <= cp <= 0x11FF or 0x3130 <= cp <= 0x318F or 0xAC00 <= cp <= 0xD7AF:
        return "hangul"
    name = unicodedata.name(char, "")
    if "CYRILLIC" in name:
        return "cyrillic"
    if "LATIN" in name:
        return "latin"
    return "other"


def _success_record(
    *,
    engine: str,
    model_id: str,
    route: str,
    sample: Mapping[str, Any],
    hypothesis: str,
    batch_wall_s: float,
    batch_size: int,
) -> dict[str, Any]:
    amortized = batch_wall_s / max(1, batch_size)
    duration = float(sample["duration_s"])
    return {
        **_record_base(engine, model_id, route, sample),
        "status": "ok",
        "hypothesis": str(hypothesis).strip(),
        "batch_wall_s": batch_wall_s,
        "batch_size": batch_size,
        "amortized_wall_s": amortized,
        "amortized_rtf": amortized / duration if duration > 0 else None,
        "timing_note": "amortized batch wall time; not end-to-end latency",
        "evidence_class": "HOST",
    }


def _error_record(
    *,
    engine: str,
    model_id: str,
    route: str,
    sample: Mapping[str, Any],
    exc: BaseException,
    batch_wall_s: float,
    batch_size: int,
) -> dict[str, Any]:
    return {
        **_record_base(engine, model_id, route, sample),
        "status": "error",
        "hypothesis": "",
        "batch_wall_s": batch_wall_s,
        "batch_size": batch_size,
        "amortized_wall_s": batch_wall_s / max(1, batch_size),
        "amortized_rtf": None,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )[-12000:],
        "evidence_class": "HOST",
    }


def _record_base(
    engine: str, model_id: str, route: str, sample: Mapping[str, Any]
) -> dict[str, Any]:
    provenance = _model_provenance(model_id)
    return {
        "schema_version": _SCHEMA_VERSION,
        "engine": engine,
        "model": model_id,
        "bank": provenance["bank"],
        "revision": provenance["revision"],
        "route": route,
        "input_index": int(sample["_input_index"]),
        "utterance_id": sample["utterance_id"],
        "language": sample["language"],
        "condition": sample["condition"],
        "audio_path": sample["audio_path"],
        "reference": sample["reference"],
        "duration_s": sample["duration_s"],
        "duration_seconds": sample["duration_s"],
        **{
            key: sample[key]
            for key in _MIRRORED_SAMPLE_KEYS
            if key in sample
        },
    }


def _checkpoint_batch(
    path: Path,
    run_id: str,
    engine: str,
    batch_index: int,
    records: list[dict[str, Any]],
    traces: list[dict[str, Any]],
) -> None:
    _append_progress(
        path,
        {
            "schema_version": _SCHEMA_VERSION,
            "run_protocol_version": _RUN_PROTOCOL_VERSION,
            "event": "batch",
            "run_id": run_id,
            "engine": engine,
            "batch_index": batch_index,
            "timestamp_utc": _utc_now(),
            "records": records,
            "partial_traces": traces,
        },
    )


def _append_progress(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _load_progress(path: Path, run_id: str) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    max_batch = -1
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    # A final torn line can occur if a Colab VM is preempted
                    # between write and fsync.  Earlier complete batches remain
                    # valid; any other malformed line is ignored but rerun.
                    continue
                if entry.get("run_id") != run_id or entry.get("event") != "batch":
                    continue
                if entry.get("run_protocol_version") != _RUN_PROTOCOL_VERSION:
                    continue
                records.extend(entry.get("records", []))
                traces.extend(entry.get("partial_traces", []))
                max_batch = max(max_batch, int(entry.get("batch_index", -1)))
    return {
        "records": _dedupe_records(records),
        "partial_traces": _dedupe_traces(traces),
        "next_batch_index": max_batch + 1,
    }


def _route_metadata(
    spec: Mapping[str, Any],
    all_samples: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    supported = set(spec["languages"])
    skipped_counts = {language: 0 for language in _LANGUAGES}
    for sample in all_samples:
        if sample["language"] not in supported:
            skipped_counts[sample["language"]] += 1
    return {
        "supported_languages": list(spec["languages"]),
        "selected_sample_count": len(selected),
        "unsupported_sample_count": len(all_samples) - len(selected),
        "unsupported_samples_by_language": {
            k: v for k, v in skipped_counts.items() if v
        },
    }


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finalize_sensevoice_probe_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    finalized = dict(entry)
    finalized["entry_sha256"] = _canonical_json_sha256(finalized)
    return finalized


def _prepare_sensevoice_fixed_context_audio(
    sample: Mapping[str, Any],
    destination: Path,
    *,
    np: Any,
    sf: Any,
) -> tuple[dict[str, Any], Path | None]:
    entry: dict[str, Any] = {
        "utterance_id": str(sample["utterance_id"]),
        "language": str(sample["language"]),
        "condition": str(sample["condition"]),
        "audio_relpath": str(sample.get("audio_relpath", "")),
        "source_audio_pcm_sha256": sample.get("audio_pcm_sha256"),
        "source_clean_pcm_sha256": sample.get("clean_pcm_sha256"),
        "source_audio_file_sha256": sample.get("audio_file_sha256"),
        "source_reference_sha256": sample.get("reference_sha256"),
        "source_num_samples": sample.get("num_samples"),
        "source_sample_rate": sample.get("sample_rate"),
        "context_seconds": _SENSEVOICE_FIXED_CONTEXT_SECONDS,
        "derived_num_samples": _SENSEVOICE_FIXED_CONTEXT_NUM_SAMPLES,
        "derived_sample_rate": _SENSEVOICE_FIXED_CONTEXT_SAMPLE_RATE,
        "scoring_included": False,
        "reference_modified": False,
        "model_invoked": False,
        "status": "preparation_error",
    }
    try:
        if entry["condition"] != "clean":
            raise ValueError("fixed-context probe accepts clean fixture inputs only")
        waveform, sample_rate = sf.read(
            sample["audio_path"], dtype="float32", always_2d=False
        )
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim != 1:
            raise ValueError(
                f"fixed-context source must be canonical mono, got {waveform.shape}"
            )
        if int(sample_rate) != _SENSEVOICE_FIXED_CONTEXT_SAMPLE_RATE:
            raise ValueError(
                f"fixed-context source rate is {sample_rate}, expected "
                f"{_SENSEVOICE_FIXED_CONTEXT_SAMPLE_RATE}"
            )
        if int(sample.get("sample_rate", -1)) != int(sample_rate):
            raise ValueError("fixture/source sample-rate metadata mismatch")
        if int(sample.get("num_samples", -1)) != int(waveform.size):
            raise ValueError("fixture/source sample-count metadata mismatch")
        if not np.isfinite(waveform).all():
            raise ValueError("fixed-context source contains non-finite PCM")
        source_pcm = np.asarray(waveform, dtype="<f4").reshape(-1)
        source_pcm_sha256 = hashlib.sha256(
            source_pcm.tobytes(order="C")
        ).hexdigest()
        if source_pcm_sha256 != sample.get("audio_pcm_sha256"):
            raise ValueError("fixture/source PCM SHA-256 mismatch")
        if source_pcm_sha256 != sample.get("clean_pcm_sha256"):
            raise ValueError("clean fixture PCM aliases disagree")
        if _sha256_file(Path(str(sample["audio_path"]))) != sample.get(
            "audio_file_sha256"
        ):
            raise ValueError("fixture/source WAV SHA-256 mismatch")
        reference = str(sample.get("reference", ""))
        if hashlib.sha256(reference.encode("utf-8")).hexdigest() != sample.get(
            "reference_sha256"
        ):
            raise ValueError("fixture/source reference SHA-256 mismatch")

        fixed = np.zeros(
            _SENSEVOICE_FIXED_CONTEXT_NUM_SAMPLES, dtype=np.float32
        )
        copied = min(int(waveform.size), _SENSEVOICE_FIXED_CONTEXT_NUM_SAMPLES)
        fixed[:copied] = waveform[:copied]
        if waveform.size < _SENSEVOICE_FIXED_CONTEXT_NUM_SAMPLES:
            transformation = "right_zero_pad"
        elif waveform.size > _SENSEVOICE_FIXED_CONTEXT_NUM_SAMPLES:
            transformation = "truncate_tail"
        else:
            transformation = "identity"
        fixed = np.ascontiguousarray(fixed, dtype=np.float32)
        destination.parent.mkdir(parents=True, exist_ok=True)
        sf.write(
            destination,
            fixed,
            _SENSEVOICE_FIXED_CONTEXT_SAMPLE_RATE,
            subtype="FLOAT",
            format="WAV",
        )
        written, written_rate = sf.read(
            destination, dtype="float32", always_2d=False
        )
        written = np.asarray(written, dtype=np.float32)
        if (
            written.ndim != 1
            or written.size != _SENSEVOICE_FIXED_CONTEXT_NUM_SAMPLES
            or int(written_rate) != _SENSEVOICE_FIXED_CONTEXT_SAMPLE_RATE
        ):
            raise ValueError("derived fixed-context WAV shape/rate mismatch")
        derived_pcm_sha256 = hashlib.sha256(
            np.asarray(written, dtype="<f4").tobytes(order="C")
        ).hexdigest()
        expected_pcm_sha256 = hashlib.sha256(
            np.asarray(fixed, dtype="<f4").tobytes(order="C")
        ).hexdigest()
        if derived_pcm_sha256 != expected_pcm_sha256:
            raise ValueError("derived fixed-context WAV PCM mismatch")
        entry.update(
            {
                "transformation": transformation,
                "copied_source_samples": copied,
                "derived_pcm_sha256": derived_pcm_sha256,
                "derived_wav_sha256": _sha256_file(destination),
                "status": "prepared",
            }
        )
        return entry, destination
    except Exception as exc:
        destination.unlink(missing_ok=True)
        entry.update(
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        return entry, None


def _run_sensevoice_fixed_context_probe(
    *,
    model: Any,
    samples: Sequence[Mapping[str, Any]],
    output_dir: Path,
    postprocess: Callable[[str], str],
    np: Any,
    sf: Any,
    probe_n_per_language: int = _SENSEVOICE_FIXED_CONTEXT_PROBE_N,
) -> dict[str, Any]:
    if probe_n_per_language <= 0:
        raise ValueError("SenseVoice fixed-context probe n must be positive")
    probe_root = output_dir / "sensevoice_fixed_context_probe"
    language_receipts: dict[str, Any] = {}
    for language in ("en", "zh"):
        selected = sorted(
            (
                sample
                for sample in samples
                if sample["language"] == language
                and sample["condition"] == "clean"
            ),
            key=lambda sample: int(sample["_input_index"]),
        )[:probe_n_per_language]
        entries: list[dict[str, Any]] = []
        for batch_start in range(0, len(selected), 16):
            batch = selected[batch_start : batch_start + 16]
            prepared: list[tuple[dict[str, Any], Path]] = []
            for batch_offset, sample in enumerate(batch):
                token = hashlib.sha256(
                    str(sample["utterance_id"]).encode("utf-8")
                ).hexdigest()[:20]
                destination = probe_root / (
                    f"{language}-{batch_start + batch_offset:03d}-{token}.wav"
                )
                entry, path = _prepare_sensevoice_fixed_context_audio(
                    sample, destination, np=np, sf=sf
                )
                if path is None:
                    entries.append(_finalize_sensevoice_probe_entry(entry))
                else:
                    prepared.append((entry, path))
            if not prepared:
                continue
            try:
                raw = model.generate(
                    input=[str(path) for _entry, path in prepared],
                    cache={},
                    language=language,
                    use_itn=False,
                    batch_size=len(prepared),
                )
                values = _coerce_output_sequence(raw, len(prepared))
                for (entry, _path), value in zip(prepared, values):
                    final_text = postprocess(_extract_text(value))
                    entry.update(
                        {
                            "model_invoked": True,
                            "status": "ok",
                            "final_text_sha256": hashlib.sha256(
                                final_text.encode("utf-8")
                            ).hexdigest(),
                            "final_text_empty": not bool(final_text.strip()),
                        }
                    )
                    entries.append(_finalize_sensevoice_probe_entry(entry))
            except Exception as exc:
                for entry, _path in prepared:
                    entry.update(
                        {
                            "model_invoked": True,
                            "status": "inference_error",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                    entries.append(_finalize_sensevoice_probe_entry(entry))
            finally:
                for _entry, path in prepared:
                    path.unlink(missing_ok=True)
        entries.sort(
            key=lambda entry: (
                str(entry.get("utterance_id", "")),
                str(entry.get("audio_relpath", "")),
            )
        )
        invoked = sum(entry.get("model_invoked") is True for entry in entries)
        succeeded = sum(entry.get("status") == "ok" for entry in entries)
        language_receipts[language] = {
            "expected": probe_n_per_language,
            "selected": len(selected),
            "invoked": invoked,
            "succeeded": succeeded,
            "failed": len(entries) - succeeded,
            "samples": entries,
        }
    try:
        probe_root.rmdir()
    except OSError:
        pass
    receipt: dict[str, Any] = {
        "schema": "lane-c-sensevoice-fixed-context-probe/1.0",
        "route": "fixed-10s-context/final-only",
        "model": ENGINE_SPECS["sensevoice"]["models"]["sensevoice"]["repo_id"],
        "bank": ENGINE_SPECS["sensevoice"]["models"]["sensevoice"]["bank"],
        "revision": ENGINE_SPECS["sensevoice"]["models"]["sensevoice"][
            "revision"
        ],
        "context_seconds": _SENSEVOICE_FIXED_CONTEXT_SECONDS,
        "sample_rate": _SENSEVOICE_FIXED_CONTEXT_SAMPLE_RATE,
        "target_num_samples": _SENSEVOICE_FIXED_CONTEXT_NUM_SAMPLES,
        "probe_n_per_language": probe_n_per_language,
        "condition": "clean",
        "main_scoring_inputs_unchanged": True,
        "scoring_included": False,
        "reference_scoring_included": False,
        "partial_api": "utterance-only/no partial API",
        "partials_emitted": False,
        "languages": language_receipts,
    }
    receipt["valid"] = all(
        language_receipts[language]["selected"] == probe_n_per_language
        and language_receipts[language]["invoked"] == probe_n_per_language
        and language_receipts[language]["succeeded"] == probe_n_per_language
        and language_receipts[language]["failed"] == 0
        for language in ("en", "zh")
    )
    receipt["receipt_sha256"] = _canonical_json_sha256(receipt)
    return receipt


def _sensevoice_context_policy(
    selected: Sequence[Mapping[str, Any]],
    fixed_context_probe: Mapping[str, Any] | None,
) -> dict[str, Any]:
    maximum_duration_s = max(
        (float(sample["duration_s"]) for sample in selected), default=None
    )
    exercised = bool(
        isinstance(fixed_context_probe, Mapping)
        and fixed_context_probe.get("valid") is True
    )
    return {
        "evaluation_input": "whole utterance",
        "main_scoring_input": "whole utterance",
        "main_scoring_inputs_unchanged": True,
        "maximum_selected_fixture_duration_s": maximum_duration_s,
        "maximum_context_s": maximum_duration_s,
        "product_fixed_context_s": _SENSEVOICE_FIXED_CONTEXT_SECONDS,
        "product_fixed_context_separately_exercised": exercised,
        "fixed_context_probe_condition": "clean",
        "fixed_context_probe_n_per_language": (
            _SENSEVOICE_FIXED_CONTEXT_PROBE_N
        ),
        "fixed_context_probe_scoring_included": False,
        "fixed_context_probe_receipt_sha256": (
            fixed_context_probe.get("receipt_sha256")
            if isinstance(fixed_context_probe, Mapping)
            else None
        ),
        "note": (
            "C1 scoring keeps each fixture utterance and reference unchanged. "
            "C4 separately invokes exact 10-second clean contexts; those final "
            "outputs are capability evidence only and are not scored."
        ),
    }


def _base_metadata(
    engine: str, run_id: str, started_at: str, progress_path: Path
) -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "run_protocol_version": _RUN_PROTOCOL_VERSION,
        "engine": engine,
        "run_id": run_id,
        "started_at_utc": started_at,
        "venue_required": "Google Colab",
        "training_performed": False,
        "progress_jsonl": str(progress_path),
        "engine_spec": ENGINE_SPECS[engine],
    }


def _utterance_only_metadata(reason: str) -> dict[str, Any]:
    return {
        "api": "utterance-only/no partial API",
        "partial_traces_retained": False,
        "reason": reason,
    }


def _runtime_metadata(distributions: Sequence[str]) -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for distribution in distributions:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    result: dict[str, Any] = {
        "python": sys.version,
        "platform": sys.platform,
        "packages": versions,
    }
    try:
        import torch  # type: ignore[import-not-found]

        result["torch"] = {
            "version": torch.__version__,
            "cuda_build": getattr(torch.version, "cuda", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        }
    except Exception as exc:
        result["torch_probe_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _require_cuda(engine: str) -> None:
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            f"{engine} requires a fresh Google Colab GPU runtime and its install step"
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"{engine} must run on a Google Colab GPU; torch.cuda.is_available() is False"
        )


def _empty_cuda_cache(torch: Any) -> None:
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _model_cache_root(output_dir: Path) -> Path:
    configured = os.environ.get("LANE_C_ASR_MODEL_CACHE")
    if configured:
        root = Path(configured).expanduser().resolve()
    elif Path("/content").is_dir():
        root = Path("/content/lane_c_asr_models")
    else:
        root = output_dir / "artifacts" / "models"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _validate_samples(samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
        raise TypeError("samples must be a sequence of mappings")
    result = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise TypeError(f"samples[{index}] is not a mapping")
        missing = [
            key
            for key in _REQUIRED_SAMPLE_KEYS
            if key != "duration_s" and key not in sample
        ]
        if "duration_s" not in sample and "duration_seconds" not in sample:
            missing.append("duration_s")
        if missing:
            raise ValueError(f"samples[{index}] is missing keys: {missing}")
        language = str(sample["language"]).lower()
        if language not in _LANGUAGES:
            raise ValueError(
                f"samples[{index}].language must be one of {_LANGUAGES}, got {language!r}"
            )
        duration_value = sample.get("duration_s", sample.get("duration_seconds"))
        duration = float(duration_value)
        if not math.isfinite(duration) or duration < 0:
            raise ValueError(f"samples[{index}].duration_s is invalid: {duration!r}")
        audio_path = str(Path(sample["audio_path"]).expanduser().resolve())
        normalized = {
            "utterance_id": str(sample["utterance_id"]),
            "language": language,
            "condition": str(sample["condition"]).lower(),
            "audio_path": audio_path,
            "reference": str(sample["reference"]),
            "duration_s": duration,
            "duration_seconds": duration,
            "_input_index": index,
        }
        for key in _MIRRORED_SAMPLE_KEYS:
            if key in sample:
                normalized[key] = sample[key]
        result.append(normalized)
    return result


def _run_id(engine: str, samples: Sequence[Mapping[str, Any]]) -> str:
    provenance_keys = (
        *_REQUIRED_SAMPLE_KEYS,
        "duration_seconds",
        *_MIRRORED_SAMPLE_KEYS,
    )
    manifest = {
        "schema_version": _SCHEMA_VERSION,
        "run_protocol_version": _RUN_PROTOCOL_VERSION,
        "engine": engine,
        "spec": ENGINE_SPECS[engine],
        "samples": [
            {key: sample[key] for key in provenance_keys if key in sample}
            for sample in samples
        ],
    }
    encoded = json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Required pinned artifact is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _coerce_output_sequence(raw: Any, expected: int) -> list[Any]:
    if isinstance(raw, tuple) and raw and isinstance(raw[0], (list, tuple)):
        raw = raw[0]
    if expected == 1 and not isinstance(raw, (list, tuple)):
        values = [raw]
    elif isinstance(raw, (list, tuple)):
        values = list(raw)
    else:
        raise TypeError(f"Engine returned non-sequence output for batch of {expected}")
    if len(values) != expected:
        raise ValueError(f"Engine returned {len(values)} outputs for batch of {expected}")
    return values


def _extract_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in ("text", "pred_text", "transcript", "hypothesis"):
            if key in value:
                return str(value[key])
    if hasattr(value, "text"):
        return str(value.text)
    return str(value)


def _chunks(values: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    if size <= 0:
        raise ValueError("batch size must be positive")
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _sample_identity(sample: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(sample["utterance_id"]),
        str(sample["condition"]),
        str(sample["audio_path"]),
    )


def _model_provenance(model_id: str) -> dict[str, str]:
    for engine_spec in ENGINE_SPECS.values():
        for model_spec in engine_spec["models"].values():
            if model_spec["repo_id"] == model_id:
                return {
                    "bank": str(model_spec["bank"]),
                    "revision": str(model_spec["revision"]),
                }
    raise KeyError(f"No pinned provenance is registered for model {model_id!r}")


def _record_identity(
    model_id: str, sample: Mapping[str, Any]
) -> tuple[str, str, str, str]:
    return (str(model_id),) + _sample_identity(sample)


def _record_identity_from_record(record: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(record.get("model", "")),
        str(record.get("utterance_id", "")),
        str(record.get("condition", "")),
        str(record.get("audio_path", "")),
    )


def _successful_record_identities(
    records: Sequence[Mapping[str, Any]],
) -> set[tuple[str, str, str, str]]:
    """Return only identities backed by a completed inference result."""

    return {
        _record_identity_from_record(record)
        for record in records
        if record.get("status") == "ok"
    }


def _dedupe_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for record in records:
        identity = _record_identity_from_record(record)
        previous = result.get(identity)
        if (
            previous is not None
            and previous.get("status") == "ok"
            and record.get("status") != "ok"
        ):
            # A later stale/repeated error must not replace a successful retry.
            continue
        result[identity] = record
    return list(result.values())


def _merge_records(
    existing: Sequence[dict[str, Any]],
    new: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge checkpoint records, replacing stale errors with successful retries."""

    return _dedupe_records([*existing, *new])


def _trace_identity(trace: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        trace.get("model"),
        trace.get("utterance_id"),
        trace.get("condition"),
        trace.get("audio_path"),
        trace.get("stream_time_s"),
        trace.get("phase"),
        trace.get("is_final"),
    )


def _dedupe_traces(traces: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    for trace in traces:
        result[_trace_identity(trace)] = trace
    return list(result.values())


def _record_sort_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    model_order = {
        "FunAudioLLM/SenseVoiceSmall": 0,
        "nvidia/parakeet-tdt-0.6b-v3": 0,
        "nvidia/canary-180m-flash": 1,
        "pfluo/k2fsa-zipformer-chinese-english-mixed": 0,
        "ai-sage/GigaAM-Multilingual": 0,
        "facebook/omniASR-CTC-300M": 0,
    }
    return (
        model_order.get(str(record.get("model")), 99),
        int(record.get("input_index", 0)),
        str(record.get("condition", "")),
    )


def _trace_sort_key(trace: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(trace.get("utterance_id", "")),
        str(trace.get("condition", "")),
        float(trace.get("stream_time_s", 0.0)),
        bool(trace.get("is_final", False)),
    )


def _engine_name(engine: str) -> str:
    name = str(engine).strip().lower()
    if name not in ENGINE_SPECS:
        raise ValueError(
            f"Unknown engine {engine!r}; choose one of {sorted(ENGINE_SPECS)}"
        )
    return name


def _utc_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat()
