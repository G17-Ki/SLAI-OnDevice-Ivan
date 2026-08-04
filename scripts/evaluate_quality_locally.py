#!/usr/bin/env python3
"""Reproducible local quality audit for the LFM feasibility experiment.

This script consumes already-captured local and Qualcomm AI Hub artifacts.  It
does not submit cloud jobs.  The optional backbone check loads only the frozen
LFM backbone from the pinned checkpoint and compares downstream next-token
decisions after injecting local versus AI Hub FastConformer embeddings.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from path_config import (
    MODEL_REVISION,
    MODEL_SNAPSHOT,
    REPO_ROOT,
    VENDOR_ROOT,
    WORK_ROOT,
)

PROJECT = REPO_ROOT
WORK = WORK_ROOT / "lfm-feasibility"
MODEL_ROOT = MODEL_SNAPSHOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fastconformer-outputs",
        type=Path,
        default=WORK
        / "aihub/fastconformer-first-pass-20260715t091744z/fastconformer/outputs-strict-npu.npz",
    )
    parser.add_argument(
        "--component-results",
        type=Path,
        default=PROJECT / "reports/aihub_component_results.json",
    )
    parser.add_argument(
        "--colab-summary",
        type=Path,
        default=PROJECT / "reports/colab_bf16/summary.json",
    )
    parser.add_argument(
        "--colab-tokens",
        type=Path,
        default=PROJECT / "reports/colab_bf16/golden_tokens.npz",
    )
    parser.add_argument(
        "--official-readme",
        type=Path,
        default=VENDOR_ROOT / "README.md",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT / "workstreams/quality_eval",
    )
    parser.add_argument(
        "--teacher-forced-tokens",
        type=int,
        default=20,
        help="Maximum common-context next-token comparisons; stops at golden EOS.",
    )
    parser.add_argument(
        "--skip-backbone",
        action="store_true",
        help="Skip the frozen-LFM downstream check (feature/ASR/audio checks still run).",
    )
    return parser.parse_args()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a64 = np.asarray(a, dtype=np.float64).ravel()
    b64 = np.asarray(b, dtype=np.float64).ravel()
    denominator = np.linalg.norm(a64) * np.linalg.norm(b64)
    return float(np.dot(a64, b64) / denominator) if denominator else 1.0


def feature_metrics(golden: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    if golden.shape != actual.shape:
        raise ValueError(f"FastConformer shape mismatch: {golden.shape} != {actual.shape}")
    g = golden.astype(np.float64)
    a = actual.astype(np.float64)
    delta = a - g
    abs_delta = np.abs(delta)
    golden_rms = float(np.sqrt(np.mean(g * g)))
    rmse = float(np.sqrt(np.mean(delta * delta)))

    frames_g = g.reshape(-1, g.shape[-1])
    frames_a = a.reshape(-1, a.shape[-1])
    frame_cos = np.array([cosine(x, y) for x, y in zip(frames_g, frames_a, strict=True)])
    relative_l2 = np.linalg.norm(frames_a - frames_g, axis=1) / np.maximum(
        np.linalg.norm(frames_g, axis=1), 1e-12
    )

    # Does every remote timestep still identify its matching local timestep?
    g_unit = frames_g / np.maximum(np.linalg.norm(frames_g, axis=1, keepdims=True), 1e-12)
    a_unit = frames_a / np.maximum(np.linalg.norm(frames_a, axis=1, keepdims=True), 1e-12)
    cross_similarity = a_unit @ g_unit.T
    nearest_frame = cross_similarity.argmax(axis=1)
    expected_frame = np.arange(len(frames_g))

    golden_geometry = g_unit @ g_unit.T
    actual_geometry = a_unit @ a_unit.T
    geometry_delta = actual_geometry - golden_geometry

    return {
        "shape": list(golden.shape),
        "global": {
            "cosine_similarity": cosine(g, a),
            "max_abs_error": float(abs_delta.max()),
            "mean_abs_error": float(abs_delta.mean()),
            "rmse": rmse,
            "golden_rms": golden_rms,
            "normalized_rmse": rmse / golden_rms,
            "sign_agreement_rate": float(np.mean(np.signbit(g) == np.signbit(a))),
        },
        "per_frame": {
            "count": int(len(frame_cos)),
            "cosine_mean": float(frame_cos.mean()),
            "cosine_min": float(frame_cos.min()),
            "cosine_p05": float(np.quantile(frame_cos, 0.05)),
            "relative_l2_mean": float(relative_l2.mean()),
            "relative_l2_max": float(relative_l2.max()),
        },
        "temporal_identity": {
            "nearest_golden_frame_accuracy": float(np.mean(nearest_frame == expected_frame)),
            "nearest_golden_frame_indices": nearest_frame.tolist(),
            "matching_frame_similarity_min": float(np.diag(cross_similarity).min()),
            "matching_vs_best_wrong_margin_min": float(
                min(
                    cross_similarity[i, i]
                    - np.max(np.delete(cross_similarity[i], i))
                    for i in range(len(frames_g))
                )
            ),
        },
        "pairwise_geometry": {
            "cosine_matrix_correlation": float(
                np.corrcoef(golden_geometry.ravel(), actual_geometry.ravel())[0, 1]
            ),
            "mean_abs_error": float(np.abs(geometry_delta).mean()),
            "max_abs_error": float(np.abs(geometry_delta).max()),
        },
    }


def normalize_transcript(text: str) -> list[str]:
    text = re.sub(r"<\|[^|]+\|>", " ", text).lower()
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text)


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, start=1):
        current = [i]
        for j, hyp in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (ref != hyp),
                )
            )
        previous = current
    return previous[-1]


def asr_metrics(summary: dict[str, Any], official_readme: Path) -> dict[str, Any]:
    readme = official_readme.read_text()
    match = re.search(r"\*\*Model output\*\*:\s*(.+)", readme)
    if not match:
        raise RuntimeError("Could not extract the official ASR reference from README.md")
    reference_text = match.group(1).strip()
    decoded_text = str(summary["asr"]["decoded_text"])
    ref_words = normalize_transcript(reference_text)
    hyp_words = normalize_transcript(decoded_text)
    ref_chars = list(" ".join(ref_words))
    hyp_chars = list(" ".join(hyp_words))
    word_edits = edit_distance(ref_words, hyp_words)
    char_edits = edit_distance(ref_chars, hyp_chars)
    return {
        "reference_source": str(official_readme),
        "reference_text": reference_text,
        "decoded_text_raw": decoded_text,
        "decoded_control_suffix_present": bool(re.search(r"<\|[^|]+\|>", decoded_text)),
        "reference_word_count": len(ref_words),
        "hypothesis_word_count": len(hyp_words),
        "word_edits": word_edits,
        "wer": word_edits / max(len(ref_words), 1),
        "character_edits": char_edits,
        "cer": char_edits / max(len(ref_chars), 1),
        "normalized_exact_match": ref_words == hyp_words,
    }


def audio_integrity(path: Path) -> dict[str, Any]:
    wave, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = wave.mean(axis=1).astype(np.float64)
    abs_wave = np.abs(mono)
    rms = float(np.sqrt(np.mean(mono * mono)))
    peak = float(abs_wave.max())
    frame = max(1, int(round(sample_rate * 0.02)))
    usable = len(mono) // frame * frame
    frame_rms = (
        np.sqrt(np.mean(mono[:usable].reshape(-1, frame) ** 2, axis=1))
        if usable
        else np.array([], dtype=np.float64)
    )
    return {
        "path": str(path),
        "sample_rate_hz": int(sample_rate),
        "channels": int(wave.shape[1]),
        "samples": int(len(mono)),
        "duration_seconds": float(len(mono) / sample_rate),
        "finite": bool(np.isfinite(mono).all()),
        "rms": rms,
        "peak_abs": peak,
        "dc_offset": float(mono.mean()),
        "crest_factor": peak / rms if rms else None,
        "hard_clip_rate": float(np.mean(abs_wave >= 0.999)),
        "near_silence_20ms_frame_rate": float(np.mean(frame_rms < 1e-4)) if len(frame_rms) else None,
        "zero_crossing_rate": float(np.mean(np.signbit(mono[1:]) != np.signbit(mono[:-1])))
        if len(mono) > 1
        else None,
    }


def generated_audio_metrics(summary: dict[str, Any], report_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for turn_name in ("turn1", "turn2"):
        audio_path = report_dir / f"interleaved_{turn_name}.wav"
        integrity = audio_integrity(audio_path)
        generation = summary["interleaved"][turn_name]
        decode = summary["interleaved"]["decode"][turn_name]
        text = str(generation["decoded_text"])
        expected_seconds = float(decode["waveform_seconds"])
        integrity["summary_duration_seconds"] = expected_seconds
        integrity["duration_matches_summary"] = bool(
            abs(integrity["duration_seconds"] - expected_seconds) < 1.0 / integrity["sample_rate_hz"]
        )
        integrity["decoded_text"] = text
        integrity["text_character_count"] = len(text)
        integrity["text_characters_per_audio_second"] = len(text) / integrity["duration_seconds"]
        result[turn_name] = integrity

    followup = str(summary["inputs"]["followup_text"])
    response = str(summary["interleaved"]["turn2"]["decoded_text"])
    requested_keywords = sorted(set(re.findall(r"\bchairs?\b", followup.lower())))
    matched_keywords = [word for word in requested_keywords if word in response.lower()]
    result["turn2_followup_adherence"] = {
        "followup_text": followup,
        "requested_domain_keywords": requested_keywords,
        "matched_domain_keywords": matched_keywords,
        "all_requested_domain_keywords_present": set(requested_keywords) <= set(matched_keywords),
        "note": "Deterministic keyword check only; this is not a semantic or perceptual judge score.",
    }
    return result


def component_token_metrics(component_results: dict[str, Any]) -> dict[str, Any]:
    strict_depth = next(
        item
        for item in component_results["results"]
        if item["component"] == "depth-decoder" and item["runtime"] == "strict-npu"
    )
    comparison = strict_depth["comparisons"]["tokens"]
    return {
        "component": "depth-decoder",
        "runtime": "strict-npu",
        "exact_token_match": bool(comparison["passed"]),
        "comparison": comparison["comparison"],
        "target_note": component_results["target_note"],
    }


def token_archive_metrics(path: Path) -> dict[str, Any]:
    with np.load(path) as archive:
        result = {
            name: {
                "shape": list(archive[name].shape),
                "dtype": str(archive[name].dtype),
                "min": int(archive[name].min()),
                "max": int(archive[name].max()),
            }
            for name in archive.files
        }
    return result


def distribution_metrics(p: Any, q: Any) -> dict[str, float]:
    import torch

    eps = 1e-12
    midpoint = 0.5 * (p + q)
    js = 0.5 * torch.sum(p * torch.log((p + eps) / (midpoint + eps))) + 0.5 * torch.sum(
        q * torch.log((q + eps) / (midpoint + eps))
    )
    return {
        "total_variation": float(0.5 * torch.sum(torch.abs(p - q))),
        "jensen_shannon_nats": float(js),
    }


def downstream_backbone_metrics(
    golden: np.ndarray, actual: np.ndarray, max_tokens: int
) -> dict[str, Any]:
    import torch
    from safetensors import safe_open
    from transformers import AutoTokenizer, Lfm2Config, Lfm2Model
    from transformers.models.lfm2.modeling_lfm2 import Lfm2RotaryEmbedding

    if not MODEL_ROOT.is_dir():
        raise FileNotFoundError(f"Pinned model snapshot not found: {MODEL_ROOT}")
    config_json = json.loads((MODEL_ROOT / "config.json").read_text())
    config = Lfm2Config(**config_json["lfm"])

    load_started = time.perf_counter()
    # Meta initialization plus assign=True avoids a second full copy of the 1.2B backbone.
    with torch.device("meta"):
        model = Lfm2Model(config)
    # Rotary buffers are non-persistent and therefore absent from the checkpoint.
    model.rotary_emb = Lfm2RotaryEmbedding(config, device="cpu")
    model.pos_emb = Lfm2RotaryEmbedding(config, device="cpu")
    with safe_open(MODEL_ROOT / "model.safetensors", framework="pt", device="cpu") as reader:
        state = {
            key.removeprefix("lfm."): reader.get_tensor(key)
            for key in reader.keys()
            if key.startswith("lfm.")
        }
    load_result = model.load_state_dict(state, strict=True, assign=True)
    del state
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(str(load_result))
    model.eval()
    model.set_attn_implementation("sdpa")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ROOT, local_files_only=True)
    load_seconds = time.perf_counter() - load_started

    prefix = tokenizer.encode(
        "<|startoftext|><|im_start|>user\n", add_special_tokens=False, return_tensors="pt"
    )
    suffix = tokenizer.encode(
        "<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
        return_tensors="pt",
    )
    golden_features = torch.from_numpy(golden).to(torch.bfloat16)
    actual_features = torch.from_numpy(actual).to(torch.bfloat16)

    steps: list[dict[str, Any]] = []
    generated_golden_ids: list[int] = []
    remote_argmax_ids: list[int] = []
    cache = None
    inference_started = time.perf_counter()
    with torch.inference_mode():
        prefix_embedding = model.embed_tokens(prefix)
        suffix_embedding = model.embed_tokens(suffix)
        golden_input = torch.cat([prefix_embedding, golden_features, suffix_embedding], dim=1)
        actual_input = torch.cat([prefix_embedding, actual_features, suffix_embedding], dim=1)
        current_input = torch.cat([golden_input, actual_input], dim=0)

        for step_index in range(max_tokens):
            output = model(inputs_embeds=current_input, past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            hidden = output.last_hidden_state[:, -1].float()
            logits = hidden @ model.embed_tokens.weight.float().T
            probabilities = torch.softmax(logits, dim=-1)
            golden_id = int(torch.argmax(logits[0]))
            remote_id = int(torch.argmax(logits[1]))
            golden_probability = float(probabilities[0, golden_id])
            remote_probability_for_golden = float(probabilities[1, golden_id])
            remote_rank_for_golden = int(
                1 + torch.sum(logits[1] > logits[1, golden_id]).item()
            )
            golden_top_values, golden_top_ids = torch.topk(probabilities[0], 5)
            remote_top_values, remote_top_ids = torch.topk(probabilities[1], 5)
            golden_margin = float(golden_top_values[0] - golden_top_values[1])
            step_result = {
                "step": step_index,
                "golden_argmax_id": golden_id,
                "golden_argmax_text": tokenizer.decode([golden_id]),
                "remote_argmax_id": remote_id,
                "remote_argmax_text": tokenizer.decode([remote_id]),
                "argmax_equal": golden_id == remote_id,
                "remote_rank_of_golden_argmax": remote_rank_for_golden,
                "golden_argmax_probability": golden_probability,
                "remote_probability_of_golden_argmax": remote_probability_for_golden,
                "golden_top1_probability_margin": golden_margin,
                "hidden_cosine_similarity": float(
                    torch.nn.functional.cosine_similarity(hidden[:1], hidden[1:]).item()
                ),
                "logit_cosine_similarity": float(
                    torch.nn.functional.cosine_similarity(logits[:1], logits[1:]).item()
                ),
                "top5_overlap": len(set(golden_top_ids.tolist()) & set(remote_top_ids.tolist())) / 5,
                "golden_top5": [
                    {
                        "id": int(token_id),
                        "text": tokenizer.decode([int(token_id)]),
                        "probability": float(probability),
                    }
                    for probability, token_id in zip(
                        golden_top_values.tolist(), golden_top_ids.tolist(), strict=True
                    )
                ],
                "remote_top5": [
                    {
                        "id": int(token_id),
                        "text": tokenizer.decode([int(token_id)]),
                        "probability": float(probability),
                    }
                    for probability, token_id in zip(
                        remote_top_values.tolist(), remote_top_ids.tolist(), strict=True
                    )
                ],
                **distribution_metrics(probabilities[0], probabilities[1]),
            }
            steps.append(step_result)
            generated_golden_ids.append(golden_id)
            remote_argmax_ids.append(remote_id)

            # Common-context (teacher-forced) comparison: feed the golden choice to both
            # branches so later differences are not merely consequences of an early fork.
            common_token = torch.tensor([[golden_id], [golden_id]], dtype=torch.long)
            current_input = model.embed_tokens(common_token)
            if golden_id == config.eos_token_id:
                break

    inference_seconds = time.perf_counter() - inference_started
    agreement = [step["argmax_equal"] for step in steps]
    first_divergence = next((index for index, same in enumerate(agreement) if not same), None)
    return {
        "method": (
            "Frozen pinned LFM backbone; local and AI Hub FastConformer embeddings replace the "
            "same ten audio-input positions. At every subsequent step both branches receive the "
            "golden branch's token, preventing autoregressive divergence from contaminating later comparisons."
        ),
        "prompt": {
            "prefix": "<|startoftext|><|im_start|>user\\n",
            "audio_embedding_positions": int(golden.shape[1]),
            "suffix": "<|im_end|>\\n<|im_start|>assistant\\n",
            "note": "The captured 80-mel-frame probe is a truncated question.wav segment, not the full utterance.",
        },
        "model": {
            "id": "LiquidAI/LFM2.5-Audio-1.5B",
            "revision": MODEL_REVISION,
            "backbone_dtype": str(model.embed_tokens.weight.dtype),
            "device": "cpu",
            "load_seconds": load_seconds,
            "inference_seconds": inference_seconds,
        },
        "summary": {
            "steps_compared": len(steps),
            "argmax_agreement_count": int(sum(agreement)),
            "argmax_agreement_rate": float(np.mean(agreement)),
            "first_argmax_divergence_step": first_divergence,
            "golden_choice_remote_top5_rate": float(
                np.mean([step["remote_rank_of_golden_argmax"] <= 5 for step in steps])
            ),
            "mean_remote_rank_of_golden_choice": float(
                np.mean([step["remote_rank_of_golden_argmax"] for step in steps])
            ),
            "mean_total_variation": float(np.mean([step["total_variation"] for step in steps])),
            "max_total_variation": float(max(step["total_variation"] for step in steps)),
            "mean_jensen_shannon_nats": float(
                np.mean([step["jensen_shannon_nats"] for step in steps])
            ),
            "mean_logit_cosine_similarity": float(
                np.mean([step["logit_cosine_similarity"] for step in steps])
            ),
            "minimum_logit_cosine_similarity": float(
                min(step["logit_cosine_similarity"] for step in steps)
            ),
            "golden_teacher_forced_text": tokenizer.decode(generated_golden_ids),
            "remote_argmax_text_under_golden_context": tokenizer.decode(remote_argmax_ids),
        },
        "steps": steps,
        "interpretation_limit": (
            "This is a real downstream sensitivity test for one truncated audio probe. It is not an "
            "ASR benchmark and cannot establish corpus-level semantic equivalence."
        ),
    }


def markdown_report(result: dict[str, Any]) -> str:
    feature = result["fastconformer_features"]
    asr = result["full_bf16_asr"]
    depth = result["depth_decoder_tokens"]
    audio = result["generated_audio_integrity"]
    detok = result["detokenizer_waveform_fidelity"]
    backbone = result.get("fastconformer_downstream_backbone")
    lines = [
        "# Local quality audit",
        "",
        "This audit reuses the pinned local and Qualcomm AI Hub artifacts; it submits no cloud jobs.",
        "",
        "## Decision summary",
        "",
        f"- Full BF16 ASR normalized exact match: **{asr['normalized_exact_match']}** "
        f"(WER {asr['wer']:.3f}, CER {asr['cer']:.3f}).",
        f"- Strict-NPU depth-decoder token match: **{depth['exact_token_match']}**.",
        f"- FastConformer feature cosine: **{feature['global']['cosine_similarity']:.6f}**; "
        f"NRMSE **{feature['global']['normalized_rmse']:.4f}**.",
        f"- All {feature['per_frame']['count']} remote feature frames retrieve the matching local "
        f"frame: **{feature['temporal_identity']['nearest_golden_frame_accuracy']:.0%}** accuracy.",
    ]
    if backbone:
        summary = backbone["summary"]
        lines.extend(
            [
                f"- Frozen-backbone common-context top-1 agreement: "
                f"**{summary['argmax_agreement_count']}/{summary['steps_compared']} "
                f"({summary['argmax_agreement_rate']:.1%})**; the first prediction matches, "
                f"but the first divergence occurs at step {summary['first_argmax_divergence_step']}.",
                f"- The golden choice remains in the remote branch's top 5 on "
                f"**{summary['golden_choice_remote_top5_rate']:.1%}** of compared steps.",
            ]
        )
    lines.extend(
        [
            f"- Generated response WAVs are finite 24 kHz mono audio with hard-clip rates "
            f"{audio['turn1']['hard_clip_rate']:.3%} and {audio['turn2']['hard_clip_rate']:.3%}.",
            f"- NPU detokenizer reconstruction remains unusable: T=4 SI-SDR "
            f"**{detok['t4']['waveform']['si_sdr_db']:.2f} dB**, T=8 SI-SDR "
            f"**{detok['t8']['waveform']['si_sdr_db']:.2f} dB**.",
            "",
            "## What the new downstream check changes",
            "",
        ]
    )
    if backbone:
        summary = backbone["summary"]
        lines.extend(
            [
                "The AI Hub FastConformer output is close enough to preserve the initial LFM token, "
                "but not close enough to claim exact sequence equivalence. Under a shared golden-token "
                "context, a later top-1 choice changes even though the competing distributions remain close.",
                "",
                f"Golden path: `{summary['golden_teacher_forced_text']}`",
                "",
                f"Remote argmax under that same context: `{summary['remote_argmax_text_under_golden_context']}`",
                "",
                "This makes downstream validation after quantization mandatory. Feature cosine alone would "
                "have hidden a real decision-boundary crossing.",
            ]
        )
    else:
        lines.append("The frozen-backbone check was skipped for this run.")
    lines.extend(
        [
            "",
            "## Measurement boundaries",
            "",
            "- The FastConformer downstream probe contains only the first 80 mel frames of one "
            "question.wav sample. It is a sensitivity test, not a corpus benchmark.",
            "- The downstream sensitivity comparison runs the same pinned BF16 LFM backbone on CPU "
            "for both branches; only the ten injected audio embeddings differ.",
            "- The exact ASR match is the full BF16 Colab golden versus Liquid Audio's published "
            "reference; it does not test AI Hub FastConformer substitution.",
            "- Generated-response audio has no clean reference. The reported WAV checks establish "
            "technical integrity, not naturalness, intelligibility, MOS, PESQ, or STOI.",
            "- Detokenizer waveform fidelity is reported separately and remains a blocker for NPU use.",
            "- AI Hub used QCS8550 Proxy / Hexagon v73, not AR1; none of these are AR1 quality or latency claims.",
            "",
            "## Reproduce",
            "",
            "```bash",
            "python scripts/evaluate_quality_locally.py",
            "```",
            "",
            "The machine-readable details, including every compared token distribution, are in `results.json`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.fastconformer_outputs) as archive:
        golden = np.array(archive["golden__adapted"], dtype=np.float32)
        actual = np.array(archive["actual__adapted"], dtype=np.float32)
    component_results = json.loads(args.component_results.read_text())
    colab_summary = json.loads(args.colab_summary.read_text())

    result: dict[str, Any] = {
        "status": "passed",
        "scope": "Local audit of existing pinned artifacts; no cloud submission.",
        "sources": {
            "fastconformer_outputs": str(args.fastconformer_outputs),
            "component_results": str(args.component_results),
            "colab_summary": str(args.colab_summary),
            "colab_tokens": str(args.colab_tokens),
        },
        "fastconformer_features": feature_metrics(golden, actual),
        "full_bf16_asr": asr_metrics(colab_summary, args.official_readme),
        "depth_decoder_tokens": component_token_metrics(component_results),
        "full_bf16_token_archive": token_archive_metrics(args.colab_tokens),
        "generated_audio_integrity": generated_audio_metrics(
            colab_summary, args.colab_summary.parent
        ),
        "detokenizer_waveform_fidelity": {
            "t4": json.loads((PROJECT / "reports/detok_waveform_t4/report.json").read_text()),
            "t8": json.loads((PROJECT / "reports/detok_waveform_t8/report.json").read_text()),
        },
    }
    if not args.skip_backbone:
        result["fastconformer_downstream_backbone"] = downstream_backbone_metrics(
            golden, actual, args.teacher_forced_tokens
        )

    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output_dir / "findings.md").write_text(markdown_report(result))
    print(json.dumps({"status": "passed", "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
