#!/usr/bin/env python3
"""Colab execution harness for Lane B's frozen B3/B4 protocol.

This script never quantizes or trains a model.  It renders the prompt with the
pinned Hugging Face tokenizer/template, downloads exactly one pre-built GGUF at
a time, verifies its frozen bytes and SHA-256, and drives the pinned llama.cpp
binaries for multiple-choice continuation likelihoods and greedy generation.

The official run uses the pinned llama.cpp commit plus an exact audited one-file
memory-allocation patch.  The patch leaves the stock ``--multiple-choice``
decode and length-normalized continuation scoring order unchanged, but sizes its
host logits buffer to the outputs in the assembled batch instead of the full
context window.  Per-item correctness is reconstructed exactly from the stock
cumulative trace; unavailable per-choice vectors remain null.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


SCHEMA_VERSION = "lane-b-b34-raw/1.0"
RENDER_SCHEMA_VERSION = "lane-b-b34-rendered/1.0"
LLAMA_CPP_COMMIT = "11924d4c17abc27383376a1ac6a24fa3e36c1c0c"
LLAMA_CPP_MULTIPLE_CHOICE_SOURCE_RELATIVE_PATH = "tools/perplexity/perplexity.cpp"
LLAMA_CPP_MULTIPLE_CHOICE_SOURCE_ORIGINAL_SHA256 = (
    "380aadd9d8000bf1b95cc3a338708385ac75240a5ba1c1f2bb34b208892a1777"
)
LLAMA_CPP_MULTIPLE_CHOICE_SOURCE_SHA256 = (
    "65051abf6eb3e68d26cb2f24b62e2fa1af1bf7ddea96a2db53f944c3f0232b78"
)
LLAMA_CPP_MULTIPLE_CHOICE_PATCH_RELATIVE_PATH = "llama_mcq_memory.patch"
LLAMA_CPP_MULTIPLE_CHOICE_PATCH_SHA256 = (
    "2539f2da48852fa44fdacdba26154e144a01b3986ff7c88dfe6974ac3e4f9dd7"
)
LLAMA_CPP_EXPECTED_SOURCE_STATUS = f" M {LLAMA_CPP_MULTIPLE_CHOICE_SOURCE_RELATIVE_PATH}"
if sys.platform == "darwin":
    LLAMA_CPP_PERPLEXITY_IMPL_LIBRARY_NAME = "libllama-perplexity-impl.dylib"
elif sys.platform == "win32":
    LLAMA_CPP_PERPLEXITY_IMPL_LIBRARY_NAME = "llama-perplexity-impl.dll"
else:
    LLAMA_CPP_PERPLEXITY_IMPL_LIBRARY_NAME = "libllama-perplexity-impl.so"
LLAMA_CPP_PERPLEXITY_IMPL_LIBRARY_RELATIVE_PATH = f"bin/{LLAMA_CPP_PERPLEXITY_IMPL_LIBRARY_NAME}"
MODEL_IDS = ("qwen35_08b", "qwen25_15b", "qwen3_06b", "falcon_h1_15b")
LANGUAGES = ("en", "zh", "es", "fr", "ru")
CHOICES = ("A", "B", "C", "D")
MCQ_CONTEXT_TOKENS = 4096
MCQ_BATCH_SIZE = 1024
MCQ_UBATCH_SIZE = 128
MCQ_PARALLEL = 4
SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent


class HarnessError(RuntimeError):
    """A protocol or execution invariant failed."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise HarnessError(f"expected JSON object in {path}")
    return value


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise HarnessError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise HarnessError(f"expected object at {path}:{line_number}")
            yield row


def write_json(path: Path, value: Any, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise HarnessError(f"refusing to overwrite existing file: {path}")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]], *, replace: bool = False) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise HarnessError(f"refusing to overwrite existing file: {path}")
    temporary = path.with_name(path.name + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    temporary.replace(path)
    return count


def run_capture(command: Sequence[str], *, cwd: Path | None = None, check: bool = True) -> str:
    process = subprocess.run(
        list(command),
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if check and process.returncode != 0:
        rendered = " ".join(command)
        raise HarnessError(f"command failed ({process.returncode}): {rendered}\n{process.stdout}")
    return process.stdout


def run_capture_bytes(command: Sequence[str], *, cwd: Path | None = None) -> bytes:
    process = subprocess.run(
        list(command),
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode != 0:
        rendered = " ".join(command)
        detail = process.stderr.decode("utf-8", errors="replace")
        raise HarnessError(f"command failed ({process.returncode}): {rendered}\n{detail}")
    return process.stdout


def validate_llama_source_patch(lane_dir: Path, llama_src: Path) -> dict[str, Any]:
    llama_src = llama_src.resolve()
    source_path = llama_src / LLAMA_CPP_MULTIPLE_CHOICE_SOURCE_RELATIVE_PATH
    patch_path = (lane_dir / LLAMA_CPP_MULTIPLE_CHOICE_PATCH_RELATIVE_PATH).resolve()
    if not source_path.is_file():
        raise HarnessError(f"missing llama.cpp multiple-choice source: {source_path}")
    if not patch_path.is_file():
        raise HarnessError(f"missing audited llama.cpp patch: {patch_path}")

    head = run_capture(["git", "rev-parse", "HEAD"], cwd=llama_src).strip()
    if head != LLAMA_CPP_COMMIT:
        raise HarnessError(f"llama.cpp HEAD is {head}, expected {LLAMA_CPP_COMMIT}")

    original_source = run_capture_bytes(
        ["git", "show", f"HEAD:{LLAMA_CPP_MULTIPLE_CHOICE_SOURCE_RELATIVE_PATH}"],
        cwd=llama_src,
    )
    original_source_sha256 = sha256_bytes(original_source)
    if original_source_sha256 != LLAMA_CPP_MULTIPLE_CHOICE_SOURCE_ORIGINAL_SHA256:
        raise HarnessError(
            "pinned llama.cpp multiple-choice source does not match the audited original: "
            f"{original_source_sha256}"
        )

    patch_sha256 = sha256_file(patch_path)
    if patch_sha256 != LLAMA_CPP_MULTIPLE_CHOICE_PATCH_SHA256:
        raise HarnessError(
            "llama.cpp MCQ patch hash mismatch: "
            f"expected {LLAMA_CPP_MULTIPLE_CHOICE_PATCH_SHA256}, got {patch_sha256}"
        )

    source_sha256 = sha256_file(source_path)
    if source_sha256 != LLAMA_CPP_MULTIPLE_CHOICE_SOURCE_SHA256:
        raise HarnessError(
            "patched llama.cpp multiple-choice source hash mismatch: "
            f"expected {LLAMA_CPP_MULTIPLE_CHOICE_SOURCE_SHA256}, got {source_sha256}"
        )

    source_status = run_capture(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=llama_src
    ).rstrip("\r\n")
    if source_status != LLAMA_CPP_EXPECTED_SOURCE_STATUS:
        raise HarnessError(
            "official B3/B4 execution requires exactly the audited one-file llama.cpp patch; "
            f"expected status {LLAMA_CPP_EXPECTED_SOURCE_STATUS!r}, got {source_status!r}"
        )
    run_capture(["git", "diff", "--check"], cwd=llama_src)
    run_capture(["git", "apply", "--reverse", "--check", str(patch_path)], cwd=llama_src)

    return {
        "llama_cpp_commit": head,
        "llama_cpp_source": str(llama_src),
        "source_tree_clean": False,
        "source_tree_patch_exact": True,
        "source_status": source_status,
        "multiple_choice_source_relative_path": LLAMA_CPP_MULTIPLE_CHOICE_SOURCE_RELATIVE_PATH,
        "multiple_choice_source_original_sha256": original_source_sha256,
        "multiple_choice_source_sha256": source_sha256,
        "multiple_choice_patch_relative_path": LLAMA_CPP_MULTIPLE_CHOICE_PATCH_RELATIVE_PATH,
        "multiple_choice_patch_sha256": patch_sha256,
    }


def lane_paths(lane_dir: Path) -> dict[str, Path]:
    return {
        "prereg": lane_dir / "b1_prereg.json",
        "prereg_sha": lane_dir / "b1_prereg.sha256",
        "mmlu": lane_dir / "protocol" / "global_mmlu_80x5.jsonl",
        "generation": lane_dir / "protocol" / "native_generation_40x5.jsonl",
    }


def read_prereg(lane_dir: Path) -> tuple[dict[str, Any], str]:
    paths = lane_paths(lane_dir)
    for path in paths.values():
        if not path.is_file():
            raise HarnessError(f"missing frozen input: {path}")
    expected = paths["prereg_sha"].read_text(encoding="utf-8").split()[0]
    actual = sha256_file(paths["prereg"])
    if actual != expected:
        raise HarnessError(f"prereg hash mismatch: expected {expected}, got {actual}")
    prereg = load_json(paths["prereg"])
    return prereg, actual


def model_config(prereg: dict[str, Any], model_id: str) -> dict[str, Any]:
    if model_id not in MODEL_IDS:
        raise HarnessError(f"unknown model {model_id!r}; choose one of {MODEL_IDS}")
    matches = [row for row in prereg["models"] if row.get("model_id") == model_id]
    if len(matches) != 1:
        raise HarnessError(f"frozen prereg does not contain exactly one row for {model_id}")
    return matches[0]


def package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def require_package_version(distribution: str, expected: str) -> None:
    actual = package_version(distribution)
    if actual != expected:
        raise HarnessError(f"{distribution} must be {expected}; found {actual or 'MISSING'}")


def validate_protocol(lane_dir: Path) -> dict[str, Any]:
    prereg, prereg_sha = read_prereg(lane_dir)
    paths = lane_paths(lane_dir)

    if not ((3, 11) <= sys.version_info[:2] < (3, 14)):
        raise HarnessError(f"frozen Python requirement is >=3.11,<3.14; found {sys.version.split()[0]}")
    if sys.byteorder != "little" or struct.calcsize("i") != 4:
        raise HarnessError("llama.cpp multiple-choice serialization requires little-endian 32-bit int layout")
    if prereg.get("branch") != "sprint/b-backbone":
        raise HarnessError(f"unexpected frozen branch: {prereg.get('branch')!r}")
    if prereg.get("languages") != list(LANGUAGES):
        raise HarnessError("frozen language order changed")
    if prereg["harness"].get("llama_cpp_commit") != LLAMA_CPP_COMMIT:
        raise HarnessError("llama.cpp commit differs from the harness constant")

    mmlu_sha = sha256_file(paths["mmlu"])
    generation_sha = sha256_file(paths["generation"])
    expected_mmlu_sha = prereg["suites"]["global_mmlu"]["manifest_sha256"]
    expected_generation_sha = prereg["suites"]["native_generation"]["manifest_sha256"]
    if mmlu_sha != expected_mmlu_sha:
        raise HarnessError(f"Global-MMLU manifest hash mismatch: {mmlu_sha}")
    if generation_sha != expected_generation_sha:
        raise HarnessError(f"native-generation manifest hash mismatch: {generation_sha}")

    mmlu_rows = list(iter_jsonl(paths["mmlu"]))
    generation_rows = list(iter_jsonl(paths["generation"]))
    mmlu_counts = collections.Counter(row.get("language") for row in mmlu_rows)
    generation_counts = collections.Counter(row.get("language") for row in generation_rows)
    if len(mmlu_rows) != 400 or mmlu_counts != collections.Counter({language: 80 for language in LANGUAGES}):
        raise HarnessError(f"unexpected MMLU counts: total={len(mmlu_rows)}, by_language={dict(mmlu_counts)}")
    if len(generation_rows) != 200 or generation_counts != collections.Counter({language: 40 for language in LANGUAGES}):
        raise HarnessError(
            f"unexpected generation counts: total={len(generation_rows)}, by_language={dict(generation_counts)}"
        )

    sample_languages: dict[str, set[str]] = collections.defaultdict(set)
    for row in mmlu_rows:
        if row.get("answer") not in CHOICES:
            raise HarnessError(f"invalid gold answer in {row.get('item_id')}: {row.get('answer')!r}")
        sample_languages[str(row.get("sample_id"))].add(str(row.get("language")))
    if len(sample_languages) != 80 or any(value != set(LANGUAGES) for value in sample_languages.values()):
        raise HarnessError("Global-MMLU sample IDs are not exactly paired across all five languages")
    if len({row.get("item_id") for row in generation_rows}) != len(generation_rows):
        raise HarnessError("native-generation item IDs are not unique")
    if any(not isinstance(row.get("rubric"), dict) for row in generation_rows):
        raise HarnessError("native-generation row missing frozen rubric")

    frozen_models: dict[str, dict[str, Any]] = {}
    for model_id in MODEL_IDS:
        row = model_config(prereg, model_id)
        for key in (
            "base_repo",
            "base_revision",
            "tokenizer_repo",
            "tokenizer_revision",
            "quant_repo",
            "quant_revision",
            "quant_file",
            "quant_format",
            "quant_file_bytes",
            "quant_file_sha256",
        ):
            if key not in row:
                raise HarnessError(f"{model_id} is missing frozen field {key}")
        if not re.fullmatch(r"[0-9a-f]{40}", str(row["tokenizer_revision"])):
            raise HarnessError(f"{model_id} tokenizer revision is not a commit hash")
        if not re.fullmatch(r"[0-9a-f]{64}", str(row["quant_file_sha256"])):
            raise HarnessError(f"{model_id} GGUF SHA-256 is malformed")
        frozen_models[model_id] = {
            key: row[key]
            for key in (
                "tokenizer_repo",
                "tokenizer_revision",
                "quant_repo",
                "quant_revision",
                "quant_file",
                "quant_format",
                "quant_file_bytes",
                "quant_file_sha256",
            )
        }

    decoding = prereg["decoding"]
    expected_generation = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "repeat_penalty": 1.0,
        "seed": 1729,
        "max_new_tokens": 192,
        "context_tokens": 4096,
        "stream": False,
        "cache_prompt": False,
    }
    for key, expected in expected_generation.items():
        if decoding["native_generation"].get(key) != expected:
            raise HarnessError(f"generation setting drifted: {key}")
    if decoding["global_mmlu"].get("candidate_answers") != list(CHOICES):
        raise HarnessError("MCQ candidate answers drifted")

    return {
        "schema_version": "lane-b-b34-validation/1.0",
        "validated_at_utc": utc_now(),
        "harness_source_sha256": sha256_file(SCRIPT_PATH),
        "lane_dir": str(lane_dir.resolve()),
        "b1_prereg_sha256": prereg_sha,
        "global_mmlu_manifest_sha256": mmlu_sha,
        "native_generation_manifest_sha256": generation_sha,
        "global_mmlu_rows": len(mmlu_rows),
        "native_generation_rows": len(generation_rows),
        "language_order": list(LANGUAGES),
        "python_version": sys.version.split()[0],
        "llama_cpp_commit": LLAMA_CPP_COMMIT,
        "models": frozen_models,
        "status": "PASS",
    }


def _pack_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    if len(encoded) > 0xFFFFFFFF:
        raise HarnessError("string is too large for llama.cpp's uint32 serialization")
    return struct.pack("<I", len(encoded)) + encoded


def _pack_answers(answers: Sequence[str], labels: Sequence[int]) -> bytes:
    if len(answers) != len(labels):
        raise HarnessError("answer and label counts differ")
    return (
        struct.pack("<I", len(answers))
        + b"".join(_pack_string(answer) for answer in answers)
        + b"".join(struct.pack("<i", int(label)) for label in labels)
    )


def build_multiple_choice_binary(rows: Sequence[dict[str, Any]]) -> bytes:
    task_blobs: list[bytes] = []
    for row in rows:
        gold = row["gold_answer"]
        labels = [1 if choice == gold else 0 for choice in CHOICES]
        task_blobs.append(
            _pack_string(row["rendered_prompt"])
            + _pack_answers(CHOICES, labels)
            + _pack_answers((), ())
        )
    header_bytes = 4 + 4 * len(task_blobs)
    positions: list[int] = []
    cursor = header_bytes
    for blob in task_blobs:
        if cursor > 0xFFFFFFFF:
            raise HarnessError("multiple-choice binary exceeds uint32 addressing")
        positions.append(cursor)
        cursor += len(blob)
    return (
        struct.pack("<I", len(task_blobs))
        + b"".join(struct.pack("<I", position) for position in positions)
        + b"".join(task_blobs)
    )


def _first_hashes(rows: Sequence[dict[str, Any]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in rows:
        language = str(row["language"])
        result.setdefault(language, str(row["prompt_sha256"]))
    return result


def _render_prompt(tokenizer: Any, messages: list[dict[str, str]], *, disable_thinking: bool) -> str:
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if disable_thinking:
        kwargs["enable_thinking"] = False
    rendered = tokenizer.apply_chat_template(messages, **kwargs)
    if not isinstance(rendered, str) or not rendered:
        raise HarnessError("tokenizer chat template returned an empty or non-string prompt")
    repeated = tokenizer.apply_chat_template(messages, **kwargs)
    if repeated != rendered:
        raise HarnessError("chat-template rendering is nondeterministic")
    if disable_thinking and not re.search(r"<think>\s*</think>", rendered):
        raise HarnessError(
            "Qwen non-thinking render lacks the preregistered empty <think></think> block; "
            "do not substitute a prompt-level instruction"
        )
    return rendered


def render_prompts(lane_dir: Path, work_dir: Path, model_id: str, *, replace: bool = False) -> dict[str, Any]:
    validation = validate_protocol(lane_dir)
    prereg, prereg_sha = read_prereg(lane_dir)
    config = model_config(prereg, model_id)
    require_package_version("transformers", prereg["harness"]["transformers_version"])
    require_package_version("huggingface-hub", prereg["harness"]["huggingface_hub_version"])

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised in Colab setup failures
        raise HarnessError("transformers is required for frozen chat-template rendering") from exc
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - pinned by the Colab setup
        raise HarnessError("huggingface_hub is required to resolve the frozen tokenizer revision") from exc

    resolved_revision = HfApi().model_info(
        repo_id=config["tokenizer_repo"],
        revision=config["tokenizer_revision"],
    ).sha
    if resolved_revision != config["tokenizer_revision"]:
        raise HarnessError(
            f"tokenizer repository resolved to {resolved_revision}, expected {config['tokenizer_revision']}"
        )

    tokenizer_cache = work_dir / "tokenizers" / model_id
    tokenizer = AutoTokenizer.from_pretrained(
        config["tokenizer_repo"],
        revision=config["tokenizer_revision"],
        cache_dir=tokenizer_cache,
        trust_remote_code=False,
    )
    tokenizer_internal_revision = getattr(tokenizer, "_commit_hash", None) or tokenizer.init_kwargs.get("_commit_hash")
    if tokenizer_internal_revision and tokenizer_internal_revision != config["tokenizer_revision"]:
        raise HarnessError(
            f"tokenizer internal commit is {tokenizer_internal_revision}, expected {config['tokenizer_revision']}"
        )
    chat_template = tokenizer.chat_template
    if not isinstance(chat_template, str) or not chat_template:
        raise HarnessError(f"{model_id} tokenizer has no chat template")

    disable_thinking = model_id in {"qwen35_08b", "qwen3_06b"}
    prompts = prereg["prompts"]
    mmlu_rows: list[dict[str, Any]] = []
    for ordinal, source in enumerate(iter_jsonl(lane_paths(lane_dir)["mmlu"])):
        language = source["language"]
        user_text = prompts["mcq_user_template"].format(
            question=source["question"],
            option_a=source["option_a"],
            option_b=source["option_b"],
            option_c=source["option_c"],
            option_d=source["option_d"],
            localized_instruction=prompts["mcq_instruction_by_language"][language],
        )
        messages = [
            {"role": "system", "content": prompts["system_by_language"][language]},
            {"role": "user", "content": user_text},
        ]
        rendered = _render_prompt(tokenizer, messages, disable_thinking=disable_thinking)
        token_count = len(tokenizer.encode(rendered, add_special_tokens=False))
        if token_count + 4 >= prereg["decoding"]["global_mmlu"]["context_tokens"]:
            raise HarnessError(f"MMLU prompt too long for frozen context: {source['item_id']}")
        mmlu_rows.append(
            {
                "schema_version": RENDER_SCHEMA_VERSION,
                "suite": "global_mmlu",
                "ordinal": ordinal,
                "model_id": model_id,
                "language": language,
                "item_id": source["item_id"],
                "sample_id": source["sample_id"],
                "gold_answer": source["answer"],
                "candidate_answers": list(CHOICES),
                "candidate_continuations": [f" {choice}" for choice in CHOICES],
                "messages": messages,
                "rendered_prompt": rendered,
                "prompt_sha256": sha256_text(rendered),
                "hf_prompt_token_count": token_count,
                "source_item": source,
            }
        )

    generation_rows: list[dict[str, Any]] = []
    for ordinal, source in enumerate(iter_jsonl(lane_paths(lane_dir)["generation"])):
        language = source["language"]
        messages = [
            {"role": "system", "content": prompts["system_by_language"][language]},
            {"role": "user", "content": prompts["native_user_template"].format(prompt=source["prompt"])},
        ]
        rendered = _render_prompt(tokenizer, messages, disable_thinking=disable_thinking)
        token_count = len(tokenizer.encode(rendered, add_special_tokens=False))
        if token_count + prereg["decoding"]["native_generation"]["max_new_tokens"] >= prereg["decoding"]["native_generation"]["context_tokens"]:
            raise HarnessError(f"generation prompt too long for frozen context: {source['item_id']}")
        generation_rows.append(
            {
                "schema_version": RENDER_SCHEMA_VERSION,
                "suite": "native_generation",
                "ordinal": ordinal,
                "model_id": model_id,
                "language": language,
                "item_id": source["item_id"],
                "category": source["category"],
                "messages": messages,
                "rendered_prompt": rendered,
                "prompt_sha256": sha256_text(rendered),
                "hf_prompt_token_count": token_count,
                "source_item": source,
            }
        )

    output_dir = work_dir / "rendered" / model_id
    output_dir.mkdir(parents=True, exist_ok=True)
    mmlu_path = output_dir / "global_mmlu.rendered.jsonl"
    generation_path = output_dir / "native_generation.rendered.jsonl"
    binary_path = output_dir / "global_mmlu.multiple_choice.bin"
    receipt_path = output_dir / "render_receipt.json"
    write_jsonl(mmlu_path, mmlu_rows, replace=replace)
    write_jsonl(generation_path, generation_rows, replace=replace)
    binary = build_multiple_choice_binary(mmlu_rows)
    if binary_path.exists() and not replace:
        raise HarnessError(f"refusing to overwrite existing file: {binary_path}")
    binary_path.write_bytes(binary)

    receipt = {
        "schema_version": "lane-b-b34-render-receipt/1.0",
        "rendered_at_utc": utc_now(),
        "harness_source_sha256": sha256_file(SCRIPT_PATH),
        "model_id": model_id,
        "b1_prereg_sha256": prereg_sha,
        "tokenizer_repo": config["tokenizer_repo"],
        "tokenizer_revision": config["tokenizer_revision"],
        "resolved_tokenizer_revision": resolved_revision,
        "tokenizer_internal_commit_hash": tokenizer_internal_revision,
        "chat_template_sha256": sha256_text(chat_template),
        "thinking_disabled_at_template": disable_thinking,
        "transformers_version": package_version("transformers"),
        "huggingface_hub_version": package_version("huggingface-hub"),
        "global_mmlu_source_sha256": validation["global_mmlu_manifest_sha256"],
        "native_generation_source_sha256": validation["native_generation_manifest_sha256"],
        "global_mmlu_rendered_sha256": sha256_file(mmlu_path),
        "native_generation_rendered_sha256": sha256_file(generation_path),
        "multiple_choice_binary_sha256": sha256_file(binary_path),
        "global_mmlu_count": len(mmlu_rows),
        "native_generation_count": len(generation_rows),
        "first_prompt_sha256": {
            "global_mmlu": _first_hashes(mmlu_rows),
            "native_generation": _first_hashes(generation_rows),
        },
        "candidate_continuations": [f" {choice}" for choice in CHOICES],
        "status": "PASS",
    }
    write_json(receipt_path, receipt, replace=replace)
    return receipt


def _safe_current_model_dir(work_dir: Path) -> Path:
    work = work_dir.resolve()
    current = (work / "current_model").resolve()
    if current.parent != work:
        raise HarnessError("current-model path escaped the work directory")
    return current


def prepare_model(lane_dir: Path, work_dir: Path, model_id: str, *, replace_current: bool) -> dict[str, Any]:
    prereg, prereg_sha = read_prereg(lane_dir)
    config = model_config(prereg, model_id)
    require_package_version("huggingface-hub", prereg["harness"]["huggingface_hub_version"])
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover
        raise HarnessError("huggingface_hub is required to download the frozen GGUF") from exc

    current = _safe_current_model_dir(work_dir)
    if current.exists():
        if not replace_current:
            raise HarnessError(
                f"{current} already exists; purge it after the previous arm or pass --replace-current"
            )
        shutil.rmtree(current)
    current.mkdir(parents=True)

    downloaded = Path(
        hf_hub_download(
            repo_id=config["quant_repo"],
            filename=config["quant_file"],
            revision=config["quant_revision"],
            local_dir=current,
        )
    ).resolve()
    if downloaded.parent != current:
        raise HarnessError(f"download resolved outside current-model directory: {downloaded}")
    actual_bytes = downloaded.stat().st_size
    actual_sha = sha256_file(downloaded)
    if actual_bytes != int(config["quant_file_bytes"]):
        raise HarnessError(f"GGUF byte mismatch: expected {config['quant_file_bytes']}, got {actual_bytes}")
    if actual_sha != config["quant_file_sha256"]:
        raise HarnessError(f"GGUF SHA mismatch: expected {config['quant_file_sha256']}, got {actual_sha}")

    receipt = {
        "schema_version": "lane-b-b34-model-receipt/1.0",
        "downloaded_at_utc": utc_now(),
        "harness_source_sha256": sha256_file(SCRIPT_PATH),
        "model_id": model_id,
        "b1_prereg_sha256": prereg_sha,
        "quant_repo": config["quant_repo"],
        "quant_revision": config["quant_revision"],
        "quant_file": config["quant_file"],
        "quant_format": config["quant_format"],
        "quant_file_bytes": actual_bytes,
        "quant_file_sha256": actual_sha,
        "relative_model_path": downloaded.name,
        "huggingface_hub_version": package_version("huggingface-hub"),
        "status": "PASS",
    }
    write_json(current / "model_receipt.json", receipt)
    return receipt


def purge_model(work_dir: Path) -> dict[str, Any]:
    current = _safe_current_model_dir(work_dir)
    if not current.exists():
        return {"removed": False, "path": str(current)}
    receipt_path = current / "model_receipt.json"
    if not receipt_path.is_file():
        raise HarnessError(f"refusing to remove unreceipted directory: {current}")
    receipt = load_json(receipt_path)
    shutil.rmtree(current)
    return {"removed": True, "path": str(current), "model_id": receipt.get("model_id")}


def verified_model_path(lane_dir: Path, work_dir: Path, model_id: str) -> tuple[Path, dict[str, Any]]:
    prereg, _ = read_prereg(lane_dir)
    config = model_config(prereg, model_id)
    current = _safe_current_model_dir(work_dir)
    receipt = load_json(current / "model_receipt.json")
    if receipt.get("model_id") != model_id:
        raise HarnessError(f"current model is {receipt.get('model_id')}, requested {model_id}")
    model_path = (current / str(receipt["relative_model_path"])).resolve()
    if model_path.parent != current or not model_path.is_file():
        raise HarnessError("current model path is invalid")
    if model_path.stat().st_size != int(config["quant_file_bytes"]):
        raise HarnessError("current GGUF byte count no longer matches the frozen artifact")
    actual_sha = sha256_file(model_path)
    if actual_sha != config["quant_file_sha256"] or actual_sha != receipt.get("quant_file_sha256"):
        raise HarnessError("current GGUF SHA-256 no longer matches the frozen artifact")
    return model_path, receipt


def multiple_choice_impl_library_receipt(llama_bin_dir: Path) -> dict[str, Any]:
    path = llama_bin_dir / LLAMA_CPP_PERPLEXITY_IMPL_LIBRARY_NAME
    if not path.is_file():
        raise HarnessError(f"missing llama-perplexity implementation library: {path}")
    return {
        "path": str(path.resolve()),
        "relative_path": LLAMA_CPP_PERPLEXITY_IMPL_LIBRARY_RELATIVE_PATH,
        "sha256": sha256_file(path),
    }


def cmake_build_configuration_receipt(llama_src: Path, llama_bin_dir: Path) -> dict[str, Any]:
    llama_src = llama_src.resolve()
    llama_bin_dir = llama_bin_dir.resolve()
    cache_path = llama_bin_dir.parent / "CMakeCache.txt"
    if not cache_path.is_file():
        raise HarnessError(f"missing CMake cache for llama.cpp build: {cache_path}")

    cache: dict[str, str] = {}
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith(("#", "//")) or "=" not in line:
            continue
        key_and_type, value = line.split("=", 1)
        key = key_and_type.split(":", 1)[0]
        if key in cache:
            raise HarnessError(f"duplicate CMake cache key: {key}")
        cache[key] = value

    required = {
        "CMAKE_BUILD_TYPE": "Release",
        "GGML_CUDA": "ON",
        "BUILD_SHARED_LIBS": "ON",
    }
    for key, expected in required.items():
        if cache.get(key) != expected:
            raise HarnessError(f"llama.cpp CMake setting {key} must be {expected}; got {cache.get(key)!r}")

    home_value = cache.get("CMAKE_HOME_DIRECTORY")
    build_value = cache.get("CMAKE_CACHEFILE_DIR")
    if not home_value or not build_value:
        raise HarnessError("llama.cpp CMake cache is missing source/build directory provenance")
    cmake_home = Path(home_value).resolve()
    cmake_build = Path(build_value).resolve()
    if cmake_home != llama_src:
        raise HarnessError(f"CMake source directory is {cmake_home}, expected {llama_src}")
    if cmake_build != cache_path.parent.resolve():
        raise HarnessError(
            f"CMake cache directory is {cmake_build}, but the validated cache is under {cache_path.parent.resolve()}"
        )
    expected_bin_dir = (cmake_build / "bin").resolve()
    if llama_bin_dir != expected_bin_dir:
        raise HarnessError(f"llama bin directory is {llama_bin_dir}, expected {expected_bin_dir}")

    return {
        "cache_path": str(cache_path.resolve()),
        "cache_sha256": sha256_file(cache_path),
        "CMAKE_HOME_DIRECTORY": str(cmake_home),
        "CMAKE_CACHEFILE_DIR": str(cmake_build),
        "CMAKE_BUILD_TYPE": required["CMAKE_BUILD_TYPE"],
        "GGML_CUDA": required["GGML_CUDA"],
        "BUILD_SHARED_LIBS": required["BUILD_SHARED_LIBS"],
        "llama_bin_dir": str(llama_bin_dir),
    }


def record_build(lane_dir: Path, work_dir: Path, llama_src: Path, llama_bin_dir: Path) -> dict[str, Any]:
    validate_protocol(lane_dir)
    source_receipt = validate_llama_source_patch(lane_dir, llama_src)
    cmake_receipt = cmake_build_configuration_receipt(llama_src, llama_bin_dir)
    binaries = {name: llama_bin_dir / name for name in ("llama-perplexity", "llama-server")}
    for name, path in binaries.items():
        if not path.is_file() or not os.access(path, os.X_OK):
            raise HarnessError(f"missing executable {name}: {path}")
    receipt = {
        "schema_version": "lane-b-b34-build-receipt/1.0",
        "recorded_at_utc": utc_now(),
        "harness_source_sha256": sha256_file(SCRIPT_PATH),
        **source_receipt,
        "ggml_cuda": True,
        "cmake_cache_sha256": cmake_receipt["cache_sha256"],
        "cmake_configuration": cmake_receipt,
        "multiple_choice_impl_library": multiple_choice_impl_library_receipt(llama_bin_dir),
        "binaries": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "version_output": run_capture([str(path), "--version"], check=False).strip(),
            }
            for name, path in binaries.items()
        },
        "nvidia_smi": run_capture(["nvidia-smi"], check=False).strip(),
        "nvcc_version": run_capture(["nvcc", "--version"], check=False).strip(),
        "status": "PASS",
    }
    write_json(work_dir / "build_receipt.json", receipt, replace=True)
    return receipt


def load_rendered(work_dir: Path, model_id: str, suite: str) -> tuple[list[dict[str, Any]], dict[str, Any], Path]:
    rendered_dir = work_dir / "rendered" / model_id
    receipt = load_json(rendered_dir / "render_receipt.json")
    if receipt.get("model_id") != model_id or receipt.get("status") != "PASS":
        raise HarnessError("render receipt does not match the requested model")
    filename = {
        "global_mmlu": "global_mmlu.rendered.jsonl",
        "native_generation": "native_generation.rendered.jsonl",
    }[suite]
    path = rendered_dir / filename
    rows = list(iter_jsonl(path))
    expected_count = 400 if suite == "global_mmlu" else 200
    if len(rows) != expected_count:
        raise HarnessError(f"rendered {suite} has {len(rows)} rows, expected {expected_count}")
    expected_sha = receipt[f"{suite}_rendered_sha256"]
    if sha256_file(path) != expected_sha:
        raise HarnessError(f"rendered {suite} hash no longer matches its receipt")
    return rows, receipt, path


def validate_build_for_run(lane_dir: Path, work_dir: Path, llama_bin_dir: Path) -> dict[str, Any]:
    receipt = load_json(work_dir / "build_receipt.json")
    if receipt.get("llama_cpp_commit") != LLAMA_CPP_COMMIT or receipt.get("status") != "PASS":
        raise HarnessError("build receipt does not match the frozen llama.cpp commit")
    if receipt.get("harness_source_sha256") != sha256_file(SCRIPT_PATH):
        raise HarnessError("build receipt was created by a different harness source")
    llama_src = Path(str(receipt.get("llama_cpp_source", "")))
    source_receipt = validate_llama_source_patch(lane_dir, llama_src)
    for key, expected in source_receipt.items():
        if receipt.get(key) != expected:
            raise HarnessError(f"build receipt llama.cpp patch field changed: {key}")
    if receipt.get("ggml_cuda") is not True:
        raise HarnessError("official run requires a GGML_CUDA=ON build receipt")
    cmake_receipt = cmake_build_configuration_receipt(llama_src, llama_bin_dir)
    if receipt.get("cmake_cache_sha256") != cmake_receipt["cache_sha256"]:
        raise HarnessError("llama.cpp CMake cache differs from the recorded build")
    if receipt.get("cmake_configuration") != cmake_receipt:
        raise HarnessError("llama.cpp CMake configuration differs from the recorded build")
    impl_library = multiple_choice_impl_library_receipt(llama_bin_dir)
    if receipt.get("multiple_choice_impl_library") != impl_library:
        raise HarnessError("llama-perplexity implementation library differs from the recorded build")
    for name in ("llama-perplexity", "llama-server"):
        path = llama_bin_dir / name
        if not path.is_file() or sha256_file(path) != receipt["binaries"][name]["sha256"]:
            raise HarnessError(f"{name} differs from the recorded build")
    return receipt


def parse_cumulative_accuracy(output: str, n_tasks: int) -> dict[int, tuple[float, str]]:
    trace: dict[int, tuple[float, str]] = {}
    trace_pattern = re.compile(r"^\s*(\d+)\s+([0-9]+(?:\.[0-9]+)?)\s*$")
    for line in output.splitlines():
        clean_line = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", line)
        match = trace_pattern.fullmatch(clean_line)
        if not match:
            continue
        n_done = int(match.group(1))
        if not 1 <= n_done <= n_tasks:
            continue
        if n_done in trace:
            raise HarnessError(f"duplicate cumulative-accuracy row for task count {n_done}")
        trace[n_done] = (float(match.group(2)), line)
    if set(trace) != set(range(1, n_tasks + 1)):
        missing = sorted(set(range(1, n_tasks + 1)) - set(trace))[:10]
        raise HarnessError(f"expected {n_tasks} cumulative-accuracy rows, found {len(trace)}; missing={missing}")
    return trace


def reconstruct_item_correctness(
    trace: dict[int, tuple[float, str]], n_tasks: int
) -> list[dict[str, Any]]:
    reconstructed: list[dict[str, Any]] = []
    previous_correct = 0
    for n_done in range(1, n_tasks + 1):
        cumulative_accuracy_pct, raw_line = trace[n_done]
        cumulative_correct = int(round(cumulative_accuracy_pct * n_done / 100.0))
        reconstructed_pct = 100.0 * cumulative_correct / n_done
        if abs(reconstructed_pct - cumulative_accuracy_pct) > 5e-7:
            raise HarnessError(
                f"task {n_done - 1} cumulative accuracy cannot be reconstructed exactly: "
                f"printed={cumulative_accuracy_pct}, reconstructed={reconstructed_pct}"
            )
        item_increment = cumulative_correct - previous_correct
        if item_increment not in {0, 1}:
            raise HarnessError(f"task {n_done - 1} has impossible cumulative-correct increment {item_increment}")
        reconstructed.append(
            {
                "correct": bool(item_increment),
                "cumulative_correct": cumulative_correct,
                "cumulative_accuracy_pct": cumulative_accuracy_pct,
                "raw_line": raw_line,
            }
        )
        previous_correct = cumulative_correct
    return reconstructed


def run_mcq(
    lane_dir: Path,
    work_dir: Path,
    llama_bin_dir: Path,
    model_id: str,
    output: Path | None,
) -> dict[str, Any]:
    prereg, prereg_sha = read_prereg(lane_dir)
    config = model_config(prereg, model_id)
    build_receipt = validate_build_for_run(lane_dir, work_dir, llama_bin_dir)
    model_path, model_receipt = verified_model_path(lane_dir, work_dir, model_id)
    rendered, render_receipt, _ = load_rendered(work_dir, model_id, "global_mmlu")
    binary_path = work_dir / "rendered" / model_id / "global_mmlu.multiple_choice.bin"
    if sha256_file(binary_path) != render_receipt["multiple_choice_binary_sha256"]:
        raise HarnessError("multiple-choice binary hash no longer matches render receipt")

    raw_dir = work_dir / "raw" / model_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    output = output or raw_dir / "global_mmlu.raw.jsonl"
    if output.exists():
        raise HarnessError(f"refusing to overwrite MCQ raw output: {output}")
    log_path = output.with_suffix(output.suffix + ".llama.log")
    receipt_path = output.with_suffix(output.suffix + ".receipt.json")

    mcq = prereg["decoding"]["global_mmlu"]
    command = [
        str(llama_bin_dir / "llama-perplexity"),
        "--model",
        str(model_path),
        "--binary-file",
        str(binary_path),
        "--multiple-choice",
        "--ctx-size",
        str(MCQ_CONTEXT_TOKENS),
        "--batch-size",
        str(MCQ_BATCH_SIZE),
        "--ubatch-size",
        str(MCQ_UBATCH_SIZE),
        "--parallel",
        str(MCQ_PARALLEL),
        "--n-gpu-layers",
        "999",
        "--seed",
        str(mcq["seed"]),
    ]
    started = utc_now()
    start_monotonic = time.monotonic()
    process = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    elapsed = time.monotonic() - start_monotonic
    log_path.write_text(process.stdout, encoding="utf-8")
    if process.returncode != 0:
        raise HarnessError(f"llama-perplexity failed ({process.returncode}); see {log_path}")

    accuracy_trace = parse_cumulative_accuracy(process.stdout, len(rendered))
    correctness_rows = reconstruct_item_correctness(accuracy_trace, len(rendered))

    raw_rows: list[dict[str, Any]] = []
    for index, prompt_row in enumerate(rendered):
        correctness = correctness_rows[index]
        correct = bool(correctness["correct"])
        gold_index = CHOICES.index(prompt_row["gold_answer"])
        raw_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "run_kind": "global_mmlu_log_likelihood",
                "evidence_class": "HOST",
                "model_id": model_id,
                "quant_format": config["quant_format"],
                "quant_file_sha256": model_receipt["quant_file_sha256"],
                "llama_cpp_commit": LLAMA_CPP_COMMIT,
                "ordinal": index,
                "item_id": prompt_row["item_id"],
                "sample_id": prompt_row["sample_id"],
                "language": prompt_row["language"],
                "prompt_sha256": prompt_row["prompt_sha256"],
                "rendered_prompt": prompt_row["rendered_prompt"],
                "candidate_answers": list(CHOICES),
                "candidate_continuations": [f" {choice}" for choice in CHOICES],
                "scoring_method": "stock llama-perplexity --multiple-choice; maximum length-normalized continuation log-likelihood",
                "choice_logprob_sums": None,
                "choice_token_counts": None,
                "choice_mean_logprobs": None,
                "predicted_index": None,
                "predicted_answer": None,
                "gold_index": gold_index,
                "gold_answer": prompt_row["gold_answer"],
                "correct": correct,
                "cumulative_correct": correctness["cumulative_correct"],
                "cumulative_accuracy_pct": correctness["cumulative_accuracy_pct"],
                "correctness_reconstruction": "difference of successive integer cumulative-correct counts reconstructed from stock 8-decimal accuracy output",
                "raw_llama_line": correctness["raw_line"],
                "source_item": prompt_row["source_item"],
            }
        )
    write_jsonl(output, raw_rows)

    receipt = {
        "schema_version": "lane-b-b34-run-receipt/1.0",
        "run_kind": "global_mmlu_log_likelihood",
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "harness_source_sha256": sha256_file(SCRIPT_PATH),
        "elapsed_seconds": elapsed,
        "model_id": model_id,
        "b1_prereg_sha256": prereg_sha,
        "model_receipt": model_receipt,
        "render_receipt_sha256": sha256_file(work_dir / "rendered" / model_id / "render_receipt.json"),
        "build_receipt_sha256": sha256_file(work_dir / "build_receipt.json"),
        "command": command,
        "execution_batching": {
            "ctx_size": MCQ_CONTEXT_TOKENS,
            "batch_size": MCQ_BATCH_SIZE,
            "ubatch_size": MCQ_UBATCH_SIZE,
            "parallel": MCQ_PARALLEL,
        },
        "candidate_continuations": [f" {choice}" for choice in CHOICES],
        "stock_multiple_choice_binary_unmodified": False,
        "stock_multiple_choice_scoring_unmodified": True,
        "multiple_choice_memory_patch_exact": True,
        "multiple_choice_patch_sha256": LLAMA_CPP_MULTIPLE_CHOICE_PATCH_SHA256,
        "per_choice_logprob_vectors_available": False,
        "per_item_correctness_source": "reconstructed from stock cumulative-accuracy trace",
        "n": len(raw_rows),
        "correct": sum(int(row["correct"]) for row in raw_rows),
        "raw_jsonl": str(output.resolve()),
        "raw_jsonl_sha256": sha256_file(output),
        "llama_log": str(log_path.resolve()),
        "llama_log_sha256": sha256_file(log_path),
        "build_receipt": build_receipt,
        "status": "PASS",
    }
    write_json(receipt_path, receipt)
    return receipt


def _http_json(url: str, body: dict[str, Any] | None = None, timeout: float = 30.0) -> dict[str, Any]:
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if body is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise HarnessError(f"expected JSON object from {url}")
    return payload


def _wait_for_server(process: subprocess.Popen[Any], port: int, timeout_seconds: float = 300.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise HarnessError(f"llama-server exited during startup with code {process.returncode}")
        try:
            health = _http_json(f"http://127.0.0.1:{port}/health", timeout=5.0)
            if health.get("status") == "ok":
                return health
        except (OSError, ValueError, urllib.error.HTTPError, HarnessError) as exc:
            last_error = exc
        time.sleep(1.0)
    raise HarnessError(f"llama-server did not become healthy: {last_error}")


def _load_existing_generation(path: Path, model_id: str) -> dict[str, str]:
    completed: dict[str, str] = {}
    if not path.exists():
        return completed
    for row in iter_jsonl(path):
        if row.get("model_id") != model_id:
            raise HarnessError(f"resume file contains a different model: {path}")
        item_id = str(row["item_id"])
        if item_id in completed:
            raise HarnessError(f"resume file contains duplicate item {item_id}")
        completed[item_id] = str(row["prompt_sha256"])
    return completed


def run_generation(
    lane_dir: Path,
    work_dir: Path,
    llama_bin_dir: Path,
    model_id: str,
    output: Path | None,
    *,
    port: int,
    resume: bool,
    limit: int | None,
) -> dict[str, Any]:
    prereg, prereg_sha = read_prereg(lane_dir)
    config = model_config(prereg, model_id)
    build_receipt = validate_build_for_run(lane_dir, work_dir, llama_bin_dir)
    model_path, model_receipt = verified_model_path(lane_dir, work_dir, model_id)
    rendered, _, _ = load_rendered(work_dir, model_id, "native_generation")
    render_receipt_path = work_dir / "rendered" / model_id / "render_receipt.json"

    raw_dir = work_dir / "raw" / model_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    output = output or raw_dir / "native_generation.raw.jsonl"
    if output.exists() and not resume:
        raise HarnessError(f"raw generation output already exists; pass --resume to continue: {output}")
    completed = _load_existing_generation(output, model_id) if resume else {}
    for row in rendered:
        if row["item_id"] in completed and completed[row["item_id"]] != row["prompt_sha256"]:
            raise HarnessError(f"resume prompt hash mismatch for {row['item_id']}")
    pending = [row for row in rendered if row["item_id"] not in completed]
    if limit is not None:
        if limit <= 0:
            raise HarnessError("--limit must be positive")
        pending = pending[:limit]

    log_path = output.with_suffix(output.suffix + ".server.log")
    receipt_path = output.with_suffix(output.suffix + ".receipt.json")
    server_command = [
        str(llama_bin_dir / "llama-server"),
        "--model",
        str(model_path),
        "--ctx-size",
        "4096",
        "--batch-size",
        "4096",
        "--ubatch-size",
        "512",
        "--parallel",
        "1",
        "--n-gpu-layers",
        "999",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    generation = prereg["decoding"]["native_generation"]
    request_template = {
        "n_predict": generation["max_new_tokens"],
        "temperature": generation["temperature"],
        "top_p": generation["top_p"],
        "top_k": generation["top_k"],
        "min_p": generation["min_p"],
        "repeat_penalty": generation["repeat_penalty"],
        "seed": generation["seed"],
        "stream": generation["stream"],
        "cache_prompt": generation["cache_prompt"],
        "stop": generation["stop_strings"],
    }

    started = utc_now()
    start_monotonic = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a" if resume else "w", encoding="utf-8") as server_log:
        process = subprocess.Popen(server_command, stdout=server_log, stderr=subprocess.STDOUT, text=True)
        try:
            health = _wait_for_server(process, port)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("a", encoding="utf-8") as raw_handle:
                for prompt_row in pending:
                    request_body = {**request_template, "prompt": prompt_row["rendered_prompt"]}
                    item_started = utc_now()
                    item_monotonic = time.monotonic()
                    response = _http_json(
                        f"http://127.0.0.1:{port}/completion",
                        request_body,
                        timeout=900.0,
                    )
                    completion = response.get("content")
                    if not isinstance(completion, str):
                        raise HarnessError(f"server returned no string completion for {prompt_row['item_id']}")
                    raw_row = {
                        "schema_version": SCHEMA_VERSION,
                        "run_kind": "native_generation_greedy",
                        "evidence_class": "HOST",
                        "model_id": model_id,
                        "quant_format": config["quant_format"],
                        "quant_file_sha256": model_receipt["quant_file_sha256"],
                        "llama_cpp_commit": LLAMA_CPP_COMMIT,
                        "ordinal": prompt_row["ordinal"],
                        "item_id": prompt_row["item_id"],
                        "language": prompt_row["language"],
                        "category": prompt_row["category"],
                        "prompt_sha256": prompt_row["prompt_sha256"],
                        "rendered_prompt": prompt_row["rendered_prompt"],
                        "request": request_body,
                        "raw_completion": completion,
                        "raw_completion_sha256": sha256_text(completion),
                        "server_response": response,
                        "started_at_utc": item_started,
                        "elapsed_seconds": time.monotonic() - item_monotonic,
                        "source_item": prompt_row["source_item"],
                    }
                    raw_handle.write(json.dumps(raw_row, ensure_ascii=False, sort_keys=True) + "\n")
                    raw_handle.flush()
                    os.fsync(raw_handle.fileno())
        finally:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)

    final_rows = list(iter_jsonl(output))
    complete_ids = {row["item_id"] for row in final_rows}
    expected_ids = {row["item_id"] for row in rendered}
    is_complete = complete_ids == expected_ids
    receipt = {
        "schema_version": "lane-b-b34-run-receipt/1.0",
        "run_kind": "native_generation_greedy",
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "harness_source_sha256": sha256_file(SCRIPT_PATH),
        "elapsed_seconds": time.monotonic() - start_monotonic,
        "model_id": model_id,
        "b1_prereg_sha256": prereg_sha,
        "model_receipt": model_receipt,
        "render_receipt_sha256": sha256_file(render_receipt_path),
        "build_receipt_sha256": sha256_file(work_dir / "build_receipt.json"),
        "build_receipt": build_receipt,
        "server_command": server_command,
        "request_template": request_template,
        "server_health": health,
        "n_expected": len(rendered),
        "n_present": len(final_rows),
        "n_run_this_invocation": len(pending),
        "is_complete": is_complete,
        "smoke_or_partial_limit": limit,
        "raw_jsonl": str(output.resolve()),
        "raw_jsonl_sha256": sha256_file(output),
        "server_log": str(log_path.resolve()),
        "server_log_sha256": sha256_file(log_path),
        "status": "PASS" if is_complete else "PARTIAL",
    }
    write_json(receipt_path, receipt, replace=resume and receipt_path.exists())
    return receipt


def default_work_dir() -> Path:
    content = Path("/content")
    return content / "lane_b_b34" if content.is_dir() else Path("/tmp/lane_b_b34")


def add_common_paths(parser: argparse.ArgumentParser, *, work: bool = False) -> None:
    parser.add_argument("--lane-dir", type=Path, default=SCRIPT_DIR, help="directory containing b1_prereg.json")
    if work:
        parser.add_argument("--work-dir", type=Path, default=default_work_dir(), help="Colab scratch/output directory")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="validate all frozen B1 inputs without loading a model")
    add_common_paths(validate_parser)
    validate_parser.add_argument("--output", type=Path)

    prepare_parser = subparsers.add_parser("prepare-model", help="download and verify one frozen GGUF")
    add_common_paths(prepare_parser, work=True)
    prepare_parser.add_argument("model_id", choices=MODEL_IDS)
    prepare_parser.add_argument("--replace-current", action="store_true")

    purge_parser = subparsers.add_parser("purge-model", help="remove the receipted current GGUF from Colab scratch")
    add_common_paths(purge_parser, work=True)

    render_parser = subparsers.add_parser("render", help="render and hash all frozen prompts for one model")
    add_common_paths(render_parser, work=True)
    render_parser.add_argument("model_id", choices=MODEL_IDS)
    render_parser.add_argument("--replace-rendered", action="store_true")

    build_parser_ = subparsers.add_parser(
        "record-build", help="verify and receipt the pinned llama.cpp build with the exact MCQ memory patch"
    )
    add_common_paths(build_parser_, work=True)
    build_parser_.add_argument("--llama-src", type=Path, required=True)
    build_parser_.add_argument("--llama-bin-dir", type=Path, required=True)

    mcq_parser = subparsers.add_parser("run-mcq", help="run frozen four-choice continuation likelihood scoring")
    add_common_paths(mcq_parser, work=True)
    mcq_parser.add_argument("model_id", choices=MODEL_IDS)
    mcq_parser.add_argument("--llama-bin-dir", type=Path, required=True)
    mcq_parser.add_argument("--output", type=Path)

    generation_parser = subparsers.add_parser("run-generation", help="run frozen greedy native generation")
    add_common_paths(generation_parser, work=True)
    generation_parser.add_argument("model_id", choices=MODEL_IDS)
    generation_parser.add_argument("--llama-bin-dir", type=Path, required=True)
    generation_parser.add_argument("--output", type=Path)
    generation_parser.add_argument("--port", type=int, default=8080)
    generation_parser.add_argument("--resume", action="store_true")
    generation_parser.add_argument("--limit", type=int, help="smoke/partial run only; receipt remains PARTIAL")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.command == "validate":
            result = validate_protocol(args.lane_dir)
            if args.output:
                write_json(args.output, result, replace=True)
        elif args.command == "prepare-model":
            result = prepare_model(
                args.lane_dir,
                args.work_dir,
                args.model_id,
                replace_current=args.replace_current,
            )
        elif args.command == "purge-model":
            result = purge_model(args.work_dir)
        elif args.command == "render":
            result = render_prompts(
                args.lane_dir,
                args.work_dir,
                args.model_id,
                replace=args.replace_rendered,
            )
        elif args.command == "record-build":
            result = record_build(
                args.lane_dir,
                args.work_dir,
                args.llama_src,
                args.llama_bin_dir,
            )
        elif args.command == "run-mcq":
            result = run_mcq(
                args.lane_dir,
                args.work_dir,
                args.llama_bin_dir,
                args.model_id,
                args.output,
            )
        elif args.command == "run-generation":
            result = run_generation(
                args.lane_dir,
                args.work_dir,
                args.llama_bin_dir,
                args.model_id,
                args.output,
                port=args.port,
                resume=args.resume,
                limit=args.limit,
            )
        else:  # pragma: no cover
            raise AssertionError(args.command)
    except HarnessError as exc:
        print(json.dumps({"status": "ERROR", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2) from exc
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
