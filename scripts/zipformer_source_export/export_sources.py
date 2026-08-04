#!/usr/bin/env python3
"""Pinned QAIHM 0.58 Zipformer source exporter.

The heavy imports and model construction happen only in ``run_export`` so that
``--self-check`` and the offline unit tests remain light on the host Mac.  The
authoritative ``.pt`` files are created through QAIHM's own component
serialization method.  Fixed-shape ONNX mirrors are then exported from the
saved TorchScript files and compared on identical deterministic inputs.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tarfile
import tempfile
from typing import Any


FORMAT_VERSION = 1
COMPONENT_ORDER = ("decoder", "joiner", "encoder")
VALIDATION_SEED = 20260723
ONNX_OPSET = 17
DEFAULT_THREADS = 4

QAIHM_VERSION = "0.58.0"
QAIHM_WHEEL_SHA256 = "2100a95f173c6c7b8f3bf65dcb710586098099522a5a901c6f918f3b981c4ada"
QAI_HUB_VERSION = "0.52.0"
TORCH_VERSION = "2.9.0"
TORCHVISION_VERSION = "0.24.0"
ONNX_VERSION = "1.18.0"
ONNXRUNTIME_VERSION = "1.22.1"
HUGGINGFACE_HUB_VERSION = "0.34.0"
K2_VERSION = "1.24.4.dev20251029+cpu.torch2.9.0"
ICEFALL_COMMIT = "693f069de73fd91d7c2009571245d97221cc3a3f"
ICEFALL_PATCH_SHA256 = (
    "bca5978ea4674c956e133880f3212b5961303023ba3b732eda3fde486a19245d"
)

CHECKPOINT_REPO = "pfluo/k2fsa-zipformer-chinese-english-mixed"
CHECKPOINT_REVISION = "6eb615ae77ecac05c5628d5c8ed7037c14a338d5"
CHECKPOINT_FILENAME = "exp/pretrained.pt"
CHECKPOINT_SIZE = 302_404_721
CHECKPOINT_SHA256 = "9fb1a61c426c3910dc7228aec7c3e5783f268c15a25eb93cf37db9faf4f8e243"

PACKAGE_NAMES = (
    "qai-hub-models",
    "qai-hub",
    "torch",
    "torchvision",
    "onnx",
    "onnxruntime",
    "huggingface-hub",
    "k2",
)

_URL_RE = re.compile(r"https?://[^\s'\"<>]+", re.IGNORECASE)
_SECRET_RE = re.compile(
    r"(?i)(authorization|access[_-]?token|refresh[_-]?token|id[_-]?token|"
    r"api[_-]?key|client[_-]?secret)(\s*[:=]\s*)([^\s,;]+)"
)
_RUNTIME_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])/(?:content|tmp|var/tmp|private/tmp)/[^\s,;]+"
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    text = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    atomic_write_bytes(path, text.encode("utf-8"))


def load_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def sanitize_error(value: object, limit: int = 1200) -> str:
    text = " ".join(str(value).split())
    text = _URL_RE.sub("<url>", text)
    text = _SECRET_RE.sub(
        lambda match: f"{match.group(1)}{match.group(2)}<redacted>", text
    )
    text = text.replace(str(Path.home()), "<home>")
    text = _RUNTIME_PATH_RE.sub("<absolute_path>", text)
    return text[:limit]


def normalize_components(values: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, str):
        candidates = values.split(",")
    else:
        candidates = list(values)
    requested: list[str] = []
    for value in candidates:
        name = str(value).strip().lower()
        if not name:
            continue
        if name not in COMPONENT_ORDER:
            raise ValueError(f"unknown Zipformer component: {name}")
        if name not in requested:
            requested.append(name)
    if not requested:
        raise ValueError("at least one component is required")
    return tuple(name for name in COMPONENT_ORDER if name in requested)


def pin_manifest() -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "qaihm": {
            "version": QAIHM_VERSION,
            "wheel_sha256": QAIHM_WHEEL_SHA256,
        },
        "checkpoint": {
            "repo": CHECKPOINT_REPO,
            "revision": CHECKPOINT_REVISION,
            "filename": CHECKPOINT_FILENAME,
            "size_bytes": CHECKPOINT_SIZE,
            "sha256": CHECKPOINT_SHA256,
        },
        "icefall": {
            "commit": ICEFALL_COMMIT,
            "qualcomm_patch_sha256": ICEFALL_PATCH_SHA256,
        },
        "export": {
            "component_order": list(COMPONENT_ORDER),
            "validation_seed": VALIDATION_SEED,
            "onnx_opset": ONNX_OPSET,
            "onnx_dynamo": False,
            "dynamic_axes": False,
        },
    }


def installed_versions() -> dict[str, str | None]:
    import importlib.metadata

    versions: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _artifact_paths(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for directory in (root / "sources", root / "validation", root / "manifests"):
        if directory.is_dir():
            candidates.extend(path for path in directory.rglob("*") if path.is_file())
    for name in (
        "export_manifest.json",
        "export_status.json",
        "worker_status.json",
        "requirements.lock.txt",
        "package_provenance.json",
        "harness_provenance.json",
    ):
        path = root / name
        if path.is_file():
            candidates.append(path)
    return sorted(set(candidates), key=lambda item: item.relative_to(root).as_posix())


def write_checksum_file(root: Path) -> Path:
    lines = [
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
        for path in _artifact_paths(root)
    ]
    checksum_path = root / "SHA256SUMS"
    atomic_write_bytes(checksum_path, ("\n".join(lines) + "\n").encode("utf-8"))
    return checksum_path


def _tar_filter(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.mode = 0o644
    return info


def refresh_checkpoint(root: Path, reason: str) -> dict[str, Any]:
    """Create an atomic, deterministic, resumable artifact tarball."""

    root.mkdir(parents=True, exist_ok=True)
    checksum_path = write_checksum_file(root)
    members = _artifact_paths(root) + [checksum_path]
    members = sorted(set(members), key=lambda item: item.relative_to(root).as_posix())

    archive = root / "checkpoint.tar"
    partial = root / ".checkpoint.partial.tar"
    partial.unlink(missing_ok=True)
    with tarfile.open(partial, "w", format=tarfile.USTAR_FORMAT) as bundle:
        for path in members:
            relative = path.relative_to(root).as_posix()
            bundle.add(path, arcname=relative, recursive=False, filter=_tar_filter)
    os.replace(partial, archive)

    previous = load_json(root / "checkpoint_meta.json", {}) or {}
    generation = int(previous.get("generation", 0)) + 1
    meta = {
        "format_version": FORMAT_VERSION,
        "generation": generation,
        "reason": sanitize_error(reason, 200),
        "archive_name": archive.name,
        "size_bytes": archive.stat().st_size,
        "sha256": sha256_file(archive),
        "members": [path.relative_to(root).as_posix() for path in members],
        "updated_at": utc_now(),
    }
    atomic_write_json(root / "checkpoint_meta.json", meta)
    return meta


def compare_arrays(expected: Any, observed: Any, np: Any) -> dict[str, Any]:
    expected_array = np.asarray(expected)
    observed_array = np.asarray(observed)
    shape_match = expected_array.shape == observed_array.shape
    dtype_match = expected_array.dtype == observed_array.dtype
    result: dict[str, Any] = {
        "expected_shape": list(expected_array.shape),
        "observed_shape": list(observed_array.shape),
        "expected_dtype": str(expected_array.dtype),
        "observed_dtype": str(observed_array.dtype),
        "shape_match": shape_match,
        "dtype_match": dtype_match,
        "rtol": 0.0 if np.issubdtype(expected_array.dtype, np.integer) else 1e-4,
        "atol": 0.0 if np.issubdtype(expected_array.dtype, np.integer) else 1e-5,
    }
    if not shape_match:
        result["passed"] = False
        return result

    if np.issubdtype(expected_array.dtype, np.integer) or np.issubdtype(
        expected_array.dtype, np.bool_
    ):
        result["exact_match"] = bool(np.array_equal(expected_array, observed_array))
        result["passed"] = bool(dtype_match and result["exact_match"])
        return result

    expected_float = expected_array.astype(np.float64, copy=False)
    observed_float = observed_array.astype(np.float64, copy=False)
    finite = bool(
        np.isfinite(expected_float).all() and np.isfinite(observed_float).all()
    )
    difference = np.abs(expected_float - observed_float)

    def metric(value: Any) -> float | None:
        converted = float(value)
        return converted if math.isfinite(converted) else None

    result.update(
        {
            "finite": finite,
            "max_abs_error": metric(difference.max(initial=0.0)),
            "mean_abs_error": metric(difference.mean()) if difference.size else 0.0,
            "rmse": metric(np.sqrt(np.mean(np.square(difference))))
            if difference.size
            else 0.0,
        }
    )
    if expected_float.size:
        expected_norm = float(np.linalg.norm(expected_float.reshape(-1)))
        observed_norm = float(np.linalg.norm(observed_float.reshape(-1)))
        denominator = expected_norm * observed_norm
        result["cosine_similarity"] = (
            metric(
                np.dot(expected_float.reshape(-1), observed_float.reshape(-1))
                / denominator
            )
            if denominator > 0.0
            else (1.0 if np.array_equal(expected_float, observed_float) else 0.0)
        )
    else:
        result["cosine_similarity"] = 1.0
    allclose = bool(
        np.allclose(
            expected_float, observed_float, rtol=1e-4, atol=1e-5, equal_nan=False
        )
    )
    result["passed"] = bool(dtype_match and finite and allclose)
    return result


def _flatten_torch_outputs(value: Any, torch: Any) -> list[Any]:
    if isinstance(value, torch.Tensor):
        return [value.detach().cpu().numpy()]
    if isinstance(value, Mapping):
        arrays: list[Any] = []
        for item in value.values():
            arrays.extend(_flatten_torch_outputs(item, torch))
        return arrays
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        arrays = []
        for item in value:
            arrays.extend(_flatten_torch_outputs(item, torch))
        return arrays
    raise TypeError(f"unsupported model output type: {type(value).__name__}")


def _spec_manifest(spec: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, tensor_spec in spec.items():
        result[name] = {
            "shape": [int(value) for value in tensor_spec[0]],
            "dtype": str(tensor_spec[1]),
        }
    return result


def component_complete(root: Path, name: str) -> bool:
    manifest = load_json(root / "manifests" / f"{name}.json", {}) or {}
    if manifest.get("status") != "passed":
        return False
    if manifest.get("checkpoint_sha256") != CHECKPOINT_SHA256:
        return False
    if manifest.get("qaihm_version") != QAIHM_VERSION:
        return False
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        return False
    for details in artifacts.values():
        if not isinstance(details, dict):
            return False
        relative = details.get("path")
        expected_hash = details.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            return False
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected_hash:
            return False
    return bool(manifest.get("all_local_outputs_pass"))


def _update_export_status(root: Path, **updates: Any) -> dict[str, Any]:
    path = root / "export_status.json"
    payload = load_json(path, {}) or {}
    payload.update(updates)
    payload["format_version"] = FORMAT_VERSION
    payload["updated_at"] = utc_now()
    atomic_write_json(path, payload)
    return payload


def _verify_checkpoint(path: Path) -> dict[str, Any]:
    size = path.stat().st_size
    digest = sha256_file(path)
    if size != CHECKPOINT_SIZE:
        raise RuntimeError(
            f"checkpoint size mismatch: expected {CHECKPOINT_SIZE}, got {size}"
        )
    if digest != CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint SHA-256 mismatch")
    return {"size_bytes": size, "sha256": digest, "verified": True}


def _validate_onnx_fixed_shapes(
    onnx_path: Path,
    input_names: list[str],
    output_names: list[str],
    onnx: Any,
) -> dict[str, Any]:
    graph = onnx.load(str(onnx_path), load_external_data=True)
    onnx.checker.check_model(graph)
    external_initializers = [
        initializer.name
        for initializer in graph.graph.initializer
        if initializer.data_location == onnx.TensorProto.EXTERNAL
        or initializer.external_data
    ]
    if external_initializers:
        raise RuntimeError("ONNX export unexpectedly used external tensor data")
    actual_inputs = [value.name for value in graph.graph.input]
    actual_outputs = [value.name for value in graph.graph.output]
    if actual_inputs != input_names:
        raise RuntimeError(f"ONNX input names mismatch: {actual_inputs}")
    if actual_outputs != output_names:
        raise RuntimeError(f"ONNX output names mismatch: {actual_outputs}")

    shapes: dict[str, list[int]] = {}
    for value in list(graph.graph.input) + list(graph.graph.output):
        dimensions: list[int] = []
        for dimension in value.type.tensor_type.shape.dim:
            if not dimension.HasField("dim_value") or dimension.dim_value <= 0:
                raise RuntimeError(f"non-fixed ONNX dimension for {value.name}")
            dimensions.append(int(dimension.dim_value))
        shapes[value.name] = dimensions
    return {
        "checker_passed": True,
        "fixed_shapes": True,
        "external_data": False,
        "inputs": actual_inputs,
        "outputs": actual_outputs,
        "shapes": shapes,
    }


def _atomic_save_npz(path: Path, arrays: Mapping[str, Any], np: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _export_component(
    root: Path,
    name: str,
    component: Any,
    torch: Any,
    np: Any,
    onnx: Any,
    ort: Any,
    make_torch_inputs: Any,
    safe_torch_onnx_export: Any,
) -> dict[str, Any]:
    sources = root / "sources"
    validation = root / "validation"
    manifests = root / "manifests"
    staging = root / ".staging" / name
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    sources.mkdir(parents=True, exist_ok=True)
    validation.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)

    spec = component.get_input_spec()
    input_names = list(spec)
    output_names = list(component.get_output_spec())
    inputs = tuple(make_torch_inputs(spec, seed=VALIDATION_SEED))

    # This is QAIHM's official serialization path, not a separately re-created trace.
    official_path = Path(component.serialize(staging, spec))
    if official_path.suffix != ".pt" or not official_path.is_file():
        raise RuntimeError(
            f"QAIHM returned unsupported serialization format: {official_path.suffix}"
        )
    pt_path = sources / f"{name}.pt"
    pt_partial = sources / f".{name}.partial.pt"
    shutil.copyfile(official_path, pt_partial)
    os.replace(pt_partial, pt_path)

    traced = torch.jit.load(str(pt_path), map_location="cpu").eval()
    onnx_stage = staging / f"{name}.onnx"
    safe_torch_onnx_export(
        traced,
        inputs,
        str(onnx_stage),
        input_names=input_names,
        output_names=output_names,
        opset_version=ONNX_OPSET,
        dynamo=False,
        do_constant_folding=True,
    )
    onnx_gate = _validate_onnx_fixed_shapes(onnx_stage, input_names, output_names, onnx)
    onnx_path = sources / f"{name}.onnx"
    os.replace(onnx_stage, onnx_path)

    with torch.inference_mode():
        reference = _flatten_torch_outputs(traced(*inputs), torch)
    if len(reference) != len(output_names):
        raise RuntimeError(
            f"TorchScript returned {len(reference)} outputs; expected {len(output_names)}"
        )

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = int(os.environ.get("OMP_NUM_THREADS", "4"))
    session_options.inter_op_num_threads = 1
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    observed = session.run(
        output_names,
        {
            key: value.detach().cpu().numpy()
            for key, value in zip(input_names, inputs, strict=True)
        },
    )
    if len(observed) != len(output_names):
        raise RuntimeError(
            f"ONNX Runtime returned {len(observed)} outputs; expected {len(output_names)}"
        )

    comparisons = {
        output_name: compare_arrays(expected, actual, np)
        for output_name, expected, actual in zip(
            output_names, reference, observed, strict=True
        )
    }
    all_outputs_pass = all(item["passed"] for item in comparisons.values())

    fixture_arrays: dict[str, Any] = {}
    for input_name, value in zip(input_names, inputs, strict=True):
        fixture_arrays[f"input__{input_name}"] = value.detach().cpu().numpy()
    for output_name, expected, actual in zip(
        output_names, reference, observed, strict=True
    ):
        fixture_arrays[f"torchscript__{output_name}"] = expected
        fixture_arrays[f"onnx__{output_name}"] = actual
    fixture_path = validation / f"{name}_seed_{VALIDATION_SEED}.npz"
    _atomic_save_npz(fixture_path, fixture_arrays, np)

    component_policy = None
    if hasattr(component, "component_precision"):
        component_policy = str(component.component_precision())
    artifact_paths = {
        "torchscript": pt_path,
        "onnx": onnx_path,
        "validation_fixture": fixture_path,
    }
    manifest = {
        "format_version": FORMAT_VERSION,
        "status": "passed" if all_outputs_pass else "failed",
        "component": name,
        "qaihm_version": QAIHM_VERSION,
        "qaihm_official_serialization": True,
        "qaihm_serialized_basename": official_path.name,
        "component_compile_policy": component_policy,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "validation_seed": VALIDATION_SEED,
        "input_spec": _spec_manifest(spec),
        "input_names": input_names,
        "output_names": output_names,
        "onnx": onnx_gate,
        "comparisons": comparisons,
        "all_local_outputs_pass": all_outputs_pass,
        "artifacts": {
            key: {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for key, path in artifact_paths.items()
        },
        "completed_at": utc_now(),
    }
    atomic_write_json(manifests / f"{name}.json", manifest)
    shutil.rmtree(staging, ignore_errors=True)
    if not all_outputs_pass:
        raise RuntimeError(f"local TorchScript/ONNX comparison failed for {name}")
    return manifest


def _write_export_manifest(
    root: Path,
    requested: tuple[str, ...],
    checkpoint: Mapping[str, Any] | None,
) -> dict[str, Any]:
    components = {
        name: load_json(root / "manifests" / f"{name}.json", {})
        for name in requested
        if (root / "manifests" / f"{name}.json").is_file()
    }
    manifest = pin_manifest()
    manifest.update(
        {
            "status": (
                "passed"
                if all(component_complete(root, name) for name in requested)
                else "incomplete"
            ),
            "requested_components": list(requested),
            "installed_versions": installed_versions(),
            "checkpoint_verification": dict(checkpoint or {}),
            "components": components,
            "updated_at": utc_now(),
        }
    )
    atomic_write_json(root / "export_manifest.json", manifest)
    return manifest


def run_export(root: Path, components: tuple[str, ...], threads: int) -> dict[str, Any]:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.chdir(root)
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = str(threads)
    os.environ.setdefault("QAIHM_STORE_ROOT", str(root / "qaihm_cache"))

    _update_export_status(
        root,
        state="starting",
        requested_components=list(components),
        completed_components=[
            name for name in components if component_complete(root, name)
        ],
    )
    if all(component_complete(root, name) for name in components):
        manifest = _write_export_manifest(root, components, None)
        _update_export_status(
            root,
            state="succeeded",
            completed_components=list(components),
            resumed_without_model_load=True,
        )
        refresh_checkpoint(root, "all requested components already verified")
        return manifest

    # Heavy imports begin here.  They are never reached by self-check or unit tests.
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from huggingface_hub import hf_hub_download as real_hf_download
    from qai_hub_models.models.zipformer import model as zipformer_model
    from qai_hub_models.utils.input_spec import make_torch_inputs
    from qai_hub_models.utils.onnx.helpers import safe_torch_onnx_export

    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    _update_export_status(root, state="downloading_pinned_checkpoint")
    checkpoint_path = Path(
        real_hf_download(
            repo_id=CHECKPOINT_REPO,
            filename=CHECKPOINT_FILENAME,
            revision=CHECKPOINT_REVISION,
            cache_dir=str(root / "checkpoint_cache"),
        )
    )
    checkpoint_verification = _verify_checkpoint(checkpoint_path)

    def pinned_download(
        repo_id: str,
        filename: str,
        cache_dir: str | os.PathLike[str] | None = None,
        **_: Any,
    ) -> str:
        del cache_dir
        if repo_id != CHECKPOINT_REPO or filename != CHECKPOINT_FILENAME:
            raise RuntimeError("unexpected model download requested by QAIHM")
        _verify_checkpoint(checkpoint_path)
        return str(checkpoint_path)

    zipformer_model.hf_hub_download = pinned_download
    _update_export_status(root, state="loading_pinned_qaihm_model")
    collection = zipformer_model.HfZipformer.from_pretrained()

    completed = [name for name in components if component_complete(root, name)]
    for name in components:
        if component_complete(root, name):
            continue
        _update_export_status(
            root,
            state="exporting_component",
            active_component=name,
            completed_components=completed,
        )
        try:
            _export_component(
                root,
                name,
                collection.components[name].eval(),
                torch,
                np,
                onnx,
                ort,
                make_torch_inputs,
                safe_torch_onnx_export,
            )
        except Exception as error:
            failure_path = root / "manifests" / f"{name}.json"
            existing = load_json(failure_path, {}) or {}
            existing.update(
                {
                    "format_version": FORMAT_VERSION,
                    "status": "failed",
                    "component": name,
                    "checkpoint_sha256": CHECKPOINT_SHA256,
                    "qaihm_version": QAIHM_VERSION,
                    "error_type": type(error).__name__,
                    "error": sanitize_error(error),
                    "failed_at": utc_now(),
                }
            )
            atomic_write_json(failure_path, existing)
            _write_export_manifest(root, components, checkpoint_verification)
            _update_export_status(
                root,
                state="failed",
                active_component=name,
                completed_components=completed,
                error_type=type(error).__name__,
                error=sanitize_error(error),
            )
            refresh_checkpoint(root, f"component {name} failed")
            raise
        completed.append(name)
        _write_export_manifest(root, components, checkpoint_verification)
        _update_export_status(
            root,
            state="component_complete",
            active_component=None,
            completed_components=completed,
        )
        refresh_checkpoint(root, f"component {name} completed")

    manifest = _write_export_manifest(root, components, checkpoint_verification)
    _update_export_status(
        root,
        state="succeeded",
        active_component=None,
        completed_components=list(components),
    )
    refresh_checkpoint(root, "source export succeeded")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/content/zipformer-source-v058"),
        help="Persistent remote run root.",
    )
    parser.add_argument(
        "--components",
        default=",".join(COMPONENT_ORDER),
        help="Comma-separated component subset; execution order remains decoder, joiner, encoder.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=int(os.environ.get("ZIPFORMER_EXPORT_THREADS", DEFAULT_THREADS)),
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Validate pins and arguments without importing ML packages or using the network.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    components = normalize_components(args.components)
    if not 1 <= args.threads <= 8:
        raise SystemExit("--threads must be between 1 and 8")
    if args.self_check:
        payload = pin_manifest()
        payload["self_check"] = "passed"
        payload["requested_components"] = list(components)
        payload["threads"] = args.threads
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    manifest = run_export(args.root, components, args.threads)
    return 0 if manifest.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
