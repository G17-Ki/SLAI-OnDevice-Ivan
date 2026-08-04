#!/usr/bin/env python3
"""Resumable exact-SM8850 AI Hub runner for fixed-shape Zipformer graphs.

The default/preflight paths are offline.  Cloud submission requires the
``run`` subcommand, ``--confirm-cloud``, and an active tmux session.
"""

from __future__ import annotations

import os


THREAD_CAP = 2
for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "ORT_NUM_THREADS",
):
    os.environ[_thread_variable] = str(THREAD_CAP)

import argparse
from collections import Counter, OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
import datetime as dt
import fcntl
import gc
import hashlib
from importlib import metadata
import json
from pathlib import Path
import re
import sys
import tempfile
import time
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 1
FIXTURE_SCHEMA_VERSION = 3
PREFLIGHT_SCHEMA_VERSION = 1
TARGET_DEVICE = "Snapdragon 8 Elite Gen 5 QRD"
TARGET_OS = "16"
TARGET_CHIPSET = "sm8850"
TARGET_HTP = "v81"
QAIRT_VERSION = "2.45"
FINAL_RUNTIME = "qnn_context_binary"
COMPILE_RUNTIME = "qnn_dlc"
COMPONENT_ORDER = ("decoder", "joiner", "encoder")
CANARY_COMPONENTS = ("decoder", "joiner")
DEFAULT_MODEL_FILES = {
    "encoder": "encoder.pt",
    "decoder": "decoder.pt",
    "joiner": "joiner.pt",
}
COMPONENT_SEED_OFFSETS = {"decoder": 11, "joiner": 23, "encoder": 37}
REDACTED_PATH = "<absolute_path>"


@dataclass(frozen=True)
class ComponentSpec:
    name: str
    graph_name: str
    inputs: tuple[tuple[str, tuple[int, ...], np.dtype[Any]], ...]
    outputs: tuple[str, ...]

    @property
    def input_map(self) -> OrderedDict[str, tuple[tuple[int, ...], np.dtype[Any]]]:
        return OrderedDict((name, (shape, dtype)) for name, shape, dtype in self.inputs)


def _encoder_spec() -> ComponentSpec:
    layers = (2, 4, 3, 2, 4)
    key_lengths = (128, 64, 32, 16, 64)
    inputs: list[tuple[str, tuple[int, ...], np.dtype[Any]]] = [
        ("x", (1, 71, 80), np.dtype("float32"))
    ]
    inputs.extend(
        (f"cached_len_{index}", (count, 1), np.dtype("int32"))
        for index, count in enumerate(layers)
    )
    inputs.extend(
        (f"cached_avg_{index}", (count, 1, 384), np.dtype("float32"))
        for index, count in enumerate(layers)
    )
    inputs.extend(
        (
            f"cached_key_{index}",
            (count, key_lengths[index], 1, 192),
            np.dtype("float32"),
        )
        for index, count in enumerate(layers)
    )
    for prefix in ("cached_val", "cached_val2"):
        inputs.extend(
            (
                f"{prefix}_{index}",
                (count, key_lengths[index], 1, 96),
                np.dtype("float32"),
            )
            for index, count in enumerate(layers)
        )
    for prefix in ("cached_conv1", "cached_conv2"):
        inputs.extend(
            (
                f"{prefix}_{index}",
                (count, 1, 384, 30),
                np.dtype("float32"),
            )
            for index, count in enumerate(layers)
        )
    outputs = (
        "encoder_out",
        *(f"new_cached_len_{index}" for index in range(5)),
        *(f"new_cached_avg_{index}" for index in range(5)),
        *(f"new_cached_key_{index}" for index in range(5)),
        *(f"new_cached_val_{index}" for index in range(5)),
        *(f"new_cached_val2_{index}" for index in range(5)),
        *(f"new_cached_conv1_{index}" for index in range(5)),
        *(f"new_cached_conv2_{index}" for index in range(5)),
    )
    return ComponentSpec("encoder", "encoder_model", tuple(inputs), tuple(outputs))


COMPONENTS = {
    "decoder": ComponentSpec(
        "decoder",
        "decoder_model",
        (("y", (1, 2), np.dtype("int32")),),
        ("decoder_out",),
    ),
    "joiner": ComponentSpec(
        "joiner",
        "joiner_model",
        (
            ("encoder_out", (1, 512), np.dtype("float32")),
            ("decoder_out", (1, 512), np.dtype("float32")),
        ),
        ("logit",),
    ),
    "encoder": _encoder_spec(),
}


def _expected_output_specs() -> dict[
    str, OrderedDict[str, tuple[tuple[int, ...], np.dtype[Any]]]
]:
    # Shapes/dtypes agree between the official qai-hub-models Zipformer contract
    # and the exact-QRD v0.55 release metadata (SM8850 / HTP 81 / QAIRT 2.45).
    encoder: OrderedDict[str, tuple[tuple[int, ...], np.dtype[Any]]] = OrderedDict()
    encoder["encoder_out"] = ((1, 16, 512), np.dtype("float32"))
    input_specs = COMPONENTS["encoder"].input_map
    for prefix in (
        "cached_len",
        "cached_avg",
        "cached_key",
        "cached_val",
        "cached_val2",
        "cached_conv1",
        "cached_conv2",
    ):
        for index in range(5):
            shape, dtype = input_specs[f"{prefix}_{index}"]
            encoder[f"new_{prefix}_{index}"] = (shape, dtype)
    return {
        "decoder": OrderedDict((("decoder_out", ((1, 512), np.dtype("float32"))),)),
        "joiner": OrderedDict((("logit", ((1, 6254), np.dtype("float32"))),)),
        "encoder": encoder,
    }


EXPECTED_OUTPUT_SPECS = _expected_output_specs()


def utc_now() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def package_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _stat_signature(stat_result: os.stat_result) -> dict[str, int]:
    return {
        "device": int(stat_result.st_dev),
        "inode": int(stat_result.st_ino),
        "bytes": int(stat_result.st_size),
        "mtime_ns": int(stat_result.st_mtime_ns),
        "ctime_ns": int(stat_result.st_ctime_ns),
    }


def stable_file_record(path: Path, attempts: int = 3) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    for _ in range(attempts):
        before = _stat_signature(path.stat())
        digest = sha256_file(path)
        after = _stat_signature(path.stat())
        if before == after:
            return {"sha256": digest, "signature": after}
    raise RuntimeError(f"File changed while it was being hashed: {path.name}")


def json_fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _redact_string(value: str, redaction_roots: Sequence[Path]) -> str:
    redacted = value
    roots = sorted(
        {str(path.expanduser().resolve()) for path in redaction_roots},
        key=len,
        reverse=True,
    )
    for root in roots:
        if root and root != "/":
            redacted = redacted.replace(root, REDACTED_PATH)
    redacted = re.sub(
        r"(?i)(authorization:\s*bearer\s+)[^\s,;]+", r"\1<redacted>", redacted
    )
    redacted = re.sub(
        r"""(?ix)
        (
          ["']?(?:api[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret)["']?
          \s*[=:]\s*["']?
        )
        [^\s,"';}]+
        """,
        r"\1<redacted>",
        redacted,
    )
    redacted = re.sub(
        r"(?i)((?:api[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret)\s*[=:]\s*)[^\s,;]+",
        r"\1<redacted>",
        redacted,
    )
    redacted = re.sub(r"(https?://[^\s?]+)\?[^\s]+", r"\1?<redacted>", redacted)
    redacted = re.sub(
        r"(?<![A-Za-z0-9])/(?:Users|home|private|var|tmp|opt|Volumes)/[^\s\"'<>]+",
        REDACTED_PATH,
        redacted,
    )
    redacted = re.sub(
        r"(?<![:/A-Za-z0-9])/(?!/)[^\s\"'<>]+",
        REDACTED_PATH,
        redacted,
    )
    redacted = re.sub(
        r"(?i)(?<![A-Za-z0-9])[A-Z]:\\[^\s\"'<>]+", REDACTED_PATH, redacted
    )
    return redacted


def sanitize_public(value: Any, redaction_roots: Sequence[Path]) -> Any:
    if isinstance(value, str):
        return _redact_string(value, redaction_roots)
    if isinstance(value, dict):
        return {
            str(key): sanitize_public(item, redaction_roots)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_public(item, redaction_roots) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def atomic_write_json(
    path: Path, value: Any, redaction_roots: Sequence[Path] = ()
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    public_value = sanitize_public(value, redaction_roots)
    payload = (
        json.dumps(public_value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as destination:
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_savez(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as destination:
            np.savez(destination, **arrays)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def exclusive_run_lock(output_dir: Path) -> Iterator[None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".zipformer_aihub_v81.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "Another Zipformer runner is already using this output directory."
            ) from error
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


class StateStore:
    def __init__(
        self, path: Path, state: dict[str, Any], redaction_roots: Sequence[Path]
    ):
        self.path = path
        self.state = state
        self.redaction_roots = tuple(redaction_roots)

    def save(self) -> None:
        self.state["revision"] = int(self.state.get("revision", 0)) + 1
        self.state["updated_utc"] = utc_now()
        atomic_write_json(self.path, self.state, self.redaction_roots)

    def event(self, kind: str, component: str | None = None, **details: Any) -> None:
        event = {"time_utc": utc_now(), "kind": kind}
        if component is not None:
            event["component"] = component
        event.update(details)
        events = self.state.setdefault("events", [])
        events.append(event)
        if len(events) > 500:
            del events[:-500]
        self.save()


def safe_error(error: BaseException, redaction_roots: Sequence[Path]) -> dict[str, str]:
    return {
        "type": type(error).__name__,
        "message": _redact_string(str(error), redaction_roots),
    }


def logical_model_ref(path: Path) -> str:
    return f"models/{path.name}"


def component_paths(args: argparse.Namespace) -> dict[str, Path]:
    model_dir = args.model_dir.expanduser().resolve()
    result: dict[str, Path] = {}
    for component in COMPONENT_ORDER:
        configured = Path(getattr(args, f"{component}_model"))
        result[component] = (
            configured.expanduser().resolve()
            if configured.is_absolute()
            else (model_dir / configured).resolve()
        )
    return result


def fixture_path(output_dir: Path, component: str) -> Path:
    return output_dir / "fixtures" / f"{component}.npz"


def component_dir(output_dir: Path, component: str) -> Path:
    return output_dir / "components" / component


def _onnx_dtype_to_numpy(type_name: str) -> np.dtype[Any] | None:
    mapping = {
        "tensor(float)": np.dtype("float32"),
        "tensor(float16)": np.dtype("float16"),
        "tensor(double)": np.dtype("float64"),
        "tensor(int64)": np.dtype("int64"),
        "tensor(int32)": np.dtype("int32"),
        "tensor(int16)": np.dtype("int16"),
        "tensor(int8)": np.dtype("int8"),
        "tensor(uint8)": np.dtype("uint8"),
        "tensor(bool)": np.dtype("bool"),
    }
    return mapping.get(type_name)


def _fixed_shape(shape: Sequence[Any], label: str) -> tuple[int, ...]:
    fixed: list[int] = []
    for dimension in shape:
        if (
            not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or dimension < 1
        ):
            raise ValueError(f"{label} has a dynamic or invalid dimension: {shape!r}")
        fixed.append(dimension)
    return tuple(fixed)


def validate_ort_interface(session: Any, spec: ComponentSpec) -> dict[str, Any]:
    actual_inputs = {node.name: node for node in session.get_inputs()}
    expected_names = list(spec.input_map)
    if set(actual_inputs) != set(expected_names):
        raise ValueError(
            f"{spec.name} input names differ: expected {expected_names}, got {list(actual_inputs)}"
        )
    input_metadata: dict[str, Any] = {}
    for name, (expected_shape, expected_dtype) in spec.input_map.items():
        node = actual_inputs[name]
        actual_shape = _fixed_shape(node.shape, f"{spec.name}:{name}")
        actual_dtype = _onnx_dtype_to_numpy(node.type)
        if actual_shape != expected_shape:
            raise ValueError(
                f"{spec.name}:{name} shape is {actual_shape}, expected {expected_shape}"
            )
        if actual_dtype != expected_dtype:
            raise ValueError(
                f"{spec.name}:{name} dtype is {node.type}, expected {expected_dtype}"
            )
        input_metadata[name] = {
            "shape": list(actual_shape),
            "dtype": str(actual_dtype),
        }
    actual_outputs = session.get_outputs()
    actual_output_names = [node.name for node in actual_outputs]
    if actual_output_names != list(spec.outputs):
        raise ValueError(
            f"{spec.name} output names/order differ: expected {list(spec.outputs)}, "
            f"got {actual_output_names}"
        )
    output_metadata: dict[str, Any] = {}
    for node in actual_outputs:
        actual_shape = _fixed_shape(node.shape, f"{spec.name}:{node.name}")
        actual_dtype = _onnx_dtype_to_numpy(node.type)
        expected_shape, expected_dtype = EXPECTED_OUTPUT_SPECS[spec.name][node.name]
        if actual_shape != expected_shape:
            raise ValueError(
                f"{spec.name}:{node.name} output shape is {actual_shape}, expected {expected_shape}"
            )
        if actual_dtype != expected_dtype:
            raise ValueError(
                f"{spec.name}:{node.name} output dtype is {node.type}, expected {expected_dtype}"
            )
        output_metadata[node.name] = {
            "shape": list(actual_shape),
            "dtype": str(actual_dtype),
        }
    return {"inputs": input_metadata, "outputs": output_metadata}


def inspect_onnx_float_source(path: Path) -> dict[str, Any]:
    """Inspect ONNX graph metadata without resolving external tensor payloads."""
    try:
        import onnx
    except ImportError as error:
        return {
            "inspection_complete": False,
            "passed": False,
            "error": f"onnx import failed: {type(error).__name__}",
            "quantized_ops": [],
            "low_precision_integer_initializers": [],
            "external_data_initializers": [],
        }

    try:
        model = onnx.load_model(str(path), load_external_data=False)
    except Exception as error:
        return {
            "inspection_complete": False,
            "passed": False,
            "error": f"ONNX metadata load failed: {type(error).__name__}: {error}",
            "quantized_ops": [],
            "low_precision_integer_initializers": [],
            "external_data_initializers": [],
        }

    low_precision_type_names = ("UINT4", "INT4", "UINT8", "INT8", "UINT16", "INT16")
    low_precision_types = {
        int(getattr(onnx.TensorProto, name))
        for name in low_precision_type_names
        if hasattr(onnx.TensorProto, name)
    }
    exact_quantized_ops = {
        "ConvInteger",
        "DequantizeLinear",
        "DynamicQuantizeLinear",
        "MatMulInteger",
        "QLinearConv",
        "QLinearMatMul",
        "QuantizeLinear",
    }
    quantized_ops: list[dict[str, str]] = []
    low_precision_initializers: list[dict[str, Any]] = []
    external_initializers: list[dict[str, Any]] = []

    def dtype_name(data_type: int) -> str:
        try:
            return str(onnx.TensorProto.DataType.Name(int(data_type)))
        except Exception:
            return f"UNKNOWN_{int(data_type)}"

    def inspect_tensor(tensor: Any, location: str, kind: str) -> None:
        data_type = int(tensor.data_type)
        is_external = bool(len(tensor.external_data)) or int(
            tensor.data_location
        ) == int(onnx.TensorProto.EXTERNAL)
        record = {
            "name": str(tensor.name or "<unnamed>"),
            "location": location,
            "kind": kind,
            "dtype": dtype_name(data_type),
            "external_data": is_external,
        }
        if is_external:
            external_initializers.append(record)
        if data_type in low_precision_types:
            low_precision_initializers.append(record)

    def is_quantized_op(op_type: str) -> bool:
        return (
            op_type in exact_quantized_ops
            or op_type.startswith("QLinear")
            or "Quantize" in op_type
            or "Dequantize" in op_type
            or op_type.endswith("Integer")
        )

    def inspect_graph(graph: Any, location: str) -> None:
        for initializer in graph.initializer:
            inspect_tensor(initializer, location, "initializer")
        for sparse in graph.sparse_initializer:
            inspect_tensor(sparse.values, location, "sparse_initializer_values")
        for node_index, node in enumerate(graph.node):
            node_location = f"{location}/node[{node_index}]"
            if is_quantized_op(str(node.op_type)):
                quantized_ops.append(
                    {
                        "op_type": str(node.op_type),
                        "domain": str(node.domain),
                        "name": str(node.name or "<unnamed>"),
                        "location": node_location,
                    }
                )
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    inspect_graph(attribute.g, f"{node_location}/{attribute.name}")
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for graph_index, child in enumerate(attribute.graphs):
                        inspect_graph(
                            child,
                            f"{node_location}/{attribute.name}[{graph_index}]",
                        )
                elif attribute.type == onnx.AttributeProto.TENSOR:
                    inspect_tensor(
                        attribute.t,
                        f"{node_location}/{attribute.name}",
                        "constant_attribute",
                    )
                elif attribute.type == onnx.AttributeProto.TENSORS:
                    for tensor_index, tensor in enumerate(attribute.tensors):
                        inspect_tensor(
                            tensor,
                            f"{node_location}/{attribute.name}[{tensor_index}]",
                            "constant_attribute",
                        )

    inspect_graph(model.graph, "graph")
    del model
    return {
        "inspection_complete": True,
        "passed": not quantized_ops and not low_precision_initializers,
        "error": None,
        "quantized_ops": quantized_ops,
        "low_precision_integer_initializers": low_precision_initializers,
        "external_data_initializers": external_initializers,
    }


def quantization_marker_scan(path: Path) -> dict[str, Any]:
    markers = (
        b"QuantizeLinear",
        b"DequantizeLinear",
        b"DynamicQuantizeLinear",
        b"ConvInteger",
        b"MatMulInteger",
        b"QLinearConv",
        b"QLinearMatMul",
    )
    found: set[str] = set()
    overlap = max(len(marker) for marker in markers) - 1
    tail = b""
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            block = tail + chunk
            for marker in markers:
                if marker in block:
                    found.add(marker.decode("ascii"))
            tail = block[-overlap:]
    filename_markers = sorted(
        set(
            re.findall(
                r"(?:u?int(?:4|8|16)|quant(?:ized|ization)?|w\d+a\d+)",
                path.name,
                re.I,
            )
        )
    )
    onnx_inspection = (
        inspect_onnx_float_source(path) if path.suffix.lower() == ".onnx" else None
    )
    onnx_passed = onnx_inspection is None or onnx_inspection["passed"] is True
    return {
        "markers": sorted(found),
        "filename_markers": filename_markers,
        "filename_suggests_quantized": bool(filename_markers),
        "onnx_semantic_inspection": onnx_inspection,
        "passed_float_source_gate": not found and not filename_markers and onnx_passed,
    }


def fixture_float_gate_valid(entry: Mapping[str, Any], model_path: Path) -> bool:
    gate = entry.get("float_source_gate", {})
    if (
        not isinstance(gate, Mapping)
        or gate.get("passed_float_source_gate") is not True
    ):
        return False
    if model_path.suffix.lower() in {".pt", ".pt2"}:
        inspection = gate.get("torch_runtime_inspection", {})
        return isinstance(inspection, Mapping) and inspection.get("passed") is True
    if model_path.suffix.lower() == ".onnx":
        inspection = gate.get("onnx_semantic_inspection", {})
        return (
            isinstance(inspection, Mapping)
            and inspection.get("inspection_complete") is True
            and inspection.get("passed") is True
        )
    return False


def inspect_torch_float_source(local_model: Any, torch: Any) -> dict[str, Any]:
    graph_texts: list[str] = []
    for attribute in ("inlined_graph", "graph"):
        try:
            graph_texts.append(str(getattr(local_model, attribute)))
        except Exception:
            pass
    graph_text = "\n".join(graph_texts).lower()
    marker_patterns = (
        "quantized::",
        "quantized_decomposed",
        "torch.ops.quantized",
        "aten::quantize_per_tensor",
        "aten::quantize_per_channel",
        "aten::dequantize",
        "quantize_per_tensor",
        "quantize_per_channel",
        "dequantize_per_tensor",
        "dequantize_per_channel",
        "_weight_int8pack",
        "int_repr",
    )
    graph_markers = sorted(marker for marker in marker_patterns if marker in graph_text)
    non_float_parameters: list[dict[str, str]] = []
    parameter_dtypes: Counter[str] = Counter()
    for name, parameter in local_model.named_parameters():
        dtype = str(parameter.dtype)
        parameter_dtypes[dtype] += 1
        if not (parameter.is_floating_point() or parameter.is_complex()):
            non_float_parameters.append({"name": name, "dtype": dtype})
    quantized_module_types = sorted(
        {
            f"{type(module).__module__}.{type(module).__name__}"
            for module in local_model.modules()
            if "quantized"
            in f"{type(module).__module__}.{type(module).__name__}".lower()
        }
    )
    quantized_buffers: list[dict[str, str]] = []
    for name, buffer in local_model.named_buffers():
        dtype = str(buffer.dtype).lower()
        if any(
            token in dtype
            for token in (
                "int4",
                "uint4",
                "int8",
                "uint8",
                "int16",
                "uint16",
                "qint",
                "quint",
            )
        ):
            quantized_buffers.append({"name": name, "dtype": str(buffer.dtype)})
    return {
        "graph_markers": graph_markers,
        "parameter_dtypes": dict(sorted(parameter_dtypes.items())),
        "non_float_parameters": non_float_parameters,
        "quantized_buffers": quantized_buffers,
        "quantized_module_types": quantized_module_types,
        "passed": not graph_markers
        and not non_float_parameters
        and not quantized_buffers
        and not quantized_module_types,
        "torch_version": str(torch.__version__),
    }


def make_inputs(spec: ComponentSpec, seed: int) -> OrderedDict[str, np.ndarray]:
    rng = np.random.default_rng(seed + COMPONENT_SEED_OFFSETS[spec.name])
    inputs: OrderedDict[str, np.ndarray] = OrderedDict()
    for name, shape, dtype in spec.inputs:
        if spec.name == "decoder":
            value = rng.integers(0, 6254, size=shape, dtype=np.int32)
        elif spec.name == "encoder" and name == "x":
            value = rng.normal(0.0, 0.35, size=shape).astype(np.float32)
        elif spec.name == "encoder":
            value = np.zeros(shape, dtype=dtype)
        elif dtype.kind == "f":
            value = rng.normal(0.0, 0.5, size=shape).astype(dtype)
        else:
            raise ValueError(f"No deterministic generator for {spec.name}:{name}")
        inputs[name] = np.ascontiguousarray(value, dtype=dtype)
    return inputs


def validate_output_arrays(
    spec: ComponentSpec, outputs: Sequence[Any]
) -> dict[str, dict[str, Any]]:
    if len(outputs) != len(spec.outputs):
        raise RuntimeError(
            f"{spec.name} returned {len(outputs)} outputs, expected {len(spec.outputs)}"
        )
    metadata: dict[str, dict[str, Any]] = {}
    expected = EXPECTED_OUTPUT_SPECS[spec.name]
    for name, value in zip(spec.outputs, outputs, strict=True):
        array = np.asarray(value)
        expected_shape, expected_dtype = expected[name]
        if array.shape != expected_shape:
            raise ValueError(
                f"{spec.name}:{name} output shape is {array.shape}, expected {expected_shape}"
            )
        if array.dtype != expected_dtype:
            raise ValueError(
                f"{spec.name}:{name} output dtype is {array.dtype}, expected {expected_dtype}"
            )
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            raise ValueError(
                f"Local golden {spec.name}:{name} contains non-finite values"
            )
        metadata[name] = {"shape": list(array.shape), "dtype": str(array.dtype)}
    return metadata


def prepare_component(
    component: str,
    model_path: Path,
    output_dir: Path,
    seed: int,
    overwrite: bool,
) -> dict[str, Any]:
    spec = COMPONENTS[component]
    destination = fixture_path(output_dir, component)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"{destination.name} already exists; pass --overwrite to regenerate it."
        )
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    source_before = stable_file_record(model_path)
    float_gate = quantization_marker_scan(model_path)
    if not float_gate["passed_float_source_gate"]:
        raise ValueError(
            f"{component} source looks quantized; exact-v81 experiment requires a float graph. "
            f"byte_markers={float_gate['markers']}, "
            f"filename_markers={float_gate['filename_markers']}, "
            f"onnx_semantic_inspection={float_gate['onnx_semantic_inspection']}"
        )

    inputs = make_inputs(spec, seed)
    suffix = model_path.suffix.lower()
    runtime_name: str
    runtime_version: str
    cleanup_object: Any
    if suffix == ".onnx":
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError(
                "onnxruntime is required to prepare ONNX goldens; use an existing "
                "lightweight environment rather than installing exporter dependencies."
            ) from error
        options = ort.SessionOptions()
        options.intra_op_num_threads = THREAD_CAP
        options.inter_op_num_threads = THREAD_CAP
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        interface = validate_ort_interface(session, spec)
        outputs = session.run(list(spec.outputs), dict(inputs))
        runtime_name = "onnxruntime"
        runtime_version = ort.__version__
        cleanup_object = session
    elif suffix in {".pt", ".pt2"}:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError(
                "torch is required to prepare TorchScript/PT2 goldens; reuse the "
                "environment that created the source artifact."
            ) from error
        torch.set_num_threads(THREAD_CAP)
        try:
            torch.set_num_interop_threads(THREAD_CAP)
        except RuntimeError:
            if torch.get_num_interop_threads() > THREAD_CAP:
                raise RuntimeError(
                    "PyTorch inter-op thread pool is already above the cap"
                )
        if suffix == ".pt":
            local_model = torch.jit.load(str(model_path), map_location="cpu")
            source_format = "torchscript"
            forward_schema = local_model.forward.schema
            schema_arguments = list(forward_schema.arguments)
            non_self_arguments = [
                argument for argument in schema_arguments if argument.name != "self"
            ]
            if len(non_self_arguments) != len(spec.inputs):
                raise ValueError(
                    f"{component} TorchScript forward has {len(non_self_arguments)} inputs, "
                    f"expected {len(spec.inputs)}"
                )
            embedded_schema = {
                "kind": "torchscript_forward_schema",
                "input_count": len(non_self_arguments),
                "input_names": [argument.name for argument in non_self_arguments],
                "schema": str(forward_schema),
            }
        else:
            exported_program = torch.export.load(str(model_path))
            user_inputs = list(exported_program.graph_signature.user_inputs)
            if len(user_inputs) != len(spec.inputs):
                raise ValueError(
                    f"{component} PT2 graph has {len(user_inputs)} user inputs, "
                    f"expected {len(spec.inputs)}"
                )
            embedded_schema = {
                "kind": "torch_export_graph_signature",
                "input_count": len(user_inputs),
                "user_inputs": [str(value) for value in user_inputs],
                "graph_signature": str(exported_program.graph_signature),
            }
            local_model = exported_program.module()
            source_format = "torch_export_pt2"
        local_model.eval()
        torch_float_gate = inspect_torch_float_source(local_model, torch)
        float_gate["torch_runtime_inspection"] = torch_float_gate
        float_gate["passed_float_source_gate"] = bool(
            float_gate["passed_float_source_gate"] and torch_float_gate["passed"]
        )
        if not float_gate["passed_float_source_gate"]:
            raise ValueError(
                f"{component} Torch source contains quantization evidence: "
                f"markers={torch_float_gate['graph_markers']}, "
                f"modules={torch_float_gate['quantized_module_types']}, "
                f"non_float_parameters={len(torch_float_gate['non_float_parameters'])}, "
                f"quantized_buffers={len(torch_float_gate['quantized_buffers'])}"
            )
        ordered_tensors = [torch.from_numpy(value) for value in inputs.values()]
        with torch.inference_mode():
            raw_outputs = local_model(*ordered_tensors)

        flattened: list[np.ndarray] = []

        def flatten(value: Any) -> None:
            if isinstance(value, torch.Tensor):
                flattened.append(value.detach().cpu().numpy())
            elif isinstance(value, (tuple, list)):
                for child in value:
                    flatten(child)
            else:
                raise TypeError(
                    f"Unsupported {component} local output type: {type(value).__name__}"
                )

        flatten(raw_outputs)
        outputs = flattened
        interface = {
            "inputs": {
                name: {"shape": list(shape), "dtype": str(dtype)}
                for name, shape, dtype in spec.inputs
            },
            "outputs": {
                name: {
                    "shape": list(np.asarray(value).shape),
                    "dtype": str(np.asarray(value).dtype),
                }
                for name, value in zip(spec.outputs, outputs)
            },
            "source_format": source_format,
            "embedded_schema": embedded_schema,
            "shape_contract": (
                "torch.export fixed input contract"
                if suffix == ".pt2"
                else "fixed AI Hub input specs applied to a traced TorchScript source"
            ),
        }
        runtime_name = "torch"
        runtime_version = torch.__version__
        cleanup_object = local_model
    else:
        raise ValueError(
            f"Unsupported source format {model_path.suffix!r}; expected .onnx, .pt, or .pt2"
        )
    golden_metadata = validate_output_arrays(spec, outputs)
    if suffix in {".pt", ".pt2"}:
        interface["outputs"] = golden_metadata
    source_after = stable_file_record(model_path)
    if source_before != source_after:
        raise RuntimeError(
            f"{component} source changed during local golden generation; no fixture was written"
        )
    arrays: OrderedDict[str, np.ndarray] = OrderedDict()
    for name, value in inputs.items():
        arrays[f"input__{name}"] = np.asarray(value)
    for name, value in zip(spec.outputs, outputs, strict=True):
        array = np.asarray(value)
        arrays[f"golden__{name}"] = array
    atomic_savez(destination, arrays)
    entry = {
        "status": "ready",
        "component": component,
        "seed": seed + COMPONENT_SEED_OFFSETS[component],
        "source_ref": logical_model_ref(model_path),
        "source_bytes": source_before["signature"]["bytes"],
        "source_sha256": source_before["sha256"],
        "float_source_gate": float_gate,
        "fixture_ref": f"fixtures/{component}.npz",
        "fixture_bytes": destination.stat().st_size,
        "fixture_sha256": sha256_file(destination),
        "interface": interface,
        "goldens": golden_metadata,
        "local_reference_runtime": {
            "name": runtime_name,
            "version": runtime_version,
        },
        "threads": THREAD_CAP,
        "prepared_utc": utc_now(),
    }
    del cleanup_object
    gc.collect()
    return entry


def load_bundle(
    path: Path, spec: ComponentSpec
) -> tuple[OrderedDict[str, np.ndarray], OrderedDict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        inputs: OrderedDict[str, np.ndarray] = OrderedDict()
        goldens: OrderedDict[str, np.ndarray] = OrderedDict()
        for name, expected_shape, expected_dtype in spec.inputs:
            key = f"input__{name}"
            if key not in archive.files:
                raise ValueError(f"{path.name} is missing {key}")
            value = np.asarray(archive[key])
            if value.shape != expected_shape or value.dtype != expected_dtype:
                raise ValueError(
                    f"{path.name}:{key} is {value.shape}/{value.dtype}; "
                    f"expected {expected_shape}/{expected_dtype}"
                )
            inputs[name] = value
        for name in spec.outputs:
            key = f"golden__{name}"
            if key not in archive.files:
                raise ValueError(f"{path.name} is missing {key}")
            value = np.asarray(archive[key])
            expected_shape, expected_dtype = EXPECTED_OUTPUT_SPECS[spec.name][name]
            if value.shape != expected_shape or value.dtype != expected_dtype:
                raise ValueError(
                    f"{path.name}:{key} is {value.shape}/{value.dtype}; "
                    f"expected {expected_shape}/{expected_dtype}"
                )
            if value.dtype.kind in "fc" and not np.isfinite(value).all():
                raise ValueError(f"{path.name}:{key} contains non-finite values")
            goldens[name] = value
        expected_keys = {
            *(f"input__{name}" for name, _, _ in spec.inputs),
            *(f"golden__{name}" for name in spec.outputs),
        }
        extras = sorted(set(archive.files) - expected_keys)
        if extras:
            raise ValueError(f"{path.name} contains unexpected arrays: {extras}")
    return inputs, goldens


def dtype_name(dtype: np.dtype[Any]) -> str:
    supported = {
        np.dtype("float32"): "float32",
        np.dtype("float16"): "float16",
        np.dtype("int64"): "int64",
        np.dtype("int32"): "int32",
        np.dtype("int16"): "int16",
        np.dtype("int8"): "int8",
        np.dtype("uint8"): "uint8",
        np.dtype("bool"): "bool",
    }
    try:
        return supported[np.dtype(dtype)]
    except KeyError as error:
        raise ValueError(f"Unsupported AI Hub dtype: {dtype}") from error


def compile_options(spec: ComponentSpec) -> str:
    options = (
        f"--target_runtime {COMPILE_RUNTIME} "
        f"--qairt_version {QAIRT_VERSION} "
        f"--output_names {','.join(spec.outputs)} "
        f"--qnn_options context_enable_graphs={spec.graph_name} "
        "--quantize_io false"
    )
    if spec.name in {"decoder", "encoder"}:
        options += " --truncate_64bit_tensors --truncate_64bit_io"
    return options


def link_options() -> str:
    return f"--qairt_version {QAIRT_VERSION}"


def profile_options(spec: ComponentSpec, iterations: int) -> str:
    return (
        f"--qairt_version {QAIRT_VERSION} "
        f"--max_profiler_iterations {iterations} "
        f"--qnn_options context_enable_graphs={spec.graph_name}"
    )


def inference_options(spec: ComponentSpec) -> str:
    return (
        f"--qairt_version {QAIRT_VERSION} "
        f"--qnn_options context_enable_graphs={spec.graph_name}"
    )


def build_run_config(
    args: argparse.Namespace,
    paths: Mapping[str, Path],
    output_dir: Path,
) -> tuple[
    dict[str, Any],
    dict[str, tuple[OrderedDict[str, np.ndarray], OrderedDict[str, np.ndarray]]],
]:
    manifest_path = output_dir / "fixtures" / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Fixture manifest is missing. Run the prepare subcommand before cloud submission."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != FIXTURE_SCHEMA_VERSION:
        raise ValueError("Unsupported fixture manifest schema")
    if manifest.get("base_seed") != args.seed:
        raise ValueError(
            f"Fixture seed is {manifest.get('base_seed')}, but --seed is {args.seed}"
        )
    bundles: dict[
        str, tuple[OrderedDict[str, np.ndarray], OrderedDict[str, np.ndarray]]
    ] = {}
    component_config: dict[str, Any] = {}
    for component in COMPONENT_ORDER:
        model_path = paths[component]
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        fixture = fixture_path(output_dir, component)
        if not fixture.is_file():
            raise FileNotFoundError(fixture)
        entry = manifest.get("components", {}).get(component)
        if not isinstance(entry, dict) or entry.get("status") != "ready":
            raise ValueError(f"Fixture manifest has no ready {component} entry")
        source_record = stable_file_record(model_path)
        fixture_before = stable_file_record(fixture)
        if entry.get("source_sha256") != source_record["sha256"]:
            raise ValueError(f"{component} source changed after golden generation")
        if entry.get("source_bytes") != source_record["signature"]["bytes"]:
            raise ValueError(f"{component} source size changed after golden generation")
        if entry.get("fixture_sha256") != fixture_before["sha256"]:
            raise ValueError(f"{component} fixture changed after golden generation")
        if not fixture_float_gate_valid(entry, model_path):
            raise ValueError(
                f"{component} fixture manifest lacks a passing local float-source inspection"
            )
        float_gate = quantization_marker_scan(model_path)
        if not float_gate["passed_float_source_gate"]:
            raise ValueError(f"{component} source fails the float-source gate")
        inputs, goldens = load_bundle(fixture, COMPONENTS[component])
        fixture_after = stable_file_record(fixture)
        if fixture_before != fixture_after:
            raise RuntimeError(f"{component} fixture changed while it was being loaded")
        bundles[component] = (inputs, goldens)
        component_config[component] = {
            "source_ref": logical_model_ref(model_path),
            "source_bytes": source_record["signature"]["bytes"],
            "source_sha256": source_record["sha256"],
            "fixture_ref": f"fixtures/{component}.npz",
            "fixture_bytes": fixture_before["signature"]["bytes"],
            "fixture_sha256": fixture_before["sha256"],
            "graph_name": COMPONENTS[component].graph_name,
            "input_specs": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in inputs.items()
            },
            "output_specs": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in goldens.items()
            },
            "options": {
                "compile": compile_options(COMPONENTS[component]),
                "link": link_options(),
                "profile": profile_options(
                    COMPONENTS[component], args.profile_iterations
                ),
                "inference": inference_options(COMPONENTS[component]),
            },
        }
    config = {
        "target": {
            "device": TARGET_DEVICE,
            "os": TARGET_OS,
            "chipset": TARGET_CHIPSET,
            "htp": TARGET_HTP,
            "required_attributes": [
                "framework:qnn",
                f"chipset:{TARGET_CHIPSET}",
                f"hexagon:{TARGET_HTP}",
            ],
            "precision": (
                "catalog-float (no QuantizeJob; floating-point external I/O stays "
                "FP32 and required token/cache I/O stays INT32; QAIRT lowers internal "
                "HTP compute to FP16)"
            ),
            "compile_runtime": COMPILE_RUNTIME,
            "final_runtime": FINAL_RUNTIME,
            "qairt_version": QAIRT_VERSION,
        },
        "protocol": {
            "base_seed": args.seed,
            "thread_cap": THREAD_CAP,
            "profile_iterations": args.profile_iterations,
            "atol": args.atol,
            "rtol": args.rtol,
            "component_order": list(COMPONENT_ORDER),
            "canary_gate": list(CANARY_COMPONENTS),
            "encoder_submission_rule": (
                "decoder and joiner must both pass strict placement and the "
                "configured tolerance-based numerical comparison"
            ),
        },
        "components": component_config,
    }
    return config, bundles


def new_state(config: dict[str, Any]) -> dict[str, Any]:
    components: dict[str, Any] = {}
    for name in COMPONENT_ORDER:
        components[name] = {
            "status": "not_started",
            "source": config["components"][name],
            "uploads": {"source_models": [], "input_datasets": []},
            "jobs": {"compile": [], "link": [], "profile": [], "inference": []},
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "updated_utc": utc_now(),
        "revision": 0,
        "environment": {
            "python": sys.version.split()[0],
            "numpy": package_version("numpy"),
            "onnx": package_version("onnx"),
            "onnxruntime": package_version("onnxruntime"),
            "torch": package_version("torch"),
            "qai_hub": package_version("qai-hub"),
            "thread_cap": THREAD_CAP,
        },
        "network_policy": {
            "preflight_and_prepare": "offline",
            "run": "explicit --confirm-cloud inside tmux only",
        },
        "config": config,
        "config_fingerprint": json_fingerprint(config),
        "canary_gate": {
            "status": "not_evaluated",
            "components": list(CANARY_COMPONENTS),
        },
        "components": components,
        "events": [],
    }


def load_or_create_state(
    state_path: Path,
    config: dict[str, Any],
    redaction_roots: Sequence[Path],
) -> StateStore:
    expected_fingerprint = json_fingerprint(config)
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("State schema does not match this runner")
        if state.get("config_fingerprint") != expected_fingerprint:
            raise ValueError(
                "Existing state belongs to a different model/fixture/target configuration. "
                "Use the original arguments or a new output directory."
            )
    else:
        state = new_state(config)
    store = StateStore(state_path, state, redaction_roots)
    store.save()
    return store


def job_snapshot(job: Any) -> dict[str, Any]:
    status = job.get_status()
    code = getattr(status.code, "value", status.code)
    return {
        "job_id": str(getattr(job, "job_id", "")),
        "status": str(code),
        "message": str(getattr(status, "message", "")),
        "finished": bool(status.finished),
        "success": bool(status.success),
        "last_polled_utc": utc_now(),
    }


def active_attempt(item: dict[str, Any], stage: str) -> dict[str, Any] | None:
    attempts = item["jobs"][stage]
    return attempts[-1] if attempts else None


def append_job_attempt(
    store: StateStore, component: str, stage: str, job: Any
) -> dict[str, Any]:
    job_id = str(getattr(job, "job_id", ""))
    if not job_id:
        raise RuntimeError(f"AI Hub returned no ID for {component} {stage} job")
    attempt = {
        "attempt": len(store.state["components"][component]["jobs"][stage]) + 1,
        "job_id": job_id,
        "submitted_utc": utc_now(),
        "status": "submitted",
        "finished": False,
        "success": False,
    }
    store.state["components"][component]["jobs"][stage].append(attempt)
    store.state["components"][component]["status"] = f"{stage}_submitted"
    store.event("job_submitted", component, stage=stage, job_id=job_id)
    return attempt


def get_remote_with_retry(
    getter: Callable[[str], Any],
    remote_id: str,
    store: StateStore,
    component: str,
    kind: str,
    attempts: int = 5,
) -> Any:
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return getter(remote_id)
        except Exception as error:  # Network/API errors are resumable.
            last_error = error
            store.event(
                "remote_lookup_retry",
                component,
                remote_kind=kind,
                remote_id=remote_id,
                attempt=attempt,
                error=safe_error(error, store.redaction_roots),
            )
            if attempt < attempts:
                time.sleep(min(30, 2**attempt))
    assert last_error is not None
    raise last_error


def wait_for_job(
    client: Any,
    store: StateStore,
    component: str,
    stage: str,
    poll_seconds: int,
    ceiling_seconds: int,
    initial_job: Any | None = None,
) -> tuple[Any, dict[str, Any] | None]:
    item = store.state["components"][component]
    attempt = active_attempt(item, stage)
    if attempt is None:
        raise RuntimeError(f"No {stage} job is recorded for {component}")
    job = initial_job or get_remote_with_retry(
        client.get_job, attempt["job_id"], store, component, f"{stage}_job"
    )
    started = time.monotonic()
    consecutive_errors = 0
    while True:
        try:
            snapshot = job_snapshot(job)
            consecutive_errors = 0
            attempt.update(snapshot)
            store.save()
        except Exception as error:
            consecutive_errors += 1
            store.event(
                "job_poll_error",
                component,
                stage=stage,
                job_id=attempt["job_id"],
                consecutive_errors=consecutive_errors,
                error=safe_error(error, store.redaction_roots),
            )
            if time.monotonic() - started >= ceiling_seconds:
                item["status"] = f"{stage}_pending_at_ceiling"
                store.save()
                return job, None
            time.sleep(min(60, poll_seconds * max(1, consecutive_errors)))
            continue
        if snapshot["finished"]:
            return job, snapshot
        if time.monotonic() - started >= ceiling_seconds:
            item["status"] = f"{stage}_pending_at_ceiling"
            store.save()
            return job, None
        time.sleep(poll_seconds)


def ensure_source_upload(
    client: Any,
    store: StateStore,
    component: str,
    model_path: Path,
) -> Any:
    item = store.state["components"][component]
    uploads = item["uploads"]["source_models"]
    validated_uploads = [
        upload for upload in uploads if upload.get("validated_against_manifest") is True
    ]
    if validated_uploads:
        model_id = validated_uploads[-1]["model_id"]
        return get_remote_with_retry(
            client.get_model, model_id, store, component, "source_model"
        )
    expected_sha256 = item["source"]["source_sha256"]
    expected_bytes = item["source"]["source_bytes"]
    source_before = stable_file_record(model_path)
    if source_before["sha256"] != expected_sha256:
        raise RuntimeError(f"{component} source hash changed before upload")
    if source_before["signature"]["bytes"] != expected_bytes:
        raise RuntimeError(f"{component} source size changed before upload")
    model = client.upload_model(
        str(model_path), name=f"lfm-demo-zipformer-v81-{component}-source"
    )
    model_id = str(getattr(model, "model_id", ""))
    if not model_id:
        raise RuntimeError(f"AI Hub returned no source model ID for {component}")
    upload_record = {
        "attempt": len(uploads) + 1,
        "model_id": model_id,
        "uploaded_utc": utc_now(),
        "source_sha256": expected_sha256,
        "source_bytes": expected_bytes,
        "validated_against_manifest": False,
        "validation_status": "pending_post_upload_check",
    }
    uploads.append(upload_record)
    item["status"] = "source_uploaded_pending_validation"
    store.event("source_uploaded", component, model_id=model_id)
    source_after = stable_file_record(model_path)
    unchanged = source_before == source_after
    matches_manifest = (
        source_after["sha256"] == expected_sha256
        and source_after["signature"]["bytes"] == expected_bytes
    )
    upload_record["validated_against_manifest"] = bool(unchanged and matches_manifest)
    upload_record["validation_status"] = (
        "passed"
        if upload_record["validated_against_manifest"]
        else "source_changed_during_upload"
    )
    upload_record["validated_utc"] = utc_now()
    store.event(
        "source_upload_validated",
        component,
        model_id=model_id,
        passed=upload_record["validated_against_manifest"],
    )
    if not upload_record["validated_against_manifest"]:
        raise RuntimeError(
            f"{component} source changed during upload; the recorded model ID will not be compiled"
        )
    item["status"] = "source_uploaded"
    store.save()
    return model


def ensure_input_upload(
    client: Any,
    store: StateStore,
    component: str,
    inputs: Mapping[str, np.ndarray],
) -> Any:
    item = store.state["components"][component]
    uploads = item["uploads"]["input_datasets"]
    if uploads:
        dataset_id = uploads[-1]["dataset_id"]
        return get_remote_with_retry(
            client.get_dataset, dataset_id, store, component, "input_dataset"
        )
    dataset_entries = {name: [value] for name, value in inputs.items()}
    dataset = client.upload_dataset(
        dataset_entries, name=f"lfm-demo-zipformer-v81-{component}-seeded-input"
    )
    dataset_id = str(getattr(dataset, "dataset_id", ""))
    if not dataset_id:
        raise RuntimeError(f"AI Hub returned no input dataset ID for {component}")
    uploads.append(
        {
            "attempt": len(uploads) + 1,
            "dataset_id": dataset_id,
            "uploaded_utc": utc_now(),
            "fixture_sha256": item["source"]["fixture_sha256"],
        }
    )
    store.event("input_dataset_uploaded", component, dataset_id=dataset_id)
    return dataset


def ensure_compile_job(
    client: Any,
    device: Any,
    source_model: Any,
    store: StateStore,
    component: str,
    inputs: Mapping[str, np.ndarray],
) -> Any:
    existing = active_attempt(store.state["components"][component], "compile")
    if existing is not None:
        return get_remote_with_retry(
            client.get_job, existing["job_id"], store, component, "compile_job"
        )
    input_specs = {
        name: (tuple(value.shape), dtype_name(value.dtype))
        for name, value in inputs.items()
    }
    job = client.submit_compile_job(
        model=source_model,
        device=device,
        input_specs=input_specs,
        options=compile_options(COMPONENTS[component]),
        name=f"lfm-demo-zipformer-v81-{component}-dlc",
        retry=False,
    )
    append_job_attempt(store, component, "compile", job)
    return job


def ensure_target_model(
    client: Any,
    job: Any,
    attempt: dict[str, Any],
    store: StateStore,
    component: str,
    stage: str,
) -> Any:
    model_id = attempt.get("target_model_id")
    if model_id:
        return get_remote_with_retry(
            client.get_model, model_id, store, component, f"{stage}_target_model"
        )
    target = job.get_target_model()
    if target is None:
        raise RuntimeError(
            f"Successful {component} {stage} job returned no target model"
        )
    model_id = str(getattr(target, "model_id", ""))
    if not model_id:
        raise RuntimeError(f"Successful {component} {stage} job returned no model ID")
    attempt["target_model_id"] = model_id
    attempt["target_model_recorded_utc"] = utc_now()
    store.event("target_model_recorded", component, stage=stage, model_id=model_id)
    return target


def ensure_link_job(
    client: Any,
    device: Any,
    compiled_model: Any,
    store: StateStore,
    component: str,
) -> Any:
    existing = active_attempt(store.state["components"][component], "link")
    if existing is not None:
        return get_remote_with_retry(
            client.get_job, existing["job_id"], store, component, "link_job"
        )
    job = client.submit_link_job(
        [compiled_model],
        device=device,
        options=link_options(),
        name=f"lfm-demo-zipformer-v81-{component}-context",
    )
    append_job_attempt(store, component, "link", job)
    return job


def ensure_profile_job(
    client: Any,
    device: Any,
    context_model: Any,
    store: StateStore,
    component: str,
    iterations: int,
) -> Any:
    existing = active_attempt(store.state["components"][component], "profile")
    if existing is not None:
        return get_remote_with_retry(
            client.get_job, existing["job_id"], store, component, "profile_job"
        )
    job = client.submit_profile_job(
        model=context_model,
        device=device,
        options=profile_options(COMPONENTS[component], iterations),
        name=f"lfm-demo-zipformer-v81-{component}-profile",
        retry=False,
    )
    append_job_attempt(store, component, "profile", job)
    return job


def ensure_inference_job(
    client: Any,
    device: Any,
    context_model: Any,
    dataset: Any,
    store: StateStore,
    component: str,
) -> Any:
    existing = active_attempt(store.state["components"][component], "inference")
    if existing is not None:
        return get_remote_with_retry(
            client.get_job, existing["job_id"], store, component, "inference_job"
        )
    job = client.submit_inference_job(
        model=context_model,
        device=device,
        inputs=dataset,
        options=inference_options(COMPONENTS[component]),
        name=f"lfm-demo-zipformer-v81-{component}-inference",
        retry=False,
    )
    append_job_attempt(store, component, "inference", job)
    return job


def classify_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    execution_detail = profile.get("execution_detail", []) or []
    if not isinstance(execution_detail, list):
        execution_detail = []
    operations: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for index, raw in enumerate(execution_detail):
        layer = raw if isinstance(raw, dict) else {"raw": raw}
        compute_unit = str(layer.get("compute_unit", "UNSPECIFIED"))
        normalized = compute_unit.upper()
        if "CPU" in normalized:
            classification = "cpu_fallback"
        elif any(token in normalized for token in ("NPU", "HTP", "HEXAGON")):
            classification = "hexagon_npu"
        else:
            classification = "unsupported_or_export_failure"
        counts[classification] += 1
        operations.append(
            {
                "index": index,
                "classification": classification,
                "compute_unit": compute_unit,
                "operation": layer,
            }
        )
    strict_npu = bool(operations) and counts["hexagon_npu"] == len(operations)
    return {
        "profile_available": True,
        "operation_count": len(operations),
        "counts": {
            "hexagon_npu": counts["hexagon_npu"],
            "cpu_fallback": counts["cpu_fallback"],
            "unsupported_or_export_failure": counts["unsupported_or_export_failure"],
        },
        "strict_npu_pass": strict_npu,
        "operations": operations,
    }


def unavailable_placement(stage: str, error: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "profile_available": False,
        "failed_stage": stage,
        "classification": "unsupported_or_export_failure",
        "counts": {
            "hexagon_npu": 0,
            "cpu_fallback": 0,
            "unsupported_or_export_failure": 1,
        },
        "strict_npu_pass": False,
        "operations": [],
        "error": dict(error),
    }


def profile_metrics(profile: Mapping[str, Any]) -> dict[str, Any]:
    summary = profile.get("execution_summary", {}) or {}
    if not isinstance(summary, dict):
        summary = {"raw": summary}
    latency_us = summary.get("estimated_inference_time")
    peak_bytes = summary.get("estimated_inference_peak_memory")
    try:
        latency_ms = None if latency_us is None else float(latency_us) / 1000.0
    except (TypeError, ValueError):
        latency_ms = None
    try:
        peak_memory_mib = None if peak_bytes is None else float(peak_bytes) / (1024**2)
    except (TypeError, ValueError):
        peak_memory_mib = None
    return {
        "latency_ms": latency_ms,
        "peak_memory_mib": peak_memory_mib,
        "raw_execution_summary": summary,
    }


def compare_array(
    actual: np.ndarray, golden: np.ndarray, atol: float, rtol: float
) -> dict[str, Any]:
    actual = np.asarray(actual)
    golden = np.asarray(golden)
    result: dict[str, Any] = {
        "actual_shape": list(actual.shape),
        "golden_shape": list(golden.shape),
        "actual_dtype": str(actual.dtype),
        "golden_dtype": str(golden.dtype),
    }
    if actual.shape != golden.shape:
        result.update({"passed": False, "reason": "shape_mismatch"})
        return result
    if actual.dtype.kind in "fc" and not np.isfinite(actual).all():
        result.update({"passed": False, "reason": "actual_non_finite"})
        return result
    if golden.dtype.kind in "fc" and not np.isfinite(golden).all():
        result.update({"passed": False, "reason": "golden_non_finite"})
        return result
    actual_discrete = actual.dtype.kind in "biu"
    golden_discrete = golden.dtype.kind in "biu"
    if actual_discrete or golden_discrete:
        if not (actual_discrete and golden_discrete):
            result.update({"passed": False, "reason": "discrete_dtype_class_mismatch"})
            return result
        exact = bool(np.array_equal(actual, golden))
        result.update(
            {
                "comparison": "exact_integer",
                "passed": exact,
                "mismatch_count": int(np.count_nonzero(actual != golden)),
            }
        )
        return result
    actual64 = actual.astype(np.float64, copy=False).ravel()
    golden64 = golden.astype(np.float64, copy=False).ravel()
    delta = actual64 - golden64
    absolute = np.abs(delta)
    rmse = float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0
    golden_rms = float(np.sqrt(np.mean(golden64 * golden64))) if golden64.size else 0.0
    denominator = float(np.linalg.norm(actual64) * np.linalg.norm(golden64))
    result.update(
        {
            "comparison": "allclose",
            "atol": atol,
            "rtol": rtol,
            "max_abs_error": float(absolute.max()) if absolute.size else 0.0,
            "mean_abs_error": float(absolute.mean()) if absolute.size else 0.0,
            "p95_abs_error": float(np.percentile(absolute, 95))
            if absolute.size
            else 0.0,
            "rmse": rmse,
            "normalized_rmse": None if golden_rms == 0.0 else rmse / golden_rms,
            "cosine_similarity": (
                float(np.dot(actual64, golden64) / denominator)
                if denominator
                else (1.0 if np.array_equal(actual, golden) else 0.0)
            ),
            "passed": bool(np.allclose(actual, golden, atol=atol, rtol=rtol)),
        }
    )
    return result


def target_output_names(target_model: Any) -> list[str]:
    output_spec = getattr(target_model, "output_spec", None)
    if not isinstance(output_spec, Mapping):
        return []
    names: list[str] = []
    for graph_or_name, tensors in output_spec.items():
        if isinstance(tensors, Mapping):
            names.extend(str(name) for name in tensors)
        elif isinstance(tensors, Sequence) and not isinstance(tensors, (str, bytes)):
            for tensor in tensors:
                tensor_name = getattr(tensor, "name", None)
                if tensor_name is not None:
                    names.append(str(tensor_name))
        elif getattr(tensors, "name", None) is not None:
            # Older clients may expose a flat output-name -> TensorSpec mapping.
            names.append(str(getattr(tensors, "name")))
        elif graph_or_name is not None:
            names.append(str(graph_or_name))
    return names


def compare_remote_outputs(
    remote: Mapping[str, Any],
    goldens: Mapping[str, np.ndarray],
    target_model: Any,
    atol: float,
    rtol: float,
) -> tuple[dict[str, Any], OrderedDict[str, np.ndarray]]:
    normalized: OrderedDict[str, np.ndarray] = OrderedDict()
    for name, samples in remote.items():
        if not isinstance(samples, list) or len(samples) != 1:
            raise ValueError(
                f"Remote output {name!r} does not contain exactly one sample"
            )
        normalized[str(name)] = np.asarray(samples[0])
    compiled_names = target_output_names(target_model)
    remote_items = list(normalized.items())
    comparisons: dict[str, Any] = {}
    saved: OrderedDict[str, np.ndarray] = OrderedDict()
    used_remote_names: set[str] = set()
    for index, (golden_name, golden) in enumerate(goldens.items()):
        candidates = [golden_name]
        if index < len(compiled_names):
            candidates.append(compiled_names[index])
        selected: tuple[str, np.ndarray] | None = None
        for candidate in candidates:
            if candidate in normalized and candidate not in used_remote_names:
                selected = (candidate, normalized[candidate])
                break
        if selected is None and index < len(remote_items):
            candidate_name, candidate_value = remote_items[index]
            if candidate_name not in used_remote_names:
                selected = (candidate_name, candidate_value)
        if selected is None:
            comparisons[golden_name] = {
                "passed": False,
                "reason": "missing_remote_output",
            }
            continue
        remote_name, actual = selected
        used_remote_names.add(remote_name)
        comparisons[golden_name] = {
            "remote_name": remote_name,
            **compare_array(actual, golden, atol, rtol),
        }
        saved[f"actual__{golden_name}"] = actual
        saved[f"golden__{golden_name}"] = golden
    unexpected_remote_outputs = sorted(set(normalized) - used_remote_names)
    all_pass = (
        len(comparisons) == len(goldens)
        and not unexpected_remote_outputs
        and all(result.get("passed", False) for result in comparisons.values())
    )
    return {
        "all_numerics_pass": all_pass,
        "outputs": comparisons,
        "remote_output_names": list(normalized),
        "unexpected_remote_outputs": unexpected_remote_outputs,
        "compiled_output_names": compiled_names,
    }, saved


def record_failure(
    store: StateStore,
    output_dir: Path,
    component: str,
    stage: str,
    error: Mapping[str, Any],
) -> None:
    item = store.state["components"][component]
    item["status"] = f"{stage}_failed"
    item["failure"] = {
        "stage": stage,
        "classification": "unsupported_or_export_failure",
        "error": dict(error),
    }
    placement = unavailable_placement(stage, error)
    item["artifacts"] = {
        **item.get("artifacts", {}),
        "placement": f"components/{component}/placement.json",
    }
    atomic_write_json(
        component_dir(output_dir, component) / "placement.json",
        placement,
        store.redaction_roots,
    )
    store.event("component_failed", component, stage=stage, error=dict(error))


def process_component(
    client: Any,
    device: Any,
    store: StateStore,
    component: str,
    model_path: Path,
    bundle: tuple[OrderedDict[str, np.ndarray], OrderedDict[str, np.ndarray]],
    output_dir: Path,
    args: argparse.Namespace,
) -> str:
    item = store.state["components"][component]
    existing_result = item.get("result", {})
    if (
        item.get("status") == "passed"
        and existing_result.get("accepted") is True
        and existing_result.get("strict_npu_pass") is True
        and existing_result.get("all_numerics_pass") is True
    ):
        return "passed"
    inputs, goldens = bundle
    item_dir = component_dir(output_dir, component)
    item_dir.mkdir(parents=True, exist_ok=True)
    item["status"] = "resuming" if any(item["jobs"].values()) else "starting"
    store.save()
    try:
        source_model = ensure_source_upload(client, store, component, model_path)
        compile_job = ensure_compile_job(
            client, device, source_model, store, component, inputs
        )
        compile_job, compile_snapshot = wait_for_job(
            client,
            store,
            component,
            "compile",
            args.poll_seconds,
            args.ceiling_seconds,
            compile_job,
        )
        if compile_snapshot is None:
            return "pending"
        if not compile_snapshot["success"]:
            record_failure(store, output_dir, component, "compile", compile_snapshot)
            return "failed"
        compile_attempt = active_attempt(item, "compile")
        assert compile_attempt is not None
        compiled_model = ensure_target_model(
            client, compile_job, compile_attempt, store, component, "compile"
        )

        link_job = ensure_link_job(client, device, compiled_model, store, component)
        link_job, link_snapshot = wait_for_job(
            client,
            store,
            component,
            "link",
            args.poll_seconds,
            args.ceiling_seconds,
            link_job,
        )
        if link_snapshot is None:
            return "pending"
        if not link_snapshot["success"]:
            record_failure(store, output_dir, component, "link", link_snapshot)
            return "failed"
        link_attempt = active_attempt(item, "link")
        assert link_attempt is not None
        context_model = ensure_target_model(
            client, link_job, link_attempt, store, component, "link"
        )

        profile_job = ensure_profile_job(
            client,
            device,
            context_model,
            store,
            component,
            args.profile_iterations,
        )
        input_dataset = ensure_input_upload(client, store, component, inputs)
        inference_job = ensure_inference_job(
            client,
            device,
            context_model,
            input_dataset,
            store,
            component,
        )
        profile_job, profile_snapshot = wait_for_job(
            client,
            store,
            component,
            "profile",
            args.poll_seconds,
            args.ceiling_seconds,
            profile_job,
        )
        inference_job, inference_snapshot = wait_for_job(
            client,
            store,
            component,
            "inference",
            args.poll_seconds,
            args.ceiling_seconds,
            inference_job,
        )
        if profile_snapshot is None or inference_snapshot is None:
            item["status"] = "profile_or_inference_pending"
            store.save()
            return "pending"
        failures: list[dict[str, Any]] = []
        public_profile: dict[str, Any] | None = None
        if profile_snapshot["success"]:
            try:
                downloaded_profile = profile_job.download_profile()
                if not isinstance(downloaded_profile, dict):
                    raise TypeError(
                        f"Profile response is {type(downloaded_profile).__name__}, expected dict"
                    )
                public_profile = sanitize_public(
                    downloaded_profile, store.redaction_roots
                )
                placement = classify_profile(public_profile)
                atomic_write_json(
                    item_dir / "profile.json", public_profile, store.redaction_roots
                )
            except Exception as error:
                failure = {
                    "stage": "profile_download",
                    "classification": "unsupported_or_export_failure",
                    "error": safe_error(error, store.redaction_roots),
                }
                failures.append(failure)
                placement = unavailable_placement("profile_download", failure["error"])
        else:
            failure = {
                "stage": "profile",
                "classification": "unsupported_or_export_failure",
                "error": profile_snapshot,
            }
            failures.append(failure)
            placement = unavailable_placement("profile", profile_snapshot)
        atomic_write_json(item_dir / "placement.json", placement, store.redaction_roots)

        comparison: dict[str, Any]
        saved: OrderedDict[str, np.ndarray] = OrderedDict()
        if inference_snapshot["success"]:
            try:
                remote = inference_job.download_output_data()
                if not isinstance(remote, dict):
                    raise TypeError(
                        f"Inference response is {type(remote).__name__}, expected dict"
                    )
                comparison, saved = compare_remote_outputs(
                    remote, goldens, context_model, args.atol, args.rtol
                )
            except Exception as error:
                failure = {
                    "stage": "inference_download_or_compare",
                    "classification": "unsupported_or_export_failure",
                    "error": safe_error(error, store.redaction_roots),
                }
                failures.append(failure)
                comparison = {
                    "available": False,
                    "all_numerics_pass": False,
                    "failed_stage": failure["stage"],
                    "error": failure["error"],
                }
        else:
            failure = {
                "stage": "inference",
                "classification": "unsupported_or_export_failure",
                "error": inference_snapshot,
            }
            failures.append(failure)
            comparison = {
                "available": False,
                "all_numerics_pass": False,
                "failed_stage": "inference",
                "error": inference_snapshot,
            }
        atomic_write_json(
            item_dir / "comparison.json", comparison, store.redaction_roots
        )
        if saved:
            atomic_savez(item_dir / "remote_outputs.npz", saved)
        accepted = bool(
            placement["strict_npu_pass"] and comparison["all_numerics_pass"]
        )
        item["result"] = {
            "accepted": accepted,
            "metrics": None
            if public_profile is None
            else profile_metrics(public_profile),
            "strict_npu_pass": placement["strict_npu_pass"],
            "all_numerics_pass": comparison["all_numerics_pass"],
        }
        artifacts = {
            "placement": f"components/{component}/placement.json",
            "comparison": f"components/{component}/comparison.json",
        }
        if public_profile is not None:
            artifacts["profile"] = f"components/{component}/profile.json"
        if saved:
            artifacts["remote_outputs"] = f"components/{component}/remote_outputs.npz"
        item["artifacts"] = artifacts
        if failures:
            item["failure"] = failures
            item["status"] = "profile_or_inference_failed"
        else:
            item.pop("failure", None)
            item["status"] = "passed" if accepted else "completed_with_warning"
        store.event(
            "component_completed",
            component,
            accepted=accepted,
            strict_npu_pass=placement["strict_npu_pass"],
            all_numerics_pass=comparison["all_numerics_pass"],
            failures=failures,
        )
        return "passed" if accepted else "failed"
    except Exception as error:
        public_error = safe_error(error, store.redaction_roots)
        record_failure(store, output_dir, component, "runner", public_error)
        return "failed"


def evaluate_canary_gate(state: Mapping[str, Any]) -> dict[str, Any]:
    component_states = {
        component: state["components"][component].get("status", "unknown")
        for component in CANARY_COMPONENTS
    }
    acceptance_evidence: dict[str, Any] = {}
    for component in CANARY_COMPONENTS:
        item = state["components"][component]
        result = item.get("result", {})
        acceptance_evidence[component] = {
            "status_passed": item.get("status") == "passed",
            "accepted": result.get("accepted") is True,
            "strict_npu_pass": result.get("strict_npu_pass") is True,
            "all_numerics_pass": result.get("all_numerics_pass") is True,
        }
    passed = all(
        all(component_evidence.values())
        for component_evidence in acceptance_evidence.values()
    )
    pending = any(
        "pending" in status or status in {"not_started", "starting", "resuming"}
        for status in component_states.values()
    )
    if passed:
        status = "passed"
        reason = (
            "Both canaries passed strict Hexagon placement and the configured "
            "tolerance-based numerical comparison."
        )
    elif pending:
        status = "pending"
        reason = "At least one canary has not reached a terminal accepted result."
    else:
        status = "blocked"
        reason = "At least one canary failed placement, numerics, compilation, linking, or inference."
    return {
        "status": status,
        "components": component_states,
        "acceptance_evidence": acceptance_evidence,
        "encoder_submission_allowed": passed,
        "reason": reason,
        "evaluated_utc": utc_now(),
    }


def state_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    statuses = {
        component: state["components"][component].get("status", "unknown")
        for component in COMPONENT_ORDER
    }
    return {
        "canary_gate": state.get("canary_gate", {}).get("status"),
        "component_statuses": statuses,
        "accepted": sum(status == "passed" for status in statuses.values()),
        "total": len(statuses),
    }


def offline_preflight(
    args: argparse.Namespace, cloud_dry_run: bool = False
) -> dict[str, Any]:
    model_paths = component_paths(args)
    output_dir = args.output_dir.expanduser().resolve()
    redaction_roots = (args.model_dir, output_dir, Path.home())
    model_checks: dict[str, Any] = {}
    fixture_checks: dict[str, Any] = {}
    models_ready = True
    fixtures_ready = True
    manifest_path = output_dir / "fixtures" / "manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as error:
            manifest = {"load_error": safe_error(error, redaction_roots)}
    manifest_record = {
        "exists": manifest_path.is_file(),
        "ref": "fixtures/manifest.json",
        "schema_version": manifest.get("schema_version"),
        "base_seed": manifest.get("base_seed"),
        "valid_for_requested_seed": (
            manifest.get("schema_version") == FIXTURE_SCHEMA_VERSION
            and manifest.get("base_seed") == args.seed
        ),
    }
    if "load_error" in manifest:
        manifest_record["error"] = manifest["load_error"]
    if not manifest_record["valid_for_requested_seed"]:
        fixtures_ready = False
    for component in COMPONENT_ORDER:
        model_path = model_paths[component]
        model_record: dict[str, Any] = {
            "ref": logical_model_ref(model_path),
            "exists": model_path.is_file(),
            "source_format_supported": model_path.suffix.lower()
            in {".onnx", ".pt", ".pt2"},
        }
        if model_path.is_file():
            model_record.update(
                {
                    "bytes": model_path.stat().st_size,
                    "sha256": sha256_file(model_path),
                    "float_source_gate": quantization_marker_scan(model_path),
                }
            )
            models_ready &= model_record["float_source_gate"][
                "passed_float_source_gate"
            ]
            models_ready &= model_record["source_format_supported"]
        else:
            models_ready = False
        model_checks[component] = model_record
        fixture = fixture_path(output_dir, component)
        fixture_record: dict[str, Any] = {
            "ref": f"fixtures/{component}.npz",
            "exists": fixture.is_file(),
        }
        if fixture.is_file():
            try:
                inputs, goldens = load_bundle(fixture, COMPONENTS[component])
                fixture_record.update(
                    {
                        "bytes": fixture.stat().st_size,
                        "sha256": sha256_file(fixture),
                        "valid": True,
                        "input_count": len(inputs),
                        "golden_count": len(goldens),
                    }
                )
                entry = manifest.get("components", {}).get(component, {})
                if entry.get("fixture_sha256") != fixture_record["sha256"]:
                    raise ValueError("fixture checksum differs from manifest")
                if model_record.get("sha256") != entry.get("source_sha256"):
                    raise ValueError("source checksum differs from fixture manifest")
                if not fixture_float_gate_valid(entry, model_path):
                    raise ValueError(
                        "fixture manifest lacks a passing local float-source inspection"
                    )
            except Exception as error:
                fixture_record.update(
                    {"valid": False, "error": safe_error(error, redaction_roots)}
                )
                fixtures_ready = False
        else:
            fixtures_ready = False
        fixture_checks[component] = fixture_record
    state_path = output_dir / "state.json"
    state_record: dict[str, Any] = {"exists": state_path.is_file(), "ref": "state.json"}
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state_record.update(
                {
                    "schema_version": state.get("schema_version"),
                    "revision": state.get("revision"),
                    "summary": state_summary(state),
                }
            )
        except Exception as error:
            state_record["error"] = safe_error(error, redaction_roots)
    packages = {
        "numpy": package_version("numpy"),
        "onnx": package_version("onnx"),
        "onnxruntime": package_version("onnxruntime"),
        "torch": package_version("torch"),
        "qai_hub": package_version("qai-hub"),
    }
    source_suffixes = {
        component: model_paths[component].suffix.lower()
        for component in COMPONENT_ORDER
        if model_paths[component].is_file()
    }
    local_runtime_ready = all(
        (
            suffix == ".onnx"
            and packages["onnx"] is not None
            and packages["onnxruntime"] is not None
        )
        or (suffix in {".pt", ".pt2"} and packages["torch"] is not None)
        for suffix in source_suffixes.values()
    ) and len(source_suffixes) == len(COMPONENT_ORDER)
    state_config_compatible = True
    state_config_error: dict[str, Any] | None = None
    if cloud_dry_run and models_ready and fixtures_ready:
        try:
            planned_config, _ = build_run_config(args, model_paths, output_dir)
            if state_path.is_file():
                existing_state = json.loads(state_path.read_text(encoding="utf-8"))
                if existing_state.get("schema_version") != SCHEMA_VERSION:
                    raise ValueError("state schema does not match this runner")
                state_config_compatible = existing_state.get(
                    "config_fingerprint"
                ) == json_fingerprint(planned_config)
                if not state_config_compatible:
                    raise ValueError(
                        "state fingerprint differs from the requested cloud configuration"
                    )
        except Exception as error:
            state_config_compatible = False
            state_config_error = safe_error(error, redaction_roots)
    report = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "generated_utc": utc_now(),
        "mode": "cloud_dry_run" if cloud_dry_run else "offline_preflight",
        "network_contacted": False,
        "thread_cap": THREAD_CAP,
        "target": {
            "device": TARGET_DEVICE,
            "os": TARGET_OS,
            "chipset": TARGET_CHIPSET,
            "htp": TARGET_HTP,
            "qairt_version": QAIRT_VERSION,
            "compile_runtime": COMPILE_RUNTIME,
            "final_runtime": FINAL_RUNTIME,
        },
        "packages": packages,
        "models": model_checks,
        "fixture_manifest": manifest_record,
        "fixtures": fixture_checks,
        "state": state_record,
        "guards": {
            "tmux_active": bool(os.environ.get("TMUX")),
            "cloud_confirmation_present": bool(getattr(args, "confirm_cloud", False)),
            "encoder_gate": "decoder and joiner accepted before any encoder upload or job",
        },
        "readiness": {
            "prepare_ready": models_ready and local_runtime_ready,
            "cloud_run_ready": models_ready
            and fixtures_ready
            and packages["qai_hub"] is not None
            and state_config_compatible,
            "state_config_compatible": state_config_compatible,
            "state_config_error": state_config_error,
        },
        "planned_cloud_order": [
            "decoder: source upload -> compile DLC -> link context -> profile + inference -> classify + compare",
            "joiner: source upload -> compile DLC -> link context -> profile + inference -> classify + compare",
            "evaluate strict canary gate",
            "encoder: only when both canaries are accepted; then the same staged flow",
        ],
    }
    atomic_write_json(output_dir / "preflight.json", report, redaction_roots)
    return sanitize_public(report, redaction_roots)


def command_prepare(args: argparse.Namespace) -> int:
    model_paths = component_paths(args)
    output_dir = args.output_dir.expanduser().resolve()
    redaction_roots = (args.model_dir, output_dir, Path.home())
    selected = args.component or list(COMPONENT_ORDER)
    manifest_path = output_dir / "fixtures" / "manifest.json"
    with exclusive_run_lock(output_dir):
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema_version") != FIXTURE_SCHEMA_VERSION:
                raise ValueError("Unsupported fixture manifest schema")
            if manifest.get("base_seed") != args.seed:
                raise ValueError(
                    "Existing fixture manifest uses a different seed; use another output directory."
                )
        else:
            manifest = {
                "schema_version": FIXTURE_SCHEMA_VERSION,
                "created_utc": utc_now(),
                "base_seed": args.seed,
                "thread_cap": THREAD_CAP,
                "network_contacted": False,
                "components": {},
            }
        failures = 0
        for component in selected:
            destination = fixture_path(output_dir, component)
            existing_entry = manifest.get("components", {}).get(component)
            if destination.exists() and not args.overwrite:
                try:
                    if (
                        not isinstance(existing_entry, dict)
                        or existing_entry.get("status") != "ready"
                    ):
                        raise ValueError(
                            "an NPZ exists without a matching ready manifest entry"
                        )
                    current_source = stable_file_record(model_paths[component])
                    current_fixture = stable_file_record(destination)
                    if existing_entry.get("source_sha256") != current_source["sha256"]:
                        raise ValueError(
                            "the source artifact differs from the ready manifest"
                        )
                    if (
                        existing_entry.get("source_bytes")
                        != current_source["signature"]["bytes"]
                    ):
                        raise ValueError(
                            "the source size differs from the ready manifest"
                        )
                    if (
                        existing_entry.get("fixture_sha256")
                        != current_fixture["sha256"]
                    ):
                        raise ValueError("the NPZ differs from the ready manifest")
                    load_bundle(destination, COMPONENTS[component])
                    print(
                        f"[{component}] existing fixed seeded golden reused", flush=True
                    )
                    continue
                except Exception as error:
                    failures += 1
                    manifest.setdefault("preparation_errors", []).append(
                        {
                            "component": component,
                            "time_utc": utc_now(),
                            "error": safe_error(error, redaction_roots),
                            "action": "pass --overwrite only after reviewing the mismatch",
                        }
                    )
                    manifest["updated_utc"] = utc_now()
                    atomic_write_json(manifest_path, manifest, redaction_roots)
                    print(
                        f"[{component}] existing fixture not reused: "
                        f"{_redact_string(str(error), redaction_roots)}",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue
            try:
                entry = prepare_component(
                    component,
                    model_paths[component],
                    output_dir,
                    args.seed,
                    args.overwrite,
                )
                manifest["components"][component] = entry
                print(f"[{component}] fixed seeded golden ready", flush=True)
            except Exception as error:
                failures += 1
                failure_entry = {
                    "status": "failed",
                    "component": component,
                    "source_ref": logical_model_ref(model_paths[component]),
                    "error": safe_error(error, redaction_roots),
                    "updated_utc": utc_now(),
                }
                if (
                    not isinstance(existing_entry, dict)
                    or existing_entry.get("status") != "ready"
                ):
                    manifest["components"][component] = failure_entry
                else:
                    manifest.setdefault("preparation_errors", []).append(failure_entry)
                print(
                    f"[{component}] {type(error).__name__}: {_redact_string(str(error), redaction_roots)}",
                    file=sys.stderr,
                    flush=True,
                )
            manifest["updated_utc"] = utc_now()
            atomic_write_json(manifest_path, manifest, redaction_roots)
        return 0 if failures == 0 else 2


def command_run(args: argparse.Namespace) -> int:
    if args.dry_run:
        output_dir = args.output_dir.expanduser().resolve()
        with exclusive_run_lock(output_dir):
            report = offline_preflight(args, cloud_dry_run=True)
        print(json.dumps(report["readiness"], indent=2, sort_keys=True))
        return 0 if report["readiness"]["cloud_run_ready"] else 2
    if not args.confirm_cloud:
        raise RuntimeError(
            "Cloud execution requires the explicit --confirm-cloud flag."
        )
    if not os.environ.get("TMUX"):
        raise RuntimeError(
            "Cloud execution is allowed only inside an active tmux session."
        )

    model_paths = component_paths(args)
    output_dir = args.output_dir.expanduser().resolve()
    redaction_roots = (args.model_dir, output_dir, Path.home())
    with exclusive_run_lock(output_dir):
        config, bundles = build_run_config(args, model_paths, output_dir)
        store = load_or_create_state(output_dir / "state.json", config, redaction_roots)
        try:
            import qai_hub as hub
        except ImportError as error:
            raise RuntimeError(
                "qai-hub is required for the cloud run subcommand"
            ) from error
        client = hub.Client()
        device = hub.Device(
            TARGET_DEVICE,
            TARGET_OS,
            [
                "framework:qnn",
                f"chipset:{TARGET_CHIPSET}",
                f"hexagon:{TARGET_HTP}",
            ],
        )
        canary_results: dict[str, str] = {}
        for component in CANARY_COMPONENTS:
            canary_results[component] = process_component(
                client,
                device,
                store,
                component,
                model_paths[component],
                bundles[component],
                output_dir,
                args,
            )
        store.state["canary_gate"] = evaluate_canary_gate(store.state)
        store.event(
            "canary_gate_evaluated",
            status=store.state["canary_gate"]["status"],
            encoder_submission_allowed=store.state["canary_gate"][
                "encoder_submission_allowed"
            ],
        )
        if not store.state["canary_gate"]["encoder_submission_allowed"]:
            store.state["summary"] = state_summary(store.state)
            store.save()
            return 3 if "pending" in canary_results.values() else 4
        encoder_result = process_component(
            client,
            device,
            store,
            "encoder",
            model_paths["encoder"],
            bundles["encoder"],
            output_dir,
            args,
        )
        store.state["summary"] = state_summary(store.state)
        if encoder_result == "passed":
            store.state["completed_utc"] = utc_now()
        store.save()
        if encoder_result == "pending":
            return 3
        return 0 if encoder_result == "passed" else 4


def add_path_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    for component in COMPONENT_ORDER:
        parser.add_argument(
            f"--{component}-model",
            default=DEFAULT_MODEL_FILES[component],
            help=f"Path relative to --model-dir (default: {DEFAULT_MODEL_FILES[component]}).",
        )
    parser.add_argument("--seed", type=int, default=20260723)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser(
        "preflight",
        help="Offline file/package/state check; never constructs an AI Hub client.",
    )
    add_path_arguments(preflight)
    preflight.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero unless cloud-run inputs are ready.",
    )

    prepare = subparsers.add_parser(
        "prepare",
        help="Generate fixed seeded NPZ inputs and local ONNX Runtime goldens.",
    )
    add_path_arguments(prepare)
    prepare.add_argument("--component", action="append", choices=COMPONENT_ORDER)
    prepare.add_argument("--overwrite", action="store_true")

    run = subparsers.add_parser(
        "run", help="Resume/submit the guarded exact-v81 AI Hub state machine."
    )
    add_path_arguments(run)
    run.add_argument("--profile-iterations", type=int, default=20)
    run.add_argument("--poll-seconds", type=int, default=30)
    run.add_argument("--ceiling-seconds", type=int, default=10_800)
    run.add_argument("--atol", type=float, default=5e-2)
    run.add_argument("--rtol", type=float, default=5e-2)
    run.add_argument("--confirm-cloud", action="store_true")
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="Print/write the cloud plan without importing an AI Hub client or contacting AI Hub.",
    )
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.command == "run":
        if args.profile_iterations < 1:
            parser.error("--profile-iterations must be positive")
        if args.poll_seconds < 1 or args.poll_seconds > 60:
            parser.error("--poll-seconds must be in [1, 60]")
        if args.ceiling_seconds < 1:
            parser.error("--ceiling-seconds must be positive")
        if args.atol < 0 or args.rtol < 0:
            parser.error("--atol and --rtol must be non-negative")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args, parser)
    try:
        if args.command == "preflight":
            report = offline_preflight(args)
            print(json.dumps(report["readiness"], indent=2, sort_keys=True))
            if args.strict and not report["readiness"]["cloud_run_ready"]:
                return 2
            return 0
        if args.command == "prepare":
            return command_prepare(args)
        if args.command == "run":
            return command_run(args)
        parser.error(f"Unknown command: {args.command}")
    except Exception as error:
        redaction_roots = (
            getattr(args, "model_dir", Path.cwd()),
            getattr(args, "output_dir", Path.cwd()),
            Path.home(),
        )
        print(
            f"{type(error).__name__}: {_redact_string(str(error), redaction_roots)}",
            file=sys.stderr,
        )
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
