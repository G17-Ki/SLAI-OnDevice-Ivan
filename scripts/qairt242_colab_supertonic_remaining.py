#!/usr/bin/env python3
"""Resume-safe QAIRT 2.42 Colab compiler for remaining Supertonic graphs.

This controller supports exactly two independent experiments: the pinned
Supertonic vector estimator and vocoder.  It fixes only concrete I/O metadata;
for the vocoder it also removes the one graph output duplicated in value_info.
The controller must run on Ubuntu 22.04 x86_64 with Python 3.10.  It refuses
macOS execution, another QAIRT release, CPU backend substitution, graph-node
rewrites, and any phone/NPU claim before separate physical verification.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tarfile
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
ROOT = SCRIPT_PATH.parents[1]
_REPO_BASE_CONTROLLER = ROOT / "scripts/qairt242_colab_supertonic_duration_canary.py"
BASE_CONTROLLER = (
    _REPO_BASE_CONTROLLER
    if _REPO_BASE_CONTROLLER.is_file()
    else SCRIPT_DIR / "qairt242_colab_supertonic_duration_canary.py"
)
BASE_CONTROLLER_SHA256 = (
    "bc39dc8fe71206177321c5fa9165cc3d7ea7d78ef3fbdb3b7e0add5f4a802ca2"
)
_REPO_SPEC = ROOT / (
    "reports/android_phone/model_sweep_20260729/"
    "qairt242_colab_supertonic_remaining_spec.json"
)
DEFAULT_SPEC = (
    _REPO_SPEC
    if _REPO_SPEC.is_file()
    else SCRIPT_DIR / "qairt242_colab_supertonic_remaining_spec.json"
)
EXPECTED_SPEC_SHA256 = (
    "6a843bcb2eeb7819d0e16b81fab16b7e2ab6093db51d413e7a5e624c9b8dcc40"
)
EXPECTED_EXPERIMENT_ID = (
    "qairt242_colab_supertonic_remaining_sm8850_htp_v81_v1"
)
EXPECTED_QAIRT_SHA256 = (
    "1a744ed72a813fc9dd4a29e31b1685f784786102fd45414d74915bb3351ae321"
)
EXPECTED_SUPERTONIC_SHA256 = (
    "82fa96f91c4ef8abaae3a14a3f4153facf88bed821d1f7331cec2700f432c427"
)
ALLOWED_COMPONENTS = ("vector_estimator", "vocoder")
REQUIRED_STAGES = (
    "source_model",
    "fixed_model",
    "qnn_conversion",
    "model_library",
    "context_binary",
    "deliverable",
)


def load_base_controller() -> Any:
    if not BASE_CONTROLLER.is_file():
        raise RuntimeError(f"missing base controller: {BASE_CONTROLLER}")
    if sha256_file(BASE_CONTROLLER) != BASE_CONTROLLER_SHA256:
        raise RuntimeError("base duration controller SHA-256 drift")
    loader = importlib.util.spec_from_file_location(
        "qairt242_supertonic_duration_base", BASE_CONTROLLER
    )
    if loader is None or loader.loader is None:
        raise RuntimeError("could not load base duration controller")
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


base = load_base_controller()
RepairError = base.CanaryError


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RepairError(message)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def exact_keys(value: dict[str, Any], keys: set[str], label: str) -> None:
    require(set(value) == keys, f"{label} keys changed: {sorted(value)}")


def validate_spec(spec: dict[str, Any], path: Path | None = None) -> None:
    require(spec.get("schema_version") == 1, "unexpected spec schema")
    require(
        spec.get("experiment_id") == EXPECTED_EXPERIMENT_ID,
        "unexpected experiment identity",
    )
    if path is not None:
        require(path.is_file(), f"missing spec: {path}")
        require(
            sha256_file(path) == EXPECTED_SPEC_SHA256,
            "spec SHA-256 does not match controller trust root",
        )

    policy = spec.get("execution_policy", {})
    require(policy.get("mac_execution_allowed") is False, "macOS must stay blocked")
    require(policy.get("credentials_required") is False, "credentials are forbidden")
    require(
        policy.get("network_hosts")
        == ["softwarecenter.qualcomm.com", "github.com"],
        "network allowlist changed",
    )
    require(
        policy.get("allowed_components") == list(ALLOWED_COMPONENTS),
        "component allowlist changed",
    )
    fallback = policy.get("fallback_policy", {})
    expected_fallback = {
        "allow_other_qairt_version": False,
        "allow_qairt_245_context_in_qairt_242_phone_runtime": False,
        "allow_unverified_download": False,
        "allow_cpu_backend": False,
        "allow_graph_rewrite": False,
        "allow_unpinned_component": False,
    }
    require(fallback == expected_fallback, "fallback policy changed")

    dependencies = spec.get("host_dependencies", {})
    require(dependencies.get("ubuntu_version") == "22.04", "Ubuntu pin changed")
    require(
        dependencies.get("python_version_prefix") == "3.10.",
        "Python pin changed",
    )
    require(dependencies.get("clang_major") == 14, "clang pin changed")
    require(
        dependencies.get("minimum_free_bytes") == 16 * 1024**3,
        "free-space floor changed",
    )
    require(
        dependencies.get("python_packages")
        == {
            "matplotlib": "3.8.4",
            "numpy": "1.26.4",
            "onnx": "1.16.1",
            "onnxruntime": "1.18.0",
            "onnxsim": "0.4.36",
            "protobuf": "3.20.3",
            "pyyaml": "6.0.2",
        },
        "Python package pins changed",
    )

    archives = spec.get("archives", {})
    qairt = archives.get("qairt", {})
    supertonic = archives.get("supertonic", {})
    require(qairt.get("sha256") == EXPECTED_QAIRT_SHA256, "QAIRT hash changed")
    require(qairt.get("bytes") == 1543955191, "QAIRT bytes changed")
    require(
        qairt.get("build_id") == "2.42.0.251225135753_193295",
        "QAIRT build changed",
    )
    require(
        qairt.get("url", "").endswith("/2.42.0.251225/v2.42.0.251225.zip"),
        "QAIRT URL changed",
    )
    require(
        supertonic.get("sha256") == EXPECTED_SUPERTONIC_SHA256,
        "Supertonic hash changed",
    )
    require(supertonic.get("bytes") == 128774318, "Supertonic bytes changed")

    components = spec.get("components", {})
    require(set(components) == set(ALLOWED_COMPONENTS), "component set changed")
    expected_sources = {
        "vector_estimator": (
            "vector_estimator.int8.onnx",
            78400833,
            "20cd86fa5c6effedfda0e7cffe5b0569ca401c440a0c3a1d72bf39286c0db3fd",
            "denoised_latent",
        ),
        "vocoder": (
            "vocoder.int8.onnx",
            25991073,
            "e923d60f53f95eb1ce235f1dc33ec56d9c057823c96fa6f8acf98f32b0da6152",
            "wav_tts",
        ),
    }
    for name, (filename, size, digest, output_name) in expected_sources.items():
        component = components[name]
        source = component.get("source", {})
        require(source.get("archive_basename") == filename, f"{name} file changed")
        require(source.get("bytes") == size, f"{name} size changed")
        require(source.get("sha256") == digest, f"{name} hash changed")
        contract = component.get("onnx_contract", {})
        require(contract.get("ir_version") == 9, f"{name} IR changed")
        require(contract.get("graph_name") == "main_graph", f"{name} graph changed")
        require(
            set(contract.get("outputs", {})) == {output_name},
            f"{name} output changed",
        )
        for tensor in [
            *contract.get("inputs", {}).values(),
            *contract.get("outputs", {}).values(),
        ]:
            require(tensor.get("dtype") == "float32", f"{name} dtype changed")
            require(bool(tensor.get("source_shape")), f"{name} source shape missing")
            require(bool(tensor.get("fixed_shape")), f"{name} fixed shape missing")
    require(
        components["vector_estimator"]["onnx_contract"].get(
            "remove_duplicate_io_value_info"
        )
        == [],
        "vector metadata mutation changed",
    )
    vocoder_contract = components["vocoder"]["onnx_contract"]
    require(
        vocoder_contract.get("remove_duplicate_io_value_info") == ["wav_tts"],
        "vocoder metadata repair changed",
    )
    sanitized = vocoder_contract.get("metadata_only_sanitized_identity", {})
    require(sanitized.get("bytes_before_fixed_shapes") == 25991018, "repair size drift")
    require(
        sanitized.get("sha256_before_fixed_shapes")
        == "db8cd48bf21464fc23114941dfe4d1ed421754920823b5ff02eb0ff1889e20df",
        "repair hash drift",
    )

    target = spec.get("target", {})
    require(
        target
        == {
            "chipset": "SM8850",
            "device_generation": "Snapdragon 8 Elite Gen 5",
            "soc_model": 87,
            "dsp_arch": "v81",
            "target_abi": "aarch64-android",
            "backend": "HTP",
            "pd_session": "unsigned",
            "graph_optimization_level": 3,
            "vtcm_mb": 8,
            "qnn_precision": "FLOAT16",
        },
        "target contract changed",
    )
    require(
        spec.get("output_contract")
        == {
            "required_manifest_status": "context_generated_unexecuted",
            "include_runtime_binaries_in_bundle": False,
            "phone_execution_proven": False,
            "npu_execution_proven": False,
        },
        "output contract changed",
    )
    base.tool_inventory  # Ensure the pinned base exposes the required helper.


def load_spec(path: Path) -> tuple[Path, dict[str, Any]]:
    resolved = path.expanduser().resolve()
    spec = read_json(resolved)
    require(isinstance(spec, dict), "spec root must be an object")
    validate_spec(spec, resolved)
    return resolved, spec


def component_workspace(workspace: Path, component: str) -> Path:
    root = base.safe_workspace(workspace)
    require(component in ALLOWED_COMPONENTS, "component is not allowed")
    result = root / "components" / component
    result.mkdir(parents=True, exist_ok=True)
    return result


def initial_state(
    spec_path: Path, spec: dict[str, Any], component: str
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_id": EXPECTED_EXPERIMENT_ID,
        "component": component,
        "spec_path_name": spec_path.name,
        "spec_sha256": EXPECTED_SPEC_SHA256,
        "controller_sha256": sha256_file(Path(__file__).resolve()),
        "base_controller_sha256": BASE_CONTROLLER_SHA256,
        "claim_boundary": spec["claim_boundary"],
        "created_utc": base.utc_now(),
        "updated_utc": base.utc_now(),
        "status": "initialized",
        "archives": {},
        "stages": {},
    }


def load_or_create_state(
    state_path: Path,
    spec_path: Path,
    spec: dict[str, Any],
    component: str,
) -> dict[str, Any]:
    if state_path.is_file():
        state = read_json(state_path)
        require(isinstance(state, dict), "state root must be an object")
        require(state.get("schema_version") == 1, "unexpected state schema")
        require(state.get("experiment_id") == EXPECTED_EXPERIMENT_ID, "state ID drift")
        require(state.get("component") == component, "state component drift")
        require(state.get("spec_sha256") == EXPECTED_SPEC_SHA256, "state spec drift")
        require(
            state.get("controller_sha256") == sha256_file(Path(__file__).resolve()),
            "controller changed after state creation; use a new workspace",
        )
        require(
            state.get("base_controller_sha256") == BASE_CONTROLLER_SHA256,
            "base controller changed after state creation",
        )
        return state
    state = initial_state(spec_path, spec, component)
    base.atomic_write_json(state_path, state)
    return state


def extract_component(
    archive: Path,
    destination: Path,
    component: dict[str, Any],
) -> dict[str, Any]:
    source_spec = component["source"]
    basename = source_spec["archive_basename"]
    with tarfile.open(archive, "r:bz2") as source:
        matches = [
            member
            for member in source.getmembers()
            if Path(member.name).name == basename
        ]
        require(len(matches) == 1, f"archive must contain one {basename}")
        member = matches[0]
        require(member.isfile(), "model member is not a regular file")
        require(not member.name.startswith("/"), "absolute archive member forbidden")
        require(".." not in Path(member.name).parts, "unsafe archive member path")
        require(member.size == source_spec["bytes"], "model member size drift")
        stream = source.extractfile(member)
        require(stream is not None, "could not read model member")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as output:
            shutil.copyfileobj(stream, output, length=1024 * 1024)
    require(destination.stat().st_size == source_spec["bytes"], "model size drift")
    require(sha256_file(destination) == source_spec["sha256"], "model hash drift")
    return {
        "archive_member": member.name,
        "filename": destination.name,
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }


FIXED_MODEL_HELPER = r'''
import hashlib
import json
from pathlib import Path
import sys
import onnx
from onnx import TensorProto

source = Path(sys.argv[1])
target = Path(sys.argv[2])
metadata_path = Path(sys.argv[3])
contract = json.loads(Path(sys.argv[4]).read_text(encoding="utf-8"))

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def dtype_name(value):
    return {TensorProto.FLOAT: "float32"}.get(value, f"onnx:{value}")

def dims(value):
    result = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            result.append(int(dim.dim_value))
        else:
            result.append(str(dim.dim_param))
    return result

model = onnx.load(str(source), load_external_data=True)
if model.ir_version != contract["ir_version"]:
    raise RuntimeError("IR version drift")
opsets = {item.domain: item.version for item in model.opset_import}
if opsets != contract["opsets"]:
    raise RuntimeError(f"ONNX opset drift: {opsets}")
if model.graph.name != contract["graph_name"]:
    raise RuntimeError("graph-name drift")
inputs = {item.name: item for item in model.graph.input}
outputs = {item.name: item for item in model.graph.output}
if set(inputs) != set(contract["inputs"]):
    raise RuntimeError("input-name drift")
if set(outputs) != set(contract["outputs"]):
    raise RuntimeError("output-name drift")

node_before = hashlib.sha256(
    b"".join(item.SerializeToString() for item in model.graph.node)
).hexdigest()
initializer_before = hashlib.sha256(
    b"".join(item.SerializeToString() for item in model.graph.initializer)
).hexdigest()

source_inputs = {}
for name, expected in contract["inputs"].items():
    value = inputs[name]
    actual_dtype = dtype_name(value.type.tensor_type.elem_type)
    actual_shape = dims(value)
    if actual_dtype != expected["dtype"]:
        raise RuntimeError(f"source dtype drift for {name}: {actual_dtype}")
    if actual_shape != expected["source_shape"]:
        raise RuntimeError(f"source shape drift for {name}: {actual_shape}")
    source_inputs[name] = {"dtype": actual_dtype, "shape": actual_shape}

source_outputs = {}
for name, expected in contract["outputs"].items():
    value = outputs[name]
    actual_dtype = dtype_name(value.type.tensor_type.elem_type)
    actual_shape = dims(value)
    if actual_dtype != expected["dtype"]:
        raise RuntimeError(f"output dtype drift for {name}: {actual_dtype}")
    if actual_shape != expected["source_shape"]:
        raise RuntimeError(f"output shape drift for {name}: {actual_shape}")
    source_outputs[name] = {"dtype": actual_dtype, "shape": actual_shape}

io_names = set(inputs) | set(outputs)
duplicate_io = [item.name for item in model.graph.value_info if item.name in io_names]
expected_removed = contract["remove_duplicate_io_value_info"]
if duplicate_io != expected_removed:
    raise RuntimeError(f"duplicate I/O value_info drift: {duplicate_io}")
retained = [item for item in model.graph.value_info if item.name not in io_names]
del model.graph.value_info[:]
model.graph.value_info.extend(retained)

sanitized_identity = contract.get("metadata_only_sanitized_identity")
if sanitized_identity is not None:
    sanitized_bytes = model.SerializeToString()
    if len(sanitized_bytes) != sanitized_identity["bytes_before_fixed_shapes"]:
        raise RuntimeError("metadata-only sanitized size drift")
    if hashlib.sha256(sanitized_bytes).hexdigest() != sanitized_identity["sha256_before_fixed_shapes"]:
        raise RuntimeError("metadata-only sanitized hash drift")

for name, expected in contract["inputs"].items():
    shape = inputs[name].type.tensor_type.shape
    del shape.dim[:]
    for size in expected["fixed_shape"]:
        shape.dim.add().dim_value = int(size)

for name, expected in contract["outputs"].items():
    shape = outputs[name].type.tensor_type.shape
    del shape.dim[:]
    for size in expected["fixed_shape"]:
        shape.dim.add().dim_value = int(size)

node_after = hashlib.sha256(
    b"".join(item.SerializeToString() for item in model.graph.node)
).hexdigest()
initializer_after = hashlib.sha256(
    b"".join(item.SerializeToString() for item in model.graph.initializer)
).hexdigest()
if node_before != node_after:
    raise RuntimeError("node payload changed while fixing metadata")
if initializer_before != initializer_after:
    raise RuntimeError("initializer payload changed while fixing metadata")

onnx.checker.check_model(model, full_check=True)
target.parent.mkdir(parents=True, exist_ok=True)
onnx.save(model, str(target))
reloaded = onnx.load(str(target), load_external_data=False)
onnx.checker.check_model(reloaded, full_check=True)
metadata = {
    "schema_version": 1,
    "source": {"bytes": source.stat().st_size, "sha256": sha256(source)},
    "derived": {"bytes": target.stat().st_size, "sha256": sha256(target)},
    "ir_version": reloaded.ir_version,
    "opsets": {item.domain: item.version for item in reloaded.opset_import},
    "graph_name": reloaded.graph.name,
    "node_count": len(reloaded.graph.node),
    "initializer_count": len(reloaded.graph.initializer),
    "node_payload_sha256": node_after,
    "initializer_payload_sha256": initializer_after,
    "source_inputs": source_inputs,
    "source_outputs": source_outputs,
    "removed_duplicate_io_value_info": duplicate_io,
    "fixed_inputs": {
        item.name: {"dtype": dtype_name(item.type.tensor_type.elem_type), "shape": dims(item)}
        for item in reloaded.graph.input
    },
    "fixed_outputs": {
        item.name: {"dtype": dtype_name(item.type.tensor_type.elem_type), "shape": dims(item)}
        for item in reloaded.graph.output
    },
    "allowed_mutation": "Concrete I/O shapes and pinned duplicate-I/O value_info removal only",
}
metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
'''


def converter_command(
    sdk_root: Path,
    fixed_model: Path,
    output_cpp: Path,
    component: dict[str, Any],
    spec: dict[str, Any],
) -> list[str]:
    contract = component["onnx_contract"]
    output_name = next(iter(contract["outputs"]))
    command = [
        str(sdk_root / spec["tool_contract"]["converter"]),
        "--input_network",
        str(fixed_model),
        "--output_path",
        str(output_cpp),
        "--out_node",
        output_name,
        "--preserve_io",
        "datatype",
    ]
    for name, record in contract["inputs"].items():
        command.extend(
            ["--input_dim", name, ",".join(map(str, record["fixed_shape"]))]
        )
        command.extend(["--input_dtype", name, record["dtype"]])
    return command


def model_lib_command(
    sdk_root: Path,
    cpp: Path,
    weights: Path,
    output: Path,
    component: dict[str, Any],
    spec: dict[str, Any],
) -> list[str]:
    return [
        str(sdk_root / spec["tool_contract"]["model_lib_generator"]),
        "-c",
        str(cpp),
        "-b",
        str(weights),
        "-t",
        "x86_64-linux-clang",
        "-l",
        component["model_library_basename"],
        "-o",
        str(output),
    ]


def context_command(
    sdk_root: Path,
    model_library: Path,
    output_dir: Path,
    wrapper_config: Path,
    component: dict[str, Any],
    spec: dict[str, Any],
) -> list[str]:
    return [
        str(sdk_root / spec["tool_contract"]["context_binary_generator"]),
        "--backend",
        str(sdk_root / spec["tool_contract"]["htp_backend"]),
        "--model",
        str(model_library),
        "--binary_file",
        component["context_basename"],
        "--output_dir",
        str(output_dir),
        "--config_file",
        str(wrapper_config),
        "--soc_model",
        str(spec["target"]["soc_model"]),
        "--profiling_level",
        "basic",
        "--log_level",
        "info",
    ]


def plan(spec: dict[str, Any], name: str) -> dict[str, Any]:
    require(name in ALLOWED_COMPONENTS, "component is not allowed")
    component = spec["components"][name]
    return {
        "schema_version": 1,
        "experiment_id": EXPECTED_EXPERIMENT_ID,
        "status": "plan_only",
        "host_execution": False,
        "component": name,
        "source": copy.deepcopy(component["source"]),
        "fixed_inputs": {
            tensor_name: {
                "shape": record["fixed_shape"],
                "dtype": record["dtype"],
            }
            for tensor_name, record in component["onnx_contract"]["inputs"].items()
        },
        "target": copy.deepcopy(spec["target"]),
        "archives": {
            archive_name: {
                "url": record["url"],
                "bytes": record["bytes"],
                "sha256": record["sha256"],
            }
            for archive_name, record in spec["archives"].items()
        },
        "stages": list(REQUIRED_STAGES),
        "forbidden": [
            "macOS conversion",
            "another QAIRT version",
            "QAIRT 2.45 context mixed into QAIRT 2.42 runtime",
            "CPU backend fallback",
            "graph node or weight rewrite",
            "unverified download",
            "phone or NPU execution claim",
        ],
    }


def verify_inventory(records: Any, workspace: Path) -> None:
    require(isinstance(records, list) and bool(records), "empty output inventory")
    seen: set[str] = set()
    for record in records:
        require(
            isinstance(record, dict)
            and set(record) == {"path", "bytes", "sha256"},
            "output inventory record drift",
        )
        relative = Path(str(record["path"]))
        require(not relative.is_absolute(), "absolute inventory path")
        path = (workspace / relative).resolve()
        require(
            base.relative_to_workspace(path, workspace) == relative.as_posix(),
            f"non-canonical inventory path: {relative}",
        )
        require(relative.as_posix() not in seen, f"duplicate inventory path: {relative}")
        seen.add(relative.as_posix())
        require(
            isinstance(record["bytes"], int)
            and not isinstance(record["bytes"], bool)
            and record["bytes"] >= 0,
            f"invalid inventory size: {relative}",
        )
        base.validate_sha256(record["sha256"], f"inventory {relative}")
        require(path.is_file(), f"missing output: {relative}")
        require(path.stat().st_size == record["bytes"], f"size drift: {relative}")
        require(sha256_file(path) == record["sha256"], f"hash drift: {relative}")


def verify_file_identity(
    path: Path, expected_bytes: int, expected_sha256: str, label: str
) -> None:
    require(path.is_file(), f"missing {label}: {path}")
    require(path.stat().st_size == expected_bytes, f"{label} size drift")
    require(sha256_file(path) == expected_sha256, f"{label} hash drift")


def verify_stage_record(
    stage: Any,
    workspace: Path,
    stage_name: str,
    expected_fingerprint: str,
) -> dict[str, Any]:
    require(
        isinstance(stage, dict)
        and set(stage)
        == {
            "status",
            "fingerprint",
            "started_utc",
            "completed_utc",
            "details",
            "outputs",
        },
        f"{stage_name} stage field inventory drift",
    )
    require(stage["status"] == "complete", f"stage incomplete: {stage_name}")
    require(
        stage["fingerprint"] == expected_fingerprint,
        f"{stage_name} fingerprint drift",
    )
    for key in ("started_utc", "completed_utc"):
        require(
            isinstance(stage[key], str) and stage[key],
            f"{stage_name} {key} is missing",
        )
    verify_inventory(stage["outputs"], workspace)
    stage_root = workspace / "stages" / stage_name
    require(
        stage["outputs"] == base.inventory_files(stage_root, workspace),
        f"{stage_name} canonical output inventory drift",
    )
    require(isinstance(stage["details"], dict), f"{stage_name} details missing")
    return stage


def verify_completed(
    workspace: Path,
    state: dict[str, Any],
    spec: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    component = spec["components"][name]
    require(isinstance(state, dict), "component state root is invalid")
    require(
        set(state)
        == {
            "schema_version",
            "experiment_id",
            "component",
            "spec_path_name",
            "spec_sha256",
            "controller_sha256",
            "base_controller_sha256",
            "claim_boundary",
            "created_utc",
            "updated_utc",
            "status",
            "archives",
            "stages",
            "host",
            "sdk",
            "python_environment",
            "completed_utc",
            "result",
        },
        "component state field inventory drift",
    )
    require(state["schema_version"] == 1, "state schema drift")
    require(state["status"] == "complete", "component state is not complete")
    require(state["experiment_id"] == EXPECTED_EXPERIMENT_ID, "state ID drift")
    require(state["component"] == name, "state component drift")
    require(state["spec_path_name"] == DEFAULT_SPEC.name, "state spec-name drift")
    require(state["spec_sha256"] == EXPECTED_SPEC_SHA256, "state spec drift")
    require(
        state["controller_sha256"] == sha256_file(Path(__file__).resolve()),
        "state controller drift",
    )
    require(
        state["base_controller_sha256"] == BASE_CONTROLLER_SHA256,
        "state base-controller drift",
    )
    require(state["claim_boundary"] == spec["claim_boundary"], "state claim drift")
    for key in ("created_utc", "updated_utc", "completed_utc"):
        require(isinstance(state[key], str) and state[key], f"state {key} missing")

    archives = state["archives"]
    require(isinstance(archives, dict), "state archive inventory missing")
    require(set(archives) == {"qairt", "supertonic"}, "state archive set drift")
    for archive_name in ("qairt", "supertonic"):
        record = archives[archive_name]
        expected_archive = spec["archives"][archive_name]
        require(isinstance(record, dict), f"{archive_name} archive record drift")
        require(
            set(record)
            in (
                {"status", "filename", "bytes", "sha256", "reused"},
                {
                    "status",
                    "filename",
                    "bytes",
                    "sha256",
                    "reused",
                    "verified_utc",
                },
            ),
            f"{archive_name} archive fields drift",
        )
        require(record["status"] == "verified", f"{archive_name} is unverified")
        require(record["filename"] == expected_archive["filename"], "archive filename drift")
        require(record["bytes"] == expected_archive["bytes"], "archive size drift")
        require(record["sha256"] == expected_archive["sha256"], "archive hash drift")
        require(isinstance(record["reused"], bool), "archive reuse flag drift")

    host = state["host"]
    require(
        isinstance(host, dict)
        and set(host)
        == {
            "platform",
            "machine",
            "ubuntu_version",
            "python",
            "clang",
            "cmake",
            "workspace_free_bytes_at_preflight",
        },
        "host evidence inventory drift",
    )
    require(host["machine"].lower() in {"x86_64", "amd64"}, "host machine drift")
    require(host["ubuntu_version"] == "22.04", "host Ubuntu drift")
    require(str(host["python"]).startswith("3.10."), "host Python drift")
    require(str(host["clang"]).startswith("Ubuntu clang version 14"), "host clang drift")
    require(str(host["cmake"]).startswith("cmake version "), "host CMake drift")
    require(
        isinstance(host["workspace_free_bytes_at_preflight"], int)
        and not isinstance(host["workspace_free_bytes_at_preflight"], bool)
        and host["workspace_free_bytes_at_preflight"]
        >= spec["host_dependencies"]["minimum_free_bytes"],
        "host free-space evidence drift",
    )

    sdk = state["sdk"]
    require(
        isinstance(sdk, dict)
        and set(sdk) == {"archive_sha256", "build_id", "critical_files"},
        "SDK evidence inventory drift",
    )
    require(sdk["archive_sha256"] == EXPECTED_QAIRT_SHA256, "SDK archive drift")
    require(
        sdk["build_id"] == spec["archives"]["qairt"]["build_id"],
        "SDK build drift",
    )
    require(isinstance(sdk["critical_files"], list), "SDK critical inventory drift")
    expected_tool_paths = {
        *[
            spec["tool_contract"][key]
            for key in (
                "converter",
                "model_lib_generator",
                "context_binary_generator",
                "context_binary_utility",
                "htp_backend",
                "htp_extension",
            )
        ],
        *spec["tool_contract"]["android_runtime_inventory"],
        "bin/envsetup.sh",
        "bin/check-python-dependency",
    }
    require(
        {item.get("path") for item in sdk["critical_files"]} == expected_tool_paths,
        "SDK critical path inventory drift",
    )
    for item in sdk["critical_files"]:
        require(
            isinstance(item, dict) and set(item) == {"path", "bytes", "sha256"},
            "SDK critical record drift",
        )
        require(
            isinstance(item["bytes"], int)
            and not isinstance(item["bytes"], bool)
            and item["bytes"] > 0,
            "SDK critical size drift",
        )
        base.validate_sha256(item["sha256"], f"SDK critical {item['path']}")

    packages = state["python_environment"]
    require(
        isinstance(packages, dict)
        and set(packages) == {"python", "packages", "all_packages", "all_packages_sha256"},
        "Python environment inventory drift",
    )
    require(str(packages["python"]).startswith("3.10."), "venv Python drift")
    require(
        packages["packages"] == spec["host_dependencies"]["python_packages"],
        "pinned package drift",
    )
    require(
        packages["all_packages_sha256"]
        == base.canonical_sha256(packages["all_packages"]),
        "full package inventory hash drift",
    )

    stages = state["stages"]
    require(isinstance(stages, dict), "state stages missing")
    require(set(stages) == set(REQUIRED_STAGES), "required stage set is incomplete")

    source_fingerprint = base.stage_fingerprint(
        "source_model",
        EXPECTED_SPEC_SHA256,
        {
            "component": name,
            "archive_sha256": EXPECTED_SUPERTONIC_SHA256,
            "source_sha256": component["source"]["sha256"],
        },
    )
    source_stage = verify_stage_record(
        stages["source_model"], workspace, "source_model", source_fingerprint
    )
    require(
        {item["path"] for item in source_stage["outputs"]}
        == {
            f"stages/source_model/{component['source']['archive_basename']}"
        },
        "source output path inventory drift",
    )
    source_details = source_stage["details"]
    exact_keys(
        source_details,
        {"archive_member", "filename", "bytes", "sha256"},
        "source details",
    )
    require(
        Path(source_details["archive_member"]).name
        == component["source"]["archive_basename"],
        "source archive member drift",
    )
    require(
        source_details["filename"] == component["source"]["archive_basename"],
        "source filename drift",
    )
    require(source_details["bytes"] == component["source"]["bytes"], "source size drift")
    require(source_details["sha256"] == component["source"]["sha256"], "source hash drift")
    source_path = workspace / "stages/source_model" / source_details["filename"]
    verify_file_identity(
        source_path, source_details["bytes"], source_details["sha256"], "source model"
    )

    fixed_fingerprint = base.stage_fingerprint(
        "fixed_model",
        EXPECTED_SPEC_SHA256,
        {"component": name, "source_sha256": source_details["sha256"]},
    )
    fixed_stage = verify_stage_record(
        stages["fixed_model"], workspace, "fixed_model", fixed_fingerprint
    )
    require(
        {item["path"] for item in fixed_stage["outputs"]}
        == {
            "stages/fixed_model/fix_model.py",
            "stages/fixed_model/fixed_model.log",
            "stages/fixed_model/fixed_model_metadata.json",
            "stages/fixed_model/onnx_contract.json",
            f"stages/fixed_model/{name}.fixed.onnx",
        },
        "fixed output path inventory drift",
    )
    fixed_details = fixed_stage["details"]
    exact_keys(
        fixed_details,
        {
            "allowed_mutation",
            "source_sha256",
            "derived_sha256",
            "derived_bytes",
            "node_payload_sha256",
            "initializer_payload_sha256",
            "removed_duplicate_io_value_info",
        },
        "fixed details",
    )
    require(fixed_details["source_sha256"] == source_details["sha256"], "fixed source drift")
    require(
        fixed_details["removed_duplicate_io_value_info"]
        == component["onnx_contract"]["remove_duplicate_io_value_info"],
        "fixed metadata repair drift",
    )
    for key in ("derived_sha256", "node_payload_sha256", "initializer_payload_sha256"):
        base.validate_sha256(fixed_details[key], f"fixed {key}")
    fixed_path = workspace / f"stages/fixed_model/{name}.fixed.onnx"
    verify_file_identity(
        fixed_path,
        fixed_details["derived_bytes"],
        fixed_details["derived_sha256"],
        "fixed model",
    )
    fixed_metadata_path = workspace / "stages/fixed_model/fixed_model_metadata.json"
    require(
        (workspace / "stages/fixed_model/fix_model.py").read_text(encoding="utf-8")
        == FIXED_MODEL_HELPER,
        "fixed-model helper drift",
    )
    require(
        read_json(workspace / "stages/fixed_model/onnx_contract.json")
        == component["onnx_contract"],
        "fixed-model contract drift",
    )
    fixed_metadata = read_json(fixed_metadata_path)
    require(fixed_metadata["source"]["sha256"] == source_details["sha256"], "fixed metadata source drift")
    require(fixed_metadata["derived"]["sha256"] == fixed_details["derived_sha256"], "fixed metadata derived drift")
    require(fixed_metadata["derived"]["bytes"] == fixed_details["derived_bytes"], "fixed metadata size drift")
    require(fixed_metadata["node_payload_sha256"] == fixed_details["node_payload_sha256"], "fixed node digest drift")
    require(
        fixed_metadata["initializer_payload_sha256"]
        == fixed_details["initializer_payload_sha256"],
        "fixed initializer digest drift",
    )
    require(
        fixed_metadata["removed_duplicate_io_value_info"]
        == fixed_details["removed_duplicate_io_value_info"],
        "fixed metadata removal drift",
    )

    conversion_fingerprint = base.stage_fingerprint(
        "qnn_conversion",
        EXPECTED_SPEC_SHA256,
        {"component": name, "fixed_sha256": fixed_details["derived_sha256"]},
    )
    conversion_stage = verify_stage_record(
        stages["qnn_conversion"],
        workspace,
        "qnn_conversion",
        conversion_fingerprint,
    )
    require(
        {item["path"] for item in conversion_stage["outputs"]}
        == {
            f"stages/qnn_conversion/{component['model_library_basename']}.cpp",
            f"stages/qnn_conversion/{component['model_library_basename']}.bin",
            "stages/qnn_conversion/qnn_converter.log",
        },
        "conversion output path inventory drift",
    )
    conversion_details = conversion_stage["details"]
    exact_keys(
        conversion_details,
        {"command", "qnn_graph_name", "cpp_sha256", "weights_sha256"},
        "conversion details",
    )
    require(
        isinstance(conversion_details["command"], list)
        and all(isinstance(item, str) for item in conversion_details["command"]),
        "converter command drift",
    )
    require("2.45" not in " ".join(conversion_details["command"]), "2.45 leaked into conversion")
    cpp_path = workspace / (
        f"stages/qnn_conversion/{component['model_library_basename']}.cpp"
    )
    weights_path = cpp_path.with_suffix(".bin")
    require(sha256_file(cpp_path) == conversion_details["cpp_sha256"], "converter C++ drift")
    require(sha256_file(weights_path) == conversion_details["weights_sha256"], "converter weights drift")
    require(
        base.parse_graph_name(cpp_path) == conversion_details["qnn_graph_name"],
        "QNN graph-name drift",
    )

    model_lib_fingerprint = base.stage_fingerprint(
        "model_library",
        EXPECTED_SPEC_SHA256,
        {
            "component": name,
            "cpp_sha256": conversion_details["cpp_sha256"],
            "weights_sha256": conversion_details["weights_sha256"],
        },
    )
    model_lib_stage = verify_stage_record(
        stages["model_library"], workspace, "model_library", model_lib_fingerprint
    )
    model_lib_details = model_lib_stage["details"]
    exact_keys(
        model_lib_details,
        {"command", "library_relative", "library_bytes", "library_sha256"},
        "model-library details",
    )
    library_relative = Path(model_lib_details["library_relative"])
    require(not library_relative.is_absolute(), "model-library path is absolute")
    require(
        library_relative.name == f"lib{component['model_library_basename']}.so",
        "model-library filename drift",
    )
    model_library_path = workspace / "stages/model_library" / library_relative
    require(
        {item["path"] for item in model_lib_stage["outputs"]}
        == {
            "stages/model_library/model_lib_generator.log",
            f"stages/model_library/{library_relative.as_posix()}",
        },
        "model-library output path inventory drift",
    )
    verify_file_identity(
        model_library_path,
        model_lib_details["library_bytes"],
        model_lib_details["library_sha256"],
        "QNN model library",
    )

    context_fingerprint = base.stage_fingerprint(
        "context_binary",
        EXPECTED_SPEC_SHA256,
        {
            "component": name,
            "model_library_sha256": model_lib_details["library_sha256"],
            "qnn_graph_name": conversion_details["qnn_graph_name"],
            "tool_inventory_sha256": base.canonical_sha256(
                stages["context_binary"]["details"].get("tool_inventory")
            ),
            "target": spec["target"],
        },
    )
    context_stage = verify_stage_record(
        stages["context_binary"], workspace, "context_binary", context_fingerprint
    )
    context_details = context_stage["details"]
    exact_keys(
        context_details,
        {
            "command",
            "utility_command",
            "qnn_graph_name",
            "target",
            "context_relative",
            "context_bytes",
            "context_sha256",
            "context_metadata_sha256",
            "tool_inventory",
        },
        "context details",
    )
    require(context_details["qnn_graph_name"] == conversion_details["qnn_graph_name"], "context graph drift")
    require(context_details["target"] == spec["target"], "context target drift")
    require(
        context_details["context_relative"]
        == f"context/{component['context_basename']}.bin",
        "context path drift",
    )
    require(
        {item["path"] for item in context_stage["outputs"]}
        == {
            "stages/context_binary/backend_extensions.json",
            "stages/context_binary/context_binary_utility.log",
            "stages/context_binary/context_generator.log",
            "stages/context_binary/context_metadata.json",
            "stages/context_binary/htp_backend_config.json",
            f"stages/context_binary/{context_details['context_relative']}",
        },
        "context output path inventory drift",
    )
    require(
        context_details["context_bytes"] >= component["minimum_context_bytes"],
        "context binary is unexpectedly small",
    )
    context_path = workspace / "stages/context_binary" / context_details["context_relative"]
    verify_file_identity(
        context_path,
        context_details["context_bytes"],
        context_details["context_sha256"],
        "context binary",
    )
    context_metadata_path = workspace / "stages/context_binary/context_metadata.json"
    base.validate_sha256(context_details["context_metadata_sha256"], "context metadata")
    require(
        sha256_file(context_metadata_path) == context_details["context_metadata_sha256"],
        "context metadata drift",
    )
    read_json(context_metadata_path)
    tool_inventory = context_details["tool_inventory"]
    require(isinstance(tool_inventory, dict), "tool inventory drift")
    critical_by_path = {item["path"]: item for item in sdk["critical_files"]}
    require(
        set(tool_inventory)
        == {
            *[
                spec["tool_contract"][key]
                for key in (
                    "converter",
                    "model_lib_generator",
                    "context_binary_generator",
                    "context_binary_utility",
                    "htp_backend",
                    "htp_extension",
                )
            ],
            *spec["tool_contract"]["android_runtime_inventory"],
        },
        "tool inventory path drift",
    )
    for path, record in tool_inventory.items():
        require(
            record
            == {
                "bytes": critical_by_path[path]["bytes"],
                "sha256": critical_by_path[path]["sha256"],
            },
            f"tool inventory identity drift: {path}",
        )

    deliverable_fingerprint = base.stage_fingerprint(
        "deliverable",
        EXPECTED_SPEC_SHA256,
        {
            "component": name,
            "context_sha256": context_details["context_sha256"],
            "context_metadata_sha256": context_details["context_metadata_sha256"],
            "controller_sha256": state["controller_sha256"],
            "base_controller_sha256": BASE_CONTROLLER_SHA256,
        },
    )
    deliverable_stage = verify_stage_record(
        stages["deliverable"], workspace, "deliverable", deliverable_fingerprint
    )
    expected_result = {
        "manifest_status": spec["output_contract"]["required_manifest_status"],
        "bundle_bytes": deliverable_stage["details"].get("bundle_bytes"),
        "bundle_sha256": deliverable_stage["details"].get("bundle_sha256"),
        "phone_execution_proven": False,
        "npu_execution_proven": False,
    }
    require(deliverable_stage["details"] == expected_result, "deliverable details drift")
    require(state["result"] == expected_result, "state result drift")

    manifest_path = workspace / "stages/deliverable/result_manifest.json"
    manifest = read_json(manifest_path)
    require(isinstance(manifest, dict), "result manifest root is invalid")
    exact_keys(
        manifest,
        {
            "schema_version",
            "experiment_id",
            "component",
            "status",
            "claim_boundary",
            "created_utc",
            "spec_sha256",
            "controller_sha256",
            "base_controller_sha256",
            "host",
            "python_environment",
            "archives",
            "source_model",
            "fixed_model",
            "qnn_conversion",
            "model_library",
            "context",
            "tool_inventory",
            "runtime_binaries_bundled",
            "phone_execution_proven",
            "npu_execution_proven",
            "next_gate",
        },
        "result manifest",
    )
    require(manifest["schema_version"] == 1, "manifest schema drift")
    require(manifest["experiment_id"] == EXPECTED_EXPERIMENT_ID, "manifest ID drift")
    require(manifest["component"] == name, "manifest component drift")
    require(
        manifest["status"] == spec["output_contract"]["required_manifest_status"],
        "manifest status drift",
    )
    require(manifest["spec_sha256"] == EXPECTED_SPEC_SHA256, "manifest spec drift")
    require(
        manifest["controller_sha256"] == sha256_file(Path(__file__).resolve()),
        "manifest controller drift",
    )
    require(
        manifest["base_controller_sha256"] == BASE_CONTROLLER_SHA256,
        "manifest base-controller drift",
    )
    require(manifest["claim_boundary"] == spec["claim_boundary"], "manifest claim drift")
    require(manifest["host"] == host, "manifest host drift")
    require(manifest["python_environment"] == packages, "manifest Python drift")
    require(manifest["archives"] == archives, "manifest archive drift")
    require(manifest["source_model"] == source_details, "manifest source drift")
    require(manifest["fixed_model"] == fixed_details, "manifest fixed drift")
    require(manifest["qnn_conversion"] == conversion_details, "manifest conversion drift")
    require(manifest["model_library"] == model_lib_details, "manifest model-library drift")
    manifest_context = {
        key: value for key, value in context_details.items() if key != "tool_inventory"
    }
    require(manifest["context"] == manifest_context, "manifest context drift")
    require(manifest["tool_inventory"] == tool_inventory, "manifest tool inventory drift")
    require(manifest["runtime_binaries_bundled"] is False, "runtime was bundled")
    require(manifest["phone_execution_proven"] is False, "invalid phone claim")
    require(manifest["npu_execution_proven"] is False, "invalid NPU claim")

    bundle_name = f"qairt242_{name}_result.tar.gz"
    bundle = workspace / "stages/deliverable" / bundle_name
    digest_file = bundle.with_suffix(bundle.suffix + ".sha256")
    digest_parts = digest_file.read_text(encoding="utf-8").split()
    require(digest_parts == [deliverable_stage["details"]["bundle_sha256"], bundle.name], "bundle checksum file drift")
    expected = digest_parts[0]
    require(bundle.stat().st_size == deliverable_stage["details"]["bundle_bytes"], "bundle size drift")
    require(sha256_file(bundle) == expected, "deliverable bundle hash drift")

    bundle_sources = {
        context_path.name: context_path,
        "context_metadata.json": context_metadata_path,
        "htp_backend_config.json": workspace
        / "stages/context_binary/htp_backend_config.json",
        "backend_extensions.json": workspace
        / "stages/context_binary/backend_extensions.json",
        "context_generator.log": workspace
        / "stages/context_binary/context_generator.log",
        "context_binary_utility.log": workspace
        / "stages/context_binary/context_binary_utility.log",
        "fixed_model_metadata.json": fixed_metadata_path,
        DEFAULT_SPEC.name: workspace
        / "stages/deliverable/payload"
        / DEFAULT_SPEC.name,
        "result_manifest.json": manifest_path,
    }
    with tarfile.open(bundle, "r:gz") as archive:
        members = [item for item in archive.getmembers() if item.isfile()]
        require({item.name for item in members} == set(bundle_sources), "bundle member set drift")
        for member in members:
            stream = archive.extractfile(member)
            require(stream is not None, f"could not read bundle member: {member.name}")
            payload = stream.read()
            source_path_for_member = bundle_sources[member.name]
            require(len(payload) == source_path_for_member.stat().st_size, f"bundle member size drift: {member.name}")
            require(
                hashlib.sha256(payload).hexdigest() == sha256_file(source_path_for_member),
                f"bundle member hash drift: {member.name}",
            )
    return {
        "schema_version": 1,
        "experiment_id": EXPECTED_EXPERIMENT_ID,
        "component": name,
        "status": "verified_complete",
        "manifest": base.relative_to_workspace(manifest_path, workspace),
        "bundle": base.relative_to_workspace(bundle, workspace),
        "bundle_bytes": bundle.stat().st_size,
        "bundle_sha256": expected,
        "phone_execution_proven": False,
        "npu_execution_proven": False,
    }


def execute(
    spec_path: Path,
    spec: dict[str, Any],
    root_workspace: Path,
    scratch: Path,
    name: str,
) -> dict[str, Any]:
    root_workspace = base.safe_workspace(root_workspace)
    root_workspace.mkdir(parents=True, exist_ok=True)
    workspace = component_workspace(root_workspace, name)
    state_path = workspace / "qairt242_colab_state.json"
    component = spec["components"][name]
    with base.workspace_lock(root_workspace):
        state = load_or_create_state(state_path, spec_path, spec, name)
        if state.get("status") == "complete":
            return verify_completed(workspace, state, spec, name)
        try:
            state.pop("last_failure", None)
            state["status"] = "running"
            state["host"] = base.host_preflight(root_workspace, spec)
            free = shutil.disk_usage(root_workspace).free
            require(
                free >= spec["host_dependencies"]["minimum_free_bytes"],
                "remaining-graph workspace needs at least 16 GiB free",
            )
            base.write_state(state_path, state)

            qairt_archive = base.verified_download(
                "qairt", spec["archives"]["qairt"], root_workspace, state_path, state
            )
            supertonic_archive = base.verified_download(
                "supertonic",
                spec["archives"]["supertonic"],
                root_workspace,
                state_path,
                state,
            )

            source_fp = base.stage_fingerprint(
                "source_model",
                EXPECTED_SPEC_SHA256,
                {
                    "component": name,
                    "archive_sha256": EXPECTED_SUPERTONIC_SHA256,
                    "source_sha256": component["source"]["sha256"],
                },
            )

            def source_action(partial: Path) -> dict[str, Any]:
                return extract_component(
                    supertonic_archive,
                    partial / component["source"]["archive_basename"],
                    component,
                )

            source_stage = base.run_stage(
                "source_model",
                source_fp,
                workspace,
                state_path,
                state,
                source_action,
            )
            source_model = (
                workspace
                / "stages/source_model"
                / component["source"]["archive_basename"]
            )

            sdk_root, extraction_inventory = base.prepare_sdk(
                qairt_archive, scratch, root_workspace, spec
            )
            venv, env, packages = base.prepare_venv(
                scratch, sdk_root, root_workspace, spec
            )
            state["sdk"] = {
                "archive_sha256": EXPECTED_QAIRT_SHA256,
                "build_id": spec["archives"]["qairt"]["build_id"],
                "critical_files": extraction_inventory["critical_files"],
            }
            state["python_environment"] = packages
            base.write_state(state_path, state)

            fixed_fp = base.stage_fingerprint(
                "fixed_model",
                EXPECTED_SPEC_SHA256,
                {"component": name, "source_sha256": source_stage["details"]["sha256"]},
            )

            def fixed_action(partial: Path) -> dict[str, Any]:
                helper = partial / "fix_model.py"
                base.atomic_write_text(helper, FIXED_MODEL_HELPER)
                contract_path = partial / "onnx_contract.json"
                base.atomic_write_json(contract_path, component["onnx_contract"])
                output = partial / f"{name}.fixed.onnx"
                metadata = partial / "fixed_model_metadata.json"
                command = [
                    str(venv / "bin/python"),
                    str(helper),
                    str(source_model),
                    str(output),
                    str(metadata),
                    str(contract_path),
                ]
                base.run_logged(command, partial, env, partial / "fixed_model.log")
                record = read_json(metadata)
                require(
                    record["source"]["sha256"] == component["source"]["sha256"],
                    "fixed-model source drift",
                )
                require(
                    record["removed_duplicate_io_value_info"]
                    == component["onnx_contract"]["remove_duplicate_io_value_info"],
                    "fixed-model metadata repair drift",
                )
                return {
                    "allowed_mutation": record["allowed_mutation"],
                    "source_sha256": record["source"]["sha256"],
                    "derived_sha256": record["derived"]["sha256"],
                    "derived_bytes": record["derived"]["bytes"],
                    "node_payload_sha256": record["node_payload_sha256"],
                    "initializer_payload_sha256": record[
                        "initializer_payload_sha256"
                    ],
                    "removed_duplicate_io_value_info": record[
                        "removed_duplicate_io_value_info"
                    ],
                }

            fixed_stage = base.run_stage(
                "fixed_model",
                fixed_fp,
                workspace,
                state_path,
                state,
                fixed_action,
            )
            fixed_model = workspace / f"stages/fixed_model/{name}.fixed.onnx"

            conversion_fp = base.stage_fingerprint(
                "qnn_conversion",
                EXPECTED_SPEC_SHA256,
                {"component": name, "fixed_sha256": fixed_stage["details"]["derived_sha256"]},
            )

            def conversion_action(partial: Path) -> dict[str, Any]:
                cpp = partial / f"{component['model_library_basename']}.cpp"
                command = converter_command(sdk_root, fixed_model, cpp, component, spec)
                base.run_logged(command, partial, env, partial / "qnn_converter.log")
                require(cpp.is_file(), "QNN converter created no C++ output")
                weights = cpp.with_suffix(".bin")
                require(weights.is_file(), "QNN converter created no weight binary")
                graph_name = base.parse_graph_name(cpp)
                return {
                    "command": command,
                    "qnn_graph_name": graph_name,
                    "cpp_sha256": sha256_file(cpp),
                    "weights_sha256": sha256_file(weights),
                }

            conversion_stage = base.run_stage(
                "qnn_conversion",
                conversion_fp,
                workspace,
                state_path,
                state,
                conversion_action,
            )
            conversion_dir = workspace / "stages/qnn_conversion"
            cpp = conversion_dir / f"{component['model_library_basename']}.cpp"
            weights = cpp.with_suffix(".bin")

            model_lib_fp = base.stage_fingerprint(
                "model_library",
                EXPECTED_SPEC_SHA256,
                {
                    "component": name,
                    "cpp_sha256": conversion_stage["details"]["cpp_sha256"],
                    "weights_sha256": conversion_stage["details"]["weights_sha256"],
                },
            )

            def model_lib_action(partial: Path) -> dict[str, Any]:
                output = partial / "model_libs"
                command = model_lib_command(
                    sdk_root, cpp, weights, output, component, spec
                )
                base.run_logged(
                    command, partial, env, partial / "model_lib_generator.log"
                )
                library = base.find_unique(
                    output,
                    f"lib{component['model_library_basename']}.so",
                    "QNN model library",
                )
                return {
                    "command": command,
                    "library_relative": library.relative_to(partial).as_posix(),
                    "library_bytes": library.stat().st_size,
                    "library_sha256": sha256_file(library),
                }

            model_lib_stage = base.run_stage(
                "model_library",
                model_lib_fp,
                workspace,
                state_path,
                state,
                model_lib_action,
            )
            model_library = (
                workspace
                / "stages/model_library"
                / model_lib_stage["details"]["library_relative"]
            )

            tools = base.tool_inventory(sdk_root, spec)
            context_fp = base.stage_fingerprint(
                "context_binary",
                EXPECTED_SPEC_SHA256,
                {
                    "component": name,
                    "model_library_sha256": model_lib_stage["details"][
                        "library_sha256"
                    ],
                    "qnn_graph_name": conversion_stage["details"]["qnn_graph_name"],
                    "tool_inventory_sha256": base.canonical_sha256(tools),
                    "target": spec["target"],
                },
            )

            def context_action(partial: Path) -> dict[str, Any]:
                graph_name = conversion_stage["details"]["qnn_graph_name"]
                htp, wrapper = base.backend_configs(sdk_root, graph_name, spec)
                htp_path = partial / "htp_backend_config.json"
                wrapper_path = partial / "backend_extensions.json"
                base.atomic_write_json(htp_path, htp)
                wrapper["backend_extensions"]["config_file_path"] = str(htp_path)
                base.atomic_write_json(wrapper_path, wrapper)
                output_dir = partial / "context"
                output_dir.mkdir()
                command = context_command(
                    sdk_root,
                    model_library,
                    output_dir,
                    wrapper_path,
                    component,
                    spec,
                )
                base.run_logged(
                    command, partial, env, partial / "context_generator.log"
                )
                expected = output_dir / f"{component['context_basename']}.bin"
                require(expected.is_file(), "context generator created no expected binary")
                require(
                    expected.stat().st_size >= component["minimum_context_bytes"],
                    "context binary is unexpectedly small",
                )
                metadata = partial / "context_metadata.json"
                utility_command = [
                    str(
                        sdk_root
                        / spec["tool_contract"]["context_binary_utility"]
                    ),
                    "--context_binary",
                    str(expected),
                    "--json_file",
                    str(metadata),
                ]
                base.run_logged(
                    utility_command,
                    partial,
                    env,
                    partial / "context_binary_utility.log",
                )
                require(metadata.is_file(), "context utility produced no JSON")
                read_json(metadata)
                return {
                    "command": command,
                    "utility_command": utility_command,
                    "qnn_graph_name": graph_name,
                    "target": copy.deepcopy(spec["target"]),
                    "context_relative": expected.relative_to(partial).as_posix(),
                    "context_bytes": expected.stat().st_size,
                    "context_sha256": sha256_file(expected),
                    "context_metadata_sha256": sha256_file(metadata),
                    "tool_inventory": tools,
                }

            context_stage = base.run_stage(
                "context_binary",
                context_fp,
                workspace,
                state_path,
                state,
                context_action,
            )

            deliverable_fp = base.stage_fingerprint(
                "deliverable",
                EXPECTED_SPEC_SHA256,
                {
                    "component": name,
                    "context_sha256": context_stage["details"]["context_sha256"],
                    "context_metadata_sha256": context_stage["details"][
                        "context_metadata_sha256"
                    ],
                    "controller_sha256": state["controller_sha256"],
                    "base_controller_sha256": BASE_CONTROLLER_SHA256,
                },
            )

            def deliverable_action(partial: Path) -> dict[str, Any]:
                payload = partial / "payload"
                payload.mkdir()
                context_dir = workspace / "stages/context_binary"
                context_file = context_dir / context_stage["details"]["context_relative"]
                copies = {
                    context_file: payload / context_file.name,
                    context_dir / "context_metadata.json": payload
                    / "context_metadata.json",
                    context_dir / "htp_backend_config.json": payload
                    / "htp_backend_config.json",
                    context_dir / "backend_extensions.json": payload
                    / "backend_extensions.json",
                    context_dir / "context_generator.log": payload
                    / "context_generator.log",
                    context_dir / "context_binary_utility.log": payload
                    / "context_binary_utility.log",
                    workspace / "stages/fixed_model/fixed_model_metadata.json": payload
                    / "fixed_model_metadata.json",
                }
                for source, target in copies.items():
                    require(source.is_file(), f"missing deliverable source: {source.name}")
                    shutil.copy2(source, target)
                shutil.copy2(spec_path, payload / spec_path.name)
                manifest = {
                    "schema_version": 1,
                    "experiment_id": EXPECTED_EXPERIMENT_ID,
                    "component": name,
                    "status": spec["output_contract"]["required_manifest_status"],
                    "claim_boundary": spec["claim_boundary"],
                    "created_utc": base.utc_now(),
                    "spec_sha256": EXPECTED_SPEC_SHA256,
                    "controller_sha256": state["controller_sha256"],
                    "base_controller_sha256": BASE_CONTROLLER_SHA256,
                    "host": copy.deepcopy(state["host"]),
                    "python_environment": packages,
                    "archives": copy.deepcopy(state["archives"]),
                    "source_model": copy.deepcopy(source_stage["details"]),
                    "fixed_model": copy.deepcopy(fixed_stage["details"]),
                    "qnn_conversion": copy.deepcopy(conversion_stage["details"]),
                    "model_library": copy.deepcopy(model_lib_stage["details"]),
                    "context": {
                        key: copy.deepcopy(value)
                        for key, value in context_stage["details"].items()
                        if key != "tool_inventory"
                    },
                    "tool_inventory": tools,
                    "runtime_binaries_bundled": False,
                    "phone_execution_proven": False,
                    "npu_execution_proven": False,
                    "next_gate": (
                        "Load this exact context only with the exact QAIRT 2.42 "
                        "runtime in a separate candidate, keep CPU fallback disabled, "
                        "and capture app-UID HTP/FastRPC/DMA plus numerical evidence."
                    ),
                }
                base.atomic_write_json(payload / "result_manifest.json", manifest)
                shutil.copy2(
                    payload / "result_manifest.json", partial / "result_manifest.json"
                )
                bundle = partial / f"qairt242_{name}_result.tar.gz"
                base.deterministic_tar_gz(payload, bundle)
                bundle_hash = sha256_file(bundle)
                base.atomic_write_text(
                    bundle.with_suffix(bundle.suffix + ".sha256"),
                    f"{bundle_hash}  {bundle.name}\n",
                )
                return {
                    "manifest_status": manifest["status"],
                    "bundle_bytes": bundle.stat().st_size,
                    "bundle_sha256": bundle_hash,
                    "phone_execution_proven": False,
                    "npu_execution_proven": False,
                }

            deliverable_stage = base.run_stage(
                "deliverable",
                deliverable_fp,
                workspace,
                state_path,
                state,
                deliverable_action,
            )
            state["status"] = "complete"
            state["completed_utc"] = base.utc_now()
            state["result"] = copy.deepcopy(deliverable_stage["details"])
            base.write_state(state_path, state)
            return verify_completed(workspace, state, spec, name)
        except Exception as error:
            state["status"] = "failed"
            state["last_failure"] = {
                "time_utc": base.utc_now(),
                "type": type(error).__name__,
                "message": str(error),
            }
            base.write_state(state_path, state)
            base.atomic_write_json(
                workspace / "qairt242_colab_failure_report.json",
                {
                    "schema_version": 1,
                    "experiment_id": EXPECTED_EXPERIMENT_ID,
                    "component": name,
                    "status": "failed_closed",
                    "spec_sha256": EXPECTED_SPEC_SHA256,
                    "time_utc": base.utc_now(),
                    "error": state["last_failure"],
                    "state_file": state_path.name,
                    "instruction": (
                        "Preserve this component workspace. Fix only the exact "
                        "documented precondition and rerun. Do not substitute QAIRT "
                        "2.45, CPU backend execution, or another graph. The other "
                        "component may be attempted in its independent workspace."
                    ),
                },
            )
            raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run", "verify"))
    parser.add_argument("--component", required=True, choices=ALLOWED_COMPONENTS)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument(
        "--scratch",
        type=Path,
        default=Path("/content/qairt242_supertonic_remaining_scratch"),
        help="Colab-local scratch; durable outputs remain under --workspace.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec_path, spec = load_spec(args.spec)
    if args.command == "plan":
        print(json.dumps(plan(spec, args.component), indent=2, sort_keys=True))
        return 0
    if args.workspace is None:
        raise RepairError("--workspace is required for run and verify")
    root_workspace = base.safe_workspace(args.workspace)
    workspace = component_workspace(root_workspace, args.component)
    if args.command == "run":
        result = execute(
            spec_path,
            spec,
            root_workspace,
            args.scratch,
            args.component,
        )
    else:
        state_path = workspace / "qairt242_colab_state.json"
        require(state_path.is_file(), f"missing state: {state_path}")
        state = read_json(state_path)
        result = verify_completed(workspace, state, spec, args.component)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RepairError as error:
        print(f"FAIL-CLOSED: {error}", file=sys.stderr)
        raise SystemExit(2)
