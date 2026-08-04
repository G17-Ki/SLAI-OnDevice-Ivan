#!/usr/bin/env python3
"""Compile/profile fixed-shape Piper TTS components on SM8850 / HTP v81.

This runner intentionally operates below Piper's text-normalization layer.  The
official qai-hub-models neural components do not require ``piper_phonemize`` to
load, serialize, or execute, which lets Apple-Silicon Macs submit the fixed
shape graphs even though the phonemizer wheel is Linux-only.

Cloud use is deliberately explicit: invoke this script from a detached tmux
session and redirect stdout/stderr to ``run.log``.  A state file is updated as
soon as every AI Hub job is submitted, so job IDs survive an interrupted local
poller.
"""

from __future__ import annotations

import argparse
import atexit
from collections.abc import Iterable
import enum
import gc
import hashlib
import importlib
from importlib import metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
import traceback
import types
from typing import Any


THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
for _thread_env_name in THREAD_ENV_VARS:
    try:
        _thread_env_value = int(os.environ.get(_thread_env_name, "2"))
    except ValueError:
        _thread_env_value = 2
    os.environ[_thread_env_name] = str(min(max(_thread_env_value, 1), 2))

import numpy as np
import torch


torch.set_num_threads(2)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    # Importers may already have initialized the inter-op pool. The detached
    # runner also exports the thread caps before Python starts.
    pass


LANGUAGE_ENUM_NAMES = {
    "en": "ENGLISH",
    "de": "GERMAN",
    "it": "ITALIAN",
}
COMPONENT_NAMES = ("encoder", "sdp", "flow", "decoder")
MAX_PROCESS_TREE_RSS_BYTES = int(2.5 * 1024**3)
RESOURCE_GUARD_POLL_SECONDS = 2.0
ITALIAN_MODEL_RELATIVE_PATH = Path(
    ".qaihm/qai-hub-models/models/pipertts/v1/it_IT-paola-medium.onnx"
)
ITALIAN_MODEL_BYTES = 63_511_038
ITALIAN_MODEL_SHA256 = (
    "6fc918b5a0ea6137382833dddfa567bffbe6a5060c02043c87192ee59c04210c"
)
ITALIAN_CONFIG_RELATIVE_PATH = Path(
    "it/it_IT/paola/medium/it_IT-paola-medium.onnx.json"
)
ITALIAN_CONFIG_BYTES = 7_099
ITALIAN_CONFIG_SHA256 = (
    "aea19c0a7fce29fbc359b93f10e7902854401e4c95ae2ea328ae516b15d296cf"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Submit official qai-hub-models Piper components to the Snapdragon "
            "8 Elite Gen 5 QRD (SM8850 / HTP v81)."
        )
    )
    parser.add_argument(
        "--language",
        action="append",
        choices=tuple(LANGUAGE_ENUM_NAMES),
        required=True,
        help="Piper language to run; repeat for more than one.",
    )
    parser.add_argument(
        "--component",
        action="append",
        choices=COMPONENT_NAMES,
        required=True,
        help="Fixed-shape neural component to run; repeat for more than one.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Persistent run directory. State, job logs, and numerical outputs land here.",
    )
    parser.add_argument("--device", default="Snapdragon 8 Elite Gen 5 QRD")
    parser.add_argument("--device-os", default="16")
    parser.add_argument(
        "--qairt-version",
        default="2.45",
        help="AI Hub QAIRT API version. 2.45 matches the public 0.58.0 scorecards.",
    )
    parser.add_argument("--profile-iterations", type=int, default=20)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--ceiling-seconds", type=int, default=10_800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atol", type=float, default=3e-2)
    parser.add_argument("--rtol", type=float, default=3e-2)
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Load, serialize, and generate goldens without contacting AI Hub.",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop instead of recording an error and continuing to the next item.",
    )
    args = parser.parse_args()
    if args.profile_iterations < 1:
        parser.error("--profile-iterations must be positive")
    if args.poll_seconds < 1:
        parser.error("--poll-seconds must be positive")
    if args.ceiling_seconds < 1:
        parser.error("--ceiling-seconds must be positive")
    if args.atol < 0 or args.rtol < 0:
        parser.error("--atol and --rtol must be non-negative")
    if len(args.language) != 1 or len(args.component) != 1:
        parser.error(
            "Use exactly one --language and one --component per persistent output directory"
        )
    return args


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_cached_file(
    path: Path,
    expected_bytes: int,
    expected_sha256: str,
    label: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required cached {label} is absent: {path}. Refusing an implicit download."
        )
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"Cached {label} has {actual_bytes} bytes; expected {expected_bytes}: {path}"
        )
    actual_sha256 = sha256(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"Cached {label} SHA-256 is {actual_sha256}; expected {expected_sha256}: {path}"
        )
    return {
        "path": str(path),
        "bytes": actual_bytes,
        "sha256": actual_sha256,
    }


def huggingface_hub_cache() -> Path:
    explicit = os.environ.get("HF_HUB_CACHE") or os.environ.get(
        "HUGGINGFACE_HUB_CACHE"
    )
    if explicit:
        return Path(explicit).expanduser()
    hf_home = Path(
        os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    ).expanduser()
    return hf_home / "hub"


def validate_required_caches(languages: Iterable[str]) -> dict[str, Any]:
    cache_state: dict[str, Any] = {}
    if "it" not in languages:
        return cache_state

    qaihm_root = Path(os.environ.get("QAIHM_STORE_ROOT", str(Path.home()))).expanduser()
    model_path = qaihm_root / ITALIAN_MODEL_RELATIVE_PATH
    cache_state["italian_model"] = validate_cached_file(
        model_path,
        ITALIAN_MODEL_BYTES,
        ITALIAN_MODEL_SHA256,
        "Italian Piper ONNX",
    )

    repo_cache = huggingface_hub_cache() / "models--rhasspy--piper-voices"
    revision_file = repo_cache / "refs" / "main"
    if not revision_file.is_file():
        raise FileNotFoundError(
            f"Hugging Face cache revision is absent: {revision_file}. "
            "Refusing an implicit download."
        )
    revision = revision_file.read_text().strip()
    if not revision or "/" in revision or ".." in revision:
        raise RuntimeError(f"Unexpected Hugging Face cache revision: {revision!r}")
    config_path = repo_cache / "snapshots" / revision / ITALIAN_CONFIG_RELATIVE_PATH
    cache_state["italian_config"] = {
        **validate_cached_file(
            config_path,
            ITALIAN_CONFIG_BYTES,
            ITALIAN_CONFIG_SHA256,
            "Italian Piper config",
        ),
        "revision": revision,
    }
    return cache_state


def process_tree_rss_bytes(root_pid: int) -> int:
    completed = subprocess.run(
        ["/bin/ps", "-axo", "pid=,ppid=,rss="],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    rows: list[tuple[int, int, int]] = []
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) != 3:
            continue
        pid, parent_pid, rss_kib = (int(field) for field in fields)
        rows.append((pid, parent_pid, rss_kib))
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent_pid, _ in rows:
            if parent_pid in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    return sum(rss_kib for pid, _, rss_kib in rows if pid in descendants) * 1024


def macos_memory_pressure_level() -> int | None:
    if sys.platform != "darwin":
        return None
    completed = subprocess.run(
        ["/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return int(completed.stdout.strip())


class ResourceGuard:
    """Abort the detached runner when the agreed Mac resource gate is crossed."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.root_pid = os.getpid()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.peak_rss_bytes = 0
        self.last_persisted = 0.0
        self.aborted = False

    def snapshot(self) -> dict[str, Any]:
        rss_bytes = process_tree_rss_bytes(self.root_pid)
        self.peak_rss_bytes = max(self.peak_rss_bytes, rss_bytes)
        return {
            "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "root_pid": self.root_pid,
            "process_tree_rss_bytes": rss_bytes,
            "process_tree_rss_gib": rss_bytes / 1024**3,
            "peak_process_tree_rss_bytes": self.peak_rss_bytes,
            "peak_process_tree_rss_gib": self.peak_rss_bytes / 1024**3,
            "rss_limit_bytes": MAX_PROCESS_TREE_RSS_BYTES,
            "rss_limit_gib": MAX_PROCESS_TREE_RSS_BYTES / 1024**3,
            "memory_pressure_level": macos_memory_pressure_level(),
        }

    @staticmethod
    def violation(snapshot: dict[str, Any]) -> str | None:
        if snapshot["process_tree_rss_bytes"] > MAX_PROCESS_TREE_RSS_BYTES:
            return (
                f"process-tree RSS {snapshot['process_tree_rss_gib']:.3f} GiB "
                f"exceeds {MAX_PROCESS_TREE_RSS_BYTES / 1024**3:.1f} GiB"
            )
        pressure = snapshot["memory_pressure_level"]
        if pressure is not None and pressure != 1:
            return f"kern.memorystatus_vm_pressure_level is {pressure}, expected 1"
        return None

    def check_now(self, persist: bool = False) -> dict[str, Any]:
        snapshot = self.snapshot()
        reason = self.violation(snapshot)
        if reason is not None:
            self.aborted = True
            record = {"status": "aborted", "reason": reason, **snapshot}
            write_json(self.path, record)
            print(f"[resource-guard] {reason}; terminating", flush=True)
            raise SystemExit(86)
        if persist:
            write_json(self.path, {"status": "running", **snapshot})
            self.last_persisted = time.monotonic()
        return snapshot

    def monitor(self) -> None:
        while not self.stop_event.wait(RESOURCE_GUARD_POLL_SECONDS):
            try:
                snapshot = self.snapshot()
                reason = self.violation(snapshot)
                if reason is not None:
                    self.aborted = True
                    write_json(
                        self.path,
                        {"status": "aborted", "reason": reason, **snapshot},
                    )
                    print(f"[resource-guard] {reason}; terminating", flush=True)
                    os._exit(86)
                if time.monotonic() - self.last_persisted >= 30:
                    write_json(self.path, {"status": "running", **snapshot})
                    self.last_persisted = time.monotonic()
            except BaseException as error:
                self.aborted = True
                write_json(
                    self.path,
                    {
                        "status": "aborted",
                        "reason": "resource monitor failed closed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "time_utc": time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                        ),
                    },
                )
                print(
                    f"[resource-guard] monitor failed closed: {type(error).__name__}: {error}",
                    flush=True,
                )
                os._exit(87)

    def start(self) -> None:
        self.check_now(persist=True)
        self.thread = threading.Thread(
            target=self.monitor,
            name="piper-resource-guard",
            daemon=True,
        )
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5)
        if self.aborted:
            return
        try:
            snapshot = self.snapshot()
            write_json(self.path, {"status": "stopped", **snapshot})
        except BaseException:
            pass


def ensure_optional_import_stubs() -> type[enum.Enum]:
    """Avoid optional preprocessing/dataset imports not used by component export."""

    if importlib.util.find_spec("piper_phonemize") is None:
        phonemizer = types.ModuleType("piper_phonemize")
        phonemizer._DIR = ""  # type: ignore[attr-defined]
        sys.modules["piper_phonemize"] = phonemizer

    class TTSLanguage(enum.Enum):
        ENGLISH = "ENGLISH"
        SPANISH = "SPANISH"
        ITALIAN = "ITALIAN"
        GERMAN = "GERMAN"
        CHINESE = "CHINESE"

    common_voice = types.ModuleType("qai_hub_models.datasets.common_voice")
    common_voice.TTSLanguage = TTSLanguage  # type: ignore[attr-defined]
    common_voice.LANG_CODE_MAP = {  # type: ignore[attr-defined]
        TTSLanguage.ENGLISH: "EN",
        TTSLanguage.SPANISH: "ES",
        TTSLanguage.ITALIAN: "IT",
        TTSLanguage.GERMAN: "DE",
        TTSLanguage.CHINESE: "ZH",
    }
    # Only used as an exported calibration-dataset class reference. This runner
    # does not quantize or instantiate the dataset.
    common_voice.CommonVoiceText = object  # type: ignore[attr-defined]
    sys.modules["qai_hub_models.datasets.common_voice"] = common_voice
    return TTSLanguage


def job_snapshot(job: Any) -> dict[str, Any]:
    status = job.get_status()
    return {
        "job_id": getattr(job, "job_id", None),
        "url": getattr(job, "url", None),
        "status": status.code,
        "message": status.message,
        "finished": bool(status.finished),
        "success": bool(status.success),
    }


def submitted_job_record(job: Any) -> dict[str, Any]:
    """Persist an accepted job ID without making a second network request."""

    job_id = getattr(job, "job_id", None)
    if not job_id:
        raise RuntimeError("AI Hub accepted a submission object without a job ID")
    return {
        "job_id": job_id,
        "url": getattr(job, "url", None),
        "status": "SUBMITTED",
        "message": "recorded immediately after submission",
        "finished": False,
        "success": False,
    }


def poll_jobs(
    jobs: dict[str, Any],
    state: dict[str, Any],
    state_path: Path,
    stage: str,
    poll_seconds: int,
    ceiling_seconds: int,
) -> dict[str, dict[str, Any]]:
    started = time.monotonic()
    while True:
        snapshot = {name: job_snapshot(job) for name, job in jobs.items()}
        state.setdefault("polls", []).append(
            {
                "stage": stage,
                "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "jobs": snapshot,
            }
        )
        state.setdefault("jobs", {})[stage] = snapshot
        write_json(state_path, state)
        print(json.dumps({"stage": stage, "jobs": snapshot}, indent=2), flush=True)
        if all(item["finished"] for item in snapshot.values()):
            return snapshot
        if time.monotonic() - started >= ceiling_seconds:
            state.setdefault("pending_at_ceiling", {})[stage] = {
                name: item for name, item in snapshot.items() if not item["finished"]
            }
            write_json(state_path, state)
            return snapshot
        time.sleep(poll_seconds)


def normalize_local_outputs(value: Any) -> list[np.ndarray]:
    values: Iterable[Any]
    if isinstance(value, (tuple, list)):
        values = value
    else:
        values = (value,)
    result = []
    for item in values:
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().numpy()
        result.append(np.asarray(item))
    return result


def build_component(
    shared: Any,
    tts_language: type[enum.Enum],
    language: str,
    component_name: str,
) -> Any:
    language_value = getattr(tts_language, LANGUAGE_ENUM_NAMES[language])
    generator = shared.get_model(language_value)
    if component_name == "encoder":
        return shared.Encoder(generator)
    if component_name == "sdp":
        return shared.SDP(generator, shared.SPEED[language_value])
    if component_name == "flow":
        return shared.Flow(generator)
    if component_name == "decoder":
        return shared.Decoder(generator)
    raise ValueError(component_name)


def prepare_local_artifacts(
    component: Any,
    item_dir: Path,
    seed: int,
) -> tuple[Path, dict[str, list[np.ndarray]], dict[str, np.ndarray]]:
    from qai_hub_models import TargetRuntime

    np.random.seed(seed)
    torch.manual_seed(seed)
    input_spec = component.get_input_spec()
    inputs = component.sample_inputs(
        input_spec,
        use_channel_last_format=TargetRuntime.VOICE_AI.channel_last_native_execution,
    )
    ordered_inputs = [
        torch.from_numpy(np.asarray(inputs[name][0])) for name in input_spec.keys()
    ]
    with torch.inference_mode():
        local_outputs = normalize_local_outputs(component(*ordered_inputs))
    output_names = list(component.get_output_spec())
    if len(output_names) != len(local_outputs):
        raise RuntimeError(
            f"Output-spec mismatch: names={output_names}, tensors={len(local_outputs)}"
        )
    goldens = dict(zip(output_names, local_outputs, strict=True))
    arrays: dict[str, np.ndarray] = {}
    for name, samples in inputs.items():
        if len(samples) != 1:
            raise ValueError(f"Expected one sample for {name}, got {len(samples)}")
        arrays[f"input__{name}"] = np.asarray(samples[0])
    arrays.update({f"golden__{name}": value for name, value in goldens.items()})
    np.savez(item_dir / "golden_io.npz", **arrays)

    source_dir = item_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(component.serialize(source_dir, input_spec))
    return source_path, inputs, goldens


def ordered_remote_outputs(outputs: dict[str, Any]) -> list[tuple[str, np.ndarray]]:
    ordered = []
    for name, samples in outputs.items():
        if not isinstance(samples, list) or len(samples) != 1:
            raise ValueError(f"Expected one output sample for {name!r}")
        ordered.append((name, np.asarray(samples[0])))
    return ordered


def compare_outputs(
    remote: dict[str, Any],
    goldens: dict[str, np.ndarray],
    target_model: Any,
    atol: float,
    rtol: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    from submit_component_aihub import compare_output, compiled_output_names

    target_names = compiled_output_names(target_model)
    ordered = ordered_remote_outputs(remote)
    comparisons: dict[str, Any] = {}
    saved: dict[str, np.ndarray] = {}
    for index, (golden_name, golden) in enumerate(goldens.items()):
        candidate = target_names[index] if index < len(target_names) else None
        if golden_name in remote:
            samples = remote[golden_name]
            if not isinstance(samples, list) or len(samples) != 1:
                raise ValueError(
                    f"Expected one output sample for exact-name match {golden_name!r}"
                )
            actual = np.asarray(samples[0])
            remote_name = golden_name
        elif candidate is not None and candidate in remote:
            actual = np.asarray(remote[candidate][0])
            remote_name = candidate
        elif index < len(ordered):
            remote_name, actual = ordered[index]
        else:
            comparisons[golden_name] = {
                "passed": False,
                "error": "Remote output is missing",
            }
            continue
        comparisons[golden_name] = {
            "remote_name": remote_name,
            **compare_output(actual, golden, atol, rtol),
        }
        saved[f"actual__{golden_name}"] = actual
        saved[f"golden__{golden_name}"] = golden
    return comparisons, saved


def download_job_logs(job: Any, output_dir: Path) -> dict[str, Any]:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        return {"status": "downloaded", "files": job.download_job_logs(str(output_dir))}
    except Exception as error:  # Logs are supplementary; never hide the main result.
        return {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
        }


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / "state.json"
    if state_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing AI Hub state: {state_path}. "
            "Use resume_aihub_tts_v81.py with the recorded job IDs instead."
        )
    os.environ["HF_HUB_OFFLINE"] = "1"
    cache_state = validate_required_caches(args.language)
    resource_guard = ResourceGuard(args.output_dir / "resource_guard.json")
    resource_guard.start()
    atexit.register(resource_guard.stop)
    tts_language = ensure_optional_import_stubs()

    import qai_hub as hub
    from qai_hub_models import Precision, TargetRuntime
    from qai_hub_models.utils.input_spec import to_hub_input_specs
    from submit_encoder_aihub import classify_profile, profile_metrics

    shared = importlib.import_module("qai_hub_models.models._shared.pipertts.model")
    try:
        qai_hub_models_version = metadata.version("qai_hub_models")
    except Exception:
        qai_hub_models_version = "unknown"

    state: dict[str, Any] = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "qai_hub": getattr(hub, "__version__", "unknown"),
            "qai_hub_models": qai_hub_models_version,
            "torch_intraop_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "thread_environment": {
                name: os.environ.get(name) for name in THREAD_ENV_VARS
            },
        },
        "cache_preflight": cache_state,
        "resource_guard": resource_guard.check_now(),
        "target": {
            "device": args.device,
            "os": args.device_os,
            "chipset": "sm8850",
            "htp": "v81",
            "runtime": "voice_ai (QNN DLC compile -> context-binary link)",
            "qairt_version": args.qairt_version,
        },
        "languages": args.language,
        "components": args.component,
        "local_only": args.local_only,
        "items": {},
        "jobs": {},
        "polls": [],
    }
    write_json(state_path, state)

    client = None if args.local_only else hub.Client()
    device = hub.Device(args.device, args.device_os, ["framework:qnn"])

    for language in args.language:
        for component_name in args.component:
            item_name = f"piper_{language}_{component_name}"
            item_dir = args.output_dir / item_name
            item_dir.mkdir(parents=True, exist_ok=True)
            item: dict[str, Any] = {
                "language": language,
                "component": component_name,
                "status": "preparing_local_artifact",
            }
            state["items"][item_name] = item
            write_json(state_path, state)
            print(f"[{item_name}] loading official component", flush=True)

            try:
                component = build_component(
                    shared, tts_language, language, component_name
                )
                source_path, inputs, goldens = prepare_local_artifacts(
                    component, item_dir, args.seed
                )
                input_spec = component.get_input_spec()
                item["local"] = {
                    "source_path": str(source_path),
                    "source_bytes": source_path.stat().st_size,
                    "source_sha256": sha256(source_path),
                    "input_specs": {
                        name: {
                            "shape": list(spec.shape),
                            "dtype": spec.dtype,
                        }
                        for name, spec in input_spec.items()
                    },
                    "output_names": list(goldens),
                    "output_shapes": {
                        name: list(value.shape) for name, value in goldens.items()
                    },
                    "golden_io": str(item_dir / "golden_io.npz"),
                }
                item["status"] = "local_artifact_ready"
                write_json(state_path, state)
                if args.local_only:
                    continue

                extra_compile = f"--qairt_version {args.qairt_version}"
                compile_options = component.get_hub_compile_options(
                    TargetRuntime.VOICE_AI,
                    Precision.float,
                    extra_compile,
                    device,
                    component.context_graph_name,
                )
                link_options = component.get_hub_link_options(
                    TargetRuntime.VOICE_AI,
                    f"--qairt_version {args.qairt_version}",
                )
                profile_options = component.get_hub_profile_options(
                    TargetRuntime.VOICE_AI,
                    (
                        f"--qairt_version {args.qairt_version} "
                        f"--max_profiler_iterations {args.profile_iterations}"
                    ),
                )
                inference_options = component.get_hub_profile_options(
                    TargetRuntime.VOICE_AI,
                    f"--qairt_version {args.qairt_version}",
                )
                item["options"] = {
                    "compile": compile_options,
                    "link": link_options,
                    "profile": profile_options,
                    "inference": inference_options,
                }
                item["status"] = "ready_to_submit"
                write_json(state_path, state)

                assert client is not None
                resource_guard.check_now(persist=True)
                source_model = hub.upload_model(
                    str(source_path), name=f"lfm-demo-{item_name}-source"
                )
                item["source_model_id"] = source_model.model_id
                item["status"] = "source_uploaded"
                write_json(state_path, state)
                resource_guard.check_now(persist=True)
                compile_job = client.submit_compile_job(
                    model=source_model,
                    device=device,
                    input_specs=to_hub_input_specs(input_spec),
                    options=compile_options,
                    name=f"lfm-demo-v81-{item_name}-dlc",
                    retry=False,
                )
                item["jobs"] = {"compile": submitted_job_record(compile_job)}
                item["status"] = "compile_submitted"
                write_json(state_path, state)
                compile_snapshot = poll_jobs(
                    {item_name: compile_job},
                    state,
                    state_path,
                    f"{item_name}:compile",
                    args.poll_seconds,
                    args.ceiling_seconds,
                )[item_name]
                item["jobs"]["compile"] = compile_snapshot
                if not compile_snapshot["success"]:
                    item["status"] = (
                        "compile_pending" if not compile_snapshot["finished"] else "compile_failed"
                    )
                    item["job_logs"] = {
                        "compile": download_job_logs(
                            compile_job, item_dir / "job_logs" / "compile"
                        )
                    }
                    write_json(state_path, state)
                    continue

                dlc_model = compile_job.get_target_model()
                if dlc_model is None:
                    raise RuntimeError("Compile succeeded but returned no target DLC")
                resource_guard.check_now(persist=True)
                link_job = client.submit_link_job(
                    [dlc_model],
                    device=device,
                    options=link_options,
                    name=f"lfm-demo-v81-{item_name}-context",
                )
                item["jobs"]["link"] = submitted_job_record(link_job)
                item["status"] = "link_submitted"
                write_json(state_path, state)
                link_snapshot = poll_jobs(
                    {item_name: link_job},
                    state,
                    state_path,
                    f"{item_name}:link",
                    args.poll_seconds,
                    args.ceiling_seconds,
                )[item_name]
                item["jobs"]["link"] = link_snapshot
                if not link_snapshot["success"]:
                    item["status"] = (
                        "link_pending" if not link_snapshot["finished"] else "link_failed"
                    )
                    item["job_logs"] = {
                        "compile": download_job_logs(
                            compile_job, item_dir / "job_logs" / "compile"
                        ),
                        "link": download_job_logs(
                            link_job, item_dir / "job_logs" / "link"
                        ),
                    }
                    write_json(state_path, state)
                    continue

                target_model = link_job.get_target_model()
                if target_model is None:
                    raise RuntimeError("Link succeeded but returned no context binary")
                resource_guard.check_now(persist=True)
                profile_job = client.submit_profile_job(
                    model=target_model,
                    device=device,
                    options=profile_options,
                    name=f"lfm-demo-v81-{item_name}-profile",
                    retry=False,
                )
                item["jobs"]["profile"] = submitted_job_record(profile_job)
                item["status"] = "profile_submitted"
                write_json(state_path, state)
                resource_guard.check_now(persist=True)
                inference_job = client.submit_inference_job(
                    model=target_model,
                    device=device,
                    inputs=inputs,
                    options=inference_options,
                    name=f"lfm-demo-v81-{item_name}-inference",
                    retry=False,
                )
                item["jobs"]["inference"] = submitted_job_record(inference_job)
                item["status"] = "profile_and_inference_submitted"
                write_json(state_path, state)
                dependent = poll_jobs(
                    {"profile": profile_job, "inference": inference_job},
                    state,
                    state_path,
                    f"{item_name}:profile_and_inference",
                    args.poll_seconds,
                    args.ceiling_seconds,
                )
                item["jobs"].update(dependent)
                if not all(entry["success"] for entry in dependent.values()):
                    item["status"] = (
                        "remote_jobs_pending"
                        if any(not entry["finished"] for entry in dependent.values())
                        else "profile_or_inference_failed"
                    )
                    write_json(state_path, state)
                    continue

                profile = profile_job.download_profile()
                remote = inference_job.download_output_data()
                if not isinstance(profile, dict):
                    raise TypeError(f"Profile response is {type(profile).__name__}")
                if not isinstance(remote, dict):
                    raise TypeError(f"Inference response is {type(remote).__name__}")
                write_json(item_dir / "profile.json", profile)
                placement = classify_profile(profile)
                write_json(item_dir / "placement.json", placement)
                comparisons, saved = compare_outputs(
                    remote, goldens, target_model, args.atol, args.rtol
                )
                np.savez(item_dir / "remote_outputs.npz", **saved)
                all_numerics_pass = bool(comparisons) and all(
                    result.get("passed", False) for result in comparisons.values()
                )
                strict_npu = (
                    placement["npu_runtime_layers"] > 0
                    and placement["cpu_fallback_runtime_layers"] == 0
                    and placement["other_runtime_layers"] == 0
                )
                item["result"] = {
                    "metrics": profile_metrics(profile),
                    "placement": placement,
                    "strict_npu_pass": strict_npu,
                    "comparisons": comparisons,
                    "all_numerics_pass": all_numerics_pass,
                }
                item["status"] = (
                    "passed"
                    if strict_npu and all_numerics_pass
                    else "completed_with_numerical_or_placement_warning"
                )
                item["job_logs"] = {
                    "compile": download_job_logs(
                        compile_job, item_dir / "job_logs" / "compile"
                    ),
                    "link": download_job_logs(link_job, item_dir / "job_logs" / "link"),
                    "profile": download_job_logs(
                        profile_job, item_dir / "job_logs" / "profile"
                    ),
                    "inference": download_job_logs(
                        inference_job, item_dir / "job_logs" / "inference"
                    ),
                }
                write_json(state_path, state)
            except Exception as error:
                item["status"] = "error"
                item["error"] = {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                }
                write_json(state_path, state)
                print(f"[{item_name}] {type(error).__name__}: {error}", flush=True)
                if args.stop_on_failure:
                    raise
            finally:
                if "component" in locals():
                    del component
                gc.collect()

    state["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    statuses = {name: item["status"] for name, item in state["items"].items()}
    state["summary"] = {
        "statuses": statuses,
        "passed": sum(status == "passed" for status in statuses.values()),
        "total": len(statuses),
    }
    write_json(state_path, state)
    print(json.dumps(state["summary"], indent=2), flush=True)
    resource_guard.stop()
    atexit.unregister(resource_guard.stop)


if __name__ == "__main__":
    main()
