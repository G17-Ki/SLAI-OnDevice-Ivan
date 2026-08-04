#!/usr/bin/env python3
"""Audit cached native Whisper frontends against the pinned QAI/HF golden.

The script performs only lightweight host compilation and preprocessing.  It
does not build Android, run an ASR model, use the network, invoke ADB, or touch
an APK.  Generated inputs, binaries, and arrays stay under ignored
``artifacts/``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "scripts"
CPP_SOURCE = SCRIPT_DIR / "whisper_native_frontend_audit.cc"
GOLDEN_MODULE_PATH = SCRIPT_DIR / "prepare_whisper_ort_qnn_golden.py"
GOLDEN_MANIFEST = (
    REPO_ROOT
    / "reports/android_phone/streaming_response/whisper_v81"
    / "whisper_ort_qnn_frontend_golden_v1.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts/whisper_native_frontend_audit"

CACHE_ROOT = Path(
    os.environ.get(
        "LMF_QNN_DEPS_ROOT",
        Path.home()
        / ".cache/lmf-demo/npu-multilingual/build/build-arm64-qnn/_deps",
    )
)
DEFAULT_KISSFFT_ROOT = CACHE_ROOT / "kissfft-src"
DEFAULT_KALDI_PARENT = CACHE_ROOT / "kaldi_native_fbank-src"
DEFAULT_KISSFFT_ARCHIVE = (
    CACHE_ROOT
    / "kissfft-subbuild/kissfft-populate-prefix/src"
    / "febd4caeed32e33ad8b2e0bb5ea77542c40f18ec.zip"
)
DEFAULT_KALDI_ARCHIVE = (
    CACHE_ROOT
    / "kaldi_native_fbank-subbuild/kaldi_native_fbank-populate-prefix/src"
    / "v1.22.3.tar.gz"
)

EXPECTED_GOLDEN_MANIFEST_SHA256 = (
    "38a4ee71438c052dd3a5e69c8a607049097d59fd8a8d3a002537784aa07e8cce"
)
EXPECTED_HANN_SHA256 = (
    "c692d19f0b2bd745f0dfc967261c3e3d21a2473e485a677d219e9b4a73e0218e"
)
EXPECTED_MEL_SHA256 = "d18b48f33345d958ea71098bb990b5e32cebc78d9a2cf65ca0d7302f34ebe8a9"
EXPECTED_ARCHIVE_SHA256 = {
    "kissfft": "497103e664168ebe39580b757adbe616f6cf85a16572af581ca7bc42d0ab13fd",
    "kaldi_native_fbank": (
        "9176cc66fc7ce1edf85cf355b06e320c57db6297df74277f575183468893cf61"
    ),
}
EXPECTED_KISSFFT_SOURCE_SHA256 = {
    "kiss_fft.c": "3e8af99313c7e94f952ac12db46d3d5c5dd4a9bf93e0c32d43efa5fe39724051",
    "kiss_fftr.c": "128b651120725e8d55f6b9edf5d44b06dc022b260894eac078c6ff8aa7159561",
    "kiss_fft.h": "9d6c7ddf520932c0aad669c86c064a24e571e594681be9f1491c14c9c9c75693",
    "kiss_fftr.h": "a0f72c1bddc3126d7a51ad80e965b58f6af77e4419aa882a0b11951e783632a8",
    "_kiss_fft_guts.h": (
        "dd1731891d0fff0e297198847de39216c5c8b9677ea4fe53160d1f6c74218d40"
    ),
}
EXPECTED_KALDI_SOURCE_SHA256 = {
    "feature-window.cc": (
        "cac43e0def16acb3032491ea65d4cdfc9e2523ef18d8202524a58b43b7fe67d2"
    ),
    "feature-window.h": (
        "8a181e9111ed9c31b3d2c3fcc14cb52ef106294e0edb78ed69788300c3992a59"
    ),
    "kaldi-math.cc": (
        "11b57da15d691a94e61b8a3471c7dbad5318125412b00456aa10e4f6af30d47d"
    ),
    "kaldi-math.h": (
        "f4aed12e9e1932b97e47a2b8c33d75b6ab96e31c8d4687ebb61c6ab1783011a8"
    ),
    "mel-computations.cc": (
        "8d02d0edc209295bfa76d1a1a08def51f634a1458fac4408fefd4bdbb7b57e43"
    ),
    "mel-computations.h": (
        "3a85adb8f7e51598f6f46a0d2b6bdef4c0836efbc925dd0a5e1983598f3f4341"
    ),
    "whisper-feature.cc": (
        "191a0ffb1fbeaf8da31094bdc3a7dc1de9ae72b33ac711b224612f4981e27cb7"
    ),
    "whisper-feature.h": (
        "cefee5c8568de7a0185bdc1117c1ff6f8251bdd48da006391e20ffbfa60b0dc0"
    ),
}

N_SAMPLES = 480_000
N_FFT = 400
HOP = 160
N_BINS = 201
N_MELS = 80
N_FRAMES = 3_000
BOUNDED_MAX_U16_DELTA = 1
BOUNDED_MAX_DIFFERENT_FRACTION = 0.001
TORCH_INTRAOP_THREADS = 1
TORCH_INTEROP_THREADS = 1


@dataclass(frozen=True)
class Fixture:
    name: str
    path: Path
    wav_sha256: str
    pcm_sha256: str
    frames: int
    float_sha256: str
    quantized_sha256: str


FIXTURES = {
    "en": Fixture(
        name="en",
        path=SCRIPT_DIR / "phone_acceptance_fixtures/question_en.wav",
        wav_sha256=("b5fa81a74e608624b52fb3e4cf895d6982a455022597a0da0ad3dc52866f8b69"),
        pcm_sha256=("e46de49ab58c33d12ffa8502f8a12d9ed0810bb38ece8f7522be57ef236cff4f"),
        frames=52_819,
        float_sha256=(
            "8864a2c385f2f768e7719297735cc857a2d3c938c9f775f9e706506f516c6f41"
        ),
        quantized_sha256=(
            "29c478d40bb98b7782280dcfbc98662cee514df7e62c180c5ba849db24a2f6a1"
        ),
    ),
    "ja": Fixture(
        name="ja",
        path=SCRIPT_DIR / "phone_acceptance_fixtures/question_ja.wav",
        wav_sha256=("0da6ee3b9514a96597bf9225a1bcdf7cae86608e095e157490899a443948b761"),
        pcm_sha256=("439ae3efa582f5da8a277c8e3d1130d3c545f7b9cd89c84d50a695ccb4b773c2"),
        frames=39_710,
        float_sha256=(
            "629ce935d0adb27ca28392f3999d0886353608e2acca0802daca4e466cda0767"
        ),
        quantized_sha256=(
            "8cf37ba9a60f5cd91453dae2d721e9fd7fec69919582223bf02a3f76f8421a80"
        ),
    ),
    "zh": Fixture(
        name="zh",
        path=SCRIPT_DIR / "phone_acceptance_fixtures/question_zh.wav",
        wav_sha256=("bfc02160e9b86c487dc122a7e501e71d411ada2d7a5b94ad59b1bf0f8e149c63"),
        pcm_sha256=("86495ac40c43df1256ec14bec1c52cc253eac1aebdf4f6e3235e7e0c12dfc245"),
        frames=41_717,
        float_sha256=(
            "1c3b7eba4ce9f33369ee80bd4c5b1621ebd2488abdafc9b27ea7be8c0ae3b66c"
        ),
        quantized_sha256=(
            "6d18da5e1ec42bbd8161881a598e835c365138377b3d42364ff360e4849d4263"
        ),
    ),
    "ko": Fixture(
        name="ko",
        path=SCRIPT_DIR / "phone_acceptance_fixtures/question_ko.wav",
        wav_sha256=("0ce8353dc9faa844cc8214c613f182a127338b71b139955ccb903ca806a7b3de"),
        pcm_sha256=("643c9d9c697cb38a5381b4eb268c763445906560eb25b374eb9d9419b9684db5"),
        frames=33_432,
        float_sha256=(
            "e374d6e02d10f3872c18d57cded7855c6547f54ec34921413ae8d91a2707758f"
        ),
        quantized_sha256=(
            "808b38d3c4198c15566116562eb7ad61fbafc01912a747ae439047fb9d7eb1c5"
        ),
    ),
    "ru": Fixture(
        name="ru",
        path=SCRIPT_DIR / "phone_acceptance_fixtures/question_ru.wav",
        wav_sha256=("3545b8a20e3f320174a4e4dee57ff77268f30f79526eef91c86c778e3c180683"),
        pcm_sha256=("dd30bd3a3f4c59a43595d01fdf58115bb66de24e49bef3df2a308944cda5f9e0"),
        frames=24_315,
        float_sha256=(
            "c7dbc5927caacd2272880acad28ebd16de5504eef173bf4b10e99df86d465279"
        ),
        quantized_sha256=(
            "0e0c25a24157edabe3c1d193e60e8b59886aa9b2ba9b1581fb5c90c2eaf3bbc4"
        ),
    ),
}

FIXTURE_PROFILES: dict[str, tuple[str, ...]] = {
    # Keep the historical invocation and ordering unchanged.
    "v1": ("en", "ja"),
    # Additive multilingual audit profile; v1 assets and trust roots remain intact.
    "v2": ("en", "zh", "ru", "ko", "ja"),
}

_torch_thread_policy_configured = False

BUILD_FLAVORS = {
    "portable": ["-O2", "-fno-fast-math", "-ffp-contract=off"],
    # The cached Android arm64 KissFFT objects were built with -O3
    # -ffast-math.  This host flavor is only an approximation; target parity
    # still has to be measured on the phone.
    "android_fast_math": ["-O3", "-ffast-math", "-DNDEBUG"],
}
MODES = (
    "kissfft-direct-scalar",
    "kissfft-hypot-scalar",
    "kissfft-hypot-fma",
    "kaldi-stock",
)


class AuditError(RuntimeError):
    pass


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise AuditError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise AuditError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")
    return {"bytes": path.stat().st_size, "path": str(path), "sha256": actual}


def load_golden_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "prepare_whisper_ort_qnn_golden", GOLDEN_MODULE_PATH
    )
    if spec is None or spec.loader is None:
        raise AuditError(f"cannot import golden generator: {GOLDEN_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify_sources(
    kissfft_root: Path,
    kaldi_parent: Path,
    kissfft_archive: Path,
    kaldi_archive: Path,
) -> dict[str, Any]:
    records: dict[str, Any] = {
        "archives": {
            "kissfft": require_hash(
                kissfft_archive,
                EXPECTED_ARCHIVE_SHA256["kissfft"],
                "KissFFT archive",
            ),
            "kaldi_native_fbank": require_hash(
                kaldi_archive,
                EXPECTED_ARCHIVE_SHA256["kaldi_native_fbank"],
                "kaldi-native-fbank archive",
            ),
        },
        "kissfft_sources": {},
        "kaldi_native_fbank_sources": {},
    }
    for name, expected in EXPECTED_KISSFFT_SOURCE_SHA256.items():
        records["kissfft_sources"][name] = require_hash(
            kissfft_root / name, expected, f"KissFFT source {name}"
        )
    kaldi_csrc = kaldi_parent / "kaldi-native-fbank/csrc"
    for name, expected in EXPECTED_KALDI_SOURCE_SHA256.items():
        records["kaldi_native_fbank_sources"][name] = require_hash(
            kaldi_csrc / name, expected, f"kaldi-native-fbank source {name}"
        )
    records["golden_manifest"] = require_hash(
        GOLDEN_MANIFEST,
        EXPECTED_GOLDEN_MANIFEST_SHA256,
        "frontend golden manifest",
    )
    records["audit_cpp"] = {
        "bytes": CPP_SOURCE.stat().st_size,
        "path": str(CPP_SOURCE),
        "sha256": sha256_file(CPP_SOURCE),
    }
    return records


def read_fixture(fixture: Fixture) -> tuple[np.ndarray, dict[str, Any]]:
    wav_record = require_hash(fixture.path, fixture.wav_sha256, f"{fixture.name} WAV")
    with wave.open(str(fixture.path), "rb") as stream:
        params = (
            stream.getnchannels(),
            stream.getsampwidth(),
            stream.getframerate(),
            stream.getnframes(),
            stream.getcomptype(),
        )
        pcm = stream.readframes(stream.getnframes())
    expected_params = (1, 2, 16_000, fixture.frames, "NONE")
    if params != expected_params:
        raise AuditError(
            f"{fixture.name} WAV parameters mismatch: expected "
            f"{expected_params}, got {params}"
        )
    actual_pcm_sha256 = sha256_bytes(pcm)
    if actual_pcm_sha256 != fixture.pcm_sha256:
        raise AuditError(
            f"{fixture.name} PCM SHA-256 mismatch: expected {fixture.pcm_sha256}, "
            f"got {actual_pcm_sha256}"
        )
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
    samples /= np.float32(32_768.0)
    padded = np.zeros(N_SAMPLES, dtype=np.float32)
    padded[: samples.size] = samples
    return padded, {
        **wav_record,
        "frames": fixture.frames,
        "pcm_sha256": actual_pcm_sha256,
        "padded_samples": N_SAMPLES,
    }


def load_torch_one_thread() -> tuple[Any, Any]:
    """Import Torch and fail closed unless host work is limited to one thread."""
    global _torch_thread_policy_configured

    # These variables cover the common OpenMP/BLAS backends if Torch has not
    # yet been imported. The runtime setters below remain authoritative.
    os.environ["OMP_NUM_THREADS"] = str(TORCH_INTRAOP_THREADS)
    os.environ["MKL_NUM_THREADS"] = str(TORCH_INTRAOP_THREADS)
    try:
        import torch
        import torch.nn.functional as torch_functional
    except ImportError as error:
        raise AuditError("PyTorch is required for the pinned reference") from error

    if not _torch_thread_policy_configured:
        torch.set_num_threads(TORCH_INTRAOP_THREADS)
        try:
            torch.set_num_interop_threads(TORCH_INTEROP_THREADS)
        except RuntimeError as error:
            if torch.get_num_interop_threads() != TORCH_INTEROP_THREADS:
                raise AuditError(
                    "cannot enforce the one-thread Torch interop policy"
                ) from error
        _torch_thread_policy_configured = True
    if torch.get_num_threads() != TORCH_INTRAOP_THREADS:
        raise AuditError("cannot enforce the one-thread Torch intraop policy")
    if torch.get_num_interop_threads() != TORCH_INTEROP_THREADS:
        raise AuditError("cannot enforce the one-thread Torch interop policy")
    return torch, torch_functional


def torch_thread_policy() -> dict[str, Any]:
    torch, _ = load_torch_one_thread()
    return {
        "intraop_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "omp_num_threads": os.environ["OMP_NUM_THREADS"],
        "mkl_num_threads": os.environ["MKL_NUM_THREADS"],
    }


def torch_reference(
    samples: np.ndarray, mel_filters: np.ndarray
) -> dict[str, np.ndarray]:
    torch, torch_functional = load_torch_one_thread()

    with torch.inference_mode():
        waveform = torch.from_numpy(samples).to(dtype=torch.float32, device="cpu")
        hann = torch.hann_window(
            N_FFT, periodic=True, dtype=torch.float32, device="cpu"
        )
        padded = torch_functional.pad(
            waveform.reshape(1, 1, -1), (N_FFT // 2, N_FFT // 2), mode="reflect"
        ).reshape(-1)
        frame0 = padded[:N_FFT] * hann
        spectrum = torch.stft(
            waveform,
            N_FFT,
            HOP,
            window=hann,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        power = spectrum[..., :-1].abs() ** 2
        mel_tensor = torch.from_numpy(mel_filters).to(dtype=torch.float32)
        mel_energies = mel_tensor.T @ power
        features = torch.clamp(mel_energies, min=1e-10).log10()
        features = torch.maximum(features, features.max() - 8.0)
        features = (features + 4.0) / 4.0

        result = {
            "hann": hann.numpy().astype(np.float32, copy=False),
            "frame0": frame0.numpy().astype(np.float32, copy=False),
            "power": power.numpy().astype(np.float32, copy=False),
            "mel_energies": mel_energies.numpy().astype(np.float32, copy=False),
            "features": features.numpy().astype(np.float32, copy=False),
        }
    for key, value in result.items():
        result[key] = np.ascontiguousarray(value, dtype=np.float32)
    return result


def quantize_qai(features: np.ndarray, golden: Any) -> np.ndarray:
    return np.ascontiguousarray(
        golden.quantize_uint16(
            features,
            scale=golden.FEATURE_QUANTIZATION["scale"],
            zero_point=golden.FEATURE_QUANTIZATION["zero_point"],
        ),
        dtype="<u2",
    )


def array_sha256(value: np.ndarray, dtype: str) -> str:
    return sha256_bytes(value.astype(dtype, copy=False).tobytes(order="C"))


def write_array(path: Path, value: np.ndarray, dtype: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = value.astype(dtype, copy=False).tobytes(order="C")
    if path.exists() and path.read_bytes() == data:
        return
    path.write_bytes(data)


def run_checked(
    command: list[str], *, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    process = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        rendered = " ".join(command)
        raise AuditError(
            f"command failed ({process.returncode}): {rendered}\n"
            f"stdout:\n{process.stdout}\nstderr:\n{process.stderr}"
        )
    return process


def compile_binary(
    *,
    flavor: str,
    output_dir: Path,
    kissfft_root: Path,
    kaldi_parent: Path,
) -> dict[str, Any]:
    cc = shutil.which("cc")
    cxx = shutil.which("c++")
    if cc is None or cxx is None:
        raise AuditError("host C and C++ compilers are required")
    build_dir = output_dir / "build" / flavor
    build_dir.mkdir(parents=True, exist_ok=True)
    flags = BUILD_FLAVORS[flavor]
    kiss_fft_object = build_dir / "kiss_fft.o"
    kiss_fftr_object = build_dir / "kiss_fftr.o"
    binary = build_dir / "whisper_native_frontend_audit"

    commands = [
        [
            cc,
            *flags,
            "-std=c99",
            "-I",
            str(kissfft_root),
            "-c",
            str(kissfft_root / "kiss_fft.c"),
            "-o",
            str(kiss_fft_object),
        ],
        [
            cc,
            *flags,
            "-std=c99",
            "-I",
            str(kissfft_root),
            "-c",
            str(kissfft_root / "kiss_fftr.c"),
            "-o",
            str(kiss_fftr_object),
        ],
    ]
    for command in commands:
        run_checked(command)

    kaldi_csrc = kaldi_parent / "kaldi-native-fbank/csrc"
    link_command = [
        cxx,
        *flags,
        "-std=c++17",
        "-I",
        str(kaldi_parent),
        "-I",
        str(kissfft_root),
        str(CPP_SOURCE),
        str(kaldi_csrc / "feature-window.cc"),
        str(kaldi_csrc / "kaldi-math.cc"),
        str(kaldi_csrc / "mel-computations.cc"),
        str(kaldi_csrc / "whisper-feature.cc"),
        str(kiss_fft_object),
        str(kiss_fftr_object),
        "-o",
        str(binary),
    ]
    run_checked(link_command)
    version = run_checked([cxx, "--version"]).stdout.splitlines()[0]
    return {
        "binary": binary,
        "binary_sha256": sha256_file(binary),
        "compiler": version,
        "flags": flags,
        "commands": commands + [link_command],
    }


def compare_float(reference: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    reference = np.ascontiguousarray(reference, dtype=np.float32)
    actual = np.ascontiguousarray(actual, dtype=np.float32)
    if reference.shape != actual.shape:
        raise AuditError(f"float shape mismatch: {reference.shape} vs {actual.shape}")
    bitwise = reference.view(np.uint32) != actual.view(np.uint32)
    absolute = np.abs(reference.astype(np.float64) - actual.astype(np.float64))
    return {
        "actual_sha256": array_sha256(actual, "<f4"),
        "bitwise_different_elements": int(np.count_nonzero(bitwise)),
        "elements": int(reference.size),
        "maximum_absolute_error": float(absolute.max(initial=0.0)),
        "mean_absolute_error": float(absolute.mean()),
        "reference_sha256": array_sha256(reference, "<f4"),
    }


def first_quantized_differences(
    reference: np.ndarray, actual: np.ndarray, limit: int = 8
) -> list[dict[str, Any]]:
    indices = np.flatnonzero(reference.reshape(-1) != actual.reshape(-1))[:limit]
    values: list[dict[str, Any]] = []
    for flat_index in indices.tolist():
        mel, frame = divmod(flat_index, N_FRAMES)
        expected = int(reference.reshape(-1)[flat_index])
        observed = int(actual.reshape(-1)[flat_index])
        values.append(
            {
                "index": [0, mel, frame],
                "actual": observed,
                "delta": observed - expected,
                "reference": expected,
            }
        )
    return values


def compare_quantized(reference: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    reference = np.ascontiguousarray(reference, dtype="<u2")
    actual = np.ascontiguousarray(actual, dtype="<u2")
    if reference.shape != actual.shape:
        raise AuditError(
            f"quantized shape mismatch: {reference.shape} vs {actual.shape}"
        )
    different = reference != actual
    absolute = np.abs(reference.astype(np.int32) - actual.astype(np.int32))
    count = int(np.count_nonzero(different))
    return {
        "actual_sha256": array_sha256(actual, "<u2"),
        "different_elements": count,
        "different_fraction": count / int(reference.size),
        "elements": int(reference.size),
        "first_differences": first_quantized_differences(reference, actual),
        "maximum_absolute_difference": int(absolute.max(initial=0)),
        "mean_absolute_difference": float(absolute.mean()),
        "reference_sha256": array_sha256(reference, "<u2"),
        "strict_byte_parity": count == 0,
    }


def read_native_array(path: Path, dtype: str, shape: tuple[int, ...]) -> np.ndarray:
    expected = int(np.prod(shape))
    value = np.fromfile(path, dtype=dtype)
    if value.size != expected:
        raise AuditError(
            f"native output size mismatch for {path}: expected {expected}, "
            f"got {value.size}"
        )
    return np.ascontiguousarray(value.reshape(shape))


def torch_postprocess_power(
    power: np.ndarray, mel_filters: np.ndarray, golden: Any
) -> tuple[np.ndarray, np.ndarray]:
    torch, _ = load_torch_one_thread()

    with torch.inference_mode():
        power_tensor = torch.from_numpy(np.ascontiguousarray(power, dtype=np.float32))
        mel_tensor = torch.from_numpy(
            np.ascontiguousarray(mel_filters, dtype=np.float32)
        )
        mel_energies = mel_tensor.T @ power_tensor
        features = torch.clamp(mel_energies, min=1e-10).log10()
        features = torch.maximum(features, features.max() - 8.0)
        features = (features + 4.0) / 4.0
        feature_array = np.ascontiguousarray(features.numpy(), dtype=np.float32)
    return feature_array, quantize_qai(feature_array, golden)


def run_native_mode(
    *,
    binary: Path,
    mode: str,
    prefix: Path,
    pcm_path: Path,
    hann_path: Path,
    mel_path: Path,
) -> dict[str, Any]:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    process = run_checked(
        [
            str(binary),
            mode,
            str(pcm_path),
            str(hann_path),
            str(mel_path),
            str(prefix),
        ]
    )
    elapsed = time.monotonic() - started
    try:
        native_record = json.loads(process.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise AuditError(f"invalid native timing output: {process.stdout!r}") from error
    return {
        "native_seconds": float(native_record["seconds"]),
        "wall_seconds": elapsed,
    }


def audit_fixture(
    *,
    fixture: Fixture,
    reference: dict[str, np.ndarray],
    reference_quantized: np.ndarray,
    mel_filters: np.ndarray,
    golden: Any,
    build_records: dict[str, dict[str, Any]],
    flavors: Iterable[str],
    modes: Iterable[str],
    output_dir: Path,
    pcm_path: Path,
    hann_path: Path,
    mel_path: Path,
) -> dict[str, Any]:
    variants: dict[str, Any] = {}
    for flavor in flavors:
        binary = Path(build_records[flavor]["binary"])
        for mode in modes:
            variant_id = f"{flavor}/{mode}"
            prefix = output_dir / "outputs" / fixture.name / flavor / mode
            timing = run_native_mode(
                binary=binary,
                mode=mode,
                prefix=prefix,
                pcm_path=pcm_path,
                hann_path=hann_path,
                mel_path=mel_path,
            )
            native_frame0 = read_native_array(
                Path(f"{prefix}.frame0.f32le.bin"), "<f4", (N_FFT,)
            )
            native_mel = read_native_array(
                Path(f"{prefix}.mel_energies.f32le.bin"),
                "<f4",
                (N_MELS, N_FRAMES),
            )
            native_features = read_native_array(
                Path(f"{prefix}.features.f32le.bin"),
                "<f4",
                (N_MELS, N_FRAMES),
            )
            native_quantized = read_native_array(
                Path(f"{prefix}.features.u16le.bin"),
                "<u2",
                (N_MELS, N_FRAMES),
            )
            record: dict[str, Any] = {
                "frame0": compare_float(reference["frame0"], native_frame0),
                "mel_energies": compare_float(reference["mel_energies"], native_mel),
                "normalized_features": compare_float(
                    reference["features"], native_features
                ),
                "quantized_features": compare_quantized(
                    reference_quantized, native_quantized
                ),
                "timing": timing,
            }
            power_path = Path(f"{prefix}.power.f32le.bin")
            if power_path.is_file():
                native_power = read_native_array(power_path, "<f4", (N_BINS, N_FRAMES))
                torch_features, torch_quantized = torch_postprocess_power(
                    native_power, mel_filters, golden
                )
                record["power"] = compare_float(reference["power"], native_power)
                record["torch_postprocess_of_native_power"] = {
                    "normalized_features": compare_float(
                        reference["features"], torch_features
                    ),
                    "quantized_features": compare_quantized(
                        reference_quantized, torch_quantized
                    ),
                }
            if mode == "kaldi-stock":
                native_hann = read_native_array(
                    Path(f"{prefix}.hann.f32le.bin"), "<f4", (N_FFT,)
                )
                native_filters = read_native_array(
                    Path(f"{prefix}.mel_filters.f32le.bin"),
                    "<f4",
                    (N_MELS, N_BINS),
                )
                record["stock_tables"] = {
                    "hann": compare_float(reference["hann"], native_hann),
                    "mel_filters": compare_float(
                        np.ascontiguousarray(mel_filters.T), native_filters
                    ),
                }
            variants[variant_id] = record
    return {"variants": variants}


def summarize(result: dict[str, Any]) -> dict[str, Any]:
    fixture_names = list(result["fixtures"])
    strict_candidate_ids: set[str] | None = None
    bounded_candidate_ids: set[str] | None = None
    stock_rejected = True
    stock_seen = False
    aggregate: dict[str, dict[str, Any]] = {}
    for fixture_name in fixture_names:
        variants = result["fixtures"][fixture_name]["variants"]
        strict_passing = {
            variant_id
            for variant_id, record in variants.items()
            if variant_id.split("/", 1)[1].startswith("kissfft-")
            and record["quantized_features"]["strict_byte_parity"]
        }
        bounded_passing = {
            variant_id
            for variant_id, record in variants.items()
            if variant_id.split("/", 1)[1].startswith("kissfft-")
            and record["quantized_features"]["maximum_absolute_difference"]
            <= BOUNDED_MAX_U16_DELTA
            and record["quantized_features"]["different_fraction"]
            <= BOUNDED_MAX_DIFFERENT_FRACTION
        }
        strict_candidate_ids = (
            strict_passing
            if strict_candidate_ids is None
            else strict_candidate_ids & strict_passing
        )
        bounded_candidate_ids = (
            bounded_passing
            if bounded_candidate_ids is None
            else bounded_candidate_ids & bounded_passing
        )
        for variant_id, record in variants.items():
            quantized = record["quantized_features"]
            if variant_id.endswith("/kaldi-stock"):
                stock_seen = True
                if quantized["strict_byte_parity"]:
                    stock_rejected = False
            if not variant_id.split("/", 1)[1].startswith("kissfft-"):
                continue
            item = aggregate.setdefault(
                variant_id,
                {
                    "different_elements": 0,
                    "fixtures": 0,
                    "maximum_absolute_difference": 0,
                    "maximum_feature_error": 0.0,
                },
            )
            item["different_elements"] += quantized["different_elements"]
            item["fixtures"] += 1
            item["maximum_absolute_difference"] = max(
                item["maximum_absolute_difference"],
                quantized["maximum_absolute_difference"],
            )
            item["maximum_feature_error"] = max(
                item["maximum_feature_error"],
                record["normalized_features"]["maximum_absolute_error"],
            )
    strict_candidates = sorted(strict_candidate_ids or [])
    bounded_candidates = sorted(bounded_candidate_ids or [])
    best_aggregate = None
    if aggregate:
        variant_id, metrics = min(
            aggregate.items(),
            key=lambda item: (
                item[1]["different_elements"],
                item[1]["maximum_absolute_difference"],
                item[1]["maximum_feature_error"],
                item[0],
            ),
        )
        best_aggregate = {"variant": variant_id, **metrics}
    return {
        "adapted_bounded_candidates_across_all_fixtures": bounded_candidates,
        "adapted_strict_candidates_across_all_fixtures": strict_candidates,
        "best_adapted_aggregate": best_aggregate,
        "bounded_policy": {
            "maximum_different_fraction": BOUNDED_MAX_DIFFERENT_FRACTION,
            "maximum_u16_delta": BOUNDED_MAX_U16_DELTA,
        },
        "kissfft_bounded_equivalence": bool(bounded_candidates),
        "kissfft_strict_byte_parity": bool(strict_candidates),
        "stock_path_rejected": stock_seen and stock_rejected,
        "target_retest_required": True,
    }


def resolve_fixture_names(
    profile: str, explicit_fixtures: list[str] | None
) -> list[str]:
    if explicit_fixtures is not None:
        return list(explicit_fixtures)
    return list(FIXTURE_PROFILES[profile])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--result",
        type=Path,
        help="result JSON path (default: OUTPUT_DIR/result.json)",
    )
    parser.add_argument(
        "--fixture-profile",
        choices=sorted(FIXTURE_PROFILES),
        default="v1",
        help="v1 keeps EN/JA; v2 selects EN/ZH/RU/KO/JA",
    )
    parser.add_argument(
        "--fixtures",
        nargs="+",
        choices=sorted(FIXTURES),
        default=None,
        help="explicit fixture override (default comes from --fixture-profile)",
    )
    parser.add_argument(
        "--build-flavors",
        nargs="+",
        choices=sorted(BUILD_FLAVORS),
        default=["portable", "android_fast_math"],
    )
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--kissfft-root", type=Path, default=DEFAULT_KISSFFT_ROOT)
    parser.add_argument("--kaldi-parent", type=Path, default=DEFAULT_KALDI_PARENT)
    parser.add_argument("--kissfft-archive", type=Path, default=DEFAULT_KISSFFT_ARCHIVE)
    parser.add_argument("--kaldi-archive", type=Path, default=DEFAULT_KALDI_ARCHIVE)
    parser.add_argument(
        "--require-adapted-u16-parity",
        action="store_true",
        help="exit nonzero unless one adapted build/mode matches every selected fixture",
    )
    parser.add_argument(
        "--require-adapted-bounded-parity",
        action="store_true",
        help=(
            "exit nonzero unless one adapted build/mode stays within the "
            "documented host max-delta and differing-fraction bounds"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.fixtures = resolve_fixture_names(args.fixture_profile, args.fixtures)
    output_dir = args.output_dir.resolve()
    result_path = (args.result or output_dir / "result.json").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    golden = load_golden_module()
    sources = verify_sources(
        args.kissfft_root.resolve(),
        args.kaldi_parent.resolve(),
        args.kissfft_archive.resolve(),
        args.kaldi_archive.resolve(),
    )
    mel_filters = golden.canonical_mel_filters()
    if array_sha256(mel_filters, "<f4") != EXPECTED_MEL_SHA256:
        raise AuditError("canonical mel matrix changed")

    build_records: dict[str, dict[str, Any]] = {}
    for flavor in args.build_flavors:
        build_records[flavor] = compile_binary(
            flavor=flavor,
            output_dir=output_dir,
            kissfft_root=args.kissfft_root.resolve(),
            kaldi_parent=args.kaldi_parent.resolve(),
        )

    result: dict[str, Any] = {
        "schema_version": 1,
        "scope": {
            "adb_invoked": False,
            "android_build_run": False,
            "apk_or_phone_mutated": False,
            "host_model_inference_run": False,
            "network_used": False,
        },
        "host": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch_thread_policy": torch_thread_policy(),
        },
        "fixture_profile": args.fixture_profile,
        "selected_fixtures": args.fixtures,
        "sources": sources,
        "builds": {},
        "fixtures": {},
    }
    for flavor, record in build_records.items():
        result["builds"][flavor] = {
            **record,
            "binary": str(record["binary"]),
        }

    shared_input_dir = output_dir / "inputs"
    hann_path = shared_input_dir / "hann.f32le.bin"
    mel_path = shared_input_dir / "mel_filters.f32le.bin"
    write_array(mel_path, mel_filters, "<f4")

    for fixture_name in args.fixtures:
        fixture = FIXTURES[fixture_name]
        samples, wav_record = read_fixture(fixture)
        reference = torch_reference(samples, mel_filters)
        if array_sha256(reference["hann"], "<f4") != EXPECTED_HANN_SHA256:
            raise AuditError("Torch periodic Hann bytes changed")
        float_sha256 = array_sha256(reference["features"], "<f4")
        if float_sha256 != fixture.float_sha256:
            raise AuditError(
                f"{fixture_name} float golden changed: expected "
                f"{fixture.float_sha256}, got {float_sha256}"
            )
        reference_quantized = quantize_qai(reference["features"], golden)
        quantized_sha256 = array_sha256(reference_quantized, "<u2")
        if quantized_sha256 != fixture.quantized_sha256:
            raise AuditError(
                f"{fixture_name} uint16 golden changed: expected "
                f"{fixture.quantized_sha256}, got {quantized_sha256}"
            )
        write_array(hann_path, reference["hann"], "<f4")
        pcm_path = shared_input_dir / f"{fixture_name}.pcm.f32le.bin"
        write_array(pcm_path, samples, "<f4")
        reference_dir = output_dir / "reference" / fixture_name
        for key in ("frame0", "power", "mel_energies", "features"):
            write_array(reference_dir / f"{key}.f32le.bin", reference[key], "<f4")
        write_array(reference_dir / "features.u16le.bin", reference_quantized, "<u2")
        fixture_record = audit_fixture(
            fixture=fixture,
            reference=reference,
            reference_quantized=reference_quantized,
            mel_filters=mel_filters,
            golden=golden,
            build_records=build_records,
            flavors=args.build_flavors,
            modes=args.modes,
            output_dir=output_dir,
            pcm_path=pcm_path,
            hann_path=hann_path,
            mel_path=mel_path,
        )
        result["fixtures"][fixture_name] = {
            "audio": wav_record,
            "reference": {
                "float_features_sha256": float_sha256,
                "hann_sha256": EXPECTED_HANN_SHA256,
                "mel_filters_sha256": EXPECTED_MEL_SHA256,
                "quantized_features_sha256": quantized_sha256,
            },
            **fixture_record,
        }

    result["verdict"] = summarize(result)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "result": str(result_path),
                "verdict": result["verdict"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if (
        args.require_adapted_u16_parity
        and not result["verdict"]["kissfft_strict_byte_parity"]
    ):
        return 1
    if (
        args.require_adapted_bounded_parity
        and not result["verdict"]["kissfft_bounded_equivalence"]
    ):
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuditError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
