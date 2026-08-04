#!/usr/bin/env python3
"""Stage and run the private SenseVoice QNN CLI bundle on an exact SM8850.

This command writes raw evidence but never marks the production activation gate
as passed by itself. A reviewer must add transcript parity against independent
EN and ZH CPU goldens before creating a `status: pass` smoke result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import select_android_asr_backend as backend_gate


REMOTE_ROOT = "/data/local/tmp/lmf-sensevoice-qnn-240"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def adb(adb_path: str, serial: str, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [adb_path, "-s", serial, *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


def one_device(adb_path: str) -> str:
    result = subprocess.run(
        [adb_path, "devices"], check=True, text=True, stdout=subprocess.PIPE
    )
    devices = [
        line.split()[0]
        for line in result.stdout.splitlines()[1:]
        if line.strip().endswith("device")
    ]
    if len(devices) != 1:
        raise RuntimeError(f"expected one connected phone, found {len(devices)}")
    return devices[0]


def push(adb_path: str, serial: str, source: Path, destination: str) -> None:
    result = adb(adb_path, serial, "push", str(source), destination)
    if result.returncode:
        raise RuntimeError(result.stderr)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-manifest", type=Path, required=True)
    parser.add_argument("--runner-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--language", choices=("auto", "zh", "en", "ja", "ko", "yue"), default="auto")
    parser.add_argument("--wav", type=Path, action="append", default=[])
    args = parser.parse_args()

    try:
        bundle = json.loads(args.bundle_manifest.read_text(encoding="utf-8"))
        assets = json.loads(backend_gate.ASSETS_PATH.read_text(encoding="utf-8"))
        bundle_ok, bundle_reasons = backend_gate.validate_bundle(bundle, assets)
        if not bundle_ok:
            raise RuntimeError("invalid bundle: " + ", ".join(bundle_reasons))
        runner = args.runner_dir / "sherpa-onnx-offline"
        if not runner.is_file():
            raise RuntimeError(f"missing runner: {runner}")

        serial = one_device(args.adb)
        facts = backend_gate.probe_adb(args.adb)
        device_ok, device_reasons = backend_gate.validate_device(facts)
        if not device_ok:
            raise RuntimeError("device gate failed: " + ", ".join(device_reasons))

        paths = bundle["paths"]
        runtime = Path(paths["qnn_runtime_root"])
        model = Path(paths["sensevoice_model_root"])
        wavs = args.wav or [model / "test_wavs" / "en.wav", model / "test_wavs" / "zh.wav"]
        for wav in wavs:
            if not wav.is_file():
                raise RuntimeError(f"missing WAV: {wav}")

        adb(args.adb, serial, "shell", "rm", "-rf", REMOTE_ROOT)
        adb(args.adb, serial, "shell", "mkdir", "-p", f"{REMOTE_ROOT}/model/test_wavs")
        push(args.adb, serial, runner, f"{REMOTE_ROOT}/sherpa-onnx-offline")
        for library in sorted(args.runner_dir.glob("*.so")):
            push(args.adb, serial, library, f"{REMOTE_ROOT}/{library.name}")
        for name in sorted(backend_gate.RUNTIME_FILES):
            push(args.adb, serial, runtime / name, f"{REMOTE_ROOT}/{name}")
        push(args.adb, serial, model / "model.bin", f"{REMOTE_ROOT}/model/model.bin")
        push(args.adb, serial, model / "tokens.txt", f"{REMOTE_ROOT}/model/tokens.txt")
        for wav in wavs:
            push(args.adb, serial, wav, f"{REMOTE_ROOT}/model/test_wavs/{wav.name}")
        adb(args.adb, serial, "shell", "chmod", "755", f"{REMOTE_ROOT}/sherpa-onnx-offline")

        runs = []
        for wav in wavs:
            remote_wav = f"./model/test_wavs/{wav.name}"
            command = (
                f"cd {REMOTE_ROOT} && "
                "export LD_LIBRARY_PATH=\"$PWD\" && "
                "export ADSP_LIBRARY_PATH=\"$PWD;${ADSP_LIBRARY_PATH:-}\" && "
                "./sherpa-onnx-offline --provider=qnn "
                "--tokens=./model/tokens.txt "
                f"--sense-voice-language={args.language} "
                "--sense-voice-use-itn=true "
                "--sense-voice.qnn-backend-lib=./libQnnHtp.so "
                "--sense-voice.qnn-system-lib=./libQnnSystem.so "
                "--sense-voice.qnn-context-binary=./model/model.bin "
                f"{remote_wav}"
            )
            start = time.monotonic()
            result = adb(args.adb, serial, "shell", command, check=False)
            runs.append(
                {
                    "wav": wav.name,
                    "language": args.language,
                    "returncode": result.returncode,
                    "wall_seconds": time.monotonic() - start,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            )

        evidence = {
            "schema_version": 1,
            "status": "raw_pass" if all(run["returncode"] == 0 for run in runs) else "fail",
            "production_activation": "blocked_pending_independent_en_zh_transcript_parity",
            "provider": "qnn",
            "soc": "SM8850",
            "htp_arch": "v81",
            "qnn_version": assets["target"]["qnn_version"],
            "model_bin_sha256": sha256(model / "model.bin"),
            "device": facts,
            "runs": runs,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(evidence, indent=2, sort_keys=True))
        return 0 if evidence["status"] == "raw_pass" else 1
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
