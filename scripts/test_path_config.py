#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"


class PathConfigTest(unittest.TestCase):
    def run_python(
        self, source: str, *, environment: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(SCRIPTS)
        if environment:
            env.update(environment)
        return subprocess.run(
            [sys.executable, "-c", source],
            cwd=REPO_ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )

    def test_checkout_defaults(self) -> None:
        env = dict(os.environ)
        for name in ("LFM_DEMO_WORK_ROOT", "HF_HOME", "LFM_DEMO_MODEL_SNAPSHOT"):
            env.pop(name, None)
        result = subprocess.run(
            [sys.executable, str(SCRIPTS / "path_config.py")],
            cwd=REPO_ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        paths = json.loads(result.stdout)
        self.assertEqual(Path(paths["repo_root"]), REPO_ROOT)
        self.assertEqual(Path(paths["work_root"]), REPO_ROOT / "work")
        self.assertEqual(
            Path(paths["vendor_root"]), REPO_ROOT / "vendor" / "liquid-audio"
        )

    def test_environment_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.run_python(
                "import json; from path_config import path_record; "
                "print(json.dumps(path_record()))",
                environment={
                    "LFM_DEMO_WORK_ROOT": str(root / "mutable"),
                    "HF_HOME": str(root / "hf"),
                    "LFM_DEMO_MODEL_SNAPSHOT": str(root / "snapshot"),
                },
            )
            paths = json.loads(result.stdout)
            self.assertEqual(Path(paths["work_root"]), (root / "mutable").resolve())
            self.assertEqual(Path(paths["hf_home"]), (root / "hf").resolve())
            self.assertEqual(
                Path(paths["model_snapshot"]), (root / "snapshot").resolve()
            )

    def test_check_environment_paths_only_needs_no_model_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = subprocess.run(
                [sys.executable, str(SCRIPTS / "check_environment.py"), "--paths-only"],
                cwd=REPO_ROOT,
                env={**os.environ, "LFM_DEMO_WORK_ROOT": str(root / "work")},
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(result.stdout)
            self.assertTrue(payload["vendor_repo_present"])
            self.assertEqual(Path(payload["work_root"]), (root / "work").resolve())

    def test_colab_bundle_fails_closed(self) -> None:
        env = dict(os.environ)
        env.pop("COLAB_CLI_BUNDLE", None)
        result = subprocess.run(
            ["bash", str(SCRIPTS / "run_colab_auth_refresh.sh"), "--dry-run"],
            cwd=REPO_ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("Set COLAB_CLI_BUNDLE", result.stderr)


if __name__ == "__main__":
    unittest.main()
