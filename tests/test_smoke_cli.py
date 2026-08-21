from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_smoke_cli_writes_manifest_and_protects_outputs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = tmp_path / "smoke"

    env = os.environ.copy()
    src_path = str(repo_root / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")

    command = [
        sys.executable,
        "-m",
        "abnuts.experiments.smoke",
        "--config",
        str(repo_root / "configs" / "smoke.yaml"),
        "--out",
        str(out_dir),
    ]
    completed = subprocess.run(
        command,
        check=True,
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
    )

    manifest_path = out_dir / "manifest.json"
    assert manifest_path.exists()
    assert "Wrote smoke manifest" in completed.stdout

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "ok"
    assert manifest["backend"] == "cpu"
    assert manifest["config_hash"]
    assert manifest["config_sha256"]
    assert manifest["created_at_utc"]
    assert manifest["python_version"]
    assert manifest["platform"]
    assert manifest["num_chains"] == 8
    assert manifest["dimension"] == 4
    assert manifest["placeholder"]["num_iterations"] == 6

    preserved = subprocess.run(
        command,
        check=True,
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
    )
    assert "Preserved existing smoke manifest" in preserved.stdout
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest

    changed_command = [*command, "--dimension", "5"]
    blocked = subprocess.run(
        changed_command,
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
    )
    assert blocked.returncode == 2
    assert "--overwrite" in blocked.stderr
