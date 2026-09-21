from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def _runner() -> list[str] | None:
    executable = os.environ.get("NEON_PLAYER_EXECUTABLE")
    if executable:
        return [executable]

    root = os.environ.get("NEON_PLAYER_ROOT")
    if root:
        root_path = Path(root)
        python = os.environ.get("NEON_PLAYER_PYTHON") or sys.executable
        return [python, "-m", "pupil_labs.neon_player"]
    return None


def test_neon_player_help_smoke() -> None:
    runner = _runner()
    if runner is None:
        pytest.skip("Set NEON_PLAYER_EXECUTABLE or NEON_PLAYER_ROOT to run real Neon Player integration tests")

    env = os.environ.copy()
    root = os.environ.get("NEON_PLAYER_ROOT")
    if root:
        src = str(Path(root) / "src")
        env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run([*runner, "--help"], env=env, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr
    combined = result.stdout + result.stderr
    assert "Neon" in combined or "recording" in combined.lower()


@pytest.mark.integration
def test_real_plugin_import_probe(tmp_path: Path, repo_root: Path) -> None:
    runner = _runner()
    recording = os.environ.get("NEON_PLAYER_RECORDING")
    if runner is None or not recording:
        pytest.skip("Set runner plus NEON_PLAYER_RECORDING for end-to-end plugin test")

    fake_home = tmp_path / "home"
    plugins = fake_home / "Pupil Labs" / "Neon Player" / "plugins"
    plugins.mkdir(parents=True)
    shutil.copy(repo_root / "integration_plugin" / "dependency_probe.py", plugins / "dependency_probe.py")
    sentinel = tmp_path / "probe-ok.txt"

    env = os.environ.copy()
    env["HOME"] = str(fake_home)
    env["NEON_PLAYER_DEPENDENCY_PROBE_SENTINEL"] = str(sentinel)
    root = os.environ.get("NEON_PLAYER_ROOT")
    if root:
        src = str(Path(root) / "src")
        env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [*runner, recording, "--job", "DependencyProbe.probe"]
    result = subprocess.run(cmd, env=env, text=True, capture_output=True, timeout=180)
    assert result.returncode == 0, f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    assert sentinel.exists(), f"plugin probe did not create sentinel\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    text = sentinel.read_text(encoding="utf-8")
    assert "PIL.ImageDraw" in text
