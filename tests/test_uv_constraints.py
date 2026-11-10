from __future__ import annotations

import importlib.metadata
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from pupil_labs.neon_player.plugin_management.dependencies import (
    _write_host_constraints,
)


def _uv() -> str:
    exe = shutil.which("uv")
    if not exe:
        pytest.skip("uv is not installed")
    return exe


def _installed_version(target: Path, dist_name: str) -> str:
    for dist in importlib.metadata.distributions(path=[str(target)]):
        if dist.metadata["Name"].lower() == dist_name.lower():
            return dist.version
    raise AssertionError(f"{dist_name} not installed in {target}")


def test_constraint_forces_transitive_dependency_to_bundled_version(
    repo_root: Path, tmp_path: Path
) -> None:
    wheels = repo_root / "tests" / "fixtures" / "wheels"
    target = tmp_path / "target"
    constraints = tmp_path / "constraints.txt"
    constraints.write_text("bundledep==1.0.0\n", encoding="utf-8")

    cmd = [
        _uv(),
        "pip",
        "install",
        "--python",
        sys.executable,
        "--target",
        str(target),
        "--no-index",
        "--find-links",
        str(wheels),
        "--constraint",
        str(constraints),
        "pluginpkg==1.0.0",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)  # noqa: S603
    assert _installed_version(target, "pluginpkg") == "1.0.0"
    assert _installed_version(target, "bundledep") == "1.0.0"


def test_without_constraint_uv_selects_newer_compatible_dependency(
    repo_root: Path, tmp_path: Path
) -> None:
    wheels = repo_root / "tests" / "fixtures" / "wheels"
    target = tmp_path / "target"
    cmd = [
        _uv(),
        "pip",
        "install",
        "--python",
        sys.executable,
        "--target",
        str(target),
        "--no-index",
        "--find-links",
        str(wheels),
        "pluginpkg==1.0.0",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)  # noqa: S603
    assert _installed_version(target, "bundledep") == "2.0.0"


def test_write_host_constraints_falls_back_to_loaded_modules_when_frozen(
    tmp_path: Path,
) -> None:
    constraints_file = tmp_path / "neon_player_constraints.txt"
    fake_modules = {
        "numpy": SimpleNamespace(__version__="2.3.5"),
        "scipy": SimpleNamespace(__version__="1.17.1"),
        "qt_property_widgets": SimpleNamespace(__version__="0.10.0"),
    }
    with (
        patch("importlib.metadata.distributions", return_value=[]),
        patch.dict(sys.modules, fake_modules),
    ):
        _write_host_constraints(constraints_file)

    lines = set(constraints_file.read_text(encoding="utf-8").splitlines())
    assert "numpy==2.3.5" in lines
    assert "scipy==1.17.1" in lines
    assert "qt-property-widgets==0.10.0" in lines
