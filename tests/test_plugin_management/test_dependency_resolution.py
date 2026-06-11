
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_SRC = Path(__file__).parents[2] / "src" / "pupil_labs" / "neon_player" / "plugin_management"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_pep723 = _load_module("_pep723", _SRC / "pep723.py")
parse_pep723_dependencies = _pep723.parse_pep723_dependencies

_pkg_mgmt = _load_module("_pkg_mgmt", _SRC / "__init__.py")
check_dependencies_for_plugin = _pkg_mgmt.check_dependencies_for_plugin
get_installed_packages = _pkg_mgmt.get_installed_packages

def test_parse_valid_pep723_block():
    script = """\
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "librosa",
# ]
# ///
"""
    result = parse_pep723_dependencies(script)
    assert result is not None
    assert result.requires_python == ">=3.12"
    assert "numpy" in result.dependencies
    assert "librosa" in result.dependencies


def test_parse_pep723_no_block_returns_none():
    script = "import os\nprint('hello')\n"
    assert parse_pep723_dependencies(script) is None


def test_parse_pep723_empty_dependencies():
    script = """\
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
    result = parse_pep723_dependencies(script)
    assert result is not None
    assert result.dependencies == []


def test_parse_pep723_malformed_toml_returns_none():
    script = """\
# /// script
# this is not valid toml ][
# ///
"""
    assert parse_pep723_dependencies(script) is None


def test_parse_pep723_with_platform_markers():
    script = """\
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "mlx-whisper; sys_platform == 'darwin'",
#     "openai-whisper; sys_platform != 'darwin'",
# ]
# ///
"""
    result = parse_pep723_dependencies(script)
    assert result is not None
    assert len(result.dependencies) == 2
    assert any("mlx-whisper" in d for d in result.dependencies)
    assert any("openai-whisper" in d for d in result.dependencies)

def _write_plugin(tmp_path: Path, content: str) -> Path:
    plugin = tmp_path / "my_plugin.py"
    plugin.write_text(content)
    return plugin


def _plugin_script(deps: list[str]) -> str:
    dep_lines = "\n".join(f'#     "{d}",' for d in deps)
    return f"""\
# /// script
# requires-python = ">=3.12"
# dependencies = [
{dep_lines}
# ]
# ///
import os
"""


def test_no_pep723_block_returns_none(tmp_path: Path):
    plugin = _write_plugin(tmp_path, "import os\n")
    assert check_dependencies_for_plugin(plugin) is None


def test_all_deps_installed_returns_none(tmp_path: Path):
    plugin = _write_plugin(tmp_path, _plugin_script(["numpy"]))
    with patch(
        "_pkg_mgmt.get_installed_packages",
        return_value={"numpy"},
    ):
        assert check_dependencies_for_plugin(plugin) is None


def test_missing_dep_is_returned(tmp_path: Path):
    plugin = _write_plugin(tmp_path, _plugin_script(["numpy", "some-missing-lib"]))
    with patch(
        "_pkg_mgmt.get_installed_packages",
        return_value={"numpy"},
    ):
        result = check_dependencies_for_plugin(plugin)
        assert result is not None
        assert any("some-missing-lib" in d for d in result)
        assert not any("numpy" in d for d in result)


def test_platform_marker_darwin_skipped_on_non_darwin(tmp_path: Path):
    plugin = _write_plugin(
        tmp_path,
        _plugin_script(["mlx-whisper; sys_platform == 'darwin'"]),
    )
    with (
        patch("sys.platform", "linux"),
        patch(
            "_pkg_mgmt.get_installed_packages",
            return_value=set(),
        ),
    ):
        assert check_dependencies_for_plugin(plugin) is None


def test_platform_marker_darwin_required_on_darwin(tmp_path: Path):
    plugin = _write_plugin(
        tmp_path,
        _plugin_script(["mlx-whisper; sys_platform == 'darwin'"]),
    )
    with (
        patch("sys.platform", "darwin"),
        patch(
            "_pkg_mgmt.get_installed_packages",
            return_value=set(),
        ),
    ):
        result = check_dependencies_for_plugin(plugin)
        assert result is not None
        assert any("mlx-whisper" in d for d in result)


def test_non_darwin_marker_skipped_on_darwin(tmp_path: Path):
    plugin = _write_plugin(
        tmp_path,
        _plugin_script(["openai-whisper; sys_platform != 'darwin'"]),
    )
    with (
        patch("sys.platform", "darwin"),
        patch(
            "_pkg_mgmt.get_installed_packages",
            return_value=set(),
        ),
    ):
        assert check_dependencies_for_plugin(plugin) is None


@pytest.mark.parametrize("bad_dep", [
    "!!!invalid!!!",
    "numpy >",
    "requests ==",
    "some-pkg; unknown_env_marker == 'foo'",
])
def test_malformed_dep_string_raises_value_error(tmp_path: Path, bad_dep: str):
    plugin = _write_plugin(tmp_path, _plugin_script([bad_dep]))
    with (
        patch(
            "_pkg_mgmt.get_installed_packages",
            return_value=set(),
        ),
        pytest.raises(ValueError),
    ):
        check_dependencies_for_plugin(plugin)


def test_python_version_marker_skipped(tmp_path: Path):
    plugin = _write_plugin(
        tmp_path,
        _plugin_script(["typing_extensions; python_version < '3.0'"]),
    )
    with patch("_pkg_mgmt.get_installed_packages", return_value=set()):
        assert check_dependencies_for_plugin(plugin) is None


def test_python_version_marker_required(tmp_path: Path):
    plugin = _write_plugin(
        tmp_path,
        _plugin_script(["typing_extensions; python_version >= '3.0'"]),
    )
    with patch("_pkg_mgmt.get_installed_packages", return_value=set()):
        result = check_dependencies_for_plugin(plugin)
        assert result is not None
        assert any("typing_extensions" in d for d in result)


def test_nonexistent_plugin_file_returns_none(tmp_path: Path):
    missing = tmp_path / "ghost_plugin.py"
    assert check_dependencies_for_plugin(missing) is None


def test_plugin_directory_reads_init(tmp_path: Path):
    pkg = tmp_path / "my_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(_plugin_script(["requests"]))
    with patch(
        "_pkg_mgmt.get_installed_packages",
        return_value=set(),
    ):
        result = check_dependencies_for_plugin(pkg)
        assert result is not None
        assert any("requests" in d for d in result)
