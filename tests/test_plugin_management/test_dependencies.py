import pytest

from packaging.utils import canonicalize_name
from unittest.mock import patch

from pupil_labs.neon_player.plugin_management.dependencies import (
    _check_dependencies_for_plugin_script,
    get_installed_packages,
    is_dependency_installed
)


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


@patch("importlib.metadata.distributions", return_value=["huggingface_hub", "tf_keras"])
def test_get_installed_packages__names_are_normalized(mock_distributions):
    for package in get_installed_packages():
        assert package == canonicalize_name(package)


def test_is_dependency_installed__installed():
    installed_packages = {"numpy": "2.0"}
    assert is_dependency_installed("numpy", installed_packages)


def test_is_dependency_installed__not_installed():
    installed_packages = {"numpy": "2.0"}
    assert not is_dependency_installed("pandas", installed_packages)


def test_is_dependency_installed__installed_version_satisfies_specifier():
    installed_packages = {"numpy": "2.0"}
    assert is_dependency_installed("numpy>=2.0", installed_packages)


def test_is_dependency_installed__installed_version_does_not_satisfy_specifier():
    installed_packages = {"numpy": "2.0"}
    assert not is_dependency_installed("numpy<2.0", installed_packages)


@pytest.mark.parametrize("dependency_name", [
    "some-package",
    "some_package",
    "some.package",
    "Some_Package",
])
def test_is_dependency_installed__name_is_normalized(dependency_name):
    installed_packages = {"some-package": "1.0"}
    assert is_dependency_installed(dependency_name, installed_packages)


def test_check_dependencies_for_plugin_script__all_deps_installed():
    script = _plugin_script(["numpy", "pandas"])
    with patch(
        "pupil_labs.neon_player.plugin_management.dependencies.get_installed_packages",
        return_value={"numpy": "2.0", "pandas": "1.5"},
    ):
        assert not _check_dependencies_for_plugin_script(script, "test_plugin")


def test_check_dependencies_for_plugin_script__missing_dep():
    script = _plugin_script(["numpy", "some-missing-lib"])
    with patch(
        "pupil_labs.neon_player.plugin_management.dependencies.get_installed_packages",
        return_value={"numpy": "2.0"},
    ):
        result = _check_dependencies_for_plugin_script(script, "test_plugin")
        assert len(result) == 1
        assert "some-missing-lib" in result[0]


def test_check_dependencies_for_plugin_script__snake_dep_detected_as_installed():
    script = _plugin_script(["some_package>=1.0"])
    with patch(
        "pupil_labs.neon_player.plugin_management.dependencies.get_installed_packages",
        return_value={"some-package": "1.0"},
    ):
        result = _check_dependencies_for_plugin_script(script, "test_plugin")
        assert result == [], (
            "snake_case dep should be recognized as installed when the "
            "package is present under its canonical (hyphenated) name"
        )
