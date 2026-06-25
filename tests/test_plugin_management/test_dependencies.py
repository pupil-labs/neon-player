from pathlib import Path
from unittest.mock import patch

import pytest

from pupil_labs.neon_player.plugin_management.dependencies import (
    _check_dependencies_for_plugin_script,
    is_dependency_installed,
    get_installed_packages,
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


def test_is_dependency_installed__installed():
    # Canonical form stored in the dict (as get_installed_packages() now does)
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


# --- Regression tests: snake_case / hyphen name normalization ---


def test_is_dependency_installed__snake_dep_hyphen_installed():
    """Dependency declared with underscore, metadata stored with hyphen."""
    installed_packages = {"some-package": "1.0"}
    assert is_dependency_installed("some_package", installed_packages)


def test_is_dependency_installed__hyphen_dep_snake_installed():
    """Dependency declared with hyphen, metadata stored with underscore."""
    installed_packages = {"some_package": "1.0"}
    assert is_dependency_installed("some-package", installed_packages)


def test_is_dependency_installed__both_canonical():
    """Both sides already canonical (all-lowercase, hyphens); no regression."""
    installed_packages = {"some-package": "1.0"}
    assert is_dependency_installed("some-package", installed_packages)


def test_is_dependency_installed__snake_dep_snake_installed():
    """Both sides use underscores; should still match after canonicalization."""
    installed_packages = {"some_package": "1.0"}
    assert is_dependency_installed("some_package", installed_packages)


def test_is_dependency_installed__mixed_case_dep():
    """Dependency name with mixed case; canonical form is lowercase."""
    installed_packages = {"some-package": "1.0"}
    assert is_dependency_installed("Some_Package", installed_packages)


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


# --- Regression: snake_case deps in full plugin script flow ---


def test_check_dependencies_for_plugin_script__snake_dep_detected_as_installed():
    """Regression: a dep declared with underscore must not trigger re-install
    when the package is already present under its hyphenated canonical name."""
    script = _plugin_script(["some_package>=1.0"])
    with patch(
        "pupil_labs.neon_player.plugin_management.dependencies.get_installed_packages",
        return_value={"some-package": "1.0"},  # canonical form, as stored by pip/uv
    ):
        result = _check_dependencies_for_plugin_script(script, "test_plugin")
        assert result == [], (
            "snake_case dep should be recognized as installed when the "
            "package is present under its canonical (hyphenated) name"
        )


def test_check_dependencies_for_plugin_script__hyphen_dep_detected_as_installed():
    """Regression: a dep declared with hyphen must not trigger re-install
    when the package is already present under its underscored metadata name."""
    script = _plugin_script(["some-package>=1.0"])
    with patch(
        "pupil_labs.neon_player.plugin_management.dependencies.get_installed_packages",
        return_value={"some_package": "1.0"},
    ):
        result = _check_dependencies_for_plugin_script(script, "test_plugin")
        assert result == [], (
            "hyphenated dep should be recognized as installed when the "
            "package metadata uses underscores"
        )
