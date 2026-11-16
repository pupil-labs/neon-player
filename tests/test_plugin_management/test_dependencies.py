from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from pupil_labs.neon_player.plugin_management import dependencies
from pupil_labs.neon_player.plugin_management.dependencies import (
    _check_dependencies_for_plugin_script,
    check_dependency_constraints,
    get_installed_packages,
    is_dependency_installed,
)

ConstraintEnvironment = tuple[Path, Path]


@pytest.fixture
def constraint_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> ConstraintEnvironment:
    constraints = tmp_path / "constraints.txt"
    constraints.write_text("", encoding="utf-8")
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    monkeypatch.setattr(dependencies, "CONSTRAINTS_FILE", constraints)
    return constraints, site_packages


def _write_distribution(site_packages: Path, name: str, version: str) -> None:
    metadata = site_packages / f"{name}-{version}.dist-info" / "METADATA"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(f"Name: {name}\nVersion: {version}\n", encoding="utf-8")


def test_check_dependency_constraints__repairs_only_local_violations(
    constraint_environment: ConstraintEnvironment,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constraints, site_packages = constraint_environment
    constraints.write_text(
        "# Bundled requirements\n\n"
        "Some_Package==2.0\n"
        "matching>=1.0\nmissing==1.0\n"
        "outside==2.0\nother-package<2.0\n",
        encoding="utf-8",
    )
    _write_distribution(site_packages, "some.package", "1.0")
    _write_distribution(site_packages, "matching", "1.0")
    _write_distribution(site_packages, "other-package", "3.0")
    _write_distribution(site_packages, "unconstrained", "1.0")
    outside = tmp_path / "outside"
    _write_distribution(outside, "outside", "1.0")
    monkeypatch.syspath_prepend(str(outside))

    assert check_dependency_constraints(str(site_packages)) == [
        "Some_Package==2.0",
        "other-package<2.0",
    ]


@pytest.mark.parametrize("installed", [False, True])
def test_check_dependency_constraints__no_repairs_needed(
    constraint_environment: ConstraintEnvironment, installed: bool
) -> None:
    constraints, site_packages = constraint_environment
    constraints.write_text("matching>=1.0\nmissing==1.0\n", encoding="utf-8")
    if installed:
        _write_distribution(site_packages, "matching", "1.0")

    assert check_dependency_constraints(str(site_packages)) == []


def test_check_dependency_constraints__evaluates_markers(
    constraint_environment: ConstraintEnvironment,
) -> None:
    constraints, site_packages = constraint_environment
    constraints.write_text(
        'ignored==2.0 ; python_version < "0"\nactive==2.0 ; python_version >= "3"\n',
        encoding="utf-8",
    )
    _write_distribution(site_packages, "ignored", "1.0")
    _write_distribution(site_packages, "active", "1.0")

    assert check_dependency_constraints(str(site_packages)) == [
        'active==2.0; python_version >= "3"'
    ]


@pytest.mark.parametrize("content", [None, "not a valid requirement!!!"])
def test_check_dependency_constraints__propagates_constraint_errors(
    constraint_environment: ConstraintEnvironment, content: str | None
) -> None:
    constraints, site_packages = constraint_environment
    if content is None:
        constraints.unlink()
        error = FileNotFoundError
    else:
        constraints.write_text(content, encoding="utf-8")
        error = dependencies.InvalidRequirement

    with pytest.raises(error):
        check_dependency_constraints(str(site_packages))


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


def test_get_installed_packages__names_are_normalized() -> None:
    fake_package = SimpleNamespace(
        metadata={"name": "huggingface_hub"},
        version="1.0",
    )

    with patch(
        "importlib.metadata.distributions",
        return_value=[fake_package],
    ):
        assert get_installed_packages() == {"huggingface-hub": "1.0"}


def test_is_dependency_installed__installed() -> None:
    installed_packages = {"numpy": "2.0"}
    assert is_dependency_installed("numpy", installed_packages)


def test_is_dependency_installed__not_installed() -> None:
    installed_packages = {"numpy": "2.0"}
    assert not is_dependency_installed("pandas", installed_packages)


def test_is_dependency_installed__installed_version_satisfies_specifier() -> None:
    installed_packages = {"numpy": "2.0"}
    assert is_dependency_installed("numpy>=2.0", installed_packages)


def test_is_dependency_installed__installed_version_does_not_satisfy_specifier() -> (
    None
):
    installed_packages = {"numpy": "2.0"}
    assert not is_dependency_installed("numpy<2.0", installed_packages)


@pytest.mark.parametrize(
    "dependency_name",
    [
        "some-package",
        "some_package",
        "some.package",
        "Some_Package",
    ],
)
def test_is_dependency_installed__name_is_normalized(dependency_name: str) -> None:
    installed_packages = {"some-package": "1.0"}
    assert is_dependency_installed(dependency_name, installed_packages)


def test_check_dependencies_for_plugin_script__all_deps_installed() -> None:
    script = _plugin_script(["numpy", "pandas"])
    with patch(
        "pupil_labs.neon_player.plugin_management.dependencies.get_installed_packages",
        return_value={"numpy": "2.0", "pandas": "1.5"},
    ):
        assert not _check_dependencies_for_plugin_script(script, "test_plugin")


def test_check_dependencies_for_plugin_script__missing_dep() -> None:
    script = _plugin_script(["numpy", "some-missing-lib"])
    with patch(
        "pupil_labs.neon_player.plugin_management.dependencies.get_installed_packages",
        return_value={"numpy": "2.0"},
    ):
        result = _check_dependencies_for_plugin_script(script, "test_plugin")
        assert len(result) == 1
        assert "some-missing-lib" in result[0]


def test_check_dependencies_for_plugin_script__snake_dep_detected_as_installed() -> (
    None
):
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
