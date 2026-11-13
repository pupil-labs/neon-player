import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType

import pytest

from pupil_labs.neon_player.plugin_management.path import PassivePathExtensionHook


@pytest.fixture
def loaded_package(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    package = ModuleType("_passive_path_test")
    package.__path__ = ["existing-package-path"]
    package.__spec__ = ModuleSpec(package.__name__, loader=None, is_package=True)
    package.__spec__.submodule_search_locations = ["existing-spec-path"]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    return package


@pytest.mark.parametrize("parent", ["_passive_path_test", "_passive_path_test.nested"])
def test_find_spec__extends_loaded_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, parent: str
) -> None:
    package = ModuleType(parent)
    package.__path__ = ["existing-package-path"]
    package.__spec__ = ModuleSpec(parent, loader=None, is_package=True)
    package.__spec__.submodule_search_locations = ["existing-spec-path"]
    monkeypatch.setitem(sys.modules, parent, package)
    extra_path = tmp_path.joinpath(*parent.split("."))
    extra_path.mkdir(parents=True)
    hook = PassivePathExtensionHook(str(tmp_path))

    assert hook.find_spec(f"{parent}.child", package.__path__) is None

    assert package.__path__ == ["existing-package-path", str(extra_path)]
    assert package.__spec__.submodule_search_locations == [
        "existing-spec-path",
        str(extra_path),
    ]


@pytest.mark.parametrize("shared_locations", [False, True])
def test_find_spec__does_not_duplicate_existing_paths(
    tmp_path: Path, loaded_package: ModuleType, shared_locations: bool
) -> None:
    extra_path = tmp_path / loaded_package.__name__
    extra_path.mkdir()
    loaded_package.__path__.append(str(extra_path))
    loaded_package.__spec__.submodule_search_locations = (
        loaded_package.__path__ if shared_locations else [str(extra_path)]
    )
    hook = PassivePathExtensionHook(str(tmp_path))

    assert hook.find_spec(f"{loaded_package.__name__}.child", None) is None

    assert loaded_package.__path__ == ["existing-package-path", str(extra_path)]
    assert (
        loaded_package.__spec__.submodule_search_locations.count(str(extra_path)) == 1
    )


def test_find_spec__only_processes_each_parent_once(
    tmp_path: Path, loaded_package: ModuleType
) -> None:
    extra_path = tmp_path / loaded_package.__name__
    extra_path.mkdir()
    hook = PassivePathExtensionHook(str(tmp_path))
    hook.find_spec(f"{loaded_package.__name__}.first", None)
    loaded_package.__path__.remove(str(extra_path))
    loaded_package.__spec__.submodule_search_locations.remove(str(extra_path))

    assert hook.find_spec(f"{loaded_package.__name__}.second", None) is None

    assert loaded_package.__path__ == ["existing-package-path"]
    assert loaded_package.__spec__.submodule_search_locations == ["existing-spec-path"]


def test_find_spec__ignores_top_level_imports(
    tmp_path: Path, loaded_package: ModuleType
) -> None:
    (tmp_path / loaded_package.__name__).mkdir()
    hook = PassivePathExtensionHook(str(tmp_path))

    assert hook.find_spec(loaded_package.__name__, None) is None

    assert loaded_package.__path__ == ["existing-package-path"]
    assert loaded_package.__spec__.submodule_search_locations == ["existing-spec-path"]


@pytest.mark.parametrize("root", ["PySide6", "shiboken6"])
@pytest.mark.parametrize("suffix", ["", ".nested"])
def test_find_spec__never_extends_unsafe_packages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, root: str, suffix: str
) -> None:
    parent = root + suffix
    package = ModuleType(parent)
    package.__path__ = ["existing-package-path"]
    package.__spec__ = ModuleSpec(parent, loader=None, is_package=True)
    package.__spec__.submodule_search_locations = ["existing-spec-path"]
    monkeypatch.setitem(sys.modules, parent, package)
    tmp_path.joinpath(*parent.split(".")).mkdir(parents=True)
    hook = PassivePathExtensionHook(str(tmp_path))

    assert hook.find_spec(f"{parent}.child", None) is None

    assert package.__path__ == ["existing-package-path"]
    assert package.__spec__.submodule_search_locations == ["existing-spec-path"]
