from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _clear_demo_pkg() -> None:
    for name in list(sys.modules):
        if name == "demo_pkg" or name.startswith("demo_pkg."):
            sys.modules.pop(name, None)


def _extend_loaded_package(module, plugin_site: Path) -> bool:
    """Reference implementation of the Neon Player path-extension strategy."""
    package_dir = plugin_site.joinpath(*module.__name__.split("."))
    if not package_dir.is_dir():
        return False

    path = getattr(module, "__path__", None)
    spec = getattr(module, "__spec__", None)
    spec_locations = getattr(spec, "submodule_search_locations", None)
    if path is None or spec is None or spec_locations is None:
        return False

    value = str(package_dir)
    if value not in path:
        path.append(value)
    if value not in spec_locations:
        spec_locations.append(value)
    return True


def test_missing_python_submodule_becomes_importable(repo_root: Path, native_extension: Path) -> None:
    del native_extension  # ensure native fixture is compiled for the companion test
    app = repo_root / "tests" / "fixtures" / "app"
    plugin_site = repo_root / "tests" / "fixtures" / "plugin_site"
    _clear_demo_pkg()
    sys.path.insert(0, str(app))
    try:
        demo_pkg = importlib.import_module("demo_pkg")
        assert demo_pkg.ORIGIN == "bundled-parent"

        try:
            importlib.import_module("demo_pkg.missing")
        except ModuleNotFoundError:
            pass
        else:
            raise AssertionError("submodule unexpectedly imported before path extension")

        assert _extend_loaded_package(demo_pkg, plugin_site)
        submodule = importlib.import_module("demo_pkg.missing")
        assert submodule.VALUE == "plugin-pure-submodule"
    finally:
        sys.path.remove(str(app))
        _clear_demo_pkg()


def test_native_extension_becomes_importable(repo_root: Path, native_extension: Path) -> None:
    assert native_extension.exists()
    app = repo_root / "tests" / "fixtures" / "app"
    plugin_site = repo_root / "tests" / "fixtures" / "plugin_site"
    _clear_demo_pkg()
    sys.path.insert(0, str(app))
    try:
        demo_pkg = importlib.import_module("demo_pkg")
        try:
            importlib.import_module("demo_pkg.native_missing")
        except ModuleNotFoundError:
            pass
        else:
            raise AssertionError("native submodule unexpectedly imported before path extension")

        assert _extend_loaded_package(demo_pkg, plugin_site)
        native = importlib.import_module("demo_pkg.native_missing")
        assert native.value() == "plugin-native-submodule"
    finally:
        sys.path.remove(str(app))
        _clear_demo_pkg()


def test_extension_is_idempotent(repo_root: Path) -> None:
    app = repo_root / "tests" / "fixtures" / "app"
    plugin_site = repo_root / "tests" / "fixtures" / "plugin_site"
    _clear_demo_pkg()
    sys.path.insert(0, str(app))
    try:
        demo_pkg = importlib.import_module("demo_pkg")
        assert _extend_loaded_package(demo_pkg, plugin_site)
        assert _extend_loaded_package(demo_pkg, plugin_site)
        package_dir = str(plugin_site / "demo_pkg")
        assert list(demo_pkg.__path__).count(package_dir) == 1
        assert list(demo_pkg.__spec__.submodule_search_locations).count(package_dir) == 1
    finally:
        sys.path.remove(str(app))
        _clear_demo_pkg()
