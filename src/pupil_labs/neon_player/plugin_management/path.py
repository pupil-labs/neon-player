import importlib.abc
import importlib.machinery
import logging
import os
import sys
import types
import typing as T

_UNSAFE_EXTENSION_ROOTS = frozenset({"PySide6", "shiboken6"})
"""These modules are deliberatily excluded from the path extension mechanism
because attempts to load their parts from source code lead to app crash."""

logger = logging.getLogger(__name__)


def install_path_extension_hook(extra_path: str) -> None:
    """Install a path extension hook to load missing submodules from source code.

    Args:
        extra_path: Location of the source code to check for submodules.
    """
    sys.meta_path = [
        m for m in sys.meta_path if type(m).__name__ != "PassivePathExtensionHook"
    ]
    sys.meta_path.insert(0, PassivePathExtensionHook(extra_path))
    logger.debug(f"Installed path extension hook for {extra_path}")


def _extend_package_search_path(module: types.ModuleType, site_pkg_dir: str) -> bool:
    mod_path = getattr(module, "__path__", None)
    if not isinstance(mod_path, list) or not os.path.isdir(site_pkg_dir):
        return False

    if site_pkg_dir not in mod_path:
        mod_path.append(site_pkg_dir)

    spec = getattr(module, "__spec__", None)
    locations = getattr(spec, "submodule_search_locations", None)
    if isinstance(locations, list) and site_pkg_dir not in locations:
        locations.append(site_pkg_dir)

    return True


class PassivePathExtensionHook(importlib.abc.MetaPathFinder):
    """Passively extend `__path__` of loaded packages to resolve missing submodules."""

    def __init__(self, site_packages: str) -> None:
        self.site_packages = site_packages
        self._patched: set[str] = set()

    def find_spec(
        self,
        fullname: str,
        _path: T.Sequence[str] | None,
        _target: types.ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if "." not in fullname:
            return None

        root = fullname.split(".", 1)[0]
        if root in _UNSAFE_EXTENSION_ROOTS:
            return None

        parent = fullname.rsplit(".", 1)[0]
        if parent not in sys.modules or parent in self._patched:
            return None

        site_pkg_path = os.path.join(self.site_packages, parent.replace(".", os.sep))
        if _extend_package_search_path(sys.modules[parent], site_pkg_path):
            self._patched.add(parent)

        return None
