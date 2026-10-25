import importlib.metadata
import logging
import subprocess
import sys
import typing as T
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import NormalizedName, canonicalize_name

from pupil_labs import neon_player
from pupil_labs.neon_player.job_manager import ProgressUpdate
from pupil_labs.neon_player.plugin_management.pep723 import (
    MalformedPEP723Error,
    PythonVersionMismatchError,
    parse_pep723_dependencies,
)

logger = logging.getLogger(__name__)

PLUGINS_PACKAGES_DIR = Path.home() / "Pupil Labs" / "Neon Player" / "plugins"
SITE_PACKAGES_DIR = PLUGINS_PACKAGES_DIR / "site-packages"

SITE_PACKAGES_DIR.mkdir(parents=True, exist_ok=True)


def get_installed_packages() -> dict[NormalizedName, str]:
    """
    Get installed package names and versions in the shared site-packages.

    Dependency names are normalized according to:
    https://packaging.python.org/en/latest/specifications/name-normalization/#name-normalization.
    """
    try:
        return {
            canonicalize_name(dist.metadata["name"]): dist.metadata["version"]
            for dist in importlib.metadata.distributions()
        }
    except Exception:
        logger.exception("Could not list installed packages in plugin site-packages")
        return {}


def is_dependency_installed(
    dependency: str, installed_packages: dict[NormalizedName, str]
) -> bool:
    """Check if a dependency is already installed in the shared site-packages."""
    req = Requirement(dependency)
    req_name = canonicalize_name(req.name)
    if req_name not in installed_packages:
        return False

    installed_version = installed_packages[req_name]
    return req.specifier.contains(installed_version)


def check_dependencies_for_plugin(plugin_path: Path) -> list[str]:
    """Scan a plugin, find their dependencies, return missing ones."""
    path_to_read = plugin_path
    if plugin_path.is_dir():
        path_to_read = plugin_path / "__init__.py"
    if not path_to_read.exists():
        logger.warning(f"Plugin file on {path_to_read} does not exist.")
        return []

    script = path_to_read.read_text(encoding="utf-8")
    plugin_name = plugin_path.name

    return _check_dependencies_for_plugin_script(script, plugin_name)


def _check_dependencies_for_plugin_script(script: str, plugin_name: str) -> list[str]:
    try:
        dependencies = parse_pep723_dependencies(script)
    except PythonVersionMismatchError as e:
        logger.error(
            f"Plugin {plugin_name} requires an incompatible Python version: {e}"
        )
        raise
    except MalformedPEP723Error as e:
        logger.error(f"Plugin {plugin_name} has a malformed PEP 723 block: {e}")
        raise
    except InvalidRequirement as e:
        logger.error(f"Plugin {plugin_name} has an invalid dependency requirement: {e}")
        raise
    except Exception:
        logger.exception(f"Could not parse dependencies for {plugin_name}")
        raise

    if not dependencies:
        return []

    installed_packages = get_installed_packages()
    deps_to_install = [
        dep
        for dep in dependencies
        if not is_dependency_installed(dep, installed_packages)
    ]

    if not deps_to_install:
        return []

    logger.info(
        f"Found missing dependencies for {plugin_name} plugin: {deps_to_install}"
    )
    return deps_to_install


def install_dependencies(
    dependencies: list[str],
) -> T.Generator[ProgressUpdate, None, None]:
    """Install dependencies into the shared site-packages directory using uv."""
    if not dependencies:
        logger.info("No new dependencies to install.")
        return

    uv_cmd = str(Path(sys.argv[0]).parent / "uv") if neon_player.is_frozen() else "uv"

    pyver = sys.version_info

    command = [
        uv_cmd,
        "pip",
        "install",
        f"--python={pyver.major}.{pyver.minor}.{pyver.micro}",
        f"--target={SITE_PACKAGES_DIR}",
        *dependencies,
    ]

    try:
        proc = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.stdout:
            logger.info(proc.stdout.rstrip())
        if proc.stderr:
            logger.warning(proc.stderr.rstrip())

        yield ProgressUpdate(1.0)
        logger.info("Successfully installed dependencies.")

    except subprocess.CalledProcessError as e:
        logger.exception(f"Failed to install dependencies. Error: {e.stderr}")
        raise
