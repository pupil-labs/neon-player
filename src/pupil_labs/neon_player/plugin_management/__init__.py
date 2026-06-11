import importlib.metadata
import logging
import subprocess
import sys
from pathlib import Path

from packaging.requirements import Requirement

from pupil_labs import neon_player
from pupil_labs.neon_player.job_manager import ProgressUpdate
from pupil_labs.neon_player.plugin_management.pep723 import parse_pep723_dependencies

PLUGINS_PACKAGES_DIR = Path.home() / "Pupil Labs" / "Neon Player" / "plugins"
SITE_PACKAGES_DIR = PLUGINS_PACKAGES_DIR / "site-packages"

SITE_PACKAGES_DIR.mkdir(parents=True, exist_ok=True)


def get_installed_packages() -> set[str]:
    """Get a set of installed package names in the shared site-packages."""
    try:
        return {
            dist.metadata["name"].lower().replace("_", "-")
            for dist in importlib.metadata.distributions(path=[str(SITE_PACKAGES_DIR)])
        }
    except Exception:
        logging.warning("Could not list installed packages in plugin site-packages.")
        return set()


def check_dependencies_for_plugin(plugin_path: Path) -> tuple[str, list[str]] | None:
    """Scan a plugin, find their dependencies, return missing ones."""
    all_deps_to_install = set()
    all_dep_names = set()

    path_to_read = plugin_path
    if plugin_path.is_dir():
        path_to_read = plugin_path / "__init__.py"
    if not path_to_read.exists():
        logging.warning(f"Plugin file on {path_to_read} does not exist.")
        return

    try:
        script = path_to_read.read_text(encoding="utf-8")
        deps = parse_pep723_dependencies(script)
        if deps:
            for dep_string in deps.dependencies:
                try:
                    req = Requirement(dep_string)
                    if req.marker and not req.marker.evaluate({"sys_platform": sys.platform}):
                        continue
                    all_deps_to_install.add(dep_string)
                    all_dep_names.add(req.name.lower().replace("_", "-"))
                except Exception as req_err:
                    raise ValueError(
                        f"Plugin {plugin_path.name} has a malformed PEP 723 "
                        f"dependency: {dep_string!r}. "
                        f"Reason: {req_err}"
                    ) from req_err

    except ValueError:
        raise
    except Exception:
        logging.exception(f"Could not parse dependencies for {plugin_path.name}")

    if not all_deps_to_install:
        return  # No dependencies to check

    installed_packages = get_installed_packages()
    missing_dep_names = all_dep_names - installed_packages

    if not missing_dep_names:
        return  # All dependencies are satisfied

    deps_to_install_filtered = []

    for full_dep in all_deps_to_install:
        try:
            req_name = Requirement(full_dep).name.lower().replace("_", "-")
            if req_name in missing_dep_names:
                deps_to_install_filtered.append(full_dep)
        except Exception as req_err:
            raise ValueError(
                f"Plugin {plugin_path.name} has a malformed PEP 723 "
                f"dependency: {full_dep!r}. "
                f"Reason: {req_err}"
            ) from req_err

    deps_to_install_filtered = sorted(deps_to_install_filtered)

    logging.info(f"Found missing plugin dependencies: {deps_to_install_filtered}")

    return deps_to_install_filtered


def install_dependencies(dependencies: list[str]):
    """Install dependencies into the shared site-packages directory using uv."""
    if not dependencies:
        logging.info("No new dependencies to install.")
        return

    if neon_player.is_frozen():
        uv_cmd = Path(sys.argv[0]).parent / "uv"
    else:
        uv_cmd = "uv"

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
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.stdout:
            logging.info(proc.stdout.rstrip())
        if proc.stderr:
            logging.warning(proc.stderr.rstrip())

        yield ProgressUpdate(1.0)
        logging.info("Successfully installed dependencies.")

    except subprocess.CalledProcessError as e:
        logging.exception(f"Failed to install dependencies. Error: {e.stderr}")
        raise
