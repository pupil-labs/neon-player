import logging
import re
import sys
import tomllib

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version


class PythonVersionMismatchError(Exception):
    """Raised when a plugin requires a Python version that is not compatible
    with the current environment."""


class MalformedPEP723Error(Exception):
    """Raised when the PEP 723 block is malformed (e.g., invalid TOML)."""


def should_be_considered(dependency: str) -> bool:
    """Determine if a dependency string should be considered for installation."""
    req = Requirement(dependency)
    if req.marker and not req.marker.evaluate():
        return False
    return True


def parse_pep723_dependencies(script: str) -> list[str]:
    """Parse a code for a PEP 723 dependency block and returns the dependencies."""
    # Regex to find the /// script ... /// block
    match = re.search(
        r"^# /// script\n(.*?)\n^# ///$", script, re.MULTILINE | re.DOTALL
    )
    if not match:
        return []

    # The TOML content is captured in group 1. We need to "un-comment" it.
    toml_lines = []
    for line in match.group(1).splitlines():
        if not line.strip():
            continue

        if line.startswith("# "):
            toml_lines.append(line[2:])
        elif line.strip() == "#":
            toml_lines.append("")
        else:
            raise MalformedPEP723Error(line)

    # Parse TOML
    toml_content = "\n".join(toml_lines)
    try:
        data = tomllib.loads(toml_content)
    except tomllib.TOMLDecodeError as e:
        raise MalformedPEP723Error(toml_content) from e

    # If Python version is specified, check compatibility with Neon Player
    python_version = data.get("requires-python")
    if python_version:
        specifier = SpecifierSet(python_version)
        current_version = Version(".".join(map(str, sys.version_info[:3])))
        if not specifier.contains(current_version):
            raise PythonVersionMismatchError(python_version)

    # Find dependencies that are applicable for the current platform
    dependencies = data.get("dependencies", [])
    if not dependencies:
        return []

    return list(filter(should_be_considered, dependencies))
