import pytest
import sys

from packaging.requirements import InvalidRequirement
from pupil_labs.neon_player.plugin_management.pep723 import (
    parse_pep723_dependencies, PythonVersionMismatchError, MalformedPEP723Error
)


def test_parse_pep723_dependencies__valid_block():
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
    assert "numpy" in result
    assert "librosa" in result


def test_parse_pep723_dependencies__no_block():
    script = "import os\nprint('hello')\n"
    assert not parse_pep723_dependencies(script)


def test_parse_pep723_dependencies__empty_dependencies():
    script = """\
# /// script
# dependencies = []
# ///
"""
    assert not parse_pep723_dependencies(script)


def test_parse_pep723_dependencies__python_version_mismatch():
    script = """\
# /// script
# requires-python = "<3.11"
# dependencies = []
# ///
"""
    with pytest.raises(PythonVersionMismatchError, match="<3.11"):
        parse_pep723_dependencies(script)


def test_parse_pep723_dependencies__malformed_toml():
    script = """\
# /// script
# this is not valid toml ][
# ///
"""
    with pytest.raises(MalformedPEP723Error, match="this is not valid toml"):
        parse_pep723_dependencies(script)


def test_parse_pep723_dependencies__malformed_dependency():
    script = """\
# /// script
# dependencies = [
#     "numpy >"
# ]
# ///
"""
    with pytest.raises(InvalidRequirement):
        parse_pep723_dependencies(script)


def test_parse_pep723_dependencies__with_platform_markers():
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
    assert len(result) == 1
    if sys.platform == "darwin":
        assert "mlx-whisper" in result[0]
    else:
        assert "openai-whisper" in result[0]
