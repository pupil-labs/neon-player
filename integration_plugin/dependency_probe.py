# /// script
# dependencies = [
#   "Pillow",
# ]
# ///

from __future__ import annotations

import importlib
import os
from pathlib import Path

from pupil_labs.neon_player import Plugin, action

PIL = importlib.import_module("PIL")
ImageDraw = importlib.import_module("PIL.ImageDraw")


class DependencyProbe(Plugin):
    """End-to-end probe for bundled-parent/plugin-submodule imports."""

    @action
    def probe(self) -> None:
        sentinel = os.environ.get("NEON_PLAYER_DEPENDENCY_PROBE_SENTINEL")
        if not sentinel:
            raise RuntimeError("NEON_PLAYER_DEPENDENCY_PROBE_SENTINEL is not set")
        Path(sentinel).write_text(
            "PIL.ImageDraw\n"
            f"PIL={getattr(PIL, '__version__', 'unknown')}\n"
            f"ImageDraw={getattr(ImageDraw, '__file__', '<no-file>')}\n",
            encoding="utf-8",
        )
