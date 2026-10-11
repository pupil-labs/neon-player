import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


@pytest.fixture
def run_player(tmp_path):
    """Use real application activation in a process with isolated settings."""
    if not (
        sys.platform.startswith("linux")
        and os.environ.get("DISPLAY")
        and os.environ.get("QT_QPA_PLATFORM", "xcb") == "xcb"
    ):
        pytest.skip("Player integration checks require a native X11 display")

    def run(body):
        script = textwrap.dedent("""
            import sys
            from pathlib import Path
            from types import SimpleNamespace
            from unittest.mock import patch

            import numpy as np
            from PySide6.QtGui import QImage, QPainter
            from PySide6.QtTest import QTest
            from pupil_labs.neon_player.app import NeonPlayerApp
            from pupil_labs.neon_player.settings import RecordingSettings

            class SceneFrames:
                width, height = 640, 480
                time = np.array([0, 30_000_000], dtype=np.int64)

                def sample(self, timestamps, method="backward"):
                    return [SimpleNamespace(
                        bgr=np.full((480, 640, 3), (0, 0, 255), dtype=np.uint8),
                        width=640, height=480, idx=0,
                    )]

            class EyeFrames:
                width, height = 200, 100
                delay = 0
                missing = False

                def sample(self, timestamps):
                    if self.missing:
                        return []
                    return [SimpleNamespace(
                        gray=np.full((100, 200), 255, dtype=np.uint8),
                        width=200, height=100, time=timestamps[0] + self.delay,
                    )]

            task_home = Path(sys.argv[1])
            with patch.object(Path, "home", return_value=task_home), patch.object(
                sys, "argv", ["neon-player"]
            ):
                app = NeonPlayerApp(["neon-player"])
                window = app.main_window
                window.resize(1000, 700)
                recording = SimpleNamespace(
                    scene=SceneFrames(), eye=EyeFrames(), start_time=0,
                    stop_time=30_000_000, _rec_dir=task_home,
                    close=lambda: None,
                )
                app.recording = recording
                app.recording_settings = RecordingSettings()
                app.recording_settings.enabled_plugins = {}
                app.recording_settings.plugins_changed.connect(app.toggle_plugins_by_settings)
                window.on_recording_loaded(recording)
                window.on_recording_opened()
                try:
                    pass
        """)
        script = script.removesuffix("        pass\n")
        script += textwrap.indent(textwrap.dedent(body), "        ")
        script += "\n    finally:\n        app.unload()\n        window.video_workspace.dispose()\n        window.close()\n"
        result = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path)],
            capture_output=True, text=True, timeout=45,
            env={**os.environ, "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]},
        )
        assert result.returncode == 0, result.stdout + result.stderr

    return run
