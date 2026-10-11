"""Exercise the real application in an isolated Qt process.

The suite's shared QApplication is a minimal headless harness. A child process
lets these checks use the real MainWindow and OpenGL scene widget instead.
"""

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


def test_scene_video_workspace_keeps_application_navigation(tmp_path):
    if not (
        sys.platform.startswith("linux")
        and os.environ.get("DISPLAY")
        and os.environ.get("QT_QPA_PLATFORM", "xcb") == "xcb"
    ):
        pytest.skip("Scene workspace smoke check requires a native X11 display")

    script = textwrap.dedent("""
        import sys
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch

        import numpy as np
        from PySide6.QtCore import QPointF, Qt
        from PySide6.QtGui import QMouseEvent
        from PySide6.QtTest import QTest
        from pyside6_docking import DockFeatures, DockManager
        from pupil_labs.neon_player.app import NeonPlayerApp
        from pupil_labs.neon_player.plugins.eye_overlay import EyeOverlayPlugin
        from pupil_labs.neon_player.plugins.scene_renderer import SceneRendererPlugin
        from pupil_labs.neon_player.plugins.surface_tracking.ui import SurfaceHandle

        task_home = Path(sys.argv[1])
        with patch.object(Path, "home", return_value=task_home), patch.object(
            sys, "argv", ["neon-player"]
        ):
            app = NeonPlayerApp(["neon-player"])
            window = app.main_window
            window.resize(1000, 700)
            window.show()
            assert QTest.qWaitForWindowExposed(window)

            assert window.greeting_switcher.currentWidget() is window.splash_widget
            window.on_show_recent_action()
            assert window.greeting_switcher.currentWidget() is window.recent_widget
            window.on_show_splash_action()
            assert window.greeting_switcher.currentWidget() is window.splash_widget

            workspace = window.greeting_switcher.widget(1)
            assert isinstance(workspace, DockManager)
            scene = workspace.get_dock("scene-video")
            assert scene.content is window.video_widget
            assert scene.features == DockFeatures.NONE
            assert not workspace.isAncestorOf(window.timeline)
            assert not workspace.isAncestorOf(window.settings_panel)

            window.on_recording_opened()
            app.processEvents()
            assert window.video_widget.isVisible()
            assert window.timeline_dock.isVisible()
            assert window.settings_dock.isVisible()

            # Substitute decoded recording data at the media boundary, while
            # exercising the actual plugins and native OpenGL scene renderer.
            class SceneFrames:
                width, height = 640, 480
                time = np.array([0, 30_000_000], dtype=np.int64)

                def sample(self, timestamps, method="backward"):
                    color = (0, 0, 255) if timestamps[0] == 0 else (0, 255, 0)
                    return [SimpleNamespace(
                        bgr=np.full((480, 640, 3), color, dtype=np.uint8),
                        width=640, height=480, idx=0,
                    )]

            class EyeFrames:
                width, height = 100, 50

                def sample(self, timestamps):
                    return [SimpleNamespace(
                        gray=np.full((50, 100), 255, dtype=np.uint8),
                        width=100, height=50, time=timestamps[0],
                    )]

            app.recording = SimpleNamespace(scene=SceneFrames(), eye=EyeFrames())
            app.plugins = [SceneRendererPlugin(), EyeOverlayPlugin()]
            window.on_recording_loaded(app.recording)

            def pixel_at_scene(x, y):
                widget = window.video_widget
                widget.repaint()
                QTest.qWait(50)
                image = window.screen().grabWindow(window.winId()).toImage()
                assert not image.isNull()
                ratio = image.devicePixelRatio()
                point = widget.mapTo(window, QPointF(
                    widget.offset.x() + x * widget.scale,
                    widget.offset.y() + y * widget.scale,
                ).toPoint())
                return image.pixelColor(int(point.x() * ratio), int(point.y() * ratio)).name()

            window.set_time_in_recording(0)
            QTest.qWait(100)
            color = pixel_at_scene(480, 240)
            assert color == "#ff0000", (
                color, window.video_widget.size(), window.video_widget.isValid(),
                window.video_widget.source_size, window.video_widget.scale,
                window.video_widget.offset,
            )
            assert pixel_at_scene(40, 30) == "#ffffff"  # legacy eye overlay
            app.current_ts = 30_000_000
            window.set_time_in_recording(app.current_ts)
            app.processEvents()
            assert pixel_at_scene(480, 240) == "#00ff00"

            handle = SurfaceHandle(None, (0, 0))
            handle.setParent(window.video_widget)
            handle.resize(20, 20)
            handle.set_scene_pos(np.array([320.0, 240.0]))
            for width, height in [(1200, 850), (900, 650)]:
                window.resize(width, height)
                app.processEvents()
                widget = window.video_widget
                center = handle.mapToParent(handle.rect().center())
                mapped = widget.map_point(center)
                assert abs(mapped.x() - 320) < 3
                assert abs(mapped.y() - 240) < 3
                assert pixel_at_scene(480, 240) == "#00ff00"

            press = QPointF(handle.rect().center())
            QTest.mousePress(handle, Qt.MouseButton.LeftButton, pos=press.toPoint())
            move = QMouseEvent(
                QMouseEvent.Type.MouseMove, press + QPointF(20, 15),
                handle.mapToGlobal((press + QPointF(20, 15)).toPoint()),
                Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
            app.sendEvent(handle, move)
            QTest.mouseRelease(handle, Qt.MouseButton.LeftButton)
            assert handle.new_pos.x() > 320
            assert handle.new_pos.y() > 240
            assert handle.position_changed_debounce_timer.isActive()
            handle.position_changed_debounce_timer.stop()
            assert window.screen().grabWindow(window.winId()).save(
                str(task_home / "scene-workspace.png")
            )
            app.plugins.clear()
            app.recording = None

            app.unload()
            app.processEvents()
            assert window.greeting_switcher.currentWidget() is window.splash_widget
            assert not window.timeline_dock.isVisible()
            assert not window.settings_dock.isVisible()
            assert workspace.get_dock("scene-video").content is window.video_widget
            app.aboutToQuit.emit()
            window.close()
            print("Native scene workspace navigation passed:", app.platformName())
    """)
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=45,
        env={
            **os.environ,
            "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
