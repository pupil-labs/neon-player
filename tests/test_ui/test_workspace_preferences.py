import json

import pytest


def test_floating_layout_survives_application_restart_without_enabling_plugin(
    run_player,
):
    run_player("""
        window.show()
        app.recording_settings.enabled_plugins = {"EyeOverlayPlugin": True}
        QTest.qWait(50)
        workspace = window.video_workspace
        workspace.float_dock("eye-video")
        floating = workspace.get_dock("eye-video").content.window()
        floating.resize(420, 260)
        floating.move(40, 50)
        QTest.qWait(50)
        app.save_settings(force=True)
    """)
    run_player("""
        window.show()
        QTest.qWait(50)
        workspace = window.video_workspace
        assert workspace.get_dock("eye-video") is None
        assert not app.recording_settings.enabled_plugins["EyeOverlayPlugin"]
        app.recording_settings.enabled_plugins = {"EyeOverlayPlugin": True}
        QTest.qWait(80)
        eye = workspace.get_dock("eye-video").content
        assert eye.window() is not window
        assert eye.window().width() == 420 and eye.window().height() == 260
    """)


def test_overlay_split_and_tab_layouts_round_trip_through_global_settings(run_player):
    for mode in ("overlay", "split", "tab"):
        run_player(
            """
            import json
            from pyside6_docking import DockPosition

            window.show()
            app.recording_settings.enabled_plugins = {"EyeOverlayPlugin": True}
            QTest.qWait(50)
            workspace = window.video_workspace
            mode = "MODE"
            if mode == "overlay":
                layout = workspace.save_layout()
                layout["overlays"][0].update(x=0.4, y=0.3, width=0.35, height=0.3)
                workspace.restore_layout(layout)
            else:
                workspace.dock("eye-video", DockPosition.RIGHT if mode == "split" else DockPosition.CENTER,
                               relative_to="scene-video")
            QTest.qWait(50)
            expected = workspace.save_layout()
            (task_home / "expected.json").write_text(json.dumps(expected))
            app.save_settings(force=True)
        """.replace("MODE", mode)
        )
        run_player("""
            import json
            from PySide6.QtCore import Qt
            from PySide6.QtTest import QSignalSpy

            window.show()
            app.recording_settings.enabled_plugins = {
                "SceneRendererPlugin": True, "EyeOverlayPlugin": True,
            }
            QTest.qWait(80)
            actual = window.video_workspace.save_layout()
            expected = json.loads((task_home / "expected.json").read_text())
            assert actual["main"] == expected["main"]
            if expected.get("overlays"):
                for key in ("x", "y", "width", "height"):
                    assert abs(actual["overlays"][0][key] - expected["overlays"][0][key]) < 0.02
            eye = window.video_workspace.get_dock("eye-video").content
            app.position_changed.emit(30_000_000)
            image = eye.grab().toImage()
            assert image.pixelColor(image.width() // 4, image.height() // 2).name() == "#ffffff"
            window.activateWindow()
            assert QTest.qWaitForWindowActive(window)
            spy = QSignalSpy(window.playback_actions[0].triggered)
            QTest.keyClick(eye, Qt.Key.Key_Space)
            assert spy.count() == 1
            app.set_playback_state(False)
            image = QImage(640, 480, QImage.Format.Format_RGB32)
            painter = QPainter(image)
            app.render_to(painter, 0)
            painter.end()
            assert image.pixelColor(40, 30).name() == "#ffffff"
        """)


def test_unavailable_panel_and_new_panel_preserve_preferred_placement(run_player):
    run_player("""
        from PySide6.QtWidgets import QLabel
        from pyside6_docking import DockPosition

        window.show()
        app.recording_settings.enabled_plugins = {"EyeOverlayPlugin": True}
        workspace = window.video_workspace
        workspace.dock("eye-video", DockPosition.RIGHT, relative_to="scene-video")
        app.unload()
        original_eye = recording.eye
        recording.eye = None
        app.recording = recording
        window.on_recording_loaded(recording)
        window.on_recording_opened()
        app.toggle_plugins_by_settings()
        assert workspace.get_dock("eye-video") is None
        app.save_settings(force=True)
        assert "eye-video" in str(app.settings.video_workspace_layout)
        workspace.register_panel("example", "Example", QLabel("Example"),
                                 on_close=lambda: None, position=DockPosition.BOTTOM)
        QTest.qWait(50)
        example = workspace.get_dock("example").content
        assert example.isVisible()
        app.unload()
        recording.eye = original_eye
        app.recording = recording
        window.on_recording_loaded(recording)
        window.on_recording_opened()
        app.toggle_plugins_by_settings()
        QTest.qWait(80)
        assert workspace.get_dock("eye-video").content.isVisible()
        assert not workspace.save_layout().get("overlays")
        assert example.isVisible()
        workspace.unregister_panel("example")
    """)


def test_reset_clears_absent_placement_without_enabling_plugins(run_player):
    run_player("""
        window.show()
        app.recording_settings.enabled_plugins = {"EyeOverlayPlugin": True}
        workspace = window.video_workspace
        workspace.float_dock("eye-video")
        app.recording_settings.enabled_plugins = {"EyeOverlayPlugin": False}
        window.reset_docks()
        assert not app.recording_settings.enabled_plugins["EyeOverlayPlugin"]
        assert workspace.get_dock("eye-video") is None
        app.recording_settings.enabled_plugins = {"EyeOverlayPlugin": True}
        QTest.qWait(50)
        assert workspace.get_dock("eye-video").content.window() is window
        layout = workspace.save_layout()
        assert abs(layout["overlays"][0]["x"] - 0.02) < 0.02
        workspace.float_dock("eye-video")
        window.reset_docks()
        QTest.qWait(50)
        assert app.recording_settings.enabled_plugins["EyeOverlayPlugin"]
        assert not workspace.save_layout()["floating"]
        assert workspace.save_layout()["main"]["docks"] == ["scene-video"]
        assert workspace.save_layout()["overlays"][0]["parent_id"] == "scene-video"
        assert not window.timeline_dock.isFloating()
        assert not window.settings_dock.isFloating()
    """)


@pytest.mark.parametrize(
    "saved",
    [
        None,
        {"schema_version": 99},
        {
            "schema_version": 1,
            "main": {"type": "tabs", "docks": ["scene-video"], "active": "scene-video"},
            "floating": [],
            "hidden": [{"dock_id": "eye-video", "placement": None}],
        },
        {
            "schema_version": 1,
            "main": {
                "type": "split",
                "orientation": "horizontal",
                "sizes": [10**500, 1],
                "children": [
                    {"type": "tabs", "docks": ["scene-video"], "active": "scene-video"},
                    {"type": "tabs", "docks": ["eye-video"], "active": "eye-video"},
                ],
            },
            "floating": [],
            "hidden": [],
        },
        {
            "schema_version": 1,
            "main": {
                "type": "tabs",
                "docks": ["scene-video", "eye-video", "eye-video"],
                "active": "scene-video",
            },
            "floating": [],
            "hidden": [],
        },
        {
            "schema_version": 2,
            "main": {"type": "tabs", "docks": ["scene-video"], "active": "scene-video"},
            "floating": [],
            "hidden": [],
            "overlays": [
                {
                    "dock_id": "eye-video",
                    "parent_id": "scene-video",
                    "x": 0,
                    "y": 0,
                    "width": -1,
                    "height": 0.3,
                }
            ],
        },
        {"schema_version": 1, "main": None, "floating": [], "hidden": []},
    ],
)
def test_invalid_layout_falls_back_without_losing_other_settings(
    run_player, tmp_path, saved
):
    path = tmp_path / "Pupil Labs" / "Neon Player" / "settings.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"video_workspace_layout": saved, "show_fps": True}))
    run_player("""
        assert app.settings.show_fps
        window.show()
        app.recording_settings.plugin_states = {
            "EyeOverlayPlugin": {"offset_x": 0.6, "scale": 0.5},
        }
        app.recording_settings.enabled_plugins = {"EyeOverlayPlugin": True}
        QTest.qWait(80)
        workspace = window.video_workspace
        assert workspace.get_dock("eye-video").content.isVisible()
        assert workspace.get_dock("scene-video").content.isVisible()
        assert workspace.save_layout()["overlays"][0]["parent_id"] == "scene-video"
        plugin = app.plugins_by_class["EyeOverlayPlugin"]
        assert plugin.offset_x == 0.6 and plugin.scale == 0.5
        assert app.settings.show_fps
    """)


def test_absent_overlay_retains_its_surviving_plugin_anchor(run_player):
    run_player("""
        from PySide6.QtCore import QRectF
        from PySide6.QtWidgets import QLabel

        window.show()
        workspace = window.video_workspace
        workspace.register_panel("anchor", "Anchor", QLabel("Anchor"), on_close=lambda: None)
        workspace.register_panel("detail", "Detail", QLabel("Detail"), on_close=lambda: None,
                                 relative_to="anchor", overlay=QRectF(0.1, 0.1, 0.6, 0.6))
        QTest.qWait(50)
        layout = workspace.save_layout()
        layout["overlays"][0].update(x=0.2, y=0.2, width=0.5, height=0.5)
        workspace.restore_layout(layout)
        workspace.unregister_panel("detail")
        app.save_settings(force=True)
        workspace.register_panel("detail", "Detail", QLabel("Detail"), on_close=lambda: None,
                                 relative_to="anchor", overlay=QRectF(0.1, 0.1, 0.6, 0.6))
        QTest.qWait(50)
        overlay = workspace.save_layout()["overlays"][0]
        assert overlay["parent_id"] == "anchor"
        assert abs(overlay["x"] - 0.2) < 0.02
        assert abs(overlay["width"] - 0.5) < 0.03
        workspace.unregister_panel("detail")
        workspace.unregister_panel("anchor")
    """)


def test_absent_panel_rejoins_its_floating_tab_group(run_player):
    run_player("""
        from PySide6.QtWidgets import QLabel
        from pyside6_docking import DockFeatures

        window.show()
        workspace = window.video_workspace
        for panel_id in ("a", "b"):
            workspace.register_panel(panel_id, panel_id, QLabel(panel_id),
                                     on_close=lambda: None, features=DockFeatures.ALL)
        workspace.float_dock("a")
        floating = workspace.get_dock("a").content.window()
        floating.resize(460, 280)
        layout = workspace.save_layout()
        layout["main"] = {"type": "tabs", "docks": ["scene-video"], "active": "scene-video"}
        layout["floating"][0]["root"] = {"type": "tabs", "docks": ["a", "b"], "active": "b"}
        workspace.restore_layout(layout)
        QTest.qWait(50)
        workspace.unregister_panel("b")
        app.save_settings(force=True)
        workspace.register_panel("b", "b", QLabel("b"), on_close=lambda: None,
                                 features=DockFeatures.ALL)
        QTest.qWait(50)
        assert workspace.get_dock("a").content.window() is workspace.get_dock("b").content.window()
        layout = workspace.save_layout()
        assert len(layout["floating"]) == 1
        assert layout["floating"][0]["root"]["docks"] == ["a", "b"]
        assert layout["floating"][0]["geometry"]["width"] == 460
        workspace.unregister_panel("b")
        workspace.unregister_panel("a")
    """)
