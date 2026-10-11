def test_eye_overlay_activation_registers_one_panel(run_player):
    run_player('''
        app.recording_settings.enabled_plugins = {"EyeOverlayPlugin": True}
        app.processEvents()
        panel = window.video_workspace.get_dock("eye-video")
        assert panel is not None
        assert panel.title == "Eye video"
        plugin = app.plugins_by_class["EyeOverlayPlugin"]
        assert plugin.label == "Eye Overlay"
        assert panel.content is not window.video_widget
        app.toggle_plugins_by_settings()
        assert window.video_workspace.get_dock("eye-video") is panel
    ''')


def test_saved_export_properties_are_independent_of_panel_placement(run_player):
    run_player('''
        from PySide6.QtWidgets import QFileDialog
        from pupil_labs.neon_player.plugins.video_exporter import VideoExporter

        legacy = dict(offset_x=0.6, offset_y=0.1, scale=0.5,
                      opacity=1.0, border_width=0, border_color=[170, 170, 170, 255])
        app.recording_settings.plugin_states = {"EyeOverlayPlugin": legacy}
        app.recording_settings.enabled_plugins = {
            "SceneRendererPlugin": True, "EyeOverlayPlugin": True,
        }
        plugin = app.plugins_by_class["EyeOverlayPlugin"]
        assert plugin.offset_x == 0.6 and plugin.scale == 0.5
        window.show()
        QTest.qWait(50)
        layout = window.video_workspace.save_layout()
        placement = layout["overlays"][0]
        assert placement["parent_id"] == "scene-video"
        assert abs(placement["x"] - 0.02) < 0.02, placement
        assert abs(placement["y"] - 0.02) < 0.02
        assert abs(placement["width"] - 0.3125) < 0.03, placement

        def render(display=False):
            image = QImage(640, 480, QImage.Format.Format_RGB32)
            image.fill(0)
            painter = QPainter(image)
            app.render_to(painter, 0, for_display=display)
            painter.end()
            return image

        exported = render()
        assert exported.pixelColor(410, 70).name() == "#ffffff"
        assert exported.pixelColor(40, 30).name() == "#ff0000"
        assert render(True).pixelColor(410, 70).name() == "#ff0000"
        before = plugin.to_dict(recursive=True)
        placement.update(x=0.55, y=0.55, width=0.3, height=0.3)
        window.video_workspace.restore_layout(layout)
        assert render() == exported
        assert plugin.to_dict(recursive=True) == before

        exporter = VideoExporter()
        image = QImage(640, 480, QImage.Format.Format_RGB32)
        painter = QPainter(image)
        exporter.render_for_export(painter, 0)
        painter.end()
        assert image == exported
        exporter.copy_frame_to_clipboard()
        assert app.clipboard().pixmap().toImage().pixelColor(410, 70).name() == "#ffffff"
        path = task_home / "frame.png"
        with patch.object(QFileDialog, "getSaveFileName", return_value=(str(path), "PNG Images (*.png)")):
            exporter.export_current_frame()
        assert QImage(str(path)).pixelColor(410, 70).name() == "#ffffff"
    ''')


def test_headless_eye_overlay_keeps_export_without_a_panel(run_player):
    run_player('''
        app.args.job = ["VideoExporter.export"]
        app.toggle_plugin("SceneRendererPlugin", True, {})
        app.toggle_plugin("EyeOverlayPlugin", True, {"border_width": 0})
        assert window.video_workspace.get_dock("eye-video") is None
        image = QImage(640, 480, QImage.Format.Format_RGB32)
        painter = QPainter(image)
        app.render_to(painter, 0)
        painter.end()
        assert image.pixelColor(40, 30).name() == "#ffffff"
    ''')


def test_panel_close_disables_plugin_and_reopening_creates_a_new_panel(run_player):
    run_player('''
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QAbstractButton

        window.show()
        app.recording_settings.enabled_plugins = {"EyeOverlayPlugin": True}
        QTest.qWait(50)
        old = window.video_workspace.get_dock("eye-video")
        close = next(button for button in window.findChildren(QAbstractButton)
                     if button.accessibleName() == "Close Eye video")
        QTest.mouseClick(close, Qt.MouseButton.LeftButton)
        QTest.qWait(50)
        assert not app.recording_settings.enabled_plugins["EyeOverlayPlugin"]
        assert "EyeOverlayPlugin" not in app.plugins_by_class
        assert window.video_workspace.get_dock("eye-video") is None

        app.recording_settings.enabled_plugins = {"EyeOverlayPlugin": True}
        QTest.qWait(50)
        new = window.video_workspace.get_dock("eye-video")
        assert new is not None and new is not old
        app.toggle_plugin("EyeOverlayPlugin", False)
        assert app.recording_settings.enabled_plugins["EyeOverlayPlugin"]
        assert window.video_workspace.get_dock("eye-video") is None
    ''')


def test_eye_panel_clears_stale_and_missing_frames_and_preserves_aspect(run_player):
    run_player('''
        window.show()
        app.recording_settings.enabled_plugins = {"EyeOverlayPlugin": True}
        layout = window.video_workspace.save_layout()
        layout["overlays"][0].update(width=0.4, height=0.7)
        window.video_workspace.restore_layout(layout)
        QTest.qWait(50)
        view = window.video_workspace.get_dock("eye-video").content
        # QWidget.grab exercises the real eye presentation without OpenGL.
        def center_color():
            image = view.grab().toImage()
            return image.pixelColor(image.width() // 4, image.height() // 2).name()

        assert center_color() == "#ffffff"
        assert view.grab().toImage().pixelColor(5, 5).name() == "#000000"
        recording.eye.delay = 33_333_333
        app.position_changed.emit(30_000_000)
        assert center_color() == "#ffffff"
        recording.eye.delay = 33_333_334
        app.position_changed.emit(0)
        assert center_color() != "#ffffff"
        recording.eye.delay = 0
        app.position_changed.emit(0)
        assert center_color() == "#ffffff"
        recording.eye.missing = True
        app.position_changed.emit(0)
        assert center_color() != "#ffffff"
    ''')


def test_recording_without_eye_video_does_not_register_a_panel(run_player):
    run_player('''
        recording.eye = None
        app.recording_settings.enabled_plugins = {"EyeOverlayPlugin": True}
        assert window.video_workspace.get_dock("eye-video") is None
        app.toggle_plugin("EyeOverlayPlugin", False)
        assert app.recording_settings.enabled_plugins["EyeOverlayPlugin"]
    ''')


def test_shared_panel_api_releases_content_and_ignores_obsolete_close(run_player):
    run_player('''
        from PySide6.QtWidgets import QLabel

        workspace = window.video_workspace
        closed = []
        first_content = QLabel("First")
        workspace.register_panel("example", "Example", first_content,
                                 on_close=lambda: closed.append("first"))
        workspace.remove_dock("example")
        assert closed == []  # Reactions wait until the manager's signal returns.
        workspace.unregister_panel("example")
        assert first_content.parent() is None
        replacement = workspace.register_panel(
            "example", "Replacement", QLabel("Second"),
            on_close=lambda: closed.append("second"),
        )
        app.processEvents()
        assert closed == []
        assert workspace.get_dock("example") is replacement
        workspace.remove_dock("example")
        assert closed == []
        app.processEvents()
        assert closed == ["second"]
        assert workspace.get_dock("example") is None
    ''')
