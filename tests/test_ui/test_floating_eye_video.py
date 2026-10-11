def test_eye_video_can_pop_out_and_redock_without_replacing_content(run_player):
    run_player('''
        from pyside6_docking import DockFeatures, DockPosition

        window.show()
        app.recording_settings.enabled_plugins = {"EyeOverlayPlugin": True}
        QTest.qWait(50)
        workspace = window.video_workspace
        panel = workspace.get_dock("eye-video")
        content = panel.content
        assert panel.features & DockFeatures.FLOATABLE
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QContextMenuEvent
        from PySide6.QtWidgets import QMenu

        point = content.rect().center()
        event = QContextMenuEvent(QContextMenuEvent.Reason.Mouse, point,
                                  content.mapToGlobal(point))
        app.sendEvent(content, event)
        QTest.qWait(20)
        menu = next(widget for widget in app.topLevelWidgets()
                    if isinstance(widget, QMenu) and widget.isVisible())
        pop_out = next(action for action in menu.actions() if action.text() == "Pop out")
        assert pop_out.isEnabled()
        pop_out.trigger()
        menu.close()
        QTest.qWait(50)
        assert content.window() is not window
        assert content.isVisible()
        workspace.dock("eye-video", DockPosition.RIGHT, relative_to="scene-video")
        QTest.qWait(50)
        assert content.window() is window
        assert workspace.get_dock("eye-video").content is content
        workspace.overlay_dock("eye-video", "scene-video")
        QTest.qWait(50)
        assert content.isVisible()
    ''')


def test_playback_shortcuts_trigger_once_with_docked_and_floating_focus(run_player):
    run_player('''
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QSignalSpy
        from pyside6_docking import DockPosition

        window.show()
        app.recording_settings.enabled_plugins = {"EyeOverlayPlugin": True}
        QTest.qWait(50)
        panel = window.video_workspace.get_dock("eye-video")
        actions = [*window.playback_actions,
            window.register_action("Playback/Next test position", Qt.Key.Key_S,
                                   lambda: app.seek_to(30_000_000)),
            window.register_action("Playback/Previous test position", Qt.Key.Key_A,
                                   lambda: app.seek_to(0)),
        ]
        keys = [
            (Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier),
            (Qt.Key.Key_Right, Qt.KeyboardModifier.NoModifier),
            (Qt.Key.Key_Left, Qt.KeyboardModifier.NoModifier),
            (Qt.Key.Key_Right, Qt.KeyboardModifier.ShiftModifier),
            (Qt.Key.Key_Left, Qt.KeyboardModifier.ShiftModifier),
            (Qt.Key.Key_Period, Qt.KeyboardModifier.ShiftModifier),
            (Qt.Key.Key_Comma, Qt.KeyboardModifier.ShiftModifier),
            (Qt.Key.Key_S, Qt.KeyboardModifier.NoModifier),
            (Qt.Key.Key_A, Qt.KeyboardModifier.NoModifier),
        ]
        for floating in (False, True, False):
            if floating:
                window.video_workspace.float_dock("eye-video")
            else:
                window.video_workspace.dock("eye-video", DockPosition.RIGHT,
                                             relative_to="scene-video")
            focused_window = panel.content.window()
            focused_window.activateWindow()
            focused_window.raise_()
            assert QTest.qWaitForWindowActive(focused_window)
            panel.content.setFocus()
            QTest.qWait(50)
            for action, (key, modifiers) in zip(actions, keys):
                spy = QSignalSpy(action.triggered)
                QTest.keyClick(panel.content, key, modifiers)
                assert spy.count() == 1, (floating, action.text(), spy.count())
                if key == Qt.Key.Key_Space:
                    assert app.refresh_timer.isActive()
                if key == Qt.Key.Key_S:
                    assert app.current_ts == 30_000_000
                elif key == Qt.Key.Key_A:
                    assert app.current_ts == 0
                app.set_playback_state(False)
    ''')


def test_floating_close_disable_and_recording_changes_retire_windows(run_player):
    run_player('''
        from PySide6.QtCore import QCoreApplication, QEvent
        from shiboken6 import isValid

        window.show()
        workspace = window.video_workspace
        retired = []
        closed = []
        original_sample = recording.eye.sample
        def sample(timestamps):
            assert not closed, "Eye sample accessed after recording close"
            return original_sample(timestamps)
        recording.eye.sample = sample
        recording.close = lambda: closed.append(True)

        def flush():
            app.processEvents()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            app.processEvents()
            assert all(not isValid(w) or not w.isVisible() for w in retired)

        for _ in range(3):
            app.recording_settings.enabled_plugins = {"EyeOverlayPlugin": True}
            workspace.float_dock("eye-video")
            QTest.qWait(20)
            floating = workspace.get_dock("eye-video").content.window()
            retired.append(floating)
            floating.close()
            flush()
            assert not app.recording_settings.enabled_plugins["EyeOverlayPlugin"]
            assert "EyeOverlayPlugin" not in app.plugins_by_class
            assert workspace.get_dock("eye-video") is None

        app.recording_settings.enabled_plugins = {"EyeOverlayPlugin": True}
        workspace.float_dock("eye-video")
        retired.append(workspace.get_dock("eye-video").content.window())
        app.toggle_plugin("EyeOverlayPlugin", False)
        flush()
        assert app.recording_settings.enabled_plugins["EyeOverlayPlugin"]

        app.toggle_plugins_by_settings()
        workspace.float_dock("eye-video")
        retired.append(workspace.get_dock("eye-video").content.window())
        app.unload()
        flush()
        assert closed == [True]
        app.position_changed.emit(0)  # No retired panel may sample closed media.
        assert workspace.get_dock("eye-video") is None
        assert app.recording_settings.enabled_plugins["EyeOverlayPlugin"]

        # Load another recording through the existing application activation seam.
        replacement_recording = SimpleNamespace(
            scene=SceneFrames(), eye=EyeFrames(), start_time=0,
            stop_time=30_000_000, _rec_dir=task_home, close=lambda: None,
        )
        app.recording = replacement_recording
        window.on_recording_loaded(replacement_recording)
        window.on_recording_opened()
        app.toggle_plugins_by_settings()
        workspace.float_dock("eye-video")
        retired.append(workspace.get_dock("eye-video").content.window())
        app.aboutToQuit.emit()
        flush()
        assert "EyeOverlayPlugin" not in app.plugins_by_class
        assert app.recording is None
    ''')


def test_native_video_and_surface_mapping_survive_all_eye_arrangements(run_player):
    run_player('''
        from PySide6.QtCore import QPoint, QPointF, Qt
        from PySide6.QtGui import QMouseEvent
        from PySide6.QtWidgets import QLabel, QTabBar, QWidget
        from pyside6_docking import DockPosition
        from pupil_labs.neon_player.plugins.surface_tracking.ui import SurfaceHandle

        window.show()
        app.recording_settings.enabled_plugins = {
            "SceneRendererPlugin": True, "EyeOverlayPlugin": True,
        }
        QTest.qWait(100)
        workspace = window.video_workspace
        panel = workspace.get_dock("eye-video")
        content = panel.content
        sample = recording.eye.sample
        def eye_sample(timestamps):
            frames = sample(timestamps)
            frames[0].gray.fill(255 if timestamps[0] == 0 else 80)
            return frames
        recording.eye.sample = eye_sample
        handle = SurfaceHandle(None, (0, 0))
        handle.setParent(window.video_widget)
        handle.resize(20, 20)
        handle.set_scene_pos(np.array([320.0, 240.0]))

        def drag(widget, delta):
            start = QPointF(widget.rect().center())
            QTest.mousePress(widget, Qt.MouseButton.LeftButton, pos=start.toPoint())
            move = QMouseEvent(QMouseEvent.Type.MouseMove, start + delta,
                widget.mapToGlobal((start + delta).toPoint()),
                Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier)
            app.sendEvent(widget, move)
            QTest.mouseRelease(widget, Qt.MouseButton.LeftButton,
                               pos=(start + delta).toPoint())
            QTest.qWait(30)

        header = next(label for label in window.findChildren(QLabel)
                      if label.accessibleName() == "Move overlay Eye video")
        before = workspace.save_layout()["overlays"][0]
        drag(header, QPointF(20, 15))
        after = workspace.save_layout()["overlays"][0]
        assert after["x"] > before["x"] and after["y"] > before["y"]
        resize = next(widget for widget in window.findChildren(QWidget)
                      if widget.accessibleName() == "Resize overlay Eye video: right bottom")
        drag(resize, QPointF(30, 20))
        resized = workspace.save_layout()["overlays"][0]
        assert resized["width"] > after["width"]
        assert resized["height"] > after["height"]

        def exported(ts):
            image = QImage(640, 480, QImage.Format.Format_RGB32)
            painter = QPainter(image)
            app.render_to(painter, ts)
            painter.end()
            return image
        reference = {ts: exported(ts) for ts in (0, 30_000_000)}

        for placement in ("overlay", "split", "tab", "floating", "redocked"):
            if placement == "split" or placement == "redocked":
                workspace.dock("eye-video", DockPosition.RIGHT, relative_to="scene-video")
            elif placement == "tab":
                workspace.dock("eye-video", DockPosition.CENTER, relative_to="scene-video")
                QTest.qWait(50)
                tabs = next(bar for bar in window.findChildren(QTabBar)
                            if any(bar.tabText(i) == "Eye video" for i in range(bar.count())))
                tabs.setCurrentIndex(next(i for i in range(tabs.count())
                                          if tabs.tabText(i) == "Eye video"))
                assert not window.video_widget.isVisible()
                assert content.isVisible()
                tabs.setCurrentIndex(next(i for i in range(tabs.count())
                                          if tabs.tabText(i) == "Scene video"))
                assert window.video_widget.isVisible()
            elif placement == "floating":
                workspace.float_dock("eye-video")
            if placement == "floating":
                # Native window captures include occluding sibling windows.
                # Keep the eye window clear of the scene pixel under inspection.
                content.window().move(window.frameGeometry().topRight() + QPoint(20, 0))
            QTest.qWait(50)
            assert workspace.get_dock("eye-video").content is content
            for ts, expected in ((0, "#ff0000"), (30_000_000, "#00ff00")):
                app.seek_to(ts)
                QTest.qWait(80)
                if placement != "tab":
                    eye = content.grab().toImage()
                    assert eye.pixelColor(eye.width() // 4, eye.height() // 2).red() == (255 if ts == 0 else 80)
                scene = window.video_widget
                scene.repaint()
                # Screen captures can include the native cursor left by drags.
                QTest.mouseMove(window, QPoint(5, 5))
                QTest.qWait(30)
                window.activateWindow()
                window.raise_()
                assert QTest.qWaitForWindowActive(window)
                image = window.screen().grabWindow(window.winId()).toImage()
                image.save(str(task_home / (placement + "-" + str(ts) + ".png")))
                point = scene.mapTo(window, QPointF(
                    scene.offset.x() + 480 * scene.scale,
                    scene.offset.y() + 240 * scene.scale).toPoint())
                ratio = image.devicePixelRatio()
                assert image.pixelColor(int(point.x() * ratio), int(point.y() * ratio)).name() == expected, (placement, ts, image.pixelColor(int(point.x() * ratio), int(point.y() * ratio)).name(), scene.size(), scene.scale, scene.offset, scene.isValid())
                assert exported(ts) == reference[ts]
                mapped = scene.map_point(handle.mapToParent(handle.rect().center()))
                assert abs(mapped.x() - 320) * scene.scale < 2 and abs(mapped.y() - 240) * scene.scale < 2, (placement, mapped, scene.scale)
            drag(handle, QPointF(15, 10))
            assert handle.new_pos.x() > 320 and handle.new_pos.y() > 240
            handle.position_changed_debounce_timer.stop()
            handle.set_scene_pos(np.array([320.0, 240.0]))
        workspace.overlay_dock("eye-video", "scene-video")
        app.seek_to(0)
        QTest.qWait(50)
        assert content.isVisible()
    ''')
