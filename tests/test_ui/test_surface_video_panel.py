from textwrap import dedent

import pytest


SETUP_SURFACES = dedent('''
    from pupil_labs.neon_player.plugins.surface_tracking.tracked_surface import TrackedSurface
    from pupil_labs.neon_recording.timeseries.gaze import GazeArray

    window.show()
    QTest.qWait(50)
    # Initialize tracking without starting detector/locator subprocesses.
    app.args.job = []
    recording.gaze = np.zeros(2, dtype=[
        ("time", np.int64), ("point_x", np.float64), ("point_y", np.float64),
    ]).view(GazeArray)
    recording.gaze.time = np.array([0, 30_000_000], dtype=np.int64)
    recording.calibration = SimpleNamespace(
        scene_camera_matrix=np.eye(3), scene_distortion_coefficients=np.zeros(5),
    )
    app.toggle_plugin("SceneRendererPlugin", True)
    app.toggle_plugin("GazeDataPlugin", True)
    app.toggle_plugin("SurfaceTrackingPlugin", True)
    plugin = app.plugins_by_class["SurfaceTrackingPlugin"]
    surfaces = []
    for uid, name in (("surface-a", "Desk"), ("surface-b", "Wall")):
        surface = TrackedSurface()
        surface.uid = uid
        surface.name = name
        surface.preview_options.render_size = (640, 480)
        surface = TrackedSurface.from_dict(surface.to_dict(recursive=True))
        surfaces.append(surface)
    plugin.surfaces = list(surfaces)
    app.args.job = None
    workspace = window.video_workspace
    QTest.qWait(50)
''')


def test_view_surface_docks_once_to_right_of_scene_video(run_player):
    run_player(SETUP_SURFACES + dedent('''
        surface = surfaces[0]
        surface.view_surface()
        QTest.qWait(50)
        panel = workspace.get_dock("surface-view:surface-a")
        assert panel is not None
        assert panel.title == "Surface: Desk"
        assert panel.content.window() is window
        assert panel.content.view_widget.minimumWidth() == 400
        layout = workspace.save_layout()
        assert layout["main"]["orientation"] == "horizontal"
        assert layout["main"]["children"][0]["docks"] == ["scene-video"]
        assert layout["main"]["children"][1]["docks"] == ["surface-view:surface-a"]
        surface.view_surface()
        assert workspace.get_dock("surface-view:surface-a") is panel
    '''))


def test_view_surface_reveals_existing_tab_without_rearranging_it(run_player):
    run_player(SETUP_SURFACES + dedent('''
        from pyside6_docking import DockPosition

        for surface in surfaces:
            surface.view_surface()
        first = workspace.get_dock("surface-view:surface-a")
        workspace.float_dock("surface-view:surface-a")
        workspace.dock("surface-view:surface-b", DockPosition.CENTER,
                       relative_to="surface-view:surface-a")
        QTest.qWait(50)
        before = workspace.save_layout()
        assert not first.content.isVisible()
        surfaces[0].view_surface()
        QTest.qWait(50)
        assert workspace.get_dock("surface-view:surface-a") is first
        assert first.content.isVisible()
        after = workspace.save_layout()
        assert after["main"] == before["main"]
        assert after["floating"][0]["root"]["docks"] == before["floating"][0]["root"]["docks"]
        assert after["floating"][0]["root"]["active"] == "surface-view:surface-a"
        assert first.content.window() is not window
    '''))


def test_closing_surface_panel_keeps_tracking_and_restores_placement(run_player):
    run_player(SETUP_SURFACES + dedent('''
        import shiboken6
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QAbstractButton

        for surface in surfaces:
            surface.view_surface()
        first_content = workspace.get_dock("surface-view:surface-a").content
        other_panel = workspace.get_dock("surface-view:surface-b")
        workspace.float_dock("surface-view:surface-a")
        floating = first_content.window()
        floating.resize(850, 420)
        QTest.qWait(50)
        before = workspace.save_layout()["floating"][0]["geometry"]
        close = next(button for button in floating.findChildren(QAbstractButton)
                     if button.accessibleName() == "Close Surface: Desk")
        QTest.mouseClick(close, Qt.MouseButton.LeftButton)
        QTest.qWait(50)
        assert workspace.get_dock("surface-view:surface-a") is None
        assert not shiboken6.isValid(first_content)
        assert app.plugins_by_class["SurfaceTrackingPlugin"] is plugin
        assert plugin.surfaces == surfaces
        assert workspace.get_dock("surface-view:surface-b") is other_panel

        surfaces[0].view_surface()
        QTest.qWait(50)
        new = workspace.get_dock("surface-view:surface-a")
        assert new is not None and new.content is not first_content
        assert new.content.window() is not window
        after = workspace.save_layout()["floating"][0]["geometry"]
        # Window managers may adjust the requested top-level position.
        assert (after["width"], after["height"]) == (before["width"], before["height"])
    '''))


def test_surface_deletion_and_plugin_disable_release_panels(run_player):
    run_player(SETUP_SURFACES + dedent('''
        import shiboken6

        for surface in surfaces:
            surface.view_surface()
        first = workspace.get_dock("surface-view:surface-a").content
        second = workspace.get_dock("surface-view:surface-b").content
        plugin.surfaces = [surfaces[1]]
        QTest.qWait(50)
        assert workspace.get_dock("surface-view:surface-a") is None
        assert not shiboken6.isValid(first)
        assert workspace.get_dock("surface-view:surface-b").content is second
        surfaces[0].view_surface()
        assert workspace.get_dock("surface-view:surface-a") is None
        app.toggle_plugin("SurfaceTrackingPlugin", False)
        QTest.qWait(50)
        assert workspace.get_dock("surface-view:surface-b") is None
        assert not shiboken6.isValid(second)
        surfaces[1].view_surface()
        assert workspace.get_dock("surface-view:surface-b") is None
    '''))


def test_surface_views_stay_closed_after_restart_until_requested(run_player):
    run_player(SETUP_SURFACES + dedent('''
        surfaces[0].view_surface()
        workspace.float_dock("surface-view:surface-a")
        QTest.qWait(50)
        app.save_settings(force=True)
        app.unload()
        assert workspace.get_dock("surface-view:surface-a") is None
    '''))

    run_player(SETUP_SURFACES + dedent('''
        assert workspace.get_dock("surface-view:surface-a") is None
        surfaces[0].view_surface()
        QTest.qWait(50)
        assert workspace.get_dock("surface-view:surface-a").content.window() is not window
        assert workspace.get_dock("surface-view:surface-b") is None
    '''))


def test_view_surface_can_reopen_before_queued_close_finishes(run_player):
    run_player(SETUP_SURFACES + dedent('''
        surfaces[0].view_surface()
        original = workspace.get_dock("surface-view:surface-a")
        workspace.remove_dock("surface-view:surface-a")
        surfaces[0].view_surface()
        replacement = workspace.get_dock("surface-view:surface-a")
        assert replacement is not None and replacement is not original
        QTest.qWait(50)
        assert workspace.get_dock("surface-view:surface-a") is replacement
        assert replacement.content.isVisible()
    '''))


def test_restored_surface_controls_update_after_panel_reopens(run_player):
    run_player(SETUP_SURFACES + dedent('''
        from PySide6.QtCore import QSize
        from PySide6.QtTest import QSignalSpy

        surface = surfaces[0]
        surface.view_surface()
        workspace.remove_dock("surface-view:surface-a")
        QTest.qWait(50)
        surface.view_surface()
        changed = QSignalSpy(surface.changed)
        surface.preview_options.render_size = (800, 600)
        QTest.qWait(50)
        view = workspace.get_dock("surface-view:surface-a").content.view_widget
        assert view.source_size == QSize(800, 600)
        assert changed.count() == 1
    '''))


@pytest.mark.parametrize("placement", ["docked", "tabbed", "floating"])
def test_surface_rename_updates_existing_panel_without_rearranging(run_player, placement):
    run_player(SETUP_SURFACES + f"\nplacement = {placement!r}\n" + dedent('''
        from pyside6_docking import DockPosition
        from PySide6.QtWidgets import QAbstractButton, QTabBar

        surface = surfaces[0]
        surface.view_surface()
        if placement == "tabbed":
            surfaces[1].view_surface()
            workspace.dock("surface-view:surface-b", DockPosition.CENTER,
                           relative_to="surface-view:surface-a")
        elif placement == "floating":
            workspace.float_dock("surface-view:surface-a")
        QTest.qWait(50)
        panel = workspace.get_dock("surface-view:surface-a")
        content, parent = panel.content, panel.parentWidget()
        before = workspace.save_layout()
        surface.name = "Work desk"
        QTest.qWait(50)
        assert workspace.get_dock("surface-view:surface-a") is panel
        assert panel.title == "Surface: Work desk"
        assert panel.content is content and panel.parentWidget() is parent
        assert workspace.save_layout() == before
        tabs = next(bar for bar in content.window().findChildren(QTabBar)
                    if any(bar.tabData(i) == "surface-view:surface-a"
                           for i in range(bar.count())))
        index = next(i for i in range(tabs.count())
                     if tabs.tabData(i) == "surface-view:surface-a")
        assert tabs.tabText(index) == "Surface: Work desk"
        assert any(button.accessibleName() == "Close Surface: Work desk"
                   for button in tabs.findChildren(QAbstractButton))
        if placement == "floating":
            assert content.window().windowTitle() == "Surface: Work desk"

        # The existing duplicate-name validation must also protect the panel title.
        surface.name = "Wall"
        assert surface.name == "Work desk"
        assert panel.title == "Surface: Work desk"
    '''))


def test_closed_surface_rename_uses_current_title_when_reopened(run_player):
    run_player(SETUP_SURFACES + dedent('''
        surface = surfaces[0]
        surface.view_surface()
        workspace.remove_dock("surface-view:surface-a")
        # Rename while the queued close callback is still pending.
        surface.name = "Work desk"
        QTest.qWait(50)
        surface.name = "New desk"
        assert workspace.get_dock("surface-view:surface-a") is None
        surface.view_surface()
        assert workspace.get_dock("surface-view:surface-a").title == "Surface: New desk"
        surface.name = "Final desk"
        assert workspace.get_dock("surface-view:surface-a").title == "Surface: Final desk"
    '''))


def test_typing_surface_name_preserves_timeline_data_and_updates_labels(run_player):
    run_player(SETUP_SURFACES + dedent('''
        import pickle
        from PySide6.QtWidgets import QLineEdit

        surface = surfaces[0]
        surface.view_surface()
        timeline = window.timeline
        cache = plugin.get_cache_path()
        cache.mkdir(parents=True, exist_ok=True)
        cached_tracks = {
            "surface_visibility.pkl": [(0, 10_000_000), (20_000_000, 30_000_000)],
            "gazes.pkl": [(0, 5_000_000)],
        }
        for suffix, data in cached_tracks.items():
            (cache / f"{surface.uid}_{suffix}").write_bytes(pickle.dumps(data))
        plugin.add_visibility_timeline(surface)
        plugin.add_surface_gaze_timeline(surface)
        plots = [timeline.get_timeline_plot(f"{prefix}Desk")
                 for prefix in ("Surface: ", "Surface Gaze: ")]
        series = [tuple(plot.items) for plot in plots]
        legends = [timeline.timeline_legends[f"{prefix}Desk"]
                   for prefix in ("Surface: ", "Surface Gaze: ")]
        labels = [legend.parentItem().getItem(0, 0) for legend in legends]
        QTest.qWait(50)
        ranges = [plot.viewRange() for plot in plots]
        # A rename changes presentation only; cached data need not be available.
        for suffix in cached_tracks:
            (cache / f"{surface.uid}_{suffix}").unlink()
        name_edit = next(edit for edit in window.findChildren(QLineEdit)
                         if edit.text() == "Desk")
        name_edit.selectAll()
        for character, expected in zip("Table", ("T", "Ta", "Tab", "Tabl", "Table")):
            QTest.keyClicks(name_edit, character)
            assert surface.name == expected
            for i, prefix in enumerate(("Surface: ", "Surface Gaze: ")):
                name = f"{prefix}{expected}"
                assert timeline.get_timeline_plot(name) is plots[i]
                assert tuple(plots[i].items) == series[i]
                assert plots[i].row_name == name
                assert timeline.timeline_legends[name] is legends[i]
                assert labels[i].text == name
                assert np.allclose(plots[i].viewRange(), ranges[i]), (plots[i].viewRange(), ranges[i])
            assert workspace.get_dock("surface-view:surface-a").title == f"Surface: {expected}"
        for prefix in ("Surface: ", "Surface Gaze: "):
            assert timeline.get_timeline_plot(f"{prefix}Desk") is None

        name_edit.setText("Wall")
        assert surface.name == "Table"
        assert timeline.get_timeline_plot("Surface: Table") is plots[0]
        assert labels[0].text == "Surface: Table"
        assert workspace.get_dock("surface-view:surface-a").title == "Surface: Table"
    '''))
