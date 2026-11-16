from collections import OrderedDict
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest
from pupil_labs.camera import Camera
from pupil_labs.marker_mapper.surface import Surface, normalized_corners
from pupil_labs.neon_recording import NeonRecording
from PySide6.QtCore import QTimer

from pupil_labs.neon_player.plugins.surface_tracking import SurfaceTrackingPlugin
from pupil_labs.neon_player.plugins.surface_tracking.surface_tracking import (
    DetectedMarker,
)
from pupil_labs.neon_player.plugins.surface_tracking.tracked_surface import (
    TrackedSurface,
)
from pupil_labs.neon_player.plugins.surface_tracking.ui import MarkerEditWidget
from pupil_labs.neon_player.ui.video_render_widget import VideoRenderWidget
from tests.mocks import mock_scene_timeseries

if TYPE_CHECKING:
    from tests.conftest import MockNeonPlayerApp


@pytest.fixture
def editable_surface(
    qapp: "MockNeonPlayerApp",
    monkeypatch: pytest.MonkeyPatch,
    mock_neon_recording: Callable[..., NeonRecording],
) -> Iterator[tuple[SurfaceTrackingPlugin, TrackedSurface]]:
    recording = mock_neon_recording(scene=mock_scene_timeseries([0, 2_000_000_000]))
    monkeypatch.setattr(
        recording.scene,
        "sample",
        lambda times: [SimpleNamespace(time=t) for t in times],
    )
    video_widget = VideoRenderWidget()
    timer = QTimer()
    monkeypatch.setattr(qapp, "recording", recording)
    monkeypatch.setattr(qapp, "current_ts", 0, raising=False)
    monkeypatch.setattr(qapp, "refresh_timer", timer, raising=False)
    monkeypatch.setattr(
        qapp, "main_window", SimpleNamespace(video_widget=video_widget), raising=False
    )
    plugin = SurfaceTrackingPlugin()
    monkeypatch.setattr(qapp, "plugins_by_class", {"SurfaceTrackingPlugin": plugin})
    plugin.camera = Camera(1600, 1200, np.eye(3))
    corners = normalized_corners().astype(np.float64)
    marker = DetectedMarker(42, corners * 100 + 100)
    plugin.markers_by_frame = [[marker], [DetectedMarker(42, marker.corners + 50)]]
    surface = TrackedSurface()
    surface.uid = "desk"
    surface.tracker_surface = Surface("Desk", OrderedDict([(42, corners)]))
    plugin._surfaces = [surface]
    plugin.surface_locations[surface.uid] = [
        surface.tracker_surface.localize(markers, plugin.camera)
        for markers in plugin.markers_by_frame
    ]
    surface.location = plugin.surface_locations[surface.uid][0]
    marker_widget = MarkerEditWidget("42")
    marker_widget.setParent(video_widget)
    marker_widget.hide()
    plugin.marker_edit_widgets[42] = marker_widget
    surface.marker_edit_changed.connect(lambda: plugin.on_marker_edit_changed(surface))
    yield plugin, surface
    timer.stop()
    surface.edit = False
    video_widget.deleteLater()
