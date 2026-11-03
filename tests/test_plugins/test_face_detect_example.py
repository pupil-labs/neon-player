import runpy
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Self

import numpy as np
import pytest
from PySide6.QtGui import QImage, QPainter
from PySide6.QtTest import QSignalSpy

from pupil_labs.neon_player import Plugin


@pytest.mark.usefixtures("qapp")
def test_face_mesh_and_aoi_use_tasks_landmarks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("mediapipe")
    landmark_module = import_module(
        "mediapipe.tasks.python.components.containers.landmark"
    )

    example = runpy.run_path(
        str(Path(__file__).parents[2] / "examples/plugins/face_detect.py")
    )
    plugin = example["FaceDetection"]()
    landmarks = [
        landmark_module.NormalizedLandmark(x=0.25, y=0.5, z=0.0) for _ in range(478)
    ]
    plugin.faces = [[landmarks]]
    recording = SimpleNamespace(scene=SimpleNamespace(width=64, height=48))
    monkeypatch.setattr(Plugin, "recording", property(lambda _self: recording))
    monkeypatch.setattr(plugin, "get_scene_idx_for_time", lambda _time: 0)
    gaze = SimpleNamespace(get_gazes_for_scene=lambda _idx: [])
    monkeypatch.setattr(Plugin, "get_instance_by_name", lambda _name: gaze)
    assert plugin.get_aoi(0, "right_eye")[0] == (16.0, 24.0)
    image = QImage(64, 48, QImage.Format.Format_ARGB32)
    image.fill(0)
    painter = QPainter(image)
    try:
        plugin.render(painter, 0)
    finally:
        painter.end()
    assert image.pixelColor(16, 24).alpha() > 0


@pytest.mark.usefixtures("qapp")
def test_detection_caches_tasks_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mp = pytest.importorskip("mediapipe")
    example = runpy.run_path(
        str(Path(__file__).parents[2] / "examples/plugins/face_detect.py")
    )
    plugin = example["FaceDetection"]()
    pixels = np.full((48, 64, 3), [255, 0, 0], dtype=np.uint8)

    class Scene(list):
        time = np.array([10_000_000_000, 10_000_000_001, 10_033_000_000])

    recording = SimpleNamespace(
        scene=Scene(SimpleNamespace(time=t, rgb=pixels) for t in Scene.time)
    )
    monkeypatch.setattr(Plugin, "recording", property(lambda _self: recording))
    monkeypatch.setattr(plugin, "get_cache_path", lambda: tmp_path)
    (tmp_path / "face_landmarker.task").touch()
    landmark_module = import_module(
        "mediapipe.tasks.python.components.containers.landmark"
    )
    landmarks = [landmark_module.NormalizedLandmark(x=0.25, y=0.5, z=0.0)]
    timestamps = []
    closed = []

    class Detector:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_args: object) -> None:
            closed.append(True)

        def detect_for_video(self, image: object, timestamp_ms: int) -> object:
            np.testing.assert_array_equal(image.numpy_view()[0, 0], [255, 0, 0])
            timestamps.append(timestamp_ms)
            return SimpleNamespace(
                face_landmarks=[landmarks] if timestamp_ms == 0 else []
            )

    monkeypatch.setattr(
        mp.tasks.vision.FaceLandmarker,
        "create_from_options",
        lambda _options: Detector(),
    )
    assert len(list(plugin.bg_detect())) == 3
    assert timestamps == [0, 1, 33]
    assert closed == [True]
    changed = QSignalSpy(plugin.changed)
    plugin._load_object_cache()
    assert plugin.faces[0][0][0].x == 0.25
    assert plugin.faces[1:] == [None, None]
    assert changed.count() == 1


@pytest.mark.usefixtures("qapp")
def test_headless_load_leaves_detection_to_explicit_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("mediapipe")
    example = runpy.run_path(
        str(Path(__file__).parents[2] / "examples/plugins/face_detect.py")
    )
    plugin = example["FaceDetection"]()
    monkeypatch.setattr(plugin, "get_cache_path", lambda: tmp_path)
    monkeypatch.setattr(
        plugin, "bg_detect", lambda: pytest.fail("Detection ran before the job started")
    )
    plugin.load_object_cache()
    assert plugin.faces is None
