import json
import weakref
from collections.abc import Iterable, Sequence
from multiprocessing import get_context
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock

import cv2
import numpy as np
import numpy.typing as npt
import pupil_labs.neon_recording as nr
import pytest
from pupil_labs.video import Writer

from pupil_labs.neon_player.plugins import fixations


class _SceneFrame:
    def __init__(self, index: int, time: int, gray: npt.NDArray[np.uint8]) -> None:
        self.index = index
        self.time = time
        self._gray = gray
        self.gray_reads = 0

    @property
    def gray(self) -> npt.NDArray[np.uint8]:
        self.gray_reads += 1
        return self._gray


class _SceneFrames:
    def __init__(self, frames: Iterable[_SceneFrame]) -> None:
        self._frames = list(frames)
        self._time = np.array([frame.time for frame in self._frames], dtype=np.int64)
        self.accesses: list[int] = []
        self.timestamp_reads = 0

    @property
    def time(self) -> npt.NDArray[np.int64]:
        self.timestamp_reads += 1
        return self._time

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, key: int) -> _SceneFrame:
        frame = self._frames[key]
        self.accesses.append(frame.index)
        return frame


def _flow_image() -> npt.NDArray[np.uint8]:
    image = np.random.default_rng(42).integers(0, 256, (512, 512), dtype=np.uint8)
    return cv2.GaussianBlur(image, (5, 5), 1)


def _flow_recording(
    monkeypatch: pytest.MonkeyPatch,
    *,
    scene_times: Sequence[int] = (0, 50_000_000, 100_000_000, 150_000_000),
    gaze_times: Sequence[int] = (50_000_000,),
    fixation_intervals: Sequence[tuple[int, int]] = ((0, 100_000_000),),
) -> Mock:
    image = _flow_image()
    frames = _SceneFrames(
        _SceneFrame(
            index=index,
            time=timestamp,
            gray=cv2.warpAffine(
                image, np.float32([[1, 0, index], [0, 1, 0]]), (512, 512)
            ),
        )
        for index, timestamp in enumerate(scene_times)
    )
    gaze = np.array(
        [(timestamp, [256.0, 256.0]) for timestamp in gaze_times],
        dtype=[("time", "i8"), ("point", "f4", (2,))],
    ).view(np.recarray)
    recording = Mock(
        scene=frames,
        gaze=gaze,
        fixations=[
            SimpleNamespace(start_time=start, stop_time=stop)
            for start, stop in fixation_intervals
        ],
    )

    monkeypatch.setattr(
        fixations.FixationsPlugin, "require_recording", lambda _self: recording
    )
    return recording


def _run_flow(monkeypatch: pytest.MonkeyPatch, cache_path: Path) -> list[float]:
    monkeypatch.setattr(
        fixations.FixationsPlugin, "require_cache_path", lambda _self: cache_path
    )
    return [update.progress for update in fixations.FixationsPlugin().bg_optic_flow()]


def test_optic_flow_cache_schema_and_offsets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _flow_recording(monkeypatch)
    cache = tmp_path / "optic_flow_offsets.npz"

    progress = _run_flow(monkeypatch, tmp_path)

    with np.load(cache) as data:
        assert data.files == ["flow_data"]
        values = data["flow_data"]
    assert values.shape == (3, 4)
    np.testing.assert_array_equal(values[:, :2], [[0, 0], [0, 2], [0, 3]])
    np.testing.assert_allclose(values[:, 2:], [[1, 0], [-1, 0], [-2, 0]], atol=0.02)
    assert progress[0] == 0
    assert progress[-1] == 1.0
    assert progress == sorted(set(progress))
    assert len(progress) > 2
    assert list(tmp_path.iterdir()) == [cache]


def test_optic_flow_empty_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recording = _flow_recording(monkeypatch)
    recording.fixations = []
    cache = tmp_path / "optic_flow_offsets.npz"

    progress = _run_flow(monkeypatch, tmp_path)

    with np.load(cache) as data:
        assert data["flow_data"].shape == (0, 4)
    assert progress == [1.0]


@pytest.mark.parametrize(
    ("scene_times", "gaze_times", "fixation_interval", "expected"),
    [
        (
            [0, 100_000_000, 200_000_000],
            [50_000_000],
            (0, 100_000_000),
            [[0, 1, -1, 0], [0, 2, -2, 0]],
        ),
        (
            [0, 200_000_000],
            [100_000_000],
            (0, 100_000_000),
            [[0, 1, -1, 0]],
        ),
        (
            [0, 200_000_002],
            [100_000_001],
            (0, 100_000_001),
            [],
        ),
        (
            [0, 50_000_000, 100_000_000, 150_000_000],
            [-1, 0, 50_000_000, 100_000_000, 100_000_001],
            (0, 100_000_000),
            [[0, 0, 1, 0], [0, 2, -1, 0], [0, 3, -2, 0]],
        ),
        (
            [-1, 0, 50_000_000, 2_099_999_999, 2_100_000_000],
            [50_000_000],
            (0, 100_000_000),
            [[0, 1, 1, 0], [0, 3, -1, 0]],
        ),
        (
            [0],
            [50_000_000],
            (25_000_000, 50_000_000),
            [],
        ),
        ([], [50_000_000], (0, 100_000_000), []),
        (
            [
                1_700_000_000_000_000_000,
                1_700_000_000_000_000_001,
                1_700_000_000_100_000_001,
                1_700_000_002_100_000_000,
                1_700_000_002_100_000_002,
            ],
            [1_700_000_000_100_000_001],
            (1_700_000_000_000_000_001, 1_700_000_000_100_000_001),
            [[0, 1, 1, 0], [0, 0, 2, 0]],
        ),
    ],
    ids=[
        "nearest-tie-uses-first",
        "reference-exactly-100ms-away",
        "reference-outside-100ms",
        "inclusive-gaze-interval-uses-middle",
        "scene-interval-boundaries",
        "empty-scene-interval",
        "empty-scene-stream",
        "epoch-nanoseconds-preserve-boundary-rounding",
    ],
)
def test_optic_flow_reference_selection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    scene_times: list[int],
    gaze_times: list[int],
    fixation_interval: tuple[int, int],
    expected: list[list[float]],
) -> None:
    recording = _flow_recording(
        monkeypatch,
        scene_times=scene_times,
        gaze_times=gaze_times,
        fixation_intervals=[fixation_interval],
    )
    cache = tmp_path / "optic_flow_offsets.npz"

    _run_flow(monkeypatch, tmp_path)

    with np.load(cache) as data:
        values = data["flow_data"]
    expected_values = np.array(expected).reshape((-1, 4))
    np.testing.assert_array_equal(values[:, :2], expected_values[:, :2])
    np.testing.assert_allclose(values[:, 2:], expected_values[:, 2:], atol=0.02)
    if not expected:
        assert recording.scene.accesses == []


@pytest.mark.parametrize("has_fixations", [False, True])
def test_optic_flow_without_usable_gaze_does_not_load_scene(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, has_fixations: bool
) -> None:
    recording = _flow_recording(
        monkeypatch,
        gaze_times=[],
        fixation_intervals=[(0, 100_000_000)] if has_fixations else [],
    )
    type(recording).scene = PropertyMock(side_effect=AssertionError("scene was loaded"))
    cache = tmp_path / "optic_flow_offsets.npz"

    progress = _run_flow(monkeypatch, tmp_path)

    with np.load(cache) as data:
        assert data["flow_data"].shape == (0, 4)
    assert progress[-1] == 1.0


def test_optic_flow_reuses_timestamp_metadata_across_fixations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recording = _flow_recording(
        monkeypatch, fixation_intervals=[(0, 100_000_000), (50_000_000, 100_000_000)]
    )
    cache = tmp_path / "optic_flow_offsets.npz"

    progress = _run_flow(monkeypatch, tmp_path)

    with np.load(cache) as data:
        values = data["flow_data"]
    np.testing.assert_array_equal(
        values[:, :2], [[0, 0], [0, 2], [0, 3], [1, 2], [1, 3]]
    )
    np.testing.assert_allclose(
        values[:, 2:], [[1, 0], [-1, 0], [-2, 0], [-1, 0], [-2, 0]], atol=0.02
    )
    assert recording.scene.timestamp_reads == 1
    assert set(recording.scene.accesses) == {0, 1, 2, 3}
    assert [frame.gray_reads for frame in recording.scene._frames] == [1, 1, 1, 1]
    assert progress[0] == 0
    assert progress[-1] == 1.0
    assert progress == sorted(set(progress))


@pytest.mark.parametrize("frame_count", [150, 151])
def test_optic_flow_grayscale_memory_is_bounded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, frame_count: int
) -> None:
    recording = _flow_recording(
        monkeypatch, fixation_intervals=[(0, 100_000_000), (0, 100_000_000)]
    )
    frames = [
        _SceneFrame(index, index * 10_000_000, np.zeros((1, 1), dtype=np.uint8))
        for index in range(frame_count)
    ]
    recording.scene = _SceneFrames(frames)
    images: list[weakref.ReferenceType[npt.NDArray[np.uint8]]] = []

    def read_gray(frame: _SceneFrame) -> npt.NDArray[np.uint8]:
        frame.gray_reads += 1
        image = frame._gray.copy()
        images.append(weakref.ref(image))
        assert sum(reference() is not None for reference in images) <= 2
        return image

    monkeypatch.setattr(_SceneFrame, "gray", property(read_gray))
    monkeypatch.setattr(
        fixations, "calc_optic_flow", lambda _previous, _current, point: point
    )
    cache = tmp_path / "optic_flow_offsets.npz"

    _run_flow(monkeypatch, tmp_path)

    with np.load(cache) as data:
        values = data["flow_data"]
    assert values.shape == (2 * (frame_count - 1), 4)
    np.testing.assert_array_equal(values[:, 2:], 0)
    assert all(frame.gray_reads <= 2 for frame in frames)
    assert sum(frame.gray_reads for frame in frames) < 2 * frame_count


def test_optic_flow_batches_match_scalar_tracks_across_chunks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recording = _flow_recording(
        monkeypatch,
        scene_times=[index * 50_000_000 for index in range(14)],
        gaze_times=[index * 50_000_000 for index in range(8)],
        fixation_intervals=[
            ((index % 4) * 50_000_000, (index % 4 + 2) * 50_000_000)
            for index in range(35)
        ],
    )
    recording.gaze.point[:] = [[200 + index * 7, 220 + index * 3] for index in range(8)]
    for frame in recording.scene._frames:
        frame.index += 100
    scalar_rows = []
    track = fixations.calc_optic_flow
    for fidx, fixation in enumerate(recording.fixations):
        gaze = recording.gaze[
            (fixation.start_time <= recording.gaze.time)
            & (recording.gaze.time <= fixation.stop_time)
        ]
        reference = gaze[len(gaze) // 2]
        start, stop = np.searchsorted(
            recording.scene.time, [fixation.start_time, fixation.stop_time + 2e9]
        )
        ref_index = start + np.argmin(
            np.abs(recording.scene.time[start:stop] - reference.time)
        )
        for indices in (
            range(ref_index - 1, start - 1, -1),
            range(ref_index + 1, stop),
        ):
            point = reference.point
            image = recording.scene[ref_index].gray
            for index in indices:
                frame = recording.scene[index]
                next_image = frame.gray
                point = track(image, next_image, point)
                offset = reference.point - point
                scalar_rows.append([fidx, frame.index, *offset])
                image = next_image
    batch_sizes = []

    def track_batch(
        previous: npt.NDArray[np.uint8],
        current: npt.NDArray[np.uint8],
        points: npt.NDArray[np.float32 | np.float64],
    ) -> npt.NDArray[np.float32 | np.float64]:
        batch_sizes.append(len(points))
        return track(previous, current, points)

    monkeypatch.setattr(fixations, "calc_optic_flow", track_batch)
    cache = tmp_path / "optic_flow_offsets.npz"

    progress = _run_flow(monkeypatch, tmp_path)

    with np.load(cache) as data:
        values = data["flow_data"]
    np.testing.assert_array_equal(values[:, :2], np.array(scalar_rows)[:, :2])
    np.testing.assert_allclose(
        values[:, 2:], np.array(scalar_rows)[:, 2:], rtol=0, atol=1e-5
    )
    assert max(batch_sizes) == 32
    assert 3 in batch_sizes
    assert len(batch_sizes) < len(scalar_rows)
    assert progress == sorted(set(progress))
    assert progress[-1] == 1.0


def test_optic_flow_skips_gaps_between_fixations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recording = _flow_recording(
        monkeypatch,
        scene_times=[index * 1_000_000_000 for index in range(100)],
        gaze_times=[0, 99_000_000_000],
        fixation_intervals=[(0, 0), (99_000_000_000, 99_000_000_000)],
    )
    monkeypatch.setattr(
        fixations, "calc_optic_flow", lambda _previous, _current, points: points
    )

    _run_flow(monkeypatch, tmp_path)

    assert set(recording.scene.accesses) == {0, 1}
    assert all(frame.gray_reads == 0 for frame in recording.scene._frames[2:])


def _write_flow_recording(recording_path: Path) -> None:
    recording_path.mkdir()
    (recording_path / "info.json").write_text(json.dumps({"app_version": "2.9.31"}))
    image = _flow_image()
    for part in range(2):
        scene_path = recording_path / f"Neon Scene Camera v1 ps{part + 1}.mp4"
        with Writer(scene_path, fps=20, lossless=True) as writer:
            for index in range(part * 2, part * 2 + 2):
                writer.write_image(
                    cv2.warpAffine(
                        image, np.float32([[1, 0, index], [0, 1, 0]]), (512, 512)
                    )
                )
        np.array(
            [part * 100_000_000, part * 100_000_000 + 50_000_000], dtype="<i8"
        ).tofile(scene_path.with_suffix(".time"))
    np.array([[256, 256]], dtype="<f4").tofile(recording_path / "gaze ps1.raw")
    np.array([50_000_000], dtype="<i8").tofile(recording_path / "gaze ps1.time")
    fixation_data = np.zeros(
        2,
        dtype=[
            ("event_type", "<i4"),
            ("start_timestamp_ns", "<i8"),
            ("end_timestamp_ns", "<i8"),
            ("points_and_metrics", "<f4", (10,)),
        ],
    )
    fixation_data["event_type"] = 1
    fixation_data["end_timestamp_ns"] = 100_000_000
    fixation_data.tofile(recording_path / "fixations ps1.raw")
    np.array([0, 0], dtype="<i8").tofile(recording_path / "fixations ps1.time")


def _run_real_flow(cache_path: Path) -> list[float]:
    with nr.load(cache_path / "recording") as recording:
        plugin = Mock(spec=fixations.FixationsPlugin)
        plugin.require_recording.return_value = recording
        plugin.require_cache_path.return_value = cache_path
        return [
            update.progress
            for update in fixations.FixationsPlugin.bg_optic_flow(plugin)
        ]


def test_optic_flow_generator_with_real_recording(tmp_path: Path) -> None:
    recording_path = tmp_path / "recording"
    _write_flow_recording(recording_path)
    with get_context("spawn").Pool(1) as pool:
        progress = pool.apply_async(_run_real_flow, (tmp_path,)).get(timeout=30)

    with np.load(tmp_path / "optic_flow_offsets.npz") as data:
        values = data["flow_data"]
    np.testing.assert_array_equal(
        values[:, :2], [[0, 0], [0, 2], [0, 3], [1, 0], [1, 2], [1, 3]]
    )
    np.testing.assert_allclose(values[:, 2:], [[1, 0], [-1, 0], [-2, 0]] * 2, atol=0.02)
    assert progress == sorted(set(progress))
    assert progress[-1] == 1.0
