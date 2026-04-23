import cv2
import logging
import typing as T
from csv import DictWriter
from pathlib import Path

import av
import numpy as np
from PySide6.QtCore import QSize
from PySide6.QtGui import QColorConstants, QPainter, QIcon, QImage
from qt_property_widgets.utilities import property_params, action, action_params

import pupil_labs.video as plv
from pupil_labs import neon_player
from pupil_labs.neon_player import Plugin, asset_path
from pupil_labs.neon_player.job_manager import ProgressUpdate
from pupil_labs.neon_player.utilities import qimage_from_frame, ndarray_from_qimage


class SceneRendererPlugin(Plugin):
    label = "Scene Renderer"

    DEFAULT_BRIGHTNESS = 0.0
    DEFAULT_CONTRAST = 1.0

    def __init__(self) -> None:
        super().__init__()
        self.render_layer = 0
        self.gray = QColorConstants.Gray

        self._show_frame_index = False
        self._brightness = self.DEFAULT_BRIGHTNESS
        self._contrast = self.DEFAULT_CONTRAST

    def render(self, painter: QPainter, time_in_recording: int) -> None:
        if self.recording is None:
            painter.drawText(100, 100, "No scene data available")
            return

        if self.is_time_gray(time_in_recording):
            painter.fillRect(
                0,
                0,
                self.recording.scene.width,
                self.recording.scene.height,
                self.gray,
            )
            return

        scene_frame = self.recording.scene.sample(
            [time_in_recording], method="backward"
        )[0]
        frame_img = cv2.convertScaleAbs(
            scene_frame.bgr, alpha=self._contrast, beta=self._brightness
        )

        painter.drawImage(0, 0, qimage_from_frame(frame_img))

        if self.show_frame_index:
            font = painter.font()
            font.setPointSize(font.pointSize() * 2)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(0, scene_frame.height, str(scene_frame.idx))

    @property
    def show_frame_index(self) -> bool:
        return self._show_frame_index

    @show_frame_index.setter
    def show_frame_index(self, value: bool) -> None:
        self._show_frame_index = value
        self.changed.emit()

    @property
    @property_params(min=0, max=100)
    def brightness(self) -> int:
        return self._brightness

    @brightness.setter
    def brightness(self, value: int) -> None:
        self._brightness = value

    @property
    @property_params(min=0.0, max=3.0)
    def contrast(self) -> float:
        return self._contrast

    @contrast.setter
    def contrast(self, value: float) -> None:
        self._contrast = value

    @action
    @action_params(compact=True, icon=QIcon(str(asset_path("reset_settings.svg"))))
    def reset_settings(self) -> None:
        self.show_frame_index = False
        self.brightness = self.DEFAULT_BRIGHTNESS
        self.contrast = self.DEFAULT_CONTRAST

    @action
    @action_params(compact=True, icon=QIcon(str(asset_path("export.svg"))))
    def export_scene_video(self, destination: Path = Path()) -> None:
        app = neon_player.instance()

        if not app.headless:
            return self.job_manager.run_background_action(
                "Scene Video Export", "SceneRendererPlugin.bg_export", destination
            )

        return self.bg_export(destination)

    def bg_export(self, destination: Path = Path()) -> T.Generator[ProgressUpdate, None, None]:
        logging.getLogger("pupil_labs.video.writer").setLevel(logging.ERROR)

        recording = self.app.recording

        gray_preamble = np.arange(
            recording.start_time, recording.scene.time[0], 1e9 // 30
        )
        gray_prologue = np.arange(
            recording.scene.time[-1] + 1e9 // 30, recording.stop_time, 1e9 // 30
        )
        combined_timestamps = np.concatenate((
            gray_preamble,
            recording.scene.time,
            gray_prologue,
        ))
        start_time, stop_time = neon_player.instance().recording_settings.export_window
        combined_timestamps = combined_timestamps[
            (combined_timestamps >= start_time) & (combined_timestamps <= stop_time)
        ]
        # Find any gaps in the timestamps that are greater than 1/30 of a second
        gaps = np.where(np.diff(combined_timestamps) > 1e9 // 30)[0]

        # fill the gaps with 30 hz timestamps
        for gap in reversed(gaps):
            gap_start = combined_timestamps[gap]
            gap_end = combined_timestamps[gap + 1] - 1e9 // 60
            gap_timestamps = np.arange(gap_start, gap_end, 1e9 // 30)
            combined_timestamps = np.concatenate((
                combined_timestamps[:gap],
                gap_timestamps,
                combined_timestamps[gap + 1 :],
            ))

        with (destination / "scene_timestamps.csv").open("w") as ts_file:
            writer = DictWriter(ts_file, fieldnames=["recording id", "timestamp"])
            writer.writeheader()
            for ts in combined_timestamps:
                writer.writerow({"recording id": recording.id, "timestamp": ts})

        frame_size = QSize(
            recording.scene.width or 1600, recording.scene.height or 1200
        )

        audio_frame_timestamps = recording.audio.time[
            (recording.audio.time >= start_time) & (recording.audio.time <= stop_time)
        ]
        audio_iterator = iter(recording.audio.sample(audio_frame_timestamps))
        audio_frame = next(audio_iterator)
        audio_frame_idx = 0

        with plv.Writer(destination / "scene.mp4") as writer:

            def write_audio_frame():
                nonlocal audio_frame, audio_frame_idx

                audio_rel_ts = (audio_frame.time - start_time) / 1e9
                plv_audio_frame = plv.AudioFrame(
                    audio_frame.av_frame,
                    index=audio_frame_idx,
                    time=audio_rel_ts,
                    source=""
                )
                writer.write_frame(plv_audio_frame)
                try:
                    audio_frame = next(audio_iterator)
                    audio_frame_idx += 1
                except StopIteration:
                    audio_frame = None

            for frame_idx, ts in enumerate(combined_timestamps):
                while audio_frame and audio_frame.time < ts:
                    write_audio_frame()

                rel_ts = (ts - combined_timestamps[0]) / 1e9

                frame = QImage(frame_size, QImage.Format.Format_BGR888)
                painter = QPainter(frame)
                painter.fillRect(
                    0,
                    0,
                    frame_size.width(),
                    frame_size.height(),
                    QColorConstants.Gray
                )
                self.render(painter, int(ts))
                painter.end()

                frame_pixels = ndarray_from_qimage(frame)
                av_frame = av.VideoFrame.from_ndarray(frame_pixels, format="bgr24")

                plv_frame = plv.VideoFrame(av_frame=av_frame, index=frame_idx, time=rel_ts, source="")
                writer.write_frame(plv_frame)

                progress = (frame_idx + 1) / len(combined_timestamps)
                yield ProgressUpdate(progress)

            while audio_frame:
                write_audio_frame()
