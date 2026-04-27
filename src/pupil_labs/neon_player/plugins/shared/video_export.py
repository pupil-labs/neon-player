import av
import logging
import numpy as np
import typing as T

from csv import DictWriter
from pathlib import Path
from PySide6.QtCore import QSize
from PySide6.QtGui import QColorConstants, QPainter, QImage

import pupil_labs.video as plv
from pupil_labs.neon_player.job_manager import ProgressUpdate
from pupil_labs.neon_player.utilities import ndarray_from_qimage
from pupil_labs.neon_recording import NeonRecording


class BackgroundVideoExportMixin:
    """
    This mixin provides a method `bg_export_video` for exporting a video
    in the background, handling multiple aspects:

     * gray frames are added before and after the scene frames to match
       the recording start and stop time
     * any gaps in the scene frames that are larger than 1/fps of a second
       are filled with gray frames to maintain a consistent frame rate
     * audio frames are interleaved with the video frames in the output video

    The plugin using this mixin needs to implement and provide a render method
    for generating the video frames.
    """

    @staticmethod
    def _prepare_timestamps(
        recording: NeonRecording,
        export_window: tuple[int, int],
        fps: int = 30
    ) -> np.ndarray:
        # Add timestamps for gray frames that extend the video to match
        # recording start and stop time
        gray_preamble = np.arange(
            recording.start_time, recording.scene.time[0], 1e9 // fps
        )
        gray_prologue = np.arange(
            recording.scene.time[-1] + 1e9 // fps, recording.stop_time, 1e9 // fps
        )
        combined_timestamps = np.concatenate((
            gray_preamble,
            recording.scene.time,
            gray_prologue,
        ))

        # Filter timestamps according to the export window
        start_time, stop_time = export_window
        combined_timestamps = combined_timestamps[
            (combined_timestamps >= start_time) & (combined_timestamps <= stop_time)
        ]

        # Find any gaps in the timestamps that are greater than 1/fps of a second
        gaps = np.where(np.diff(combined_timestamps) > 1e9 // fps)[0]

        # Fill the gaps with timestamps at fps frequency
        for gap in reversed(gaps):
            gap_start = combined_timestamps[gap]
            gap_end = combined_timestamps[gap + 1] - 1e9 // (2 * fps)
            gap_timestamps = np.arange(gap_start, gap_end, 1e9 // fps)
            combined_timestamps = np.concatenate((
                combined_timestamps[:gap],
                gap_timestamps,
                combined_timestamps[gap + 1 :],
            ))

        return combined_timestamps

    def bg_export_video(
        self,
        recording: NeonRecording,
        export_window: tuple[int, int],
        render_fn: T.Callable[[QPainter, int], None],
        destination: Path,
        output_video_filename: str,
        output_timestamps_filename: str
    ) -> T.Generator[ProgressUpdate, None, None]:
        logging.getLogger("pupil_labs.video.writer").setLevel(logging.ERROR)

        combined_timestamps = self._prepare_timestamps(recording, export_window)
        with (destination / output_timestamps_filename).open("w") as ts_file:
            writer = DictWriter(ts_file, fieldnames=["recording id", "timestamp"])
            writer.writeheader()
            for ts in combined_timestamps:
                writer.writerow({"recording id": recording.id, "timestamp": ts})

        frame_size = QSize(
            recording.scene.width or 1600, recording.scene.height or 1200
        )

        start_time, stop_time = export_window
        audio_frame_timestamps = recording.audio.time[
            (recording.audio.time >= start_time) & (recording.audio.time <= stop_time)
        ]
        audio_iterator = iter(recording.audio.sample(audio_frame_timestamps))
        audio_frame = next(audio_iterator)
        audio_frame_idx = 0

        with plv.Writer(destination / output_video_filename) as writer:

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
                render_fn(painter, int(ts))
                painter.end()

                frame_pixels = ndarray_from_qimage(frame)
                av_frame = av.VideoFrame.from_ndarray(frame_pixels, format="bgr24")

                plv_frame = plv.VideoFrame(av_frame=av_frame, index=frame_idx, time=rel_ts, source="")
                writer.write_frame(plv_frame)

                progress = (frame_idx + 1) / len(combined_timestamps)
                yield ProgressUpdate(progress)

            while audio_frame:
                write_audio_frame()
