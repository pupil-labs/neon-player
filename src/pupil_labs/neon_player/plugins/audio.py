import av
import fractions
import numpy as np
import typing as T

from pathlib import Path
from PySide6.QtCore import QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QFrame,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
)
from scipy.io import wavfile

from pupil_labs import neon_player
from pupil_labs.neon_player.job_manager import ProgressUpdate
from pupil_labs.neon_recording import NeonRecording
from pupil_labs.video.reader import StreamNotFound


def _prepare_audio_frame(
    audio_data: np.ndarray, format: str, time: float, stream: av.AudioStream
) -> av.AudioFrame:
    assert stream.time_base is not None

    audio_frame = av.AudioFrame.from_ndarray(
        audio_data, format=format, layout=stream.layout.name
    )
    audio_frame.sample_rate = stream.rate
    audio_frame.time_base = stream.time_base
    audio_frame.pts = int(time / audio_frame.time_base)
    audio_frame.dts = audio_frame.pts

    return audio_frame


class AudioPlugin(neon_player.Plugin):
    label = "Audio"

    def __init__(self) -> None:
        super().__init__()
        self.audio_output = QAudioOutput(self)
        self.player = QMediaPlayer(self)
        self.player.setAudioOutput(self.audio_output)
        self.player.mediaStatusChanged.connect(self.on_media_status_changed)
        self.recording_has_audio = False
        self.has_audio_at_ts = False

        self.app.playback_state_changed.connect(self.on_playback_state_changed)
        self.app.seeked.connect(self.on_user_seeked)
        self.app.speed_changed.connect(self.on_speed_changed)

        self.cache_file = Path()

        self.volume_button = VolumeButton()
        self.volume_button.setIconSize(QSize(32, 32))
        self.volume_button.setFixedSize(QSize(36, 36))
        self.volume_button.volume_changed.connect(self.on_volume_changed)
        self.get_timeline().toolbar_layout.insertWidget(1, self.volume_button)

    def on_disabled(self) -> None:
        self.player.stop()
        self.player.setSource(QUrl())
        self.get_timeline().remove_timeline_plot("Audio")

    def on_deleted(self) -> None:
        self.get_timeline().toolbar_layout.removeWidget(self.volume_button)
        self.volume_button.deleteLater()

    def on_media_status_changed(self, status: QMediaPlayer.MediaStatus) -> None:
        if status == QMediaPlayer.MediaStatus.LoadedMedia:
            self.sync_position()

    def sync_position(self) -> None:
        if not self.recording_has_audio:
            return

        position = self.app.current_ts
        try:
            rel_time_ms = round((position - self.recording.audio.time[0]) / 1e6)

            # Delay playback if the audio has not yet started, but will
            # start eventually with the current playback speed
            self.has_audio_at_ts = rel_time_ms >= 0
            if not self.has_audio_at_ts and self.app.playback_speed > 0:
                delay_ms = -rel_time_ms / self.app.playback_speed
                QTimer.singleShot(delay_ms, self.sync_and_start_playback)
                self.player.setPosition(0)
                return

            self.player.setPosition(rel_time_ms)
        except StreamNotFound:
            pass

    def on_speed_changed(self, speed: float) -> None:
        # NOTE: On MacOS, setting negative playback rates often leads to a bus error
        # and crashes the application. To prevent this, we stop playback instead of
        # setting a negative playback rate.
        if self.app.playback_speed > 0:
            self.player.setPlaybackRate(self.app.playback_speed)
        else:
            self.player.stop()

        # Restart playback if it was previously paused due to negative playback speed
        # and now has a positive speed
        self.sync_and_start_playback()

    def on_volume_changed(self, volume: float) -> None:
        self.audio_output.setVolume(volume)

    def on_user_seeked(self, position: int) -> None:
        self.sync_position()

    def on_playback_state_changed(self, is_playing: bool) -> None:
        if is_playing:
            self.sync_and_start_playback()
        else:
            self.player.stop()

    def sync_and_start_playback(self) -> None:
        self.sync_position()
        if self.has_audio_at_ts and self.app.is_playing and self.app.playback_speed > 0:
            self.player.play()

    def on_recording_loaded(self, recording: NeonRecording) -> None:
        self.cache_file = self.get_cache_path() / "audio.wav"

        self.recording_has_audio = False
        if not self.cache_file.exists():
            if self.app.headless:
                self.extract_audio()
                self.load_audio()

            else:
                job = self.job_manager.run_background_action(
                    "Extract audio", "AudioPlugin.extract_audio"
                )
                job.finished.connect(self.load_audio)

        else:
            self.load_audio()

    def load_audio(self) -> None:
        if not self.cache_file.exists():
            return

        self.recording_has_audio = True
        self.player.setSource(QUrl.fromLocalFile(str(self.cache_file)))
        self.on_speed_changed(self.app.playback_speed)
        self.on_playback_state_changed(self.app.is_playing)

        # load the audio data into a numpy array
        _, audio_data = wavfile.read(str(self.cache_file))
        timestamps = np.arange(len(audio_data)) / self.recording.audio.rate
        timestamps *= 1e9
        timestamps += self.recording.scene.time[0]

        data = np.column_stack((timestamps, audio_data))
        timeline = self.get_timeline()
        timeline.add_timeline_line("Audio", data)

    def extract_audio(self) -> T.Generator[ProgressUpdate, None, None]:
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)

        container = av.open(str(self.cache_file), "w")
        try:
            stream = container.add_stream("pcm_s16le", rate=self.recording.audio.rate)
        except StreamNotFound:
            yield ProgressUpdate(1.0)
            return

        # Setting these parameters once and reusing them below to ensure
        # consistency across frames that come from different mp4 files
        stream.layout = self.recording.audio[0].av_frame.layout
        stream.time_base = fractions.Fraction(1, stream.rate)

        start_time = self.recording.audio.time[0]
        next_expected_frame_time = None
        frame_duration = 0
        for frame in self.recording.audio:
            raw_audio = frame.to_ndarray()
            format = frame.av_frame.format

            if next_expected_frame_time is not None:
                # fill in gaps
                gap = (frame.time - next_expected_frame_time) / 1e9
                if gap > frame_duration:
                    samples_to_gen = int(gap * stream.sample_rate)
                    silence = np.zeros([raw_audio.shape[0], samples_to_gen])

                    silence_rel_time = (next_expected_frame_time - start_time) / 1e9
                    silence_frame = _prepare_audio_frame(
                        silence.astype(np.float32), format, silence_rel_time, stream
                    )
                    for packet in stream.encode(silence_frame):
                        container.mux(packet)

            frame_duration = raw_audio.shape[1] / stream.sample_rate
            next_expected_frame_time = frame.time + frame_duration * 1e9

            rel_time = (frame.time - start_time) / 1e9
            frame_copy = _prepare_audio_frame(raw_audio, format, rel_time, stream)
            for packet in stream.encode(frame_copy):
                container.mux(packet)

            yield ProgressUpdate(frame.idx / len(self.recording.audio))

        # Flush encoder
        for packet in stream.encode(None):
            container.mux(packet)

        container.close()


class VolumeButton(QPushButton):
    volume_changed = Signal(float)

    def __init__(self) -> None:
        super().__init__()

        self.setIcon(QPixmap(neon_player.asset_path("volume-3.svg")))
        self.popup: QFrame | None = None
        self._volume = 1.0
        self.clicked.connect(self.toggle_popup)

    def toggle_popup(self) -> None:
        if self.popup and self.popup.isVisible():
            self.popup.close()
            return

        self.popup = QFrame(self, f=Qt.WindowType.Popup)

        layout = QVBoxLayout(self.popup)
        layout.setContentsMargins(11, 5, 11, 5)

        slider = QSlider(Qt.Orientation.Vertical)
        slider.sliderMoved.connect(self.on_slider_moved)
        slider.setValue(int(self._volume * 100))
        slider.setMinimumHeight(140)
        slider.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        layout.addWidget(slider, alignment=Qt.AlignmentFlag.AlignCenter)

        def set_position() -> None:
            assert self.popup is not None
            pos = self.mapToGlobal(self.rect().topLeft())
            pos.setY(pos.y() - self.popup.height())
            self.popup.move(pos)

        self.popup.show()
        QTimer.singleShot(1, set_position)

    def on_slider_moved(self, value: int) -> None:
        self.setIcon(QPixmap(neon_player.asset_path(f"volume-{value//25}.svg")))
        self._volume = value / 100
        self.volume_changed.emit(self._volume)
