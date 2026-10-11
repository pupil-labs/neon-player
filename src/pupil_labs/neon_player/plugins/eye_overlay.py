from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QColor, QPainter
from qt_property_widgets.utilities import property_params

from pupil_labs import neon_player
from pupil_labs.neon_player.ui.eye_video_widget import EyeVideoWidget
from pupil_labs.neon_player.utilities import qimage_from_frame
from pupil_labs.neon_recording import NeonRecording


class EyeOverlayPlugin(neon_player.Plugin):
    label = "Eye Overlay"

    def __init__(self) -> None:
        super().__init__()
        self.render_layer = 9
        self._offset_x = 0.02
        self._offset_y = 0.02
        self._scale = 1.0
        self._opacity = 1.0
        self._border_width = 3
        self._border_color = QColor("#aaaaaa")
        self._eye_video: EyeVideoWidget | None = None

    def on_recording_loaded(self, recording: NeonRecording) -> None:
        if self.app.headless or recording.eye is None:
            return
        eye_width, eye_height = recording.eye.width, recording.eye.height
        scene_width, scene_height = recording.scene.width, recording.scene.height
        if not eye_width or not eye_height or not scene_width or not scene_height:
            return
        self._eye_video = EyeVideoWidget()
        width = min(0.9, eye_width / scene_width)
        height = min(0.9, eye_height / scene_height + 0.08)
        self.app.main_window.video_workspace.register_panel(
            "eye-video", "Eye video", self._eye_video,
            on_close=self._on_panel_closed,
            overlay=QRectF(0.02, 0.02, width, height),
        )
        self.app.position_changed.connect(self._update_eye_video)
        self._update_eye_video(self.app.current_ts)

    def _update_eye_video(self, ts: int) -> None:
        if self._eye_video is not None:
            self._eye_video.set_frame(self.recording, ts)

    def _on_panel_closed(self) -> None:
        if self.app.plugins_by_class.get(self.__class__.__name__) is not self:
            return
        settings = self.app.recording_settings
        settings.enabled_plugins = {
            **settings.enabled_plugins, self.__class__.__name__: False,
        }

    def on_disabled(self) -> None:
        if self._eye_video is None:
            return
        self.app.position_changed.disconnect(self._update_eye_video)
        self.app.main_window.video_workspace.unregister_panel("eye-video")
        self._eye_video.deleteLater()
        self._eye_video = None

    def render_for_display(self, painter: QPainter, time_in_recording: int) -> None:
        # The eye panel owns onscreen presentation. render() remains the legacy
        # export composition, independent of the workspace arrangement.
        pass

    def render(self, painter: QPainter, time_in_recording: int) -> None:
        if self.recording is None:
            return

        eye_frame = self.recording.eye.sample([time_in_recording])[0]
        if abs(time_in_recording - eye_frame.time) / 1e9 > 1 / 30:
            return

        image = qimage_from_frame(eye_frame.gray).scaled(
            int(eye_frame.width * self._scale), int(eye_frame.height * self._scale)
        )
        painter.setOpacity(self._opacity)

        if self._border_width > 0:
            pen = painter.pen()
            pen.setWidth(self._border_width)
            pen.setColor(self._border_color)
            painter.setPen(pen)
            painter.drawRect(
                self.get_rect().adjusted(
                    -self._border_width / 2,
                    -self._border_width / 2,
                    self._border_width / 2,
                    self._border_width / 2,
                )
            )

        painter.drawImage(
            QPointF(
                self._offset_x * self.recording.scene.width,
                self._offset_y * self.recording.scene.height,
            ),
            image,
        )

        painter.setOpacity(1.0)

    def get_rect(self) -> QRectF:
        return QRectF(
            self._offset_x * self.recording.scene.width,
            self._offset_y * self.recording.scene.height,
            int(self._scale * self.recording.eye.width),
            int(self._scale * self.recording.eye.height),
        )

    @property
    @property_params(label="Export horizontal position", min=0, max=1, step=0.01)
    def offset_x(self) -> float:
        return self._offset_x

    @offset_x.setter
    def offset_x(self, value: float) -> None:
        self._offset_x = value

    @property
    @property_params(label="Export vertical position", min=0, max=1, step=0.01)
    def offset_y(self) -> float:
        return self._offset_y

    @offset_y.setter
    def offset_y(self, value: float) -> None:
        self._offset_y = value

    @property
    @property_params(label="Export scale", min=0.2, max=10, step=0.1)
    def scale(self) -> float:
        return self._scale

    @scale.setter
    def scale(self, value: float) -> None:
        self._scale = value

    @property
    @property_params(label="Export opacity", min=0, max=1, step=0.01, decimals=3)
    def opacity(self) -> float:
        return self._opacity

    @opacity.setter
    def opacity(self, value: float) -> None:
        self._opacity = value

    @property
    @property_params(label="Export border width", min=0, max=100)
    def border_width(self) -> int:
        return self._border_width

    @border_width.setter
    def border_width(self, value: int) -> None:
        self._border_width = value

    @property
    @property_params(label="Export border color")
    def border_color(self) -> QColor:
        return self._border_color

    @border_color.setter
    def border_color(self, value: QColor) -> None:
        self._border_color = value
