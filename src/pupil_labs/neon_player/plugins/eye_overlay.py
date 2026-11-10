from enum import Flag, auto

from pupil_labs.neon_recording import NeonRecording
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QMouseEvent, QPainter
from qt_property_widgets.utilities import property_params

from pupil_labs import neon_player
from pupil_labs.neon_player.utilities import qimage_from_frame


class ModifyDirection(Flag):
    TOP = auto()
    RIGHT = auto()
    BOTTOM = auto()
    LEFT = auto()
    MOVE = auto()


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

        self._mouse_mode = ModifyDirection(0)
        self._drag_position: QPointF | None = None
        self.start_geometry = QRectF()
        self.video_widget = self.app.main_window.video_widget
        self.video_widget.mouse_moved.connect(self.on_mouse_moved)
        self.video_widget.mouse_pressed.connect(self.on_mouse_pressed)

        self.scene_width: int = 0
        self.scene_height: int = 0
        self.eye_width: int = 0
        self.eye_height: int = 0

    def on_recording_loaded(self, recording: NeonRecording) -> None:
        scene = recording.scene
        if scene.width is None or scene.height is None:
            raise RuntimeError(
                "Scene video dimensions are required for the eye overlay"
            )

        eye = recording.eye
        if eye.width is None or eye.height is None:
            raise RuntimeError("Eye video dimensions are required for the eye overlay")

        self.scene_width = scene.width
        self.scene_height = scene.height
        self.eye_width = eye.width
        self.eye_height = eye.height

    def on_disabled(self) -> None:
        self.scene_width = 0
        self.scene_height = 0
        self.eye_width = 0
        self.eye_height = 0

    def on_mouse_pressed(self, event: QMouseEvent) -> None:
        self._drag_position = self.video_widget.map_point(event.pos())
        self.start_geometry = self.get_rect()

    def on_mouse_moved(self, event: QMouseEvent) -> None:
        if event.buttons() == Qt.MouseButton.LeftButton:
            self.on_drag(event)
        else:
            self.on_hover(event)

    def align(self, rect: QRectF, corner: str) -> None:
        v = getattr(self.start_geometry, corner)()
        getattr(rect, f"move{corner.title()}")(v)

    def on_drag(self, event: QMouseEvent) -> None:
        recording = self.recording
        drag_position = self._drag_position
        if recording is None or drag_position is None:
            return

        mapped_pos = self.video_widget.map_point(event.pos())
        offset = mapped_pos - drag_position

        if self._mouse_mode == ModifyDirection.MOVE:
            proposed_x = self.offset_x + offset.x() / self.scene_width
            proposed_y = self.offset_y + offset.y() / self.scene_height

            max_x = 1 - (self._scale * self.eye_width) / self.scene_width
            max_y = 1 - (self._scale * self.eye_height) / self.scene_height

            proposed_x = max(0, min(max_x, proposed_x))
            proposed_y = max(0, min(max_y, proposed_y))

            self.offset_x = proposed_x
            self.offset_y = proposed_y
            self._drag_position = mapped_pos

        else:
            rect = QRectF(self.start_geometry)
            fix_y = False

            aligns = []
            if self._mouse_mode & ModifyDirection.LEFT:
                aligns.append("right")
                rect.setLeft(max(0, rect.left() + offset.x()))
                fix_y = True
            elif self._mouse_mode & ModifyDirection.RIGHT:
                aligns.append("left")
                fix_y = True
                rect.setRight(min(self.scene_width, rect.right() + offset.x()))

            if self._mouse_mode & ModifyDirection.TOP:
                aligns.append("bottom")
                rect.setTop(max(0, rect.top() + offset.y()))
            elif self._mouse_mode & ModifyDirection.BOTTOM:
                aligns.append("top")
                rect.setBottom(min(self.scene_height, rect.bottom() + offset.y()))

            if fix_y:
                max_scale = (self.scene_height - rect.top()) / self.eye_height
                self.scale = min(max_scale, max(0.2, rect.width() / self.eye_width))
                rect.setHeight(self.scale * self.eye_height)
            else:
                max_scale = (self.scene_width - rect.left()) / self.eye_width
                self.scale = min(max_scale, max(0.2, rect.height() / self.eye_height))
                rect.setWidth(self.scale * self.eye_width)

            for a in aligns:
                self.align(rect, a)

            self.offset_x = rect.left() / self.scene_width
            self.offset_y = rect.top() / self.scene_height

        self.changed.emit()
        self.video_widget.update()

    def on_hover(self, event: QMouseEvent) -> None:
        edge_margin = 20

        pos = self.video_widget.map_point(event.pos())
        rect = self.get_rect()

        if rect.contains(pos):
            self._mouse_mode = ModifyDirection(0)
            if pos.x() < rect.left() + edge_margin:
                self._mouse_mode |= ModifyDirection.LEFT
            elif pos.x() > rect.right() - edge_margin:
                self._mouse_mode |= ModifyDirection.RIGHT

            if pos.y() < rect.top() + edge_margin:
                self._mouse_mode |= ModifyDirection.TOP
            elif pos.y() > rect.bottom() - edge_margin:
                self._mouse_mode |= ModifyDirection.BOTTOM

            if self._mouse_mode == ModifyDirection(0):
                self._mouse_mode = ModifyDirection.MOVE
        else:
            self._mouse_mode = ModifyDirection(0)

        self._update_cursor()

    def _update_cursor(self) -> None:
        if self._mouse_mode == ModifyDirection(0):
            self.video_widget.unsetCursor()

        else:
            tl = self._mouse_mode == (ModifyDirection.TOP | ModifyDirection.LEFT)
            br = self._mouse_mode == (ModifyDirection.BOTTOM | ModifyDirection.RIGHT)
            tr = self._mouse_mode == (ModifyDirection.TOP | ModifyDirection.RIGHT)
            bl = self._mouse_mode == (ModifyDirection.BOTTOM | ModifyDirection.LEFT)

            if tl or br:
                self.video_widget.setCursor(Qt.CursorShape.SizeFDiagCursor)
            elif tr or bl:
                self.video_widget.setCursor(Qt.CursorShape.SizeBDiagCursor)
            elif self._mouse_mode in (ModifyDirection.LEFT, ModifyDirection.RIGHT):
                self.video_widget.setCursor(Qt.CursorShape.SizeHorCursor)
            elif self._mouse_mode in (ModifyDirection.TOP, ModifyDirection.BOTTOM):
                self.video_widget.setCursor(Qt.CursorShape.SizeVerCursor)
            else:
                self.video_widget.setCursor(Qt.CursorShape.SizeAllCursor)

        self.video_widget.update()

    def render(self, painter: QPainter, time_in_recording: int) -> None:
        recording = self.recording
        if recording is None:
            return

        eye_frame = recording.eye.sample([time_in_recording])[0]
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
                self._offset_x * self.scene_width,
                self._offset_y * self.scene_height,
            ),
            image,
        )

        if self._mouse_mode != ModifyDirection(0):
            pen = painter.pen()
            pen.setWidth(7)
            pen.setColor("#6D7BE0")
            painter.setPen(pen)
            painter.setBrush(Qt.GlobalColor.transparent)
            painter.drawRect(self.get_rect().adjusted(0, 0, -1, -1))

        painter.setOpacity(1.0)

    def get_rect(self) -> QRectF:
        return QRectF(
            self._offset_x * self.scene_width,
            self._offset_y * self.scene_height,
            int(self._scale * self.eye_width),
            int(self._scale * self.eye_height),
        )

    @property
    @property_params(widget=None)
    def offset_x(self) -> float:
        return self._offset_x

    @offset_x.setter
    def offset_x(self, value: float) -> None:
        self._offset_x = value

    @property
    @property_params(widget=None)
    def offset_y(self) -> float:
        return self._offset_y

    @offset_y.setter
    def offset_y(self, value: float) -> None:
        self._offset_y = value

    @property
    @property_params(widget=None)
    def scale(self) -> float:
        return self._scale

    @scale.setter
    def scale(self, value: float) -> None:
        self._scale = value

    @property
    @property_params(min=0, max=1, step=0.01, decimals=3)
    def opacity(self) -> float:
        return self._opacity

    @opacity.setter
    def opacity(self, value: float) -> None:
        self._opacity = value

    @property
    @property_params(min=0, max=100)
    def border_width(self) -> int:
        return self._border_width

    @border_width.setter
    def border_width(self, value: int) -> None:
        self._border_width = value

    @property
    def border_color(self) -> QColor:
        return self._border_color

    @border_color.setter
    def border_color(self, value: QColor) -> None:
        self._border_color = value
