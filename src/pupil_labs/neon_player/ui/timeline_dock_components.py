import math
from enum import Enum, auto
from typing import NamedTuple

import pyqtgraph as pg
from pyqtgraph.GraphicsScene.mouseEvents import MouseDragEvent
from PySide6.QtCore import (
    QEvent,
    QObject,
    QPoint,
    QPointF,
    QRectF,
    QSize,
    Qt,
    QThread,
    QTimer,
    Signal,
)
from PySide6.QtGui import QColor, QIcon, QKeyEvent, QPainter, QPaintEvent, QPolygon
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsProxyWidget,
    QGraphicsRectItem,
    QGraphicsSceneMouseEvent,
    QGraphicsSceneWheelEvent,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QStyleOptionGraphicsItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from pupil_labs import neon_player
from pupil_labs.neon_player.ui import GUIEventNotifier


class TimelineRowRole(Enum):
    """Internal row purpose, independent of its scope and displayed name."""

    HEADER = auto()
    SEARCH = auto()
    CUSTOM = auto()


class TimelineRowKey(NamedTuple):
    """Internal identity shared by timeline plots, legends, and point actions."""

    group_name: str | None
    row_name: str
    role: TimelineRowRole


class TimelinePlotGroup:
    """Registered group handle; timeline operations manage its mutable state."""

    def __init__(
        self, name: str, collapsed: bool = False, *, searchable: bool = False
    ) -> None:
        self._name = name
        self._collapsed = collapsed
        self._searchable = searchable
        self._query = ""
        self._child_row_names: set[str] = set()

    @property
    def name(self) -> str:
        return self._name

    @property
    def collapsed(self) -> bool:
        return self._collapsed

    @property
    def searchable(self) -> bool:
        return self._searchable

    @property
    def child_row_names(self) -> frozenset[str]:
        return frozenset(self._child_row_names)


class TimelinePlotGroupControl(QToolButton):
    """Toggle for showing / hiding child rows of a timeline plot group."""

    def __init__(self, group_name: str, collapsed: bool = False) -> None:
        super().__init__()
        self._group_name = group_name
        self.setAutoRaise(True)
        self.setFixedSize(QSize(20, 20))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.set_collapsed(collapsed)

    def set_collapsed(self, collapsed: bool) -> None:
        action = "Expand" if collapsed else "Collapse"
        self.setText("▸" if collapsed else "▾")
        self.setToolTip(f"{action} {self._group_name}")
        self.setAccessibleName(f"{action} {self._group_name}")


class TimelineSearchField(QLineEdit):
    """Text editor that keeps timeline shortcuts out of search input."""

    def __init__(self) -> None:
        super().__init__()
        self.setStyleSheet("""
            background: transparent;
        """)

    def event(self, event: QEvent) -> bool:
        if (
            isinstance(event, QKeyEvent)
            and event.type() == QEvent.Type.ShortcutOverride
            and event.modifiers()
            in (
                Qt.KeyboardModifier.NoModifier,
                Qt.KeyboardModifier.ShiftModifier,
            )
        ):
            event.accept()
            return True
        return super().event(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        super().keyPressEvent(event)
        event.accept()


class TimelinePlotGroupSearchBar(QWidget):
    """Timeline-owned name search, shared by searchable groups."""

    def __init__(self, group_name: str) -> None:
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 0, 0, 0)
        icon = QLabel()
        icon.setFixedSize(12, 12)
        icon.setPixmap(
            QIcon(str(neon_player.asset_path("search.svg"))).pixmap(
                QSize(12, 12), self.devicePixelRatioF()
            )
        )
        self.field = TimelineSearchField()
        self.field.setPlaceholderText("Search")
        self.field.setClearButtonEnabled(True)
        self.field.setAccessibleName(f"Search {group_name}")
        self.field.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addWidget(icon)
        layout.addWidget(self.field, 1)
        self.field.destroyed.connect(self.deleteLater)


class TimelinePlotLabelHost(pg.GraphicsLayout):  # type: ignore[misc]
    """Own a row's title widget and legend without replacing either on refresh."""

    content_lost = Signal()
    size_hint_changed = Signal()

    def __init__(self, name: str, legend: "FixedLegend") -> None:
        super().__init__()
        self.setSpacing(0)
        self.setContentsMargins(0, 0, 0, 0)
        self.title = pg.LabelItem(name, justify="left")
        self.proxy: QGraphicsProxyWidget | None = None
        self.addItem(self.title)
        self.addItem(legend, row=1, col=0)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self._restore_timer = QTimer(self)
        self._restore_timer.setSingleShot(True)
        self._restore_timer.timeout.connect(self.content_lost)
        self._geometry_timer = QTimer(self)
        self._geometry_timer.setSingleShot(True)
        self._geometry_timer.timeout.connect(self._update_title_height)
        self._last_title_height = 0.0

    def _set_title_widget(self, widget: QWidget) -> None:
        if (
            QThread.currentThread() != self.thread()
            or widget.thread() != self.thread()
            or widget.parent() is not None
            or widget.graphicsProxyWidget() is not None
            or (widget.isWindow() and widget.windowType() != Qt.WindowType.Window)
            or widget.testAttribute(Qt.WidgetAttribute.WA_PaintOnScreen)
            or widget.testAttribute(Qt.WidgetAttribute.WA_NativeWindow)
            or widget.inherits("QOpenGLWidget")
        ):
            raise ValueError("Timeline title widgets must be live and embeddable")
        proxy = QGraphicsProxyWidget()
        proxy.setWidget(widget)
        if proxy.widget() is not widget:
            proxy.deleteLater()
            raise ValueError("Could not embed timeline title widget")
        if self.proxy is not None:
            old = self.proxy
            old_widget = old.widget()
            if old_widget is not None:
                old_widget.destroyed.disconnect(self._on_content_destroyed)
                old_widget.removeEventFilter(self)
            self.removeItem(old)
            old.deleteLater()
        else:
            self.removeItem(self.title)
            self.title.setParentItem(self)
            self.title.hide()
        self.proxy = proxy
        self.addItem(proxy, row=0, col=0)
        widget.destroyed.connect(self._on_content_destroyed)
        widget.installEventFilter(self)
        self._last_title_height = self.title_height()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.LayoutRequest:
            self._geometry_timer.start(0)
        return bool(super().eventFilter(watched, event))

    def _update_title_height(self) -> None:
        height = self.title_height()
        if height != self._last_title_height:
            self._last_title_height = height
            self.size_hint_changed.emit()

    def _on_content_destroyed(self) -> None:
        # A proxy and its QWidget delete each other. Wait until both are gone.
        self.proxy = None
        self._restore_timer.start(0)

    def title_height(self) -> float:
        if self.proxy is not None:
            widget = self.proxy.widget()
            if widget is not None:
                return float(
                    max(widget.sizeHint().height(), widget.minimumSizeHint().height())
                )
        return float(self.title.effectiveSizeHint(Qt.SizeHint.PreferredSize).height())


class ScrubbableViewBox(pg.ViewBox):  # type: ignore[misc]  # pyqtgraph does not type this base class.
    scrub_start = Signal(MouseDragEvent)
    scrub_end = Signal(MouseDragEvent)
    scrubbed = Signal(object)
    zoomed = Signal()

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        super().__init__()
        self.allow_y_panning = True

    def mousePressEvent(self, ev: QGraphicsSceneMouseEvent) -> None:
        if ev.button() == Qt.MouseButton.LeftButton:
            if Qt.KeyboardModifier.ControlModifier == ev.modifiers():
                if self.allow_y_panning:
                    self.setMouseMode(pg.ViewBox.RectMode)

            else:
                self.scrubbed.emit(ev)

        super().mousePressEvent(ev)

    def mouseDragEvent(self, ev: MouseDragEvent, axis: int | None = None) -> None:
        if ev.button() == Qt.MouseButton.MiddleButton:
            self.setMouseEnabled(x=True, y=self.allow_y_panning)
            super().mouseDragEvent(ev, axis)
            return

        if self.state["mouseMode"] == pg.ViewBox.RectMode:
            super().mouseDragEvent(ev, axis)
            if ev.finish:
                self.setMouseMode(pg.ViewBox.PanMode)

            return

        if ev.start:
            self.scrub_start.emit(ev)
        elif ev.finish:
            self.scrub_end.emit(ev)
        else:
            self.scrubbed.emit(ev)

        ev.accept()

    def wheelEvent(self, ev: QGraphicsSceneWheelEvent, axis: int | None = None) -> None:
        mouse_enabled = {
            "x": Qt.KeyboardModifier.ControlModifier in ev.modifiers(),
            "y": Qt.KeyboardModifier.ShiftModifier in ev.modifiers(),
        }
        mouse_enabled["y"] = self.allow_y_panning and mouse_enabled["y"]

        if any(mouse_enabled.values()):
            self.setMouseEnabled(**mouse_enabled)
            self.zoomed.emit()
            super().wheelEvent(ev, axis)

        ev.ignore()


class TimeAxisItem(pg.AxisItem):  # type: ignore[misc]  # pyqtgraph does not type this base class.
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, tickPen=pg.mkPen({"color": "#aaaaaa"}), **kwargs)
        self.recording_start_time_ns = 0
        self.recording_stop_time_ns = 0

        self.interval = 1.0
        self.setStyle(autoReduceTextSpace=False)

    def tickValues(
        self, minVal: float, maxVal: float, size: float
    ) -> list[tuple[float, list[float]]]:
        if self.recording_start_time_ns == 0 or self.recording_stop_time_ns == 0:
            return []

        minVal = max(minVal, self.recording_start_time_ns)
        maxVal = min(maxVal, self.recording_stop_time_ns)

        # Calculate the visible time range in seconds
        visible_range_ns = maxVal - minVal
        visible_range_sec = visible_range_ns / 1e9

        # Define nice intervals in seconds and their corresponding minor tick counts
        intervals = [
            (0.005, 5),
            (0.01, 2),
            (0.05, 5),
            (0.1, 10),
            (0.25, 5),
            (0.5, 5),
            (1.0, 10),
            (5.0, 5),
            (10.0, 10),
            (30.0, 6),
            (60.0, 6),
            (300.0, 5),
            (600.0, 10),
        ]

        # Find the largest interval that fits the current zoom level
        pixels_per_second = size / visible_range_sec if visible_range_sec > 0 else 0
        interval_sec, minor_ticks = intervals[-1]  # Start with largest interval

        # Find the largest interval where ticks won't be too close together
        for int_sec, minor_count in intervals:
            if (
                pixels_per_second * int_sec >= 120
            ):  # At least 120 pixels between major ticks
                interval_sec = int_sec
                minor_ticks = minor_count
                break

        self.interval = interval_sec

        # Calculate the first major tick at or after minVal that aligns with the
        # interval from recording start
        interval_ns = int(interval_sec * 1e9)
        minor_interval_ns = interval_ns // minor_ticks
        offset_from_start = (minVal - self.recording_start_time_ns) % interval_ns
        first_major_tick_ns = minVal - offset_from_start

        if first_major_tick_ns < self.recording_start_time_ns:
            first_major_tick_ns += interval_ns

        # Generate major and minor ticks
        major_ticks = []
        minor_tick_list = []

        current_major_tick_ns = first_major_tick_ns
        while (
            current_major_tick_ns <= maxVal + interval_ns
        ):  # Add one extra interval to ensure coverage
            if minVal <= current_major_tick_ns <= maxVal:
                major_ticks.append(current_major_tick_ns)

            # Add minor ticks between this major tick and the next
            for i in range(1, minor_ticks):
                minor_tick_ns = current_major_tick_ns + i * minor_interval_ns
                if (
                    minVal <= minor_tick_ns <= maxVal
                    and minor_tick_ns < current_major_tick_ns + interval_ns
                ):
                    minor_tick_list.append(minor_tick_ns)

            current_major_tick_ns += interval_ns

        # Always include the start time if it's in the visible range
        start_time_in_range = minVal <= self.recording_start_time_ns <= maxVal
        start_tick_is_missing = (
            not major_ticks or major_ticks[0] != self.recording_start_time_ns
        )
        if start_time_in_range and start_tick_is_missing:
            major_ticks.insert(0, self.recording_start_time_ns)

        # Return in the format expected by PyQtGraph: [(tick_scale, [ticks]), ...]
        return [(1.0, major_ticks), (0.5, minor_tick_list)]

    def tickStrings(
        self, values: list[float], _scale: float, _spacing: float
    ) -> list[str]:
        if self.recording_start_time_ns == 0:
            return ["" for _ in values]

        strings = []
        for val in values:
            if not (self.recording_start_time_ns <= val <= self.recording_stop_time_ns):
                strings.append("")
                continue

            relative_time_ns = val - self.recording_start_time_ns
            hours = relative_time_ns // (1e9 * 60 * 60)
            minutes = (relative_time_ns // (1e9 * 60)) % 60
            seconds = (relative_time_ns // 1e9) % 60
            ms = (relative_time_ns / 1e6) % 1000
            string = f"{minutes:0>2.0f}:{seconds:0>2.0f}"

            if self.interval < 1:
                string += f".{ms:0>3.0f}"

            if hours > 0:
                string = f"{hours:0>2,.0f}:{string}"

            strings.append(string)

        return strings

    def set_time_frame(self, start: int, end: int) -> None:
        self.recording_start_time_ns = start
        self.recording_stop_time_ns = end


class FixedLegend(pg.LegendItem):  # type: ignore[misc]  # pyqtgraph does not type this base class.
    def mouseDragEvent(self, event: MouseDragEvent) -> None:
        event.ignore()

    def addItem(self, *args: object, **kwargs: object) -> None:
        super().addItem(*args, **kwargs)
        self.setColumnCount(max(1, math.ceil(len(self.items) / 3)))

    def removeItem(self, *args: object, **kwargs: object) -> None:
        super().removeItem(*args, **kwargs)
        self.setColumnCount(max(1, math.ceil(len(self.items) / 3)))


class TimestampLabel(QLabel):
    def __init__(self) -> None:
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.set_time(0)
        font = self.font()
        font.setFixedPitch(True)
        self.setFont(font)

    def set_time(self, time_ns: int) -> None:
        hours = time_ns // (1e9 * 60 * 60)
        minutes = (time_ns // (1e9 * 60)) % 60
        seconds = (time_ns / 1e9) % 60
        self.setText(f"{hours:0>2,.0f}:{minutes:0>2.0f}:{seconds:0>6.3f}")


class TimelinePlotItem(pg.PlotItem):  # type: ignore[misc]  # pyqtgraph does not type this base class.
    """
    A wrapper class around [`pyqtgraph.PlotItem`][] that preserves a fixed height.
    """

    def __init__(
        self, legend: FixedLegend, row_name: str, *args: object, **kwargs: object
    ) -> None:
        super().__init__(*args, **kwargs)

        self.setDownsampling(auto=True, mode="subsample")
        self.legend_handle = legend
        self.row_key = TimelineRowKey(None, row_name, TimelineRowRole.CUSTOM)

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        self.preferred_height_1d = 25
        self.preferred_height_2d = 150
        self.adjust_size()

    @property
    def row_name(self) -> str:
        return self.row_key.row_name

    def addItem(self, *args: object, **kwargs: object) -> None:
        super().addItem(*args, **kwargs)
        self.adjust_size()

    def removeItem(self, *args: object, **kwargs: object) -> None:
        super().removeItem(*args, **kwargs)
        self.adjust_size()

    def adjust_size(self) -> None:
        height = self.preferred_height_2d if self.has_line else self.preferred_height_1d
        title_height = 0.0
        host = self.legend_handle.parentItem()
        if isinstance(host, TimelinePlotLabelHost):
            title_height = host.title_height()
            legend_height = (
                self.legend_handle.layout.effectiveSizeHint(
                    Qt.SizeHint.PreferredSize
                ).height()
                if self.legend_handle.items
                else 0
            )
            height = max(height, math.ceil(title_height + legend_height))

        self.setFixedHeight(height)

        self.legend_handle.setFixedHeight(max(0, height - title_height))
        self.legend_handle.parentItem().setFixedHeight(height)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        pen = painter.pen()
        pen.setColor("#444")
        pen.setWidth(1)
        painter.setPen(pen)

        if self.has_line:
            painter.drawLine(-2000, 0, self.width(), 0)
            painter.drawLine(-2000, self.height() - 1, self.width(), self.height() - 1)

        super().paint(painter, option, widget)

    @property
    def has_line(self) -> bool:
        for item in self.items:
            if isinstance(item, pg.PlotDataItem) and len(item.curve.xData) > 0:
                return True

        return False

    @property
    def has_bar(self) -> bool:
        return any(isinstance(item, pg.BarGraphItem) for item in self.items)


class PlotOverlay(QWidget):
    def __init__(self, linked_plot: pg.PlotItem, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.linked_plot = linked_plot

        linked_plot.vb.sigResized.connect(self.refresh_geometry)

    def get_x_pixel_for_x_value(self, x_value: float) -> float:
        vb = self.linked_plot.vb
        x_range = vb.viewRange()[0]
        return float(
            (x_value - x_range[0]) / (x_range[1] - x_range[0]) * vb.width() - 1
        )

    def refresh_geometry(self) -> None:
        vb = self.linked_plot.vb
        plot_rect = vb.mapToScene(vb.geometry()).boundingRect()
        parent = self.parentWidget()
        assert parent is not None, "Plot overlay must have a parent to refresh geometry"
        self.setGeometry(
            int(plot_rect.x()), 20, int(plot_rect.width()), parent.height() - 20
        )


class PlayHead(PlotOverlay):
    def __init__(self, linked_plot: pg.PlotItem, parent: QWidget | None = None) -> None:
        super().__init__(linked_plot, parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.color = QColor("#6D7BE0")
        self.t = 0

    def set_time(self, t: int) -> None:
        self.t = t
        self.update()

    def paintEvent(self, _event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setPen(self.color)
        painter.setBrush(self.color)

        x = self.get_x_pixel_for_x_value(self.t)

        if x < 0:
            painter.drawPolygon(
                QPolygon(
                    [
                        QPoint(0, 10),
                        QPoint(10, 0),
                        QPoint(10, 20),
                    ]
                )
            )

        elif x > self.width():
            painter.drawPolygon(
                QPolygon(
                    [
                        QPoint(self.rect().right(), 10),
                        QPoint(self.rect().right() - 10, 0),
                        QPoint(self.rect().right() - 10, 20),
                    ]
                )
            )

        else:
            painter.fillRect(QRectF(x - 1, 0, 3, self.height()), self.color)

        painter.end()


class TrimEndMarker(QGraphicsEllipseItem):
    def __init__(
        self, time: int, plot: pg.PlotItem, parent: QGraphicsItem | None = None
    ) -> None:
        super().__init__(time, -1, 0, 2, parent)
        self._time: float = time
        self._plot = plot

        class _Emitter(QObject):
            time_changed = Signal(object)

        self._emitter = _Emitter()
        self.time_changed = self._emitter.time_changed

        self.highlight_pen = pg.mkPen("#888", width=2)
        self.highlight_brush = pg.mkBrush("#00000000")
        self.normal_pen = pg.mkPen("#444", width=1)
        self.normal_brush = pg.mkBrush("#444")

        self.setPen(self.normal_pen)
        self.setBrush(self.normal_brush)

    @property
    def time(self) -> float:
        return self._time

    @time.setter
    def time(self, value: float) -> None:
        self._time = value
        self.time_changed.emit(value)
        self.update()

    def set_highlighted(self, highlighted: bool) -> None:
        if highlighted:
            self.setPen(self.highlight_pen)
            self.setBrush(self.highlight_brush)
        else:
            self.setPen(self.normal_pen)
            self.setBrush(self.normal_brush)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        scale_x = painter.worldTransform().m11()
        scale_y = painter.worldTransform().m22()
        scaled_width = 2 * abs(scale_y) / scale_x
        rect = self.rect()
        rect.setWidth(scaled_width)
        rect.setLeft(self._time - scaled_width / 2)
        self.setRect(rect)

        super().paint(painter, option, widget)

    def nearby(self, pos: QPoint | QPointF, buffer: float = 0.25) -> bool:
        rect = self.rect()
        dx = rect.width() * buffer
        dy = rect.height() * buffer

        return rect.adjusted(-dx, -dy, dx, dy).contains(pos)


class TrimDurationMarker(QGraphicsRectItem):
    def __init__(
        self,
        start_marker: TrimEndMarker,
        end_marker: TrimEndMarker,
        parent: QGraphicsItem | None = None,
    ) -> None:
        super().__init__(parent)

        self.setPen(pg.mkPen("#ccc", width=0))
        self.setBrush(pg.mkBrush("#ccc"))

        self._start_marker = start_marker
        self._end_marker = end_marker

        self._start_marker.time_changed.connect(lambda _: self._update_ends())
        self._end_marker.time_changed.connect(lambda _: self._update_ends())

        self._update_ends()

    def _update_ends(self) -> None:
        self.setRect(
            self._start_marker.time,
            -1,
            self._end_marker.time - self._start_marker.time,
            2,
        )
        self.update()


class TimelineTableContainer(GUIEventNotifier):
    def __init__(self, table: QWidget) -> None:
        super().__init__()
        self.setMouseTracking(True)

        layout = QVBoxLayout(self)

        layout.addWidget(table)
        layout.addStretch()
        layout.setContentsMargins(0, 0, 0, 0)
