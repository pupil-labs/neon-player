import contextlib
import logging
import typing as T

import numpy as np
import pyqtgraph as pg
from numpy.typing import NDArray
from pyqtgraph.GraphicsScene.mouseEvents import (
    MouseClickEvent,
    MouseDragEvent,
)
from PySide6.QtCore import QPoint, QPointF, QSize, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QIcon,
    QKeyEvent,
    QMouseEvent,
    QPen,
    QResizeEvent,
)
from PySide6.QtWidgets import (
    QComboBox,
    QGraphicsSceneMouseEvent,
    QHBoxLayout,
    QMenu,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from pupil_labs import neon_player
from pupil_labs import neon_recording as nr
from pupil_labs.neon_player.ui.timeline_dock_components import (
    FixedLegend,
    PlayHead,
    ScrubbableViewBox,
    TimeAxisItem,
    TimelineHorizontalScrollBar,
    TimelineHoverMarker,
    TimelinePlotItem,
    TimelineScrollArea,
    TimelineTableContainer,
    TimestampLabel,
    TrimDurationMarker,
    TrimEndMarker,
)
from pupil_labs.neon_player.ui.timeline_dock_tooltips import (
    TimelineTooltipConfig,
    TimelineTooltipContext,
    TimelineTooltipWidget,
    _get_hovered_timeline_series,
)
from pupil_labs.neon_player.ui.timeline_dock_utils import (
    get_clicked_data_point,
    get_clicked_plot_item,
)
from pupil_labs.neon_player.utilities import SignalDebouncer, clone_menu

logger = logging.getLogger(__name__)


class TimeLineDock(QWidget):
    """Displays the data as the plots in the Timeline dock.

    The timeline rows are currently sorted based on the following criteria:

    - plot type: horizontal bar plots, scatter plots, line plots
    - row name: in alphabetical order within each plot type

    !!! warning

        The sorting is performed automatically for each added or removed plot.
        When adding or removing multiple rows, [disable][disable_plot_sorting]
        and later [re-enable][enable_plot_sorting] the sorting for better performance:

        ```py
        timeline = self.get_timeline()
        was_sorting_enabled = timeline.disable_plot_sorting()

        # add / remove timeline rows

        if was_sorting_enabled:
            timeline.enable_plot_sorting()
        ```
    """

    TIMELINE_PADDING = 0.05

    key_pressed = Signal(QKeyEvent)
    tooltip_update_requested = Signal(QPointF, float)

    def __init__(self) -> None:
        super().__init__()
        app = neon_player.instance()

        self.recording: nr.NeonRecording | None = None

        self.timeline_plots: dict[str, TimelinePlotItem] = {}
        self.timeline_legends: dict[str, pg.LegendItem] = {}
        self.timeline_tooltips: dict[str, TimelineTooltipConfig | None] = {}
        self._tooltip_widget_rows: dict[int, tuple[QWidget, str]] = {}
        self.plot_colors = [
            QColor("#1f77b4"),
            QColor("#ff7f0e"),
            QColor("#2ca02c"),
            QColor("#d62728"),
            QColor("#9467bd"),
            QColor("#8c564b"),
            QColor("#e377c2"),
            QColor("#7f7f7f"),
            QColor("#bcbd22"),
            QColor("#17becf"),
        ]
        self.data_point_actions: dict[
            str, dict[str, T.Callable[[tuple[int, T.SupportsFloat]], object]]
        ] = {}

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.toolbar_layout = QHBoxLayout()
        self.play_button = QToolButton()
        self.play_button.setToolTip("Play/Pause")
        self.play_button.setIconSize(QSize(32, 32))
        self.play_button.setFixedSize(QSize(36, 36))
        self.play_button.setIcon(QIcon(str(neon_player.asset_path("play.svg"))))
        self.play_button.clicked.connect(
            lambda: app.get_action("Playback/Play\\Pause").trigger()
        )
        self.toolbar_layout.addWidget(self.play_button)

        self.speed_control = QComboBox()
        self.speed_control.setToolTip("Playback rate")
        self.speed_control.addItems(
            [f"{s:.3g}x" for s in app.playback_speed_options if s < 0]
        )
        self.speed_control.insertSeparator(self.speed_control.count())
        self.speed_control.addItems(
            [f" {s:.3g}x" for s in app.playback_speed_options if s > 0]
        )
        font = self.speed_control.font()
        font.setFixedPitch(True)
        self.speed_control.setFont(font)
        self.speed_control.setCurrentText(" 1x")

        self.speed_control.currentTextChanged.connect(
            lambda t: app.set_playback_speed(float(t[:-1]))
        )
        app.speed_changed.connect(self.on_playback_speed_changed)

        self.timestamp_label = TimestampLabel()
        self.toolbar_layout.addWidget(self.timestamp_label, 1)
        self.toolbar_layout.addWidget(self.speed_control)

        self.main_layout.addLayout(self.toolbar_layout)

        self.graphics_view = pg.GraphicsView()
        self.graphics_view.setBackground("transparent")
        self.graphics_layout = pg.GraphicsLayout()
        self.graphics_layout.setSpacing(0)
        self.graphics_layout.setContentsMargins(0, 0, 0, 0)
        self.graphics_view.setCentralItem(self.graphics_layout)
        self.tooltip_widget = TimelineTooltipWidget(self.graphics_view.viewport())

        self.graphics_view.scene().sigMouseClicked.connect(self.on_chart_area_clicked)
        self.graphics_view.scene().sigMouseMoved.connect(self.on_chart_area_mouse_moved)
        self.tooltip_update_requested.connect(self._update_tooltip)

        self.scroll_area = TimelineScrollArea()
        self.scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOn
        )

        self.graphics_view_container = TimelineTableContainer(self.graphics_view)
        self.graphics_view_container.mouse_pressed.connect(
            self.on_whitespace_mouse_clicked
        )
        self.graphics_view_container.mouse_moved.connect(self.on_whitespace_mouse_moved)
        self.scroll_area.setWidget(self.graphics_view_container)
        self.scroll_area.setWidgetResizable(True)
        self.main_layout.addWidget(self.scroll_area)

        self.horizontal_scroll_bar = TimelineHorizontalScrollBar()
        self.horizontal_scroll_bar.range_changed.connect(self.on_timeline_scroll)
        self.main_layout.addWidget(self.horizontal_scroll_bar)

        self.setMouseTracking(True)

        # Add a permanent timeline with timestamps
        self.timestamps_plot = self.get_timeline_plot(
            "Export window", create_if_missing=True
        )
        assert self.timestamps_plot is not None
        self.timestamps_plot.getViewBox().sigRangeChanged.connect(
            self.on_timeline_view_range_changed
        )
        self.timestamps_plot.showAxis("top")
        self.timestamps_plot.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )

        self.playhead = PlayHead(self.timestamps_plot, parent=self.graphics_view)
        self.playhead.hide()

        self.hover_marker = TimelineHoverMarker(
            self.timestamps_plot, parent=self.graphics_view
        )
        self.hover_marker.hide()

        app.playback_state_changed.connect(self.on_playback_state_changed)
        app.position_changed.connect(self.on_position_changed)
        app.recording_loaded.connect(self.on_recording_loaded)
        app.recording_unloaded.connect(self.on_recording_unloaded)

        self.dragging: TrimEndMarker | None = None
        self.trim_markers: list[TrimEndMarker] = []
        self.plot_sorting_enabled = True

    def sizeHint(self) -> QSize:
        return QSize(100, 150)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        self.key_pressed.emit(event)

    def resizeEvent(self, event: QResizeEvent) -> None:
        w = self.scroll_area.width() - self.scroll_area.verticalScrollBar().width()
        self.graphics_view.setFixedWidth(w)
        self.playhead.refresh_geometry()
        self.hover_marker.refresh_geometry()
        if self.tooltip_widget.isVisible():
            self.tooltip_widget.show_at(QCursor.pos())
        return super().resizeEvent(event)

    def on_recording_loaded(self, recording: nr.NeonRecording) -> None:
        app = neon_player.instance()
        self.recording = recording

        self.playhead.show()
        self.horizontal_scroll_bar.set_time_frame(
            (recording.start_time, recording.stop_time),
            padding=self.TIMELINE_PADDING,
        )
        duration = recording.stop_time - recording.start_time
        for plot_item in self.timeline_plots.values():
            plot_item.setXRange(recording.start_time, recording.stop_time, padding=0)
            plot_item.getViewBox().setLimits(
                xMin=recording.start_time - duration * self.TIMELINE_PADDING,
                xMax=recording.stop_time + duration * self.TIMELINE_PADDING,
            )
            axis = plot_item.getAxis("top")
            axis.set_time_frame(recording.start_time, recording.stop_time)

        self.on_timeline_view_range_changed()

        trim_plot = self.get_timeline_plot("Export window", create_if_missing=True)
        export_window = app.get_export_window()
        assert export_window is not None, (
            "Export window is required to initialize the timeline"
        )
        self.trim_markers = [TrimEndMarker(ts, plot=trim_plot) for ts in export_window]
        self.duration_marker = TrimDurationMarker(*self.trim_markers)
        for tm in [*self.trim_markers, self.duration_marker]:
            trim_plot.addItem(tm)

    def on_recording_unloaded(self) -> None:
        self.horizontal_scroll_bar.set_time_frame(None)
        trim_plot = self.get_timeline_plot("Export window", create_if_missing=False)
        if trim_plot is not None:
            trim_plot.clear()
        self.playhead.hide()
        self.hover_marker.set_cursor_time(None)
        self.tooltip_widget.hide()
        self.recording = None
        self.trim_markers = []
        self.dragging = None

    def on_playback_state_changed(self, is_playing: bool) -> None:
        icon_name = "pause.svg" if is_playing else "play.svg"
        self.play_button.setIcon(QIcon(str(neon_player.asset_path(icon_name))))

    def on_playback_speed_changed(self, speed: float) -> None:
        prefix = " " if speed > 0 else ""
        self.speed_control.setCurrentText(f"{prefix}{speed:.3g}x")

    def on_position_changed(self, t: int) -> None:
        app = neon_player.instance()
        recording = app.recording
        if recording is None:
            return

        if app.is_playing:
            view_range = self.timestamps_plot.viewRange()[0]
            playhead_adjust = t - self.playhead.t
            self.timestamps_plot.setXRange(
                view_range[0] + playhead_adjust,
                view_range[1] + playhead_adjust,
                padding=0,
            )

        self.timestamp_label.set_time(t - recording.start_time)
        self.playhead.set_time(t)

    def on_timeline_view_range_changed(self, *_args: object) -> None:
        app = neon_player.instance()
        if app.recording is None:
            return

        self.horizontal_scroll_bar.set_view_range(
            *self.timestamps_plot.getViewBox().viewRange()[0]
        )

    def on_timeline_scroll(self, start: float, stop: float) -> None:
        self.timestamps_plot.setXRange(start, stop, padding=0)

    def show_context_menu(self, global_position: QPoint) -> None:
        menu = neon_player.instance().main_window.get_menu(
            "Timeline", auto_create=False
        )
        context_menu = clone_menu(menu) if isinstance(menu, QMenu) else QMenu()
        context_menu.exec(global_position)

    def on_chart_area_mouse_moved(self, pos: QPointF) -> None:
        data_pos = self.timestamps_plot.getViewBox().mapSceneToView(pos)
        for tm in self.trim_markers:
            tm.set_highlighted(self.dragging == tm or tm.nearby(data_pos))

        if any(
            plot_item.sceneBoundingRect().contains(pos)
            for plot_item in self.timeline_plots.values()
        ):
            self.hover_marker.set_cursor_time(round(data_pos.x()))
        else:
            self.hover_marker.set_cursor_time(None)

        SignalDebouncer.debounce(
            self.tooltip_update_requested, 0.01, pos, round(data_pos.x())
        )

    def _update_tooltip(self, pos: QPointF, timestamp_ns: int) -> None:
        if self.recording is None:
            self.tooltip_widget.hide()
            return

        if not self.recording.start_time <= timestamp_ns <= self.recording.stop_time:
            self.tooltip_widget.hide()
            return

        hovered_plot_name = None
        for name, plot_item in self.timeline_plots.items():
            if plot_item.sceneBoundingRect().contains(pos):
                hovered_plot_name = name
                break

        tooltip_content = self._tooltip_content_for_hovered_row(
            hovered_plot_name, timestamp_ns
        )

        if isinstance(tooltip_content, QWidget):
            self.tooltip_widget.show_widget(tooltip_content, QCursor.pos())
        elif tooltip_content:
            self.tooltip_widget.show_text(tooltip_content, QCursor.pos())
        else:
            self.tooltip_widget.hide()

    def _tooltip_content_for_hovered_row(
        self, row_name: str | None, timestamp_ns: int
    ) -> str | QWidget | None:
        if row_name is None:
            return None

        tooltip = self.timeline_tooltips.get(row_name)
        if tooltip is None or row_name == "Export window":
            return None

        plot_item = self.timeline_plots[row_name]
        tolerance_ns = round(20 * plot_item.getViewBox().viewPixelSize()[0])
        series = _get_hovered_timeline_series(
            plot_item.items, timestamp_ns, tolerance_ns
        )
        if not series:
            return None

        origin_ns = self.recording.start_time if self.recording is not None else 0
        context = TimelineTooltipContext(
            timestamp_ns, series, origin_ns, row_name=row_name
        )

        try:
            if isinstance(tooltip, QWidget):
                update_tooltip = getattr(tooltip, "update_tooltip", None)
                if not callable(update_tooltip):
                    raise TypeError(
                        "Timeline tooltip widgets must implement "
                        "update_tooltip(context)"
                    )
                update_tooltip(context)
                return tooltip

            tooltip_text = tooltip(context)
            return tooltip_text
        except Exception:
            logger.exception("Timeline tooltip callback failed for %s", row_name)
            return None

    def set_timeline_tooltip(
        self,
        timeline_row_name: str,
        tooltip: TimelineTooltipConfig | None,
    ) -> None:
        previous = self.timeline_tooltips.get(timeline_row_name)
        current = TimeLineDock._normalize_timeline_tooltip(self, tooltip)
        new_widgets = {id(current): current} if isinstance(current, QWidget) else {}
        previous_widgets = (
            {id(previous): previous} if isinstance(previous, QWidget) else {}
        )
        if new_widgets:
            TimeLineDock._validate_timeline_tooltip_widgets(
                self, new_widgets, timeline_row_name
            )

        if new_widgets or previous_widgets:
            tooltip_widget = getattr(self, "tooltip_widget", None)
        else:
            tooltip_widget = None
        if (
            tooltip_widget is not None
            and tooltip_widget.isVisible()
            and any(widget_id not in new_widgets for widget_id in previous_widgets)
        ):
            self.tooltip_widget.hide()

        for widget_id, widget in new_widgets.items():
            if widget_id not in self._tooltip_widget_rows:
                self.tooltip_widget.own_content_widget(widget)
                self._tooltip_widget_rows[widget_id] = (
                    widget,
                    timeline_row_name,
                )

        self.timeline_tooltips[timeline_row_name] = current
        TimeLineDock._release_timeline_tooltip_widgets(
            self,
            timeline_row_name,
            tuple(
                widget
                for widget_id, widget in previous_widgets.items()
                if widget_id not in new_widgets
            ),
        )

    def _release_timeline_tooltip_widgets(
        self, timeline_row_name: str, configs: tuple[T.Any, ...]
    ) -> None:
        released: set[int] = set()
        for config in configs:
            if not isinstance(config, QWidget):
                continue
            widget_id = id(config)
            if widget_id in released:
                continue
            released.add(widget_id)
            owner = getattr(self, "_tooltip_widget_rows", {}).get(widget_id)
            if owner is None or owner[0] is not config or owner[1] != timeline_row_name:
                continue
            self._tooltip_widget_rows.pop(widget_id, None)
            self.tooltip_widget.release_content_widget(config)

    def _normalize_timeline_tooltip(
        self,
        tooltip: TimelineTooltipConfig | None,
    ) -> TimelineTooltipConfig | None:
        if tooltip is None:
            return None
        if isinstance(tooltip, QWidget):
            update_tooltip = getattr(tooltip, "update_tooltip", None)
            if not callable(update_tooltip):
                raise TypeError(
                    "Timeline tooltip widgets must implement update_tooltip(context)"
                )
            return tooltip
        if callable(tooltip):
            return tooltip
        raise TypeError("Timeline tooltip must be a callback, QWidget, or None")

    def _validate_timeline_tooltip_widgets(
        self, widgets: dict[int, QWidget], timeline_row_name: str
    ) -> None:
        for widget_id, widget in widgets.items():
            current_owner = getattr(self, "_tooltip_widget_rows", {}).get(widget_id)
            if current_owner is not None:
                if current_owner[0] is not widget:
                    raise RuntimeError("Timeline tooltip widget identity collision")
                if current_owner[1] != timeline_row_name:
                    raise ValueError(
                        "Timeline tooltip widget is already assigned to another row"
                    )
                continue
            if widget.parent() is not None:
                raise ValueError("Timeline tooltip widgets must be parentless")

    def on_whitespace_mouse_moved(self, event: QMouseEvent) -> None:
        if event.buttons() == Qt.MouseButton.LeftButton:
            self.on_chart_area_clicked(event)

    def on_whitespace_mouse_clicked(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.on_chart_area_clicked(event)

        elif event.button() == Qt.MouseButton.RightButton:
            view_pos = self.graphics_view.mapFrom(
                self.graphics_view_container, event.pos()
            )
            scene_pos = self.graphics_view.mapToScene(view_pos)

            class SyntheticEvent:
                def __init__(self, scene_pos: QPointF, screen_pos: QPoint) -> None:
                    self._scene_pos = scene_pos
                    self._screen_pos = screen_pos

                def scenePos(self) -> QPointF:
                    return self._scene_pos

                def screenPos(self) -> QPoint:
                    return self._screen_pos

                def globalPos(self) -> QPoint:
                    return self._screen_pos

                def isAccepted(self) -> bool:
                    return False

                def accept(self) -> None:
                    pass

            synth_event = SyntheticEvent(scene_pos, event.globalPos())
            self.check_for_data_item_click(synth_event)

    def on_trim_area_drag_start(self, event: MouseDragEvent) -> None:
        app = neon_player.instance()
        recording = app.recording
        if recording is None:
            return

        data_pos = self.timestamps_plot.getViewBox().mapSceneToView(event.scenePos())
        for tm in self.trim_markers:
            if tm.nearby(data_pos, 0.5):
                self.dragging = tm
                break
        else:
            self.on_trim_area_dragged(event)

    def on_trim_area_dragged(self, event: MouseDragEvent) -> None:
        app = neon_player.instance()
        recording = app.recording
        if recording is None:
            return

        data_pos = self.timestamps_plot.getViewBox().mapSceneToView(event.scenePos())
        if self.dragging is None:
            self.on_chart_area_clicked(event)
            return

        self.dragging.time = max(
            min(data_pos.x(), recording.stop_time), recording.start_time
        )
        app.seek_to(int(self.dragging.time))
        settings = app.recording_settings
        assert settings is not None, (
            "Recording settings are required to drag the trim area"
        )
        settings.export_window = self.get_export_window()

    def on_trim_area_drag_end(self, event: MouseDragEvent) -> None:
        self.dragging = None
        data_pos = self.timestamps_plot.getViewBox().mapSceneToView(event.scenePos())
        for tm in self.trim_markers:
            tm.set_highlighted(self.dragging == tm or tm.nearby(data_pos))

    def on_chart_area_clicked(
        self, event: QGraphicsSceneMouseEvent | MouseClickEvent | MouseDragEvent
    ) -> None:
        app = neon_player.instance()
        recording = app.recording
        if recording is None:
            return

        click_types = [QGraphicsSceneMouseEvent, MouseClickEvent]
        if any(isinstance(event, cls) for cls in click_types):
            data_pos = self.timestamps_plot.getViewBox().mapSceneToView(
                event.scenePos()
            )
            for tm in self.trim_markers:
                if tm.nearby(data_pos, 0.5):
                    return

        if (
            event.buttons() == Qt.MouseButton.LeftButton
            or event.button() == Qt.MouseButton.LeftButton
        ):
            first_plot_item = next(iter(self.timeline_plots.values()))

            if hasattr(event, "scenePos"):
                mouse_point = first_plot_item.getViewBox().mapSceneToView(
                    event.scenePos()
                )
            else:
                mouse_point = first_plot_item.getViewBox().mapSceneToView(event.pos())

            time_ns = int(mouse_point.x())

            time_ns = max(recording.start_time, time_ns)
            time_ns = min(recording.stop_time, time_ns)
            app.seek_to(time_ns)

            return

        if event.button() == Qt.MouseButton.RightButton:
            self.check_for_data_item_click(event)

    def check_for_data_item_click(self, event: MouseClickEvent) -> None:
        if event.isAccepted():
            return

        nearby_items = self.graphics_layout.scene().itemsNearEvent(event)
        clicked_plot_item = get_clicked_plot_item(nearby_items)
        clicked_data_point = get_clicked_data_point(clicked_plot_item, event)
        if clicked_plot_item is None or clicked_data_point is None:
            self.show_context_menu(QCursor.pos())
            event.accept()
            return

        row_name = clicked_plot_item.row_name
        self.on_data_point_clicked(row_name, clicked_data_point)

    @T.overload
    def get_timeline_plot(
        self, timeline_row_name: str, create_if_missing: T.Literal[True]
    ) -> TimelinePlotItem: ...

    @T.overload
    def get_timeline_plot(
        self, timeline_row_name: str, create_if_missing: bool = False
    ) -> TimelinePlotItem | None: ...

    def get_timeline_plot(
        self, timeline_row_name: str, create_if_missing: bool = False
    ) -> TimelinePlotItem | None:
        """Get or create a timeline plot (i.e., a row in the timeline).

        Args:
            timeline_row_name: The name of the timeline plot.
            create_if_missing: Whether to create the plot if it doesn't
                exist already.

        Returns:
            plot: The created/existing plot, or `None` if the plot
                did not exist and was not created.
        """
        if timeline_row_name in self.timeline_plots:
            return self.timeline_plots[timeline_row_name]

        if not create_if_missing:
            return None

        logger.debug(f"Adding plot '{timeline_row_name}' to timeline")

        row = self.graphics_layout.nextRow()
        is_timestamps_row = timeline_row_name == "Export window"

        if is_timestamps_row:
            time_axis = TimeAxisItem(orientation="top")
        else:
            time_axis = TimeAxisItem(
                orientation="top", showValues=False, pen=pg.mkPen(color="#ffff0000")
            )

        app = neon_player.instance()
        recording = app.recording
        if recording is not None:
            time_axis.set_time_frame(recording.start_time, recording.stop_time)

        vb = ScrubbableViewBox()
        if is_timestamps_row:
            vb.allow_y_panning = False
            vb.scrub_start.connect(self.on_trim_area_drag_start)
            vb.scrubbed.connect(self.on_trim_area_dragged)
            vb.scrub_end.connect(self.on_trim_area_drag_end)
        else:
            vb.scrubbed.connect(self.on_chart_area_clicked)

        vb.zoomed.connect(self.graphics_view_container.repaint)

        legend = FixedLegend()
        legend.layout.setContentsMargins(0, 0, 0, 50)
        legend.layout.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Minimum,
        )
        legend_container = pg.GraphicsLayout()
        legend_container.setSpacing(0)
        legend_container.setContentsMargins(0, 0, 0, 0)
        legend_label = pg.LabelItem(timeline_row_name, justify="left")

        legend_container.addItem(legend_label)
        legend_container.addItem(legend, row=1, col=0)
        legend_container.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Fixed,
        )
        plot_item = TimelinePlotItem(
            legend=legend,
            row_name=timeline_row_name,
            axisItems={"top": time_axis},
            viewBox=vb,
        )

        if is_timestamps_row:
            plot_item.preferred_height_1d = 50
            plot_item.adjust_size()
            legend_label.anchor((0.5, 0), (0.5, 0), (0, 20))

        legend.layout.setSpacing(0)
        self.timeline_legends[timeline_row_name] = legend

        self.graphics_layout.addItem(legend_container, row=row, col=0)
        self.graphics_layout.addItem(plot_item, row=row, col=1)

        plot_item.setMouseEnabled(x=False, y=False)
        plot_item.hideButtons()
        plot_item.setMenuEnabled(False)
        plot_item.setClipToView(True)
        plot_item.hideAxis("right")
        plot_item.hideAxis("bottom")
        plot_item.showGrid(x=True, y=False, alpha=0.3)

        if recording is not None:
            duration = recording.stop_time - recording.start_time
            plot_item.getViewBox().setLimits(
                xMin=recording.start_time - duration * self.TIMELINE_PADDING,
                xMax=recording.stop_time + duration * self.TIMELINE_PADDING,
            )

        self.timeline_plots[timeline_row_name] = plot_item

        if not is_timestamps_row and self.timestamps_plot:
            plot_item.setXRange(*self.timestamps_plot.viewRange()[0], padding=0)
            plot_item.setXLink(self.timestamps_plot)

        return plot_item

    def get_timeline_series(
        self, plot_name: str, series_name: str
    ) -> pg.PlotDataItem | None:
        """Access timeline data series by their name.

        Args:
            plot_name: The name of the timeline row.
            series_name: Name shown for the series in the row legend.

        Returns:
            series: The data series, if it exists.
        """
        plot_item = self.get_timeline_plot(plot_name)
        if plot_item is None:
            return None

        for series in plot_item.items:
            if hasattr(series, "name") and series.name == series_name:
                return series

        return None

    def add_timeline_plot(
        self,
        timeline_row_name: str,
        data: list[tuple[int, T.SupportsFloat]]
        | NDArray[np.integer[T.Any] | np.floating[T.Any]],
        plot_name: str = "",
        color: QColor | None = None,
        tooltip: TimelineTooltipConfig | None = None,
        **kwargs: T.Any,  # noqa: ANN401
    ) -> TimelinePlotItem | None:
        """Add a new plot to the timeline.

        The newly created plot appears as one of the rows in the timeline.

        Args:
            timeline_row_name: Label for the timeline row, shown to the left
                of the plot. If a row with the provided name exists, the data
                are added as another series to the already existing plot.
            data: Each data point is a ``(timestamp, value)`` pair.
                Timestamps are in nanoseconds.
            plot_name: Optional name that shown in the row legend if one row
                displays multiple time series.
            color: Optional line color. If the color not provided, the default
                color palette is used.
            tooltip: TODO.
            **kwargs: Additional plotting options passed to pyqtgraph.

        Returns:
            The timeline plot, or ``None`` if no recording is open.
        """
        app = neon_player.instance()
        recording = app.recording
        if recording is None:
            return None

        plot_item = self.get_timeline_plot(timeline_row_name, True)

        if tooltip is not None:
            self.set_timeline_tooltip(timeline_row_name, tooltip)

        if color is None:
            plot_index = len(plot_item.items)
            color = self.plot_colors[plot_index % len(self.plot_colors)]

        if "pen" not in kwargs:
            kwargs["pen"] = pg.mkPen(color=color, width=2, cap="flat")

        legend = self.timeline_legends[timeline_row_name]
        data = np.asarray(data)
        if len(data) > 0:
            plot_data_item = plot_item.plot(
                data[:, 0], data[:, 1], name=plot_name, **kwargs
            )
        else:
            plot_data_item = plot_item.plot(x=[], y=[], name=plot_name, **kwargs)

        plot_data_item.name = plot_name
        if kwargs.get("is_event"):
            plot_data_item.is_event = True
        if plot_name != "":
            legend.addItem(plot_data_item, plot_name)

        if self.plot_sorting_enabled:
            self.sort_plots()

        return plot_item

    def fix_scroll_size(self) -> None:
        h = sum([p.preferredHeight() for p in self.timeline_plots.values()])
        self.graphics_view.setFixedHeight(h)
        self.playhead.refresh_geometry()
        self.hover_marker.refresh_geometry()

    def remove_timeline_plot(self, plot_name: str) -> None:
        """Remove a timeline plot and all the data series it contains.

        Args:
            plot_name: Label of the row to remove.
        """
        plot = self.get_timeline_plot(plot_name)
        if plot is None:
            return

        with contextlib.suppress(ValueError):
            if plot_name in self.timeline_tooltips:
                self.tooltip_widget.hide()
            self.graphics_layout.removeItem(
                self.timeline_legends[plot_name].parentItem()
            )
            self.graphics_layout.removeItem(self.timeline_plots[plot_name])
            del self.timeline_plots[plot_name]
            del self.timeline_legends[plot_name]
            tooltip_config = self.timeline_tooltips.pop(plot_name, None)
            if tooltip_config is not None:
                TimeLineDock._release_timeline_tooltip_widgets(
                    self, plot_name, (tooltip_config,)
                )

        if self.plot_sorting_enabled:
            self.sort_plots()

    def remove_timeline_series(self, plot_name: str, series_name: str) -> None:
        """Remove one named data series, removing its row if it becomes empty.

        Args:
            plot_name: Label of the row containing the series.
            series_name: Name shown for the series in the row legend.
        """
        if plot_name not in self.timeline_plots:
            return

        plot = self.get_timeline_plot(plot_name)
        if plot is None:
            return

        series = self.get_timeline_series(plot_name, series_name)
        if series is None:
            return

        plot.removeItem(series)
        if plot_name in self.timeline_legends:
            legend = self.timeline_legends[plot_name]
            legend.removeItem(series_name)

        if len(plot.items) == 0:
            self.remove_timeline_plot(plot_name)

    def on_data_point_clicked(
        self, timeline_name: str, data_point: tuple[int, T.SupportsFloat]
    ) -> None:
        if timeline_name not in self.data_point_actions:
            return

        context_menu = QMenu()

        for action_name, callback in self.data_point_actions[timeline_name].items():
            action = context_menu.addAction(action_name)
            action.triggered.connect(lambda _, cb=callback: cb(data_point))

        context_menu.exec(QCursor.pos())

    def add_timeline_line(
        self,
        timeline_row_name: str,
        data: list[tuple[int, T.SupportsFloat]]
        | NDArray[np.integer[T.Any] | np.floating[T.Any]],
        plot_name: str = "",
        tooltip: TimelineTooltipConfig | None = None,
        show_y_limit_labels: bool | None = None,
        units: str | None = None,
        **kwargs: T.Any,  # noqa: ANN401
    ) -> TimelinePlotItem | None:
        """Add a line graph to a timeline row.

        Args:
            timeline_row_name: Label for the timeline row.
            data: Timestamp and value pairs; timestamps are in nanoseconds.
            plot_name: Optional name shown in the row legend.
            tooltip: TODO.
            show_y_limit_labels: Whether to show labels for y-axis limits.
                The limits are shown by default for line plots.
            units: Units to display in the labels of y-axis limits.
            **kwargs: Additional line options passed to pyqtgraph.

        Returns:
            The timeline row, or ``None`` if no recording is open.
        """
        plot_existed = timeline_row_name in self.timeline_plots
        plot_item = self.add_timeline_plot(timeline_row_name, data, plot_name, tooltip=tooltip, **kwargs)
        if plot_item is None:
            return None

        # Show y-axis limits by default for line plots
        if show_y_limit_labels is None and not plot_existed:
            show_y_limit_labels = True
        plot_item.configure_y_limit_labels(show_y_limit_labels, units)

        return plot_item

    def add_timeline_scatter(
        self,
        name: str,
        data: list[tuple[int, T.SupportsFloat]] | np.ndarray,
        item_name: str = "",
        tooltip: TimelineTooltipConfig | None = None,
    ) -> TimelinePlotItem | None:
        """Add a scatter plot to a timeline row.

        Args:
            name: Label for the timeline row.
            data: Timestamp and value pairs; timestamps are in nanoseconds.
            item_name: Optional name shown in the row legend.
            tooltip: TODO.

        Returns:
            The timeline row, or ``None`` if no recording is open.
        """
        return self.add_timeline_plot(
            name,
            data,
            item_name,
            pen=None,
            symbol="d",
            symbolBrush=pg.mkColor("white"),
            tooltip=tooltip,
        )

    def add_timeline_broken_bar(
        self,
        timeline_row_name: str,
        data: list[tuple[int, int, T.SupportsFloat]]
        | list[tuple[int, int]]
        | NDArray[np.integer[T.Any] | np.floating[T.Any]],
        item_name: str = "",
        color: str = "white",
        tooltip: TimelineTooltipConfig | None = None,
    ) -> TimelinePlotItem | None:
        """Add horizontal bars to show time ranges on the timeline.

        Args:
            timeline_row_name: Label for the timeline row.
            data: Time ranges as ``(start, end)`` pairs, or
                ``(start, end, value)`` triples. Timestamps are in nanoseconds;
                values control the bar colors when supplied.
            item_name: Optional name shown in the row legend.
            color: Color name or pyqtgraph color map name.
            tooltip: TODO.

        Returns:
            The timeline row, or ``None`` if no recording is open.
        """
        plot_widget = self.get_timeline_plot(timeline_row_name, True)

        if tooltip is not None:
            self.set_timeline_tooltip(timeline_row_name, tooltip)

        plot_widget.getViewBox().allow_y_panning = False
        # plot_widget.getViewBox().setMouseEnabled(y=False)
        # plot_widget.setMenuEnabled(False)

        if len(data) > 0:
            columns = list(zip(*data, strict=False))
            starts_raw, stops_raw = (
                np.array(columns[0], dtype=np.int64),
                np.array(columns[1], dtype=np.int64),
            )

            valid_mask = stops_raw >= starts_raw
            if not np.all(valid_mask):
                stops_raw = np.maximum(starts_raw, stops_raw)

            values = (
                np.array(columns[2], dtype=np.float64)
                if len(columns) > 2
                else np.full(len(starts_raw), np.nan)
            )
            color_values = values if not np.all(np.isnan(values)) else np.array([])
            pens, brushes = _resolve_bar_colors(color_values, color, len(starts_raw))

            bars = pg.BarGraphItem(
                x0=starts_raw,
                x1=stops_raw,
                y0=-0.4,
                y1=0.4,
                pens=pens,
                brushes=brushes,
                name=item_name,
            )
            bars.bar_values = values
            plot_widget.addItem(bars)
        else:
            bars = []

        plot_widget.getViewBox().autoRange(padding=0.7)

        if item_name and timeline_row_name in self.timeline_legends:
            legend = self.timeline_legends[timeline_row_name]
            legend.addItem(bars, name=item_name)

        if self.plot_sorting_enabled:
            self.sort_plots()

        return plot_widget

    def sort_plots(self) -> None:
        items_to_move = {}
        for move_row in range(self.graphics_layout.currentRow, 1, -1):
            legend = self.graphics_layout.getItem(move_row, 0)
            plot = self.graphics_layout.getItem(move_row, 1)

            if plot is None:
                continue

            if plot.has_line:
                prefix = "30"
            elif plot.has_bar:
                prefix = "10"
            else:
                prefix = "20"

            sort_key = f"{prefix}-{legend.rows[0][0].text.lower()}"
            items_to_move[sort_key] = (legend, plot)

            self.graphics_layout.removeItem(legend)
            self.graphics_layout.removeItem(plot)

            del self.graphics_layout.rows[move_row]

        self.graphics_layout.currentRow = 1

        sorted_keys = sorted(items_to_move.keys())

        for key in sorted_keys:
            legend, plot = items_to_move[key]
            row = self.graphics_layout.nextRow()
            self.graphics_layout.addItem(legend, row=row, col=0)
            self.graphics_layout.addItem(plot, row=row, col=1)

        self.fix_scroll_size()

    def enable_plot_sorting(self) -> None:
        """Enable plot sorting and sort plots."""
        self.plot_sorting_enabled = True
        self.sort_plots()

    def disable_plot_sorting(self) -> bool:
        """Disable plot sorting.

        Returns:
            was_enabled: Whether the sorting was enabled prior to that call.
        """
        was_enabled = self.plot_sorting_enabled
        self.plot_sorting_enabled = False
        return was_enabled

    def register_data_point_action(
        self,
        row_name: str,
        action_name: str,
        callback: T.Callable[[tuple[int, T.SupportsFloat]], object],
    ) -> None:
        if row_name not in self.data_point_actions:
            self.data_point_actions[row_name] = {}

        self.data_point_actions[row_name][action_name] = callback

    def unregister_data_point_action(self, row_name: str, action_name: str) -> None:
        if row_name not in self.data_point_actions:
            return

        if action_name not in self.data_point_actions[row_name]:
            return

        del self.data_point_actions[row_name][action_name]

    def reset_view(self) -> None:
        app = neon_player.instance()
        recording = app.recording
        if recording is None:
            return

        for plot_item in self.timeline_plots.values():
            padding = 0.7 if plot_item.has_bar else None
            plot_item.getViewBox().autoRange(padding=padding)

        self.timestamps_plot.getViewBox().setRange(
            xRange=[recording.start_time, recording.stop_time]
        )

        self.playhead.refresh_geometry()

    def get_export_window(self) -> tuple[int, int]:
        times = [int(tm.time) for tm in self.trim_markers]
        times.sort()
        return times[0], times[1]

    def set_export_window(self, times: tuple[int, int]) -> None:
        self.trim_markers[0].time = times[0]
        self.trim_markers[1].time = times[1]


def _resolve_bar_colors(
    values: NDArray[np.float64], color_arg: str = "white", count: int = 1
) -> tuple[list[QPen], list[QBrush]]:
    """Determine pen and brush colors for bars based on values and color arg."""
    if values.size == 0 or np.all(np.isnan(values)):
        c = pg.mkColor(color_arg)
        pen = pg.mkPen(c)
        brush = pg.mkBrush(c)
        return [pen] * count, [brush] * count

    min_v, max_v = np.min(values), np.max(values)
    if max_v == min_v:
        norm_values = np.full_like(values, 0.5)
    else:
        norm_values = (values - min_v) / (max_v - min_v)

    try:
        cmap = pg.colormap.get(color_arg)
    except FileNotFoundError:
        cmap = pg.colormap.get("viridis")

    colors = cmap.map(norm_values, mode="qcolor")
    return [pg.mkPen(c) for c in colors], [pg.mkBrush(c) for c in colors]
