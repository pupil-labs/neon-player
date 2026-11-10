import logging
import typing as T
from functools import partial

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
    QGraphicsProxyWidget,
    QGraphicsSceneMouseEvent,
    QHBoxLayout,
    QMenu,
    QScrollArea,
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
    TimelinePlotGroup,
    TimelinePlotGroupControl,
    TimelinePlotItem,
    TimelineRowKey,
    TimelineRowRole,
    TimelineTableContainer,
    TimestampLabel,
    TrimDurationMarker,
    TrimEndMarker,
)
from pupil_labs.neon_player.ui.timeline_dock_utils import (
    get_clicked_data_point,
    get_clicked_plot_item,
)
from pupil_labs.neon_player.utilities import clone_menu

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

    key_pressed = Signal(QKeyEvent)

    def __init__(self) -> None:
        super().__init__()
        app = neon_player.instance()

        self.timeline_plots: dict[TimelineRowKey, TimelinePlotItem] = {}
        self.timeline_groups: dict[str, TimelinePlotGroup] = {}
        self.timeline_group_controls: dict[str, TimelinePlotGroupControl] = {}
        self.timeline_group_control_proxies: dict[str, QGraphicsProxyWidget] = {}
        self.hidden_timeline_rows: set[TimelineRowKey] = set()
        self.timeline_legends: dict[TimelineRowKey, pg.LegendItem] = {}
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
            TimelineRowKey, dict[str, T.Callable[[tuple[int, T.SupportsFloat]], object]]
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

        self.graphics_view.scene().sigMouseClicked.connect(self.on_chart_area_clicked)
        self.graphics_view.scene().sigMouseMoved.connect(self.on_chart_area_mouse_moved)

        self.scroll_area = QScrollArea()
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

        self.setMouseTracking(True)

        # Add a permanent timeline with timestamps
        self.timestamps_plot = self.get_timeline_plot(
            "Export window", create_if_missing=True
        )
        self.timestamps_plot.showAxis("top")
        self.timestamps_plot.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )

        self.playhead = PlayHead(self.timestamps_plot, parent=self.graphics_view)
        self.playhead.hide()

        app.playback_state_changed.connect(self.on_playback_state_changed)
        app.position_changed.connect(self.on_position_changed)
        app.recording_loaded.connect(self.on_recording_loaded)
        app.recording_unloaded.connect(self.on_recording_unloaded)

        self.dragging: TrimEndMarker | None = None
        self.trim_markers: list[TrimEndMarker] = []
        self.plot_sorting_enabled = True

    def add_timeline_group(
        self, name: str, *, collapsed: bool = False
    ) -> TimelinePlotGroup:
        """Create a flat group whose main row displays the group name.

        Args:
            name: Name shared by the group and its main row. It must be unique
                among groups and ungrouped rows. Child rows may use the same name.
            collapsed: Whether to initially hide the group's children.

        Returns:
            The registered group handle.

        Raises:
            ValueError: The name is already used by a group or ungrouped row.
        """
        if (
            name in self.timeline_groups
            or self._timeline_row_key(name) in self.timeline_plots
        ):
            raise ValueError(f"Global timeline name '{name}' already exists")
        group = TimelinePlotGroup(name, collapsed=collapsed)
        self.timeline_groups[name] = group
        key = TimelineRowKey(name, name, TimelineRowRole.HEADER)
        self._get_timeline_plot_by_key(key, True)
        self._create_timeline_group_control(group)
        if self.plot_sorting_enabled:
            self.sort_plots()
        return group

    def get_timeline_group(self, name: str) -> TimelinePlotGroup | None:
        """Return the registered group handle, or None when absent."""
        return self.timeline_groups.get(name)

    def remove_timeline_group(self, group: str | TimelinePlotGroup) -> None:
        """Remove a group's main row, children, point actions, and expander."""
        resolved = self._resolve_timeline_group(group)
        assert resolved is not None
        for key in list(self.timeline_plots):
            if key.group_name == resolved.name:
                self._remove_timeline_plot_by_key(key)
        for key in list(self.data_point_actions):
            if key.group_name == resolved.name:
                self.data_point_actions.pop(key)
        proxy = self.timeline_group_control_proxies.pop(resolved.name)
        self.graphics_layout.removeItem(proxy)
        proxy.deleteLater()
        self.timeline_group_controls.pop(resolved.name)
        self.timeline_groups.pop(resolved.name)
        resolved._child_row_names.clear()
        if self.plot_sorting_enabled:
            self.sort_plots()
        else:
            self._refresh_timeline_layout()

    def _create_timeline_group_control(self, group: TimelinePlotGroup) -> None:
        button = TimelinePlotGroupControl(group.name, group.collapsed)
        button.clicked.connect(partial(self._toggle_timeline_group, group))
        proxy = QGraphicsProxyWidget()
        proxy.setWidget(button)
        header = self.timeline_plots[
            TimelineRowKey(group.name, group.name, TimelineRowRole.HEADER)
        ]
        row = next(
            index
            for index in self.graphics_layout.rows
            if self.graphics_layout.getItem(index, 2) is header
        )
        self.graphics_layout.addItem(proxy, row=row, col=0)
        self.timeline_group_controls[group.name] = button
        self.timeline_group_control_proxies[group.name] = proxy

    def _toggle_timeline_group(
        self, group: TimelinePlotGroup, _checked: bool = False
    ) -> None:
        resolved = self._resolve_timeline_group(group)
        assert resolved is not None
        resolved._collapsed = not resolved.collapsed
        for key in self.timeline_plots:
            if (
                key.group_name == resolved.name
                and key.role is not TimelineRowRole.HEADER
            ):
                self._set_timeline_row_visible(key, not resolved.collapsed)
        self.timeline_group_controls[resolved.name].set_collapsed(resolved.collapsed)
        self._refresh_timeline_layout()

    def _set_timeline_row_visible(self, key: TimelineRowKey, visible: bool) -> None:
        self.timeline_plots[key].setVisible(visible)
        self.timeline_legends[key].parentItem().setVisible(visible)
        if visible:
            self.hidden_timeline_rows.discard(key)
        else:
            self.hidden_timeline_rows.add(key)

    def _refresh_timeline_layout(self) -> None:
        self.graphics_layout.layout.invalidate()
        self.graphics_layout.updateGeometry()
        self.fix_scroll_size()

    def sizeHint(self) -> QSize:
        return QSize(100, 150)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        self.key_pressed.emit(event)

    def resizeEvent(self, event: QResizeEvent) -> None:
        w = self.scroll_area.width() - self.scroll_area.verticalScrollBar().width()
        self.graphics_view.setFixedWidth(w)
        self.playhead.refresh_geometry()
        return super().resizeEvent(event)

    def on_recording_loaded(self, recording: nr.NeonRecording) -> None:
        app = neon_player.instance()

        self.playhead.show()
        for plot_item in self.timeline_plots.values():
            plot_item.setXRange(recording.start_time, recording.stop_time, padding=0)
            duration = recording.stop_time - recording.start_time
            plot_item.getViewBox().setLimits(
                xMin=recording.start_time - duration * 0.05,
                xMax=recording.stop_time + duration * 0.05,
            )
            axis = plot_item.getAxis("top")
            axis.set_time_frame(recording.start_time, recording.stop_time)

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
        trim_plot = self.get_timeline_plot("Export window", create_if_missing=False)
        if trim_plot is not None:
            trim_plot.clear()
        self.playhead.hide()
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

        self._show_data_point_actions(clicked_plot_item.row_key, clicked_data_point)

    @T.overload
    def get_timeline_plot(
        self,
        timeline_row_name: str,
        create_if_missing: T.Literal[True],
        *,
        group: str | TimelinePlotGroup | None = None,
    ) -> TimelinePlotItem: ...

    @T.overload
    def get_timeline_plot(
        self,
        timeline_row_name: str,
        create_if_missing: bool = False,
        *,
        group: str | TimelinePlotGroup | None = None,
    ) -> TimelinePlotItem | None: ...

    def get_timeline_plot(
        self,
        timeline_row_name: str,
        create_if_missing: bool = False,
        *,
        group: str | TimelinePlotGroup | None = None,
    ) -> TimelinePlotItem | None:
        """Get or create a timeline plot (i.e., a row in the timeline).

        Args:
            timeline_row_name: The name of the timeline plot.
            create_if_missing: Whether to create the plot if it doesn't
                exist already.
            group: Group name or exact handle; None targets an ungrouped row.

        Returns:
            plot: The created/existing plot, or `None` if the plot
                did not exist and was not created.
        """
        key = self._timeline_row_key(timeline_row_name, group)
        return self._get_timeline_plot_by_key(key, create_if_missing)

    def _get_timeline_plot_by_key(
        self, key: TimelineRowKey, create_if_missing: bool = False
    ) -> TimelinePlotItem | None:
        if key in self.timeline_plots:
            return self.timeline_plots[key]

        if not create_if_missing:
            return None

        self._validate_timeline_row_creation(key)
        timeline_row_name = key.row_name
        logger.debug(f"Adding plot '{timeline_row_name}' to timeline")

        row = self.graphics_layout.nextRow()
        is_timestamps_row = key == TimelineRowKey(
            None, "Export window", TimelineRowRole.CUSTOM
        )

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

        plot_item.row_key = key

        if is_timestamps_row:
            plot_item.preferred_height_1d = 50
            plot_item.adjust_size()
            legend_label.anchor((0.5, 0), (0.5, 0), (0, 20))

        legend.layout.setSpacing(0)
        self.timeline_legends[key] = legend

        self.graphics_layout.addItem(legend_container, row=row, col=1)
        self.graphics_layout.addItem(plot_item, row=row, col=2)

        plot_item.setMouseEnabled(x=False, y=False)
        plot_item.hideButtons()
        plot_item.setMenuEnabled(False)
        plot_item.setClipToView(True)
        plot_item.hideAxis("left")
        plot_item.hideAxis("right")
        plot_item.hideAxis("bottom")
        plot_item.showGrid(x=True, y=False, alpha=0.3)

        if recording is not None:
            duration = recording.stop_time - recording.start_time
            plot_item.getViewBox().setLimits(
                xMin=recording.start_time - duration * 0.05,
                xMax=recording.stop_time + duration * 0.05,
            )

        self.timeline_plots[key] = plot_item
        self._record_timeline_child(key)

        if not is_timestamps_row and self.timestamps_plot:
            plot_item.setXRange(*self.timestamps_plot.viewRange()[0], padding=0)
            plot_item.setXLink(self.timestamps_plot)

        return plot_item

    def _record_timeline_child(self, key: TimelineRowKey) -> None:
        if key.group_name is None or key.role is not TimelineRowRole.CUSTOM:
            return

        row_group = self.timeline_groups[key.group_name]
        row_group._child_row_names.add(key.row_name)
        if row_group.collapsed:
            self._set_timeline_row_visible(key, False)
            self._refresh_timeline_layout()

    def get_timeline_series(
        self,
        plot_name: str,
        series_name: str,
        *,
        group: str | TimelinePlotGroup | None = None,
    ) -> pg.PlotDataItem | None:
        """Access timeline data series by their name.

        Args:
            plot_name: The name of the timeline row.
            series_name: Name shown for the series in the row legend.
            group: Group name or exact handle; None targets an ungrouped row.

        Returns:
            series: The data series, if it exists.
        """
        plot_item = self.get_timeline_plot(plot_name, group=group)
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
        *,
        group: str | TimelinePlotGroup | None = None,
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
            group: Group name or exact handle; None targets an ungrouped row.
            **kwargs: Additional plotting options passed to pyqtgraph.

        Returns:
            The timeline plot, or ``None`` if no recording is open.
        """
        key = self._timeline_row_key(timeline_row_name, group)
        self._validate_timeline_row_creation(key)
        app = neon_player.instance()
        recording = app.recording
        if recording is None:
            return None

        plot_item = self.get_timeline_plot(timeline_row_name, True, group=group)

        if color is None:
            plot_index = len(plot_item.items)
            color = self.plot_colors[plot_index % len(self.plot_colors)]

        if "pen" not in kwargs:
            kwargs["pen"] = pg.mkPen(color=color, width=2, cap="flat")

        legend = self.timeline_legends[key]
        plot_data_item = plot_item.plot(data, name=plot_name, **kwargs)
        plot_data_item.name = plot_name
        if plot_name != "":
            legend.addItem(plot_data_item, plot_name)

        if self.plot_sorting_enabled:
            self.sort_plots()

        return plot_item

    def fix_scroll_size(self) -> None:
        h = sum(
            plot.preferredHeight()
            for key, plot in self.timeline_plots.items()
            if key not in self.hidden_timeline_rows
        )
        self.graphics_view.setFixedHeight(h)
        self.playhead.refresh_geometry()

    def remove_timeline_plot(
        self, plot_name: str, *, group: str | TimelinePlotGroup | None = None
    ) -> None:
        """Remove a row in the requested scope, including its series and actions."""
        key = self._timeline_row_key(plot_name, group)
        self._remove_timeline_plot_by_key(key)
        if self.plot_sorting_enabled:
            self.sort_plots()
        else:
            self._refresh_timeline_layout()

    def _remove_timeline_plot_by_key(self, key: TimelineRowKey) -> None:
        plot = self.timeline_plots.pop(key, None)
        if plot is not None:
            if key.group_name is not None:
                plot.setXLink(None)
                plot.close()
            self.graphics_layout.removeItem(plot)
            if key.group_name is not None:
                plot.deleteLater()
        legend = self.timeline_legends.pop(key, None)
        if legend is not None:
            container = legend.parentItem()
            self.graphics_layout.removeItem(container)
            container.deleteLater()
        self.hidden_timeline_rows.discard(key)
        self.data_point_actions.pop(key, None)
        if key.group_name is not None and key.role is TimelineRowRole.CUSTOM:
            self.timeline_groups[key.group_name]._child_row_names.discard(key.row_name)

    def remove_timeline_series(
        self,
        plot_name: str,
        series_name: str,
        *,
        group: str | TimelinePlotGroup | None = None,
    ) -> None:
        """Remove one named data series, removing its row if it becomes empty.

        Args:
            plot_name: Label of the row containing the series.
            series_name: Name shown for the series in the row legend.
            group: Group name or exact handle; None targets an ungrouped row.
        """
        key = self._timeline_row_key(plot_name, group)
        if key not in self.timeline_plots:
            return

        plot = self.get_timeline_plot(plot_name, group=group)
        if plot is None:
            return

        series = self.get_timeline_series(plot_name, series_name, group=group)
        if series is None:
            return

        plot.removeItem(series)
        if key in self.timeline_legends:
            legend = self.timeline_legends[key]
            legend.removeItem(series_name)

        if len(plot.items) == 0:
            self.remove_timeline_plot(plot_name, group=group)

    def on_data_point_clicked(
        self,
        timeline_name: str,
        data_point: tuple[int, T.SupportsFloat],
        *,
        group: str | TimelinePlotGroup | None = None,
    ) -> None:
        self._show_data_point_actions(
            self._timeline_row_key(timeline_name, group), data_point
        )

    def _show_data_point_actions(
        self, key: TimelineRowKey, data_point: tuple[int, T.SupportsFloat]
    ) -> None:
        if key not in self.data_point_actions:
            return

        context_menu = QMenu()

        for action_name, callback in self.data_point_actions[key].items():
            action = context_menu.addAction(action_name)
            action.triggered.connect(lambda _, cb=callback: cb(data_point))

        context_menu.exec(QCursor.pos())

    def add_timeline_line(
        self,
        timeline_row_name: str,
        data: list[tuple[int, T.SupportsFloat]]
        | NDArray[np.integer[T.Any] | np.floating[T.Any]],
        plot_name: str = "",
        *,
        group: str | TimelinePlotGroup | None = None,
        **kwargs: T.Any,  # noqa: ANN401
    ) -> TimelinePlotItem | None:
        """Add a line graph to a timeline row.

        Args:
            timeline_row_name: Label for the timeline row.
            data: Timestamp and value pairs; timestamps are in nanoseconds.
            plot_name: Optional name shown in the row legend.
            group: Group name or exact handle; None targets an ungrouped row.
            **kwargs: Additional line options passed to pyqtgraph.

        Returns:
            The timeline row, or ``None`` if no recording is open.
        """
        return self.add_timeline_plot(
            timeline_row_name, data, plot_name, group=group, **kwargs
        )

    def add_timeline_scatter(
        self,
        name: str,
        data: list[tuple[int, T.SupportsFloat]]
        | NDArray[np.integer[T.Any] | np.floating[T.Any]],
        item_name: str = "",
        *,
        group: str | TimelinePlotGroup | None = None,
    ) -> TimelinePlotItem | None:
        """Add a scatter plot to a timeline row.

        Args:
            name: Label for the timeline row.
            data: Timestamp and value pairs; timestamps are in nanoseconds.
            item_name: Optional name shown in the row legend.
            group: Group name or exact handle; None targets an ungrouped row.

        Returns:
            The timeline row, or ``None`` if no recording is open.
        """
        return self.add_timeline_plot(
            name,
            data,
            item_name,
            group=group,
            pen=None,
            symbol="d",
            symbolBrush=pg.mkColor("white"),
        )

    def add_timeline_broken_bar(
        self,
        timeline_row_name: str,
        data: list[tuple[int, int, T.SupportsFloat]]
        | list[tuple[int, int]]
        | NDArray[np.integer[T.Any] | np.floating[T.Any]],
        item_name: str = "",
        color: str = "white",
        *,
        group: str | TimelinePlotGroup | None = None,
    ) -> TimelinePlotItem | None:
        """Add horizontal bars to show time ranges on the timeline.

        Args:
            timeline_row_name: Label for the timeline row.
            data: Time ranges as ``(start, end)`` pairs, or
                ``(start, end, value)`` triples. Timestamps are in nanoseconds;
                values control the bar colors when supplied.
            item_name: Optional name shown in the row legend.
            group: Group name or exact handle; None targets an ungrouped row.
            color: Color name or pyqtgraph color map name.

        Returns:
            The timeline row, or ``None`` if no recording is open.
        """
        key = self._timeline_row_key(timeline_row_name, group)
        self._validate_timeline_row_creation(key)
        if neon_player.instance().recording is None:
            return None
        plot_widget = self.get_timeline_plot(timeline_row_name, True, group=group)

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
                x0=starts_raw, x1=stops_raw, y0=-0.4, y1=0.4, pens=pens, brushes=brushes
            )
            plot_widget.addItem(bars)
        else:
            bars = []

        plot_widget.getViewBox().autoRange(padding=0.7)

        key = self._timeline_row_key(timeline_row_name, group)
        if item_name and key in self.timeline_legends:
            legend = self.timeline_legends[key]
            legend.addItem(bars, name=item_name)

        if self.plot_sorting_enabled:
            self.sort_plots()

        return plot_widget

    def sort_plots(self) -> None:
        rows: dict[TimelineRowKey, tuple[T.Any, T.Any, TimelinePlotItem]] = {}
        for row in range(self.graphics_layout.currentRow, 1, -1):
            plot = self.graphics_layout.getItem(row, 2)
            if plot is None:
                continue
            control = self.graphics_layout.getItem(row, 0)
            legend = self.graphics_layout.getItem(row, 1)
            rows[plot.row_key] = control, legend, plot
            for item in (control, legend, plot):
                if item is not None:
                    self.graphics_layout.removeItem(item)
            del self.graphics_layout.rows[row]

        blocks: list[
            tuple[tuple[int, str, str], list[tuple[T.Any, T.Any, TimelinePlotItem]]]
        ] = []
        for group in self.timeline_groups.values():
            keys = [key for key in rows if key.group_name == group.name]
            keys.sort(key=lambda key: self._timeline_group_row_sort_key(rows[key][2]))
            blocks.append(
                (
                    (20, group.name.lower(), group.name),
                    [rows.pop(key) for key in keys],
                )
            )
        for item in rows.values():
            priority, name = self._timeline_plot_sort_key(item[2])
            blocks.append(((priority, name, ""), [item]))

        self.graphics_layout.currentRow = 1
        for _, block in sorted(blocks, key=lambda entry: entry[0]):
            for control, legend, plot in block:
                row = self.graphics_layout.nextRow()
                if control is not None:
                    self.graphics_layout.addItem(control, row=row, col=0)
                self.graphics_layout.addItem(legend, row=row, col=1)
                self.graphics_layout.addItem(plot, row=row, col=2)
        self._refresh_timeline_layout()

    @staticmethod
    def _timeline_plot_sort_key(plot: TimelinePlotItem) -> tuple[int, str]:
        priority = 20
        if plot.row_key.role is TimelineRowRole.CUSTOM:
            priority = 30 if plot.has_line else 10 if plot.has_bar else 20
        return priority, plot.row_name.lower()

    @classmethod
    def _timeline_group_row_sort_key(
        cls, plot: TimelinePlotItem
    ) -> tuple[int, int, str]:
        roles = {
            TimelineRowRole.HEADER: 0,
            TimelineRowRole.SEARCH: 1,
            TimelineRowRole.CUSTOM: 2,
        }
        priority, name = cls._timeline_plot_sort_key(plot)
        return roles[plot.row_key.role], priority, name

    def _validate_timeline_row_creation(self, key: TimelineRowKey) -> None:
        if key.group_name is None and key.row_name in self.timeline_groups:
            raise ValueError(f"Global timeline name '{key.row_name}' already exists")

    def _resolve_timeline_group(
        self, group: str | TimelinePlotGroup | None
    ) -> TimelinePlotGroup | None:
        if group is None:
            return None
        if isinstance(group, str):
            return self.timeline_groups[group]
        if self.timeline_groups.get(group.name) is not group:
            raise ValueError("Timeline row group does not belong to this timeline")
        return group

    def _timeline_row_key(
        self, row_name: str, group: str | TimelinePlotGroup | None = None
    ) -> TimelineRowKey:
        resolved = self._resolve_timeline_group(group)
        return TimelineRowKey(
            resolved.name if resolved is not None else None,
            row_name,
            TimelineRowRole.CUSTOM,
        )

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
        *,
        group: str | TimelinePlotGroup | None = None,
    ) -> None:
        key = self._timeline_row_key(row_name, group)
        if key not in self.data_point_actions:
            self.data_point_actions[key] = {}

        self.data_point_actions[key][action_name] = callback

    def unregister_data_point_action(
        self,
        row_name: str,
        action_name: str,
        *,
        group: str | TimelinePlotGroup | None = None,
    ) -> None:
        key = self._timeline_row_key(row_name, group)
        if key not in self.data_point_actions:
            return

        if action_name not in self.data_point_actions[key]:
            return

        del self.data_point_actions[key][action_name]

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
