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
from PySide6.QtCore import QEvent, QObject, QPoint, QPointF, QSize, Qt, QTimer, Signal
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
    TimelinePinnedAxis,
    TimelinePlotGroup,
    TimelinePlotGroupControl,
    TimelinePlotGroupSearchBar,
    TimelinePlotItem,
    TimelinePlotLabelHost,
    TimelineRowKey,
    TimelineRowRole,
    TimelineScrollArea,
    TimelineTableContainer,
    TimestampLabel,
    TrimDurationMarker,
    TrimEndMarker,
)
from pupil_labs.neon_player.ui.timeline_dock_tooltips import (
    TimelineTooltipConfig,
    TimelineTooltipContext,
    TimelineTooltipOverlay,
    TimelineTooltipWidget,
    _get_hovered_timeline_series,
    _normalize_timeline_tooltip,
)
from pupil_labs.neon_player.ui.timeline_dock_utils import (
    get_clicked_data_point,
    get_clicked_plot_item,
)
from pupil_labs.neon_player.utilities import SlotDebouncer, clone_menu

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

    def __init__(self) -> None:
        super().__init__()
        app = neon_player.instance()

        self._axis_geometry_timer = QTimer(self)
        self._axis_geometry_timer.setSingleShot(True)
        self._axis_geometry_timer.timeout.connect(self._sync_pinned_axis_geometry)

        self.recording: nr.NeonRecording | None = None

        self.timeline_plots: dict[TimelineRowKey, TimelinePlotItem] = {}
        self.timeline_groups: dict[str, TimelinePlotGroup] = {}
        self.timeline_group_controls: dict[str, TimelinePlotGroupControl] = {}
        self.timeline_group_control_proxies: dict[str, QGraphicsProxyWidget] = {}
        self.hidden_timeline_rows: set[TimelineRowKey] = set()
        self.timeline_legends: dict[TimelineRowKey, pg.LegendItem] = {}
        self.timeline_tooltips: dict[TimelineRowKey, TimelineTooltipConfig | None] = {}
        self._tooltip_widget_rows: dict[int, TimelineRowKey] = {}
        self._searchbars: dict[str, TimelinePlotGroupSearchBar] = {}

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
        self.main_layout.setSpacing(0)
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

        self.scroll_area = TimelineScrollArea()
        self.scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOn
        )
        SlotDebouncer.debounce(
            self.scroll_area.verticalScrollBar().valueChanged,
            self._on_scroll_or_playback,
            delay=0.03,
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
        self.timestamps_plot.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.timestamps_plot.getViewBox().sigXRangeChanged.connect(
            lambda: self._on_scroll_or_playback(0)
        )

        self.pinned_axis = TimelinePinnedAxis(
            self.timestamps_plot.getViewBox(),
        )
        self.main_layout.insertWidget(1, self.pinned_axis)
        self._axis_geometry_timer.start(0)

        self.playhead = PlayHead(self.timestamps_plot, parent=self.graphics_view)
        self.playhead.hide()

        self.hover_marker = TimelineHoverMarker(
            self.timestamps_plot, self.pinned_axis, parent=self.graphics_view
        )
        self.hover_marker.hide()
        self.scroll_area.installEventFilter(self)
        self.graphics_view.viewport().installEventFilter(self)

        self.tooltip_widget = TimelineTooltipOverlay(self.scroll_area)

        # Ensure at least 10 ms pass between successive tooltip updates
        self._pending_tooltip_update: tuple[QPointF, int] | None = None
        self._context_menu_open = False
        self._tooltip_update_timer = QTimer(self)
        self._tooltip_update_timer.setSingleShot(True)
        self._tooltip_update_timer.setInterval(10)
        self._tooltip_update_timer.timeout.connect(self._update_pending_tooltip)

        app.playback_state_changed.connect(self.on_playback_state_changed)
        app.position_changed.connect(self.on_position_changed)
        app.recording_loaded.connect(self.on_recording_loaded)
        app.recording_unloaded.connect(self.on_recording_unloaded)

        self.dragging: TrimEndMarker | None = None
        self.trim_markers: list[TrimEndMarker] = []
        self.plot_sorting_enabled = True

    def add_timeline_group(
        self, name: str, *, collapsed: bool = False, searchable: bool = False
    ) -> TimelinePlotGroup:
        """Create a flat group whose main row displays the group name.

        Args:
            name: Name shared by the group and its main row. It must be unique
                among groups and ungrouped rows. Child rows may use the same name.
            collapsed: Whether to initially hide the group's children.
            searchable: Show a search field that allows filtering child rows by name.

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
        group = TimelinePlotGroup(name, collapsed=collapsed, searchable=searchable)
        self.timeline_groups[name] = group
        key = TimelineRowKey(name, name, TimelineRowRole.HEADER)
        self._get_timeline_plot_by_key(key, True)
        self._create_timeline_group_control(group)
        if group.searchable:
            self._create_timeline_search(group)
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
        self._dispose_timeline_search(resolved)
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
        self._set_group_collapsed(resolved, not resolved.collapsed)
        self._refresh_timeline_layout()

    def _set_group_collapsed(self, group: TimelinePlotGroup, collapsed: bool) -> None:
        group._collapsed = collapsed
        self._apply_group_visibility(group)
        self.timeline_group_controls[group.name].set_collapsed(collapsed)

    def expand_all_groups(self) -> None:
        """Expand all timeline groups while retaining their search queries."""
        self._set_all_groups_collapsed(False)

    def collapse_all_groups(self) -> None:
        """Collapse all timeline groups while retaining their search queries."""
        self._set_all_groups_collapsed(True)

    def _set_all_groups_collapsed(self, collapsed: bool) -> None:
        for group in self.timeline_groups.values():
            self._set_group_collapsed(group, collapsed)
        self._refresh_timeline_layout()

    def _create_timeline_search(self, group: TimelinePlotGroup) -> None:
        key = TimelineRowKey(group.name, "Search", TimelineRowRole.SEARCH)
        plot = self._get_timeline_plot_by_key(key, True)
        assert plot is not None
        container = self.timeline_legends[key].parentItem()
        assert isinstance(container, TimelinePlotLabelHost)
        search = TimelinePlotGroupSearchBar(group.name)
        container._set_title_widget(search)
        self._searchbars[group.name] = search
        container.content_lost.connect(
            partial(self._restore_timeline_search, group, search)
        )
        search.field.textChanged.connect(
            partial(self._on_search_changed, group, search)
        )
        plot.preferred_height_1d = max(25, search.sizeHint().height())
        plot.adjust_size()
        self._apply_group_visibility(group)

    def _restore_timeline_search(
        self, group: TimelinePlotGroup, search: TimelinePlotGroupSearchBar
    ) -> None:
        if (
            self.timeline_groups.get(group.name) is group
            and self._searchbars.get(group.name) is search
        ):
            self._dispose_timeline_search(group)
            self._create_timeline_search(group)
            self.sort_plots()

    def _dispose_timeline_search(self, group: TimelinePlotGroup) -> None:
        self._searchbars.pop(group.name, None)
        group._query = ""
        self._remove_timeline_plot_by_key(
            TimelineRowKey(group.name, "Search", TimelineRowRole.SEARCH)
        )

    def _on_search_changed(
        self, group: TimelinePlotGroup, search: TimelinePlotGroupSearchBar, text: str
    ) -> None:
        if (
            self.timeline_groups.get(group.name) is not group
            or self._searchbars.get(group.name) is not search
        ):
            return
        group._query = text.lower()
        self._apply_group_visibility(group)
        self._refresh_timeline_layout()

    def _apply_group_visibility(self, group: TimelinePlotGroup) -> None:
        for key in self.timeline_plots:
            if key.group_name == group.name:
                visible = key.role is TimelineRowRole.HEADER or (
                    not group.collapsed
                    and (
                        key.role is TimelineRowRole.SEARCH
                        or group._query in key.row_name.lower()
                    )
                )
                self._set_timeline_row_visible(key, visible)

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

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.Leave and (
            watched is self.scroll_area or watched is self.graphics_view.viewport()
        ):
            self._clear_hover_state()
        return super().eventFilter(watched, event)

    def _clear_hover_state(self) -> None:
        for plot in self.timeline_plots.values():
            plot.set_hovered(False)
        self.hover_marker.set_cursor_time(None)
        self._pending_tooltip_update = None
        self._tooltip_update_timer.stop()
        self.tooltip_widget.hide()

    def _on_scroll_or_playback(self, _value: int) -> None:
        cursor_pos = QCursor.pos()
        scroll_viewport = self.scroll_area.viewport()
        scroll_area_pos = scroll_viewport.mapFromGlobal(cursor_pos)
        if not scroll_viewport.rect().contains(scroll_area_pos):
            return

        viewport = self.graphics_view.viewport()
        viewport_pos = viewport.mapFromGlobal(cursor_pos)
        if not viewport.rect().contains(viewport_pos):
            return

        self.on_chart_area_mouse_moved(self.graphics_view.mapToScene(viewport_pos))

    def resizeEvent(self, event: QResizeEvent) -> None:
        w = self.scroll_area.viewport().width()
        if self.graphics_view.width() != w:
            self.graphics_view.setFixedWidth(w)
        self.playhead.refresh_geometry()
        self.hover_marker.refresh_geometry()
        if self.tooltip_widget.isVisible():
            self.tooltip_widget.show_at(QCursor.pos())
        super().resizeEvent(event)
        self._axis_geometry_timer.start(0)

    def _sync_pinned_axis_geometry(self) -> None:
        """Match the pinned axis's data area to the timeline plot area."""
        if not hasattr(self, "pinned_axis"):
            return

        viewport_width = self.scroll_area.viewport().width()
        if self.graphics_view.width() != viewport_width:
            self.graphics_view.setFixedWidth(viewport_width)
            self._axis_geometry_timer.start(0)
            return

        self.pinned_axis.sync_geometry(
            self,
            self.graphics_view,
            self.timestamps_plot.getViewBox(),
            self.scroll_area.frameWidth(),
        )
        self.hover_marker.label.refresh_geometry()

    def on_recording_loaded(self, recording: nr.NeonRecording) -> None:
        app = neon_player.instance()
        self.recording = recording

        for group in self.timeline_groups.values():
            if group.searchable and group.name not in self._searchbars:
                self._create_timeline_search(group)
        self.sort_plots()

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
        self.pinned_axis.set_time_frame(recording.start_time, recording.stop_time)

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
        for group in self.timeline_groups.values():
            if group.searchable:
                self._dispose_timeline_search(group)
                self._apply_group_visibility(group)
        self._refresh_timeline_layout()
        self.horizontal_scroll_bar.set_time_frame(None)
        trim_plot = self.get_timeline_plot("Export window", create_if_missing=False)
        if trim_plot is not None:
            trim_plot.clear()
        self.playhead.hide()
        self.hover_marker.set_cursor_time(None)
        self.tooltip_widget.hide()
        self.recording = None
        self.pinned_axis.set_time_frame(0, 0)
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
        self._context_menu_open = True
        self._pending_tooltip_update = None
        self._tooltip_update_timer.stop()
        try:
            context_menu.exec(global_position)
        finally:
            self._context_menu_open = False

    def on_chart_area_mouse_moved(self, pos: QPointF) -> None:
        if self._context_menu_open:
            return

        data_pos = self.timestamps_plot.getViewBox().mapSceneToView(pos)
        for tm in self.trim_markers:
            tm.set_highlighted(self.dragging == tm or tm.nearby(data_pos))

        hovered_plot = next(
            (
                plot
                for plot in self.timeline_plots.values()
                if plot.sceneBoundingRect().contains(pos)
            ),
            None,
        )
        for plot in self.timeline_plots.values():
            plot.set_hovered(plot is hovered_plot)
        self.hover_marker.set_cursor_time(round(data_pos.x()) if hovered_plot else None)

        self._pending_tooltip_update = (pos, round(data_pos.x()))
        if not self._tooltip_update_timer.isActive():
            self._tooltip_update_timer.start()

    def _update_pending_tooltip(self) -> None:
        if self._context_menu_open:
            self._pending_tooltip_update = None
            return

        if self._pending_tooltip_update is None:
            return

        pos, timestamp_ns = self._pending_tooltip_update
        self._pending_tooltip_update = None
        self._update_tooltip(pos, timestamp_ns)

    def _update_tooltip(self, pos: QPointF, timestamp_ns: int) -> None:
        if self.recording is None:
            self.tooltip_widget.hide()
            return

        if not self.recording.start_time <= timestamp_ns <= self.recording.stop_time:
            self.tooltip_widget.hide()
            return

        hovered_row_key = None
        for key, plot_item in self.timeline_plots.items():
            if plot_item.sceneBoundingRect().contains(pos):
                hovered_row_key = key
                break

        tooltip_content = self._tooltip_content_for_hovered_row(
            hovered_row_key, timestamp_ns
        )

        if isinstance(tooltip_content, QWidget):
            self.tooltip_widget.show_widget(tooltip_content, QCursor.pos())
        elif tooltip_content:
            self.tooltip_widget.show_text(tooltip_content, QCursor.pos())
        else:
            self.tooltip_widget.hide()

    def _tooltip_content_for_hovered_row(
        self, key: TimelineRowKey | None, timestamp_ns: int
    ) -> str | QWidget | None:
        if key is None:
            return None

        tooltip = self.timeline_tooltips.get(key)
        if tooltip is None or key == self._timeline_row_key("Export window"):
            return None

        plot_item = self.timeline_plots[key]
        tolerance_ns = round(20 * plot_item.getViewBox().viewPixelSize()[0])
        series = _get_hovered_timeline_series(
            plot_item.items, timestamp_ns, tolerance_ns
        )
        if not series:
            return None

        origin_ns = self.recording.start_time if self.recording is not None else 0
        context = TimelineTooltipContext(
            timestamp_ns, series, origin_ns, row_name=key.row_name
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
            logger.exception("Timeline tooltip callback failed for %s", key.row_name)
            return None

    def set_timeline_tooltip(
        self,
        timeline_row_name: str,
        tooltip: TimelineTooltipConfig | None,
        *,
        group: str | TimelinePlotGroup | None = None,
    ) -> None:
        """Set or clear the tooltip configuration for a timeline row.

        Args:
            timeline_row_name: Label of the row to configure.
            tooltip: Callback, custom tooltip widget, or ``None`` to clear.
            group: Group name or exact handle; None targets an ungrouped row.
        """
        key = self._timeline_row_key(timeline_row_name, group)
        previous = self.timeline_tooltips.get(key)
        current = _normalize_timeline_tooltip(tooltip)
        if current is previous:
            return

        if isinstance(current, TimelineTooltipWidget):
            self._validate_timeline_tooltip_widget(current, key)

        if self.tooltip_widget.isVisible():
            self.tooltip_widget.hide()

        if isinstance(current, TimelineTooltipWidget):
            self.tooltip_widget.own_content_widget(current)
            self._tooltip_widget_rows[id(current)] = key

        self.timeline_tooltips[key] = current
        self._release_timeline_tooltip_widget(key, previous)

    def _release_timeline_tooltip_widget(
        self, key: TimelineRowKey, tooltip: TimelineTooltipConfig | None
    ) -> None:
        if not isinstance(tooltip, TimelineTooltipWidget):
            return

        widget_id = id(tooltip)
        owner = self._tooltip_widget_rows.get(widget_id)
        if owner is None or owner != key:
            return

        self._tooltip_widget_rows.pop(widget_id, None)
        self.tooltip_widget.release_content_widget(tooltip)

    def _validate_timeline_tooltip_widget(
        self, widget: TimelineTooltipWidget, key: TimelineRowKey
    ) -> None:
        current_owner = self._tooltip_widget_rows.get(id(widget))
        if widget.parent() is not None:
            raise ValueError("Timeline tooltip widgets must be parentless")

        if current_owner is None:
            return

        if current_owner != key:
            raise ValueError(
                "Timeline tooltip widget is already assigned to another row"
            )

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
        legend.layout.setContentsMargins(0, 0, 0, 0)
        legend.layout.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Minimum,
        )
        legend_container = TimelinePlotLabelHost(timeline_row_name, legend)
        legend_label = legend_container.title
        plot_item = TimelinePlotItem(
            legend=legend,
            row_name=timeline_row_name,
            axisItems={"top": time_axis},
            viewBox=vb,
        )

        plot_item.row_key = key
        legend_container.size_hint_changed.connect(plot_item.adjust_size)
        legend_container.size_hint_changed.connect(self._refresh_timeline_layout)

        if is_timestamps_row:
            plot_item.preferred_height_1d = 30
            plot_item.adjust_size()
            legend_label.anchor((0.5, 0), (0.5, 0), (0, 2.5))

        legend.layout.setSpacing(0)
        self.timeline_legends[key] = legend

        self.graphics_layout.addItem(legend_container, row=row, col=1)
        self.graphics_layout.addItem(plot_item, row=row, col=2)
        self._axis_geometry_timer.start(0)

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
        if row_group.collapsed or row_group._query not in key.row_name.lower():
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
        tooltip: TimelineTooltipConfig | None = None,
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
            tooltip: Optional tooltip content for the row, implemented either
                as a
                [`TimelineTooltipCallback`][pupil_labs.neon_player.ui.timeline_dock_tooltips.TimelineTooltipCallback]
                that returns tooltip text, or a parentless subclass of
                [`TimelineTooltipWidget`][]. ``None`` keeps the current
                tooltip (by default, no tooltip is shown). The tooltip is
                updated while hovering a nearby data point.
            group: Group name or exact handle; None targets an ungrouped row.
            **kwargs: Additional plotting options passed to pyqtgraph.

        Returns:
            plot_item: The timeline plot, or ``None`` if no recording is open.
        """
        key = self._timeline_row_key(timeline_row_name, group)
        self._validate_timeline_row_creation(key)
        app = neon_player.instance()
        recording = app.recording
        if recording is None:
            return None

        plot_item = self.get_timeline_plot(timeline_row_name, True, group=group)

        if tooltip is not None:
            self.set_timeline_tooltip(timeline_row_name, tooltip, group=group)

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
        plot_item.adjust_size()

        if self.plot_sorting_enabled:
            self.sort_plots()
        self._axis_geometry_timer.start(0)

        return plot_item

    def fix_scroll_size(self) -> None:
        h = sum(
            plot.preferredHeight()
            for key, plot in self.timeline_plots.items()
            if key not in self.hidden_timeline_rows
        )
        self.graphics_view.setFixedHeight(h)
        self.playhead.refresh_geometry()
        self.hover_marker.refresh_geometry()

    def remove_timeline_plot(
        self, plot_name: str, *, group: str | TimelinePlotGroup | None = None
    ) -> None:
        """Remove a timeline plot and all the data series it contains.

        Args:
            plot_name: Label of the row to remove.
            group: Group name or exact handle; None targets an ungrouped row.
        """
        key = self._timeline_row_key(plot_name, group)
        self._remove_timeline_plot_by_key(key)
        if self.plot_sorting_enabled:
            self.sort_plots()
        else:
            self._refresh_timeline_layout()
        self._axis_geometry_timer.start(0)

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
        if key in self.timeline_tooltips:
            self.tooltip_widget.hide()
            tooltip_config = self.timeline_tooltips.pop(key, None)
            if tooltip_config is not None:
                self._release_timeline_tooltip_widget(key, tooltip_config)
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
        else:
            plot.adjust_size()
            self._refresh_timeline_layout()

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
        tooltip: TimelineTooltipConfig | None = None,
        show_y_limit_labels: bool | None = None,
        units: str | None = None,
        group: str | TimelinePlotGroup | None = None,
        **kwargs: T.Any,  # noqa: ANN401
    ) -> TimelinePlotItem | None:
        """Add a line graph to a timeline row.

        Args:
            timeline_row_name: Label for the timeline row.
            data: Timestamp and value pairs; timestamps are in nanoseconds.
            plot_name: Optional name shown in the row legend.
            tooltip: Optional tooltip callback or custom widget for the row; see
                [`add_timeline_plot`][] for details on the expected format.
            show_y_limit_labels: Show labels for y-axis limits on hover over
                the plot. The limits are shown by default for line plots.
            units: Units to display in the labels of y-axis limits.
            group: Group name or exact handle; None targets an ungrouped row.
            **kwargs: Additional line options passed to pyqtgraph.

        Returns:
            plot_item: The timeline row, or ``None`` if no recording is open.
        """
        plot_existed = (
            self._timeline_row_key(timeline_row_name, group) in self.timeline_plots
        )
        plot_item = self.add_timeline_plot(
            timeline_row_name, data, plot_name, tooltip=tooltip, group=group, **kwargs
        )
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
        group: str | TimelinePlotGroup | None = None,
    ) -> TimelinePlotItem | None:
        """Add a scatter plot to a timeline row.

        Args:
            name: Label for the timeline row.
            data: Timestamp and value pairs; timestamps are in nanoseconds.
            item_name: Optional name shown in the row legend.
            tooltip: Optional tooltip callback or custom widget for the row; see
                [`add_timeline_plot`][] for details on the expected format.
            group: Group name or exact handle; None targets an ungrouped row.

        Returns:
            plot_item: The timeline row, or ``None`` if no recording is open.
        """
        return self.add_timeline_plot(
            name,
            data,
            item_name,
            group=group,
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
        group: str | TimelinePlotGroup | None = None,
    ) -> TimelinePlotItem | None:
        """Add horizontal bars to show time ranges on the timeline.

        Args:
            timeline_row_name: Label for the timeline row.
            data: Time ranges as ``(start, end)`` pairs, or
                ``(start, end, value)`` triples. Timestamps are in nanoseconds;
                values control the bar colors when supplied.
            item_name: Optional name shown in the row legend.
            color: Color name or pyqtgraph color map name.
            tooltip: Optional tooltip callback or custom widget for the row; see
                [`add_timeline_plot`][] for details on the expected format.
            group: Group name or exact handle; None targets an ungrouped row.

        Returns:
            plot_item: The timeline row, or ``None`` if no recording is open.
        """
        key = self._timeline_row_key(timeline_row_name, group)
        self._validate_timeline_row_creation(key)
        if neon_player.instance().recording is None:
            return None
        plot_widget = self.get_timeline_plot(timeline_row_name, True, group=group)

        if tooltip is not None:
            self.set_timeline_tooltip(timeline_row_name, tooltip, group=group)

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

        key = self._timeline_row_key(timeline_row_name, group)
        if item_name and key in self.timeline_legends:
            legend = self.timeline_legends[key]
            legend.addItem(bars, name=item_name)

        if self.plot_sorting_enabled:
            self.sort_plots()
        self._axis_geometry_timer.start(0)

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
                plot.adjust_size()
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
            xRange=[recording.start_time, recording.stop_time],
            padding=self.TIMELINE_PADDING,
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
