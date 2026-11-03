import pytest

from pupil_labs.neon_player.ui.timeline_dock import TimeLineDock
from pupil_labs.neon_player.ui.timeline_dock_components import TimelinePlotItem


class DummyLegend:
    def setFixedHeight(self, _height: int) -> None:
        pass

    def parentItem(self) -> "DummyLegend":
        return self


@pytest.mark.usefixtures("qapp")
def test_timeline_plot_item_left_axis_is_a_border_without_ticks_or_values() -> None:
    plot = TimelinePlotItem(DummyLegend(), "row")
    left_axis = plot.getAxis("left")

    assert left_axis.isVisible()
    assert left_axis.style["showValues"] is False
    assert left_axis.style["tickLength"] == 0


@pytest.mark.usefixtures("qapp")
def test_smart_size_plot_item_y_limit_labels_follow_visible_range() -> None:
    plot = TimelinePlotItem(DummyLegend(), "row")
    view_box = plot.getViewBox()

    assert plot._y_units == ""
    assert plot._y_range_labels == []

    view_box.setRange(xRange=(0, 100), yRange=(-0.1, 0.1), padding=0)
    plot.configure_y_limit_labels(True, "m")

    assert plot.show_y_limit_labels
    assert plot.y_units == "m"
    upper_label, lower_label = plot._y_range_labels
    assert upper_label.toPlainText() == "0.100 m"
    assert lower_label.toPlainText() == "-0.100 m"
    assert upper_label.textItem.defaultTextColor().name() == "#a09fa6"
    assert upper_label.textItem.font().pointSize() == 10
    assert upper_label.pos().x() < 0.5
    assert upper_label.pos().y() > 0.099
    assert lower_label.pos().y() < -0.099

    line_item = plot.plot([0, 100], [0, 0])
    assert upper_label.zValue() > line_item.zValue()
    assert lower_label.zValue() > line_item.zValue()

    view_box.setRange(xRange=(50, 150), padding=0)
    assert upper_label.pos().x() > 50
    assert lower_label.pos().x() > 50

    view_box.setRange(yRange=(-10, 10), padding=0)

    assert upper_label.toPlainText() == "10.0 m"
    assert lower_label.toPlainText() == "-10.0 m"

    plot.configure_y_limit_labels(False)
    assert not plot.show_y_limit_labels
    assert plot._y_range_labels == []


class DummyTimelineDock:
    def __init__(self) -> None:
        self.timeline_plots: dict[str, DummyLinePlot] = {}

    def get_timeline_plot(
        self, row_name: str, _create_if_missing: bool = False
    ) -> "DummyLinePlot | None":
        return self.timeline_plots.get(row_name)

    def add_timeline_plot(
        self,
        row_name: str,
        _data: list[tuple[int, float]],
        _plot_name: str = "",
        **_kwargs: object,
    ) -> "DummyLinePlot":
        plot = self.timeline_plots.setdefault(row_name, DummyLinePlot())
        return plot


class DummyLinePlot:
    def __init__(self) -> None:
        self.show_y_limit_labels = False
        self.units: str | None = None

    def configure_y_limit_labels(self, show: bool | None, units: str | None) -> None:
        if show is not None:
            self.show_y_limit_labels = show
        self.units = units


@pytest.mark.usefixtures("qapp")
def test_add_timeline_line_defaults_labels_and_preserves_existing() -> None:
    dock = DummyTimelineDock()

    TimeLineDock.add_timeline_line(dock, "new", [])
    assert dock.timeline_plots["new"].show_y_limit_labels

    existing_plot = DummyLinePlot()
    dock.timeline_plots["existing"] = existing_plot
    TimeLineDock.add_timeline_line(dock, "existing", [], units="deg")

    assert not existing_plot.show_y_limit_labels
    assert existing_plot.units == "deg"


@pytest.mark.usefixtures("qapp")
def test_add_timeline_line_explicit_visibility_overrides_default() -> None:
    dock = DummyTimelineDock()

    TimeLineDock.add_timeline_line(
        dock, "hidden", [], show_y_limit_labels=False, units="mm"
    )

    plot = dock.timeline_plots["hidden"]
    assert not plot.show_y_limit_labels
    assert plot.units == "mm"
