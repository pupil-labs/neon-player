import pytest

from pupil_labs.neon_player.ui.timeline_dock_components import (
    TimelineHorizontalScrollBar,
)


@pytest.mark.usefixtures("qapp")
def test_timeline_horizontal_scrollbar__hidden_for_full_recording_view() -> None:
    scroll_bar = TimelineHorizontalScrollBar()
    scroll_bar.set_time_frame((100, 1100))

    scroll_bar.set_view_range(100, 1100)

    assert not scroll_bar.isVisible()
    assert scroll_bar.minimum() == 0
    assert scroll_bar.maximum() == 0


@pytest.mark.usefixtures("qapp")
def test_timeline_horizontal_scrollbar__tracks_view_and_scrolls_time_range() -> None:
    scroll_bar = TimelineHorizontalScrollBar()
    scroll_bar.set_time_frame((100, 1100))
    changed_ranges = []
    scroll_bar.range_changed.connect(
        lambda start, stop: changed_ranges.append((start, stop))
    )

    scroll_bar.set_view_range(200, 400)

    assert scroll_bar.isVisible()
    assert scroll_bar.minimum() == 0
    assert scroll_bar.maximum() == 800_000
    assert scroll_bar.pageStep() == 200_000
    assert scroll_bar.value() == 100_000
    assert changed_ranges == []

    scroll_bar.setValue(scroll_bar.maximum())

    assert changed_ranges == [(900, 1100)]
