import pytest

from pupil_labs.neon_player.ui.timeline_dock_components import (
    TimelineHorizontalScrollBar,
)


@pytest.mark.usefixtures("qapp")
def test_timeline_horizontal_scrollbar__hidden_for_full_recording_view() -> None:
    scroll_bar = TimelineHorizontalScrollBar()
    scroll_bar.set_time_frame((100, 1100), padding=0.05)

    scroll_bar.set_view_range(100, 1100)

    assert not scroll_bar.isVisible()
    assert scroll_bar.minimum() == 0
    assert scroll_bar.maximum() > 0
    assert scroll_bar.value() > 0


@pytest.mark.usefixtures("qapp")
def test_timeline_horizontal_scrollbar__tracks_view_and_scrolls_time_range() -> None:
    scroll_bar = TimelineHorizontalScrollBar()
    scroll_bar.set_time_frame((100, 1100), padding=0.05)
    changed_ranges = []
    scroll_bar.range_changed.connect(
        lambda start, stop: changed_ranges.append((start, stop))
    )

    scroll_bar.set_view_range(200, 400)

    assert scroll_bar.isVisible()
    assert scroll_bar.minimum() == 0
    assert scroll_bar.maximum() == 818_182
    assert scroll_bar.pageStep() == 181_818
    assert scroll_bar.value() == 136_364
    assert changed_ranges == []

    scroll_bar.setValue(scroll_bar.minimum())
    assert changed_ranges[0] == pytest.approx((50, 250))

    scroll_bar.setValue(scroll_bar.maximum())

    assert changed_ranges[1] == pytest.approx((950, 1150))
