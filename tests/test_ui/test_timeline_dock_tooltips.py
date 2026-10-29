from pupil_labs.neon_player.ui.timeline_dock_tooltips import (
    TimelineScatterSeriesHover,
    TimelineTooltipContext,
    default_scatter_tooltip,
)


def test_default_scatter_tooltip_uses_name_count_and_context_timestamp() -> None:
    context = TimelineTooltipContext(
        timestamp_ns=600_000_000,
        series=[
            TimelineScatterSeriesHover(
                name="trial.begin",
                timestamp_ns=600_000_000,
                index=1,
                value=0.0,
            )
        ],
        origin_ns=100_000_000,
        row_name="Events - trial.begin",
    )

    tooltip = default_scatter_tooltip(name="trial.begin", total=2)
    assert tooltip(context) == ("<b>trial.begin</b> (2/2)<br>00:00:00.500")
