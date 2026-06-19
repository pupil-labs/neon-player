from pupil_labs.neon_player.ui.timeline_dock_utils import get_clicked_data_point


def test_get_clicked_data_point__plot_item_none():
    assert get_clicked_data_point(None, None) is None
