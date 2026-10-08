import numpy as np

from pupil_labs.neon_player.utilities import find_ranged_index


def test_find_ranged_index():
    #                   fix1              fix2
    #                   <--->             <---->
    gaze_ts = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    fixation_start = np.array([0, 6])
    fixation_stop = np.array([2, 8])
    expected = np.array([0, 0, -1, -1, -1, -1, 1, 1, -1])

    result = find_ranged_index(gaze_ts, fixation_start, fixation_stop)
    assert np.array_equal(result, expected)
