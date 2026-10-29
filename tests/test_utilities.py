import numpy as np
import pytest
from PySide6.QtGui import QImage

from pupil_labs.neon_player.utilities import find_ranged_index, ndarray_from_qimage


def test_find_ranged_index() -> None:
    #                   fix1              fix2
    #                   <--->             <---->
    gaze_ts = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    fixation_start = np.array([0, 6])
    fixation_stop = np.array([2, 8])
    expected = np.array([0, 0, -1, -1, -1, -1, 1, 1, -1])

    result = find_ranged_index(gaze_ts, fixation_start, fixation_stop)
    assert np.array_equal(result, expected)


def test_find_ranged_index_empty() -> None:
    values = np.array([], dtype=np.int64)
    boundaries = np.array([0, 2], dtype=np.int64)
    result = find_ranged_index(values, boundaries, boundaries + 1)
    assert result.shape == (0,)
    assert result.dtype == np.dtype(np.intp)


@pytest.mark.parametrize(
    ("image_format", "dtype"),
    [(QImage.Format.Format_RGB32, np.uint8), (QImage.Format.Format_RGBA64, np.uint16)],
)
def test_qimage_array_preserves_pixel_dtype(
    image_format: QImage.Format, dtype: type
) -> None:
    image = QImage(2, 2, image_format)
    image.fill(0xFF123456)
    pixels = ndarray_from_qimage(image)
    assert pixels.dtype == np.dtype(dtype)
    assert pixels.shape == (2, 2, 4)
