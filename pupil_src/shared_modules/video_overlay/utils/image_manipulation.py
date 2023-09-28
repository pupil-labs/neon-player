"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import abc
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def float_to_int(value: float) -> int:
    return int(value) if np.isfinite(value) else 0


class ImageManipulator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def apply_to(self, image, parameter, **kwargs):
        raise NotImplementedError


class ScaleTransform(ImageManipulator):
    def apply_to(self, image, parameter, **kwargs):
        """parameter: scale factor as float"""
        return cv2.resize(image, (0, 0), fx=parameter, fy=parameter)


class HorizontalFlip(ImageManipulator):
    def apply_to(self, image, parameter, *, is_fake_frame, **kwargs):
        """parameter: boolean indicating if image should be flipped"""
        if parameter and not is_fake_frame:
            return np.fliplr(image)
        else:
            return image


class VerticalFlip(ImageManipulator):
    def apply_to(self, image, parameter, *, is_fake_frame, **kwargs):
        """parameter: boolean indicating if image should be flipped"""
        if parameter and not is_fake_frame:
            return np.flipud(image)
        else:
            return image
