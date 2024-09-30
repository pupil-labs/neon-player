"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import cv2
from plugin import Plugin
from pyglui import ui


class Image_Adjustments(Plugin):
    icon_chr = chr(0xE04B)
    icon_font = "pupil_icons"

    def __init__(self, g_pool, brightness=0, contrast=1.0):
        super().__init__(g_pool)

        self.order = 0.4
        self.menu = None

        self.brightness = brightness
        self.contrast = contrast

    def recent_events(self, events):
        frame = events.get("frame")
        if not frame:
            return

        frame.img[:] = cv2.convertScaleAbs(frame.img, alpha=self.contrast, beta=self.brightness)

    def init_ui(self):
        self.add_menu()
        self.menu.label = "Image Adjustments"
        self.menu.append(
            ui.Slider("brightness", self, min=0, max=100, label="Brightness")
        )
        self.menu.append(
            ui.Slider("contrast", self, min=1.0, max=3.0, label="Contrast")
        )

    def deinit_ui(self):
        self.remove_menu()

    def get_init_dict(self):
        return {
            "brightness": self.brightness,
            "contrast": self.contrast,
        }
