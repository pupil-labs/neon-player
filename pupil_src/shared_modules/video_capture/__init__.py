"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
"""
Video Capture provides the interface to get frames from diffferent backends.
Backends consist of a manager and at least one source class. The manager
is a Pupil plugin that provides an GUI that lists all available sources. The
source provides the stream of image frames.

These backends are available:
- UVC: Local USB sources
- Fake: Fallback, static grid image
- File: Loads video from file
"""

import logging
import os
from glob import glob

import numpy as np

from .base_backend import (
    Base_Manager,
    Base_Source,
    EndofVideoError,
    InitialisationError,
    StreamError,
)
from .file_backend import File_Manager, File_Source, FileSeekError

logger = logging.getLogger(__name__)


source_classes = [File_Source]
manager_classes = [File_Manager]

