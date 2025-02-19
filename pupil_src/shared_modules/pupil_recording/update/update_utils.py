"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import logging
import typing as T
from pathlib import Path

import pupil_labs.neon_recording as nr

import av
import camera_models as cm
import numpy as np

from ..recording import PupilRecording

logger = logging.getLogger(__name__)


def _try_patch_world_instrinsics_file(rec_dir: str, videos: T.Sequence[Path]) -> None:
    """Tries to create a reasonable world.intrinsics file from a set of videos."""
    if not videos:
        return

    recording = nr.load(Path(rec_dir).parent)

    resolution = (recording.scene.width, recording.scene.height)
    intrinsics = {
        "camera_matrix": recording.calibration.scene_camera_matrix,
        "dist_coefs": recording.calibration.scene_distortion_coefficients,
        "cam_type": "radial"
    }

    camera = cm.Camera_Model._from_raw_intrinsics("world", resolution, intrinsics)
    camera.save(rec_dir, "world")


_ConversionCallback = T.Callable[[np.array], np.array]


def _rewrite_times(
    recording: PupilRecording,
    dtype: str,
    conversion: T.Optional[_ConversionCallback] = None,
) -> None:
    """Load raw times (assuming dtype), apply conversion and save as _timestamps.npy."""
    for path in recording.files().raw_time():
        timestamps = np.fromfile(str(path), dtype=dtype)
        new_name = f"{path.stem}_timestamps_unix.npy"
        timestamp_loc = path.parent / new_name
        np.save(str(timestamp_loc), timestamps)

        if conversion is not None:
            timestamps = conversion(timestamps)

        new_name = f"{path.stem}_timestamps.npy"
        logger.info(f"Creating {new_name}")
        timestamp_loc = path.parent / new_name
        np.save(str(timestamp_loc), timestamps)
