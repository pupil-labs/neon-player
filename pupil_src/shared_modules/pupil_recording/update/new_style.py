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
import re
from pathlib import Path

import camera_models as cm
import file_methods as fm
from version_utils import get_version, parse_version

from ..info import RecordingInfoFile
from ..recording import PupilRecording
from ..recording_utils import InvalidRecordingException

logger = logging.getLogger(__name__)


def recording_update_to_latest_new_style(rec_dir: str):
    info_file = RecordingInfoFile.read_file_from_recording(rec_dir)
    check_min_player_version(info_file)

    # incremental upgrade ...
    if info_file.meta_version < parse_version("2.2"):
        info_file = update_newstyle_21_22(rec_dir)
    if info_file.meta_version < parse_version("2.3"):
        info_file = update_newstyle_22_23(rec_dir)
    if info_file.meta_version < parse_version("2.4"):
        info_file = update_newstyle_23_24(rec_dir)


def check_min_player_version(info_file: RecordingInfoFile):
    if info_file.min_player_version > get_version():
        player_out_of_date = (
            "Recording requires a newer version of Player: "
            f"{info_file.min_player_version}"
        )
        raise InvalidRecordingException(reason=player_out_of_date)


def check_for_worldless_recording_new_style(rec_dir):
    logger.info("Checking for world-less recording...")
    rec_dir = Path(rec_dir)

    recording = PupilRecording(rec_dir)
    world_videos = recording.files().core().world().videos()
    if not world_videos:
        logger.info("No world video found. Constructing an artificial replacement.")
        fake_world_version = 2
        fake_world_object = {"version": fake_world_version}
        fake_world_path = rec_dir / "world.fake"
        fm.save_object(fake_world_object, fake_world_path)


def update_newstyle_21_22(rec_dir: str):
    # Used to make Pupil v2.0 recordings backwards incompatible with v1.x
    old_info_file = RecordingInfoFile.read_file_from_recording(rec_dir)
    new_info_file = RecordingInfoFile.create_empty_file(
        rec_dir, fixed_version=parse_version("2.2")
    )
    new_info_file.update_writeable_properties_from(old_info_file)
    new_info_file.save_file()
    return new_info_file


def update_newstyle_22_23(rec_dir: str):
    old_info_file = RecordingInfoFile.read_file_from_recording(rec_dir)
    new_info_file = RecordingInfoFile.create_empty_file(
        rec_dir, fixed_version=parse_version("2.3")
    )
    new_info_file.update_writeable_properties_from(old_info_file)
    new_info_file.save_file()
    return new_info_file


def update_newstyle_23_24(rec_dir: str):
    old_info_file = RecordingInfoFile.read_file_from_recording(rec_dir)
    new_info_file = RecordingInfoFile.create_empty_file(
        rec_dir, fixed_version=parse_version("2.4")
    )
    new_info_file.update_writeable_properties_from(old_info_file)
    new_info_file.save_file()
    return new_info_file
