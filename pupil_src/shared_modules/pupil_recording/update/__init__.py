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
from types import SimpleNamespace
from pathlib import Path
import shutil

from video_capture.file_backend import File_Source

from ..info import RecordingInfoFile
from ..recording import PupilRecording
from ..recording_utils import (
    InvalidRecordingException,
    RecordingType,
    get_recording_type,
)
from .neon import transform_neon_to_corresponding_new_style
from .new_style import (
    check_for_worldless_recording_new_style,
    recording_update_to_latest_new_style,
)

logger = logging.getLogger(__name__)


def update_recording(rec_dir: str):
    recording_type = get_recording_type(rec_dir)

    if recording_type != RecordingType.NEON:
        is_neon = False
        if recording_type == RecordingType.NEW_STYLE:
            recording_info = RecordingInfoFile.read_file_from_recording(rec_dir)
            is_neon = (recording_info.recording_software_name == "Neon")

        if not is_neon:
            raise InvalidRecordingException(
                "Neon Player only supports Neon recordings",
                recovery="Try Pupil Player instead",
            )

        # It's an (already) converted Neon recording folder - nothing to do here
        return

    # Check if already converted
    new_path = Path(rec_dir) / "neon_player"
    if new_path.exists():
        return str(new_path)

    # Copy entire contents to a subfolder and convert that
    rec_dir_tmp = f"{new_path}.tmp"
    shutil.rmtree(rec_dir_tmp, ignore_errors=True)

    shutil.copytree(rec_dir, rec_dir_tmp)
    rec_dir = str(new_path)

    # NOTE: there is an issue with PI recordings, where sometimes multiple parts of
    # the recording are stored as an .mjpeg and .mp4, but for the same part number.
    # The recording is un-usable in this case, since the time information is lost.
    # Trying to open the recording will crash in the lookup-table generation. We
    # just gracefully exit here and display an error message.
    mjpeg_world_videos = (
        PupilRecording.FileFilter(rec_dir_tmp).pi().world().filter_patterns(".mjpeg$")
    )
    if mjpeg_world_videos:
        videos = [
            path.name
            for path in PupilRecording.FileFilter(rec_dir_tmp).pi().world().videos()
        ]
        logger.error(
            "Found mjpeg world videos for this Pupil Invisible recording! Videos:\n"
            + ",\n".join(videos)
        )
        raise InvalidRecordingException(
            "This recording cannot be opened in Player.",
            recovery="Please reach out to info@pupil-labs.com for support!",
        )

    transform_neon_to_corresponding_new_style(rec_dir_tmp)

    _assert_compatible_meta_version(rec_dir_tmp)

    check_for_worldless_recording_new_style(rec_dir_tmp)

    # update to latest
    recording_update_to_latest_new_style(rec_dir_tmp)

    # generate lookup tables once at the start of player, so we don't pause later for
    # compiling large lookup tables when they are needed
    _generate_all_lookup_tables(rec_dir_tmp)

    shutil.move(rec_dir_tmp, rec_dir)

    return rec_dir


def _assert_compatible_meta_version(rec_dir: str):
    # This will throw InvalidRecordingException if we cannot open the recording due
    # to meta info version or min_player_version mismatches.
    PupilRecording(rec_dir)


def _generate_all_lookup_tables(rec_dir: str):
    recording = PupilRecording(rec_dir)
    videosets = [
        recording.files().core().world().videos(),
        recording.files().core().eye0().videos(),
        recording.files().core().eye1().videos(),
    ]
    for videos in videosets:
        if not videos:
            continue
        source_path = videos[0].resolve()
        File_Source(
            SimpleNamespace(), source_path=source_path, fill_gaps=True, timing=None
        )


__all__ = ["update_recording"]
