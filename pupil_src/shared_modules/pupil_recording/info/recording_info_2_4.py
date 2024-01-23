"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
from version_utils import ParsedVersion, parse_version

from . import RecordingInfoFile
from . import recording_info_utils as utils
from .recording_info_2_3 import _RecordingInfoFile_2_3


class _RecordingInfoFile_2_4(_RecordingInfoFile_2_3):
    @property
    def meta_version(self) -> ParsedVersion:
        return parse_version("2.4")

    @property
    def min_player_version(self) -> ParsedVersion:
        return parse_version("4.1")

    @property
    def _private_key_schema(self) -> RecordingInfoFile._KeyValueSchema:
        return {
            **super()._private_key_schema,
            # overwrite meta_version key from parent
            "meta_version": (utils.validator_version_string, lambda _: "2.4"),
            "min_player_version": (utils.validator_version_string, lambda _: "4.1"),
        }
