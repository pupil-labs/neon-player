"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import argparse
import sys
import typing as T
from gettext import gettext as _


class HelpfulArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that prints the full help message on error."""

    def error(self, message: str):
        # NOTE: This is mostly argparse source code with slight adjustments
        args = {"prog": self.prog, "message": message}
        self._print_message(_("%(prog)s: error: %(message)s\n") % args, sys.stderr)
        self.print_help(sys.stderr)
        self.exit(2)


class PupilArgParser:
    def parse(self, running_from_bundle: bool, **defaults: T.Any):
        """Parse command line arguments for Pupil apps."""
        self.apps = {
            "capture": "real-time processing and recording",
            "player": "process, visualize, and export recordings",
            "service": "low latency real-time processing with constrained feature set",
        }

        self.main_parser = HelpfulArgumentParser(allow_abbrev=False)
        self.main_parser.description = "process, visualize, and export Neon recordings"
        self.main_parser.add_argument("--version", action="store_true", help="show version")
        self.main_parser.add_argument(
            "--debug", action="store_true", help="display debug log messages"
        )
        self.main_parser.add_argument(
            "--profile", action="store_true", help="profile the application's CPU time"
        )
        self.main_parser.add_argument(
            "recording", default="", nargs="?", help="path to recording"
        )
        self.main_parser.set_defaults(**defaults)

        return self.main_parser.parse_known_args()
