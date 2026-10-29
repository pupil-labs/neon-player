import sys

from pupil_labs.neon_player.app import NeonPlayerApp


def main() -> None:
    app = NeonPlayerApp(sys.argv)
    sys.exit(app.run())
