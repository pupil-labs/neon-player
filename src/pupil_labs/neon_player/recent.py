from datetime import datetime
from pathlib import Path
from pupil_labs.neon_recording import NeonRecording


def get_recording_metadata(path: Path, recording: NeonRecording) -> dict[str, str]:
    start_time = datetime.fromtimestamp(recording.info["start_time"] / 1e9)
    recorded_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
    last_opened_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    wearer = recording.wearer["name"]

    return {
        "path": str(path.resolve()),
        "name": path.name,
        "wearer": wearer,
        "recorded": recorded_str,
        "last_opened": last_opened_str,
    }
