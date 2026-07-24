import json
import logging

from dataclasses import dataclass
from datetime import datetime
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QWidget

from pupil_labs import neon_player
from pupil_labs.neon_recording import NeonRecording


@dataclass
class TextField:
    label: str
    widget: QLabel


class TextFieldGroup:
    def __init__(self, label: str, fields: dict[str, str]) -> None:
        self.label = label
        self.fields = {}
        for name, label in fields.items():
            widget = QLabel("-")
            widget.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            widget.setCursor(Qt.CursorShape.IBeamCursor)

            self.fields[name] = TextField(label=label, widget=widget)

    def __getitem__(self, key: str) -> TextField:
        return self.fields[key]


class RecordingInfoDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        self.setLayout(main_layout)

        self.groups = {
            "recording": TextFieldGroup(
                label="Recording details",
                fields={
                    "id": "ID",
                    "wearer": "Wearer",
                    "duration": "Duration",
                    "date": "Recorded",
                    "template": "Template",
                    "size": "Size",
                }
            ),
            "device": TextFieldGroup(
                label="Device",
                fields={
                    "serial": "Device Serial",
                    "gaze_mode": "Gaze Mode",
                }
            ),
            "frame": TextFieldGroup(
                label="Frame",
                fields={
                    "name": "Frame Name",
                }
            ),
            "companion": TextFieldGroup(
                label="Companion Device",
                fields={
                    "name": "Companion Device Name",
                    "device": "Companion Device",
                    "app_version": "Companion Device App Version",
                }
            ),
        }

        for idx, group in enumerate(self.groups.values()):
            if idx:
                main_layout.addSpacing(10)
                separator = QLabel("<hr>")
                separator.setTextFormat(Qt.TextFormat.RichText)
                main_layout.addWidget(separator)
                main_layout.addSpacing(10)

            group_label = QLabel(f"<h2>{group.label}</h2>")
            main_layout.addWidget(group_label)
            for field in group.fields.values():
                field_label = QLabel(f"{field.label}:")
                field_label.setFont(QFont("Arial", 16, QFont.Weight.Normal))
                main_layout.addWidget(field_label)
                main_layout.addWidget(field.widget)
                main_layout.addSpacing(5)
            main_layout.addSpacing(10)


    def update(self, data: dict[str, dict[str, str]]) -> None:
        for group_name, group_data in data.items():
            if group_name not in self.groups:
                raise ValueError(f"Unknown group name: {group_name}")

            group = self.groups[group_name]
            for field_name, value in group_data.items():
                if field_name not in group.fields:
                    raise ValueError(f"Unknown field {field_name} in group {group_name}")

                group[field_name].widget.setText(value)

    def on_recording_loaded(self, recording: NeonRecording) -> None:
        try:
            self.update(self._load_recording_info(recording))
        except FileNotFoundError:
            logging.warning("Failed to load information about recording")

        try:
            self.update(self._load_wearer_info(recording))
        except FileNotFoundError:
            logging.warning("Failed to load information about wearer")

        try:
            self.update(self._load_template_info(recording))
        except FileNotFoundError:
            logging.warning("Failed to load information about template")

    @staticmethod
    def _load_recording_info(recording: NeonRecording) -> dict[str, dict[str, str]]:
        start_time = datetime.fromtimestamp(recording.info["start_time"] / 1e9)
        start_time_str = start_time.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        # TODO: duration

        return {
            "recording": {
                "id": recording.info.get("recording_id", "-"),
                "date": start_time_str,
            },
            "device": {
                "serial": recording.info.get("module_serial_number", "-"),
                "gaze_mode": recording.info.get("gaze_mode", "-"),
            },
            "frame": {
                "name": recording.info.get("frame_name", "-"),
            },
            "companion": {
                "name": recording.info.get("android_device_name", "-"),
                "device": recording.info.get("android_device_model", "-"),
                "app_version": recording.info.get("app_version", "-"),
            },
        }

    @staticmethod
    def _load_wearer_info(recording: NeonRecording) -> str:
        return {
            "recording": {
                "wearer": recording.wearer["name"]
            }
        }

    @staticmethod
    def _load_template_info(recording: NeonRecording) -> str:
        with open(recording._rec_dir / "template.json", "r") as f:
            template_data = json.load(f)

        return {
            "recording": {
                "template": template_data.get("name", "-")
            }
        }
