import json
import logging

from dataclasses import dataclass
from datetime import datetime, timedelta
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QDialog, QFrame, QScrollArea, QVBoxLayout, QLabel, QWidget

from pupil_labs.neon_recording import NeonRecording
from pupil_labs.neon_player.ui.constants import Color


def get_recording_size(recording: NeonRecording) -> int:
    manifest_file = recording._rec_dir / "manifest.json"
    if not manifest_file.exists():
        logging.warning(
            f"Manifest file not found in {recording._rec_dir}. Cannot "
            f"calculate recording size."
        )
        return 0

    try:
        manifest = json.load(open(manifest_file, "r"))
        manifest_size = sum(entry["size"] for entry in manifest)
        manifest_file_size = manifest_file.stat().st_size
        return manifest_size + manifest_file_size
    except Exception as e:
        logging.warning(
            f"Error reading manifest file in {recording._rec_dir}: {e}. Cannot "
            f"calculate recording size."
        )
        return 0


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
            widget.setObjectName("FieldValue")
            widget.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            widget.setCursor(Qt.CursorShape.IBeamCursor)

            self.fields[name] = TextField(label=label, widget=widget)

    def __getitem__(self, key: str) -> TextField:
        return self.fields[key]


class RecordingInfoDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)
        self.setFixedSize(600, 600)
        self.setStyleSheet(f"""
            QLabel#FieldGroupHeading {{
                color: {Color.Text.Primary};
                font-size: 14px;
                font-weight: 500;
                line-height: 24px;
            }}

            QLabel#FieldLabel {{
                color: {Color.Text.Primary};
                font-size: 12px;
                font-weight: 400;
                line-height: 16px;
            }}

            QLabel#FieldValue {{
                color: {Color.Text.Secondary};
                font-size: 12px;
                font-weight: 400;
                line-height: 16px;
            }}
        """)

        # Scrollable content widget
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(20, 20, 20, 20)

        scroll_area = QScrollArea()
        scroll_area.setWidget(content)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)

        dialog_layout = QVBoxLayout(self)
        dialog_layout.setContentsMargins(0, 0, 0, 0)
        dialog_layout.addWidget(scroll_area)
        self.setLayout(dialog_layout)

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
                separator = QFrame()
                separator.setFrameShape(QFrame.Shape.HLine)
                separator.setFrameShadow(QFrame.Shadow.Raised)
                separator.setFixedHeight(1)
                separator.setStyleSheet("background-color: #292d2d; border: none;")
                content_layout.addWidget(separator)
                content_layout.addSpacing(5)

            group_label = QLabel(f"<h2>{group.label}</h2>")
            group_label.setObjectName("FieldGroupHeading")
            content_layout.addWidget(group_label)
            content_layout.addSpacing(5)
            for field in group.fields.values():
                field_label = QLabel(f"{field.label}:")
                field_label.setObjectName("FieldLabel")
                field_label.setFont(QFont("Arial", 16, QFont.Weight.Normal))
                content_layout.addWidget(field_label)
                content_layout.addWidget(field.widget)
                content_layout.addSpacing(5)

    def update(self, data: dict[str, dict[str, str]]) -> None:
        for group_name, group_data in data.items():
            if group_name not in self.groups:
                raise ValueError(f"Unknown group name: {group_name}")

            group = self.groups[group_name]
            for field_name, value in group_data.items():
                if field_name not in group.fields:
                    raise ValueError(f"Unknown field {field_name} in group {group_name}")

                group[field_name].widget.setText(value)

    def set_recording(self, recording: NeonRecording) -> None:
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
        duration = timedelta(seconds=recording.info["duration"] / 1e9)
        size = get_recording_size(recording) / (1024 * 1024)  # size in MB

        return {
            "recording": {
                "id": recording.info.get("recording_id", "-"),
                "date": start_time_str,
                "duration": str(duration),
                "size": f"{size:.2f} MB",
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
