"""
Neon Player provides a set of built-in widgets that plugin developers can use
to include plugin settings in the Neon Player UI. Each setting should be defined
as a property (using the standard `@property` decorator) in the plugin class. By
default, the widget is selected automatically based on the type hint of the value
returned by the property. The widgets appear in the same order as the properties
are defined in the plugin class.

It is also possible to implement a custom widget by subclassing `PropertyWidget`
and implementing the required methods. The custom widget can then be associated
with a property using the `widget` argument of the `@property_params` decorator.

To add buttons that trigger actions provided by the plugin, define a method in the
plugin class and apply the `@action` decorator to it.

The plugin described in this file showcases the available built-in widgets and
how to implement a custom widget. For built-in widgets, optional parameters that
control the behavior of the widget are shown in the `@property_params` decorator.
"""

from enum import Enum
import logging
from pathlib import Path
from pupil_labs.neon_player import Plugin, action
from PySide6.QtCore import Qt, QObject, Signal
from PySide6.QtGui import QFont, QColor
from PySide6.QtWidgets import QLabel, QPushButton
from qt_property_widgets.utilities import (
    property_params, FilePath, action_params, PersistentPropertiesMixin
)
from qt_property_widgets.widgets import PropertyWidget, MultiLineTextWidget


class ClickMeWidget(PropertyWidget):
    value_changed = Signal(int)

    @staticmethod
    def from_property_impl(prop: property) -> "ClickMeWidget":
        return ClickMeWidget()

    @staticmethod
    def from_type(cls: type) -> "ClickMeWidget":
        return ClickMeWidget()

    def __init__(self):
        super().__init__()
        self._value = 0

        self.button = QPushButton("Click me!", self)
        self.button.setCursor(Qt.PointingHandCursor)
        self.button.clicked.connect(self.on_button_clicked)
        self.label = QLabel("", self)
        self._update_label_text()

        self.grid_layout.addWidget(self.label, 0, 0)
        self.grid_layout.addWidget(self.button, 0, 1)

    def _update_label_text(self):
        self.label.setText(f"Clicks so far: {self._value}")

    def on_button_clicked(self):
        self.value += 1

    @property
    def value(self) -> int:
        return self._value

    @value.setter
    def value(self, new_value: int) -> None:
        self._value = new_value
        self._update_label_text()
        self.value_changed.emit(self._value)

class OptionsEnum(Enum):
    First = 1
    Second = 2
    Third = 3


class CustomType(PersistentPropertiesMixin, QObject):
    changed = Signal()

    def __init__(self):
        super().__init__()
        self._name = ""
        self._value = 0

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value
        self.changed.emit()

    @property
    def value(self) -> int:
        return self._value

    @value.setter
    def value(self, value: int) -> None:
        self._value = value
        self.changed.emit()


class DemoWidgets(Plugin):
    label = "Widgets Demo"

    def __init__(self):
        super().__init__()

        self._bool_property = False
        self._int_property = 0
        self._click_property = 0
        self._float_property = 1.0
        self._text_property = "Demo Text (max. length = 30)"
        self._long_text_property = (
            "By default, text properties are edited using a single-line text input, "
            "as shown above. For longer texts, you can use the MultiLineTextWidget, "
            "which needs to be selected explicitly in the @property_params decorator."
        )
        self._dropdown_property = OptionsEnum.First
        self._font_property = QFont("Arial", 12, QFont.Bold)
        self._color_property = QColor(0, 0, 255, 128)
        self._path_property = Path(__file__).parent
        self._file_path_property = FilePath(__file__)
        self._list_property = ["Item 1", "Item 2", "Item 3"]

        custom_item_1 = CustomType()
        custom_item_1.name = "Custom Item 1"
        custom_item_1.value = 1
        custom_item_2 = CustomType()
        custom_item_2.name = "Custom Item 2"
        custom_item_2.value = 2

        # NOTE: custom types must emit a signal on state change, and this
        # signal must be connected to the `changed` signal of the plugin, so that
        # the plugin knows when to save the properties and update the UI
        custom_item_1.changed.connect(self.changed.emit)
        custom_item_2.changed.connect(self.changed.emit)
        self._custom_type_list = [custom_item_1, custom_item_2]

    @property
    def bool_property(self) -> bool:
        return self._bool_property

    @bool_property.setter
    def bool_property(self, value: bool) -> None:
        self._bool_property = value

    @property
    @property_params(min=0, max=10, step=1)
    def int_property(self) -> int:
        return self._int_property

    @int_property.setter
    def int_property(self, value: int) -> None:
        self._int_property = value

    @property
    @property_params(min=0.0, max=2.0, decimals=1, step=0.1)
    def float_property(self) -> float:
        return self._float_property

    @float_property.setter
    def float_property(self, value: float) -> None:
        self._float_property = value

    @property
    @property_params(max_length=30)
    def text_property(self) -> str:
        return self._text_property

    @text_property.setter
    def text_property(self, value: str) -> None:
        self._text_property = value

    @property
    @property_params(widget=MultiLineTextWidget)
    def long_text_property(self) -> str:
        return self._long_text_property

    @long_text_property.setter
    def long_text_property(self, value: str) -> None:
        self._long_text_property = value

    @property
    def dropdown_property(self) -> OptionsEnum:
        return self._dropdown_property

    @dropdown_property.setter
    def dropdown_property(self, value: OptionsEnum) -> None:
        self._dropdown_property = value

    @property
    def font_property(self) -> QFont:
        return self._font_property

    @font_property.setter
    def font_property(self, value: QFont) -> None:
        self._font_property = value

    @property
    def color_property(self) -> QColor:
        return self._color_property

    @color_property.setter
    def color_property(self, value: QColor) -> None:
        self._color_property = value

    @property
    @property_params(dialog_title="Select a folder")
    def path_to_a_folder(self) -> Path:
        return self._path_property

    @path_to_a_folder.setter
    def path_to_a_folder(self, value: Path) -> None:
        self._path_property = value

    @property
    def path_to_a_file(self) -> FilePath:
        return self._file_path_property

    @path_to_a_file.setter
    def path_to_a_file(self, value: FilePath) -> None:
        self._file_path_property = value

    @property
    @property_params(
        prevent_add=False,             # set to True to hide the "Add" button
        add_button_text="Add item",    # changes the text of the "Add" button
    )
    def list_property(self) -> list[str]:
        return self._list_property

    @list_property.setter
    def list_property(self, value: list[str]) -> None:
        self._list_property = value

    @property
    @property_params(
        prevent_add=True,
        item_params={
            "label_field": "name"
        }
    )
    def custom_type_list(self) -> list[CustomType]:
        return self._custom_type_list

    @custom_type_list.setter
    def custom_type_list(self, value: list[CustomType]) -> None:
        self._custom_type_list = value

    @property
    @property_params(widget=ClickMeWidget)
    def custom_widget_property(self) -> int:
        return self._click_property

    @custom_widget_property.setter
    def custom_widget_property(self, value: int) -> None:
        self._click_property = value

    @action
    @action_params(
        compact=True,
        # icon=QIcon.fromTheme("document-edit")  # uncomment to set a custom icon
    )
    def print_to_console(self) -> None:
        logging.info(f"Current values of properties: {self.to_dict()}")

    @action
    def default_action(self, param1: str, param2: str) -> None:
        logging.info(f"Default action triggered with params: {param1}, {param2}")
