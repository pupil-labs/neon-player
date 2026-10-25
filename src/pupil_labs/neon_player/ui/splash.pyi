from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)

class Ui_Splash:
    gridLayout: QGridLayout
    content: QWidget
    verticalLayout_3: QVBoxLayout
    label: QLabel
    horizontalLayout: QHBoxLayout
    horizontalSpacer: QSpacerItem
    dropbox: QWidget
    verticalLayout_2: QVBoxLayout
    label_3: QLabel
    browse_button: QPushButton
    horizontalSpacer_2: QSpacerItem
    recent: QHBoxLayout
    recent_spacer_right: QSpacerItem
    recent_button: QPushButton
    recent_spacer_left: QSpacerItem
    logo: QLabel

    def setupUi(self, Splash: QWidget) -> None: ...
    def retranslateUi(self, Splash: QWidget) -> None: ...
