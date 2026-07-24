from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTableWidget,
)
from PySide6.QtSvgWidgets import QSvgWidget

from pupil_labs.neon_player.ui.constants import Color


def create_heading_with_icon(
    heading: str,
    icon_path: str | Path,
    heading_level: int = 2,
    icon_size: tuple[int, int] = (24, 24),
) -> QHBoxLayout:
    layout = QHBoxLayout()
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)
    icon_widget = QSvgWidget(str(icon_path))
    icon_widget.setFixedSize(*icon_size)
    layout.addWidget(icon_widget)
    layout.addWidget(QLabel(f"<h{heading_level}>{heading}</h{heading_level}>"))
    layout.addStretch()
    return layout


class _RowColorDelegate(QStyledItemDelegate):
    def __init__(self, table, normal, hovered, selected, parent=None):
        super().__init__(parent)
        self._table = table
        self.normal = QColor(normal)
        self.hovered = QColor(hovered)
        self.selected = QColor(selected)

    def paint(self, painter, option, index):
        opt = QStyleOptionViewItem(option)
        row = index.row()

        if option.state & QStyle.StateFlag.State_Selected:
            painter.fillRect(option.rect, self.selected)
        elif row == self._table._hovered_row:
            painter.fillRect(option.rect, self.hovered)
        else:
            painter.fillRect(option.rect, self.normal)

        # NOTE: clear the selected and hovered state flags to prevent the
        # default painting behavior from overriding our custom colors
        opt.state &= ~QStyle.StateFlag.State_Selected
        opt.state &= ~QStyle.StateFlag.State_MouseOver

        super().paint(painter, opt, index)


class HoverRowTable(QTableWidget):
    def __init__(
        self,
        *args,
        normal_color: str = "transparent",
        hover_color: str = Color.State.Hover,
        selected_color: str = Color.State.Selected,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._hovered_row = -1
        self.setMouseTracking(True)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.verticalHeader().setDefaultSectionSize(60)

        delegate = _RowColorDelegate(self, normal_color, hover_color, selected_color)
        self.setItemDelegate(delegate)

        self.setStyleSheet("""
            QTableWidget, QHeaderView {
                background: transparent;
                border: none;
            }
            QTableWidget::item {
                border-bottom: 1px solid #292d2d;
                padding-right: 20px;
            }

            QHeaderView::section {
                background-color: transparent;
                border: none;
                color: #a09fa6;
                font-size: 10pt;
                font-weight: normal;
            }
            QHeaderView::section:hover {
                background-color: #292d2d;
            }
        """)

    def _set_hovered_row(self, row: int):
        if row != self._hovered_row:
            self._hovered_row = row
            self.viewport().update()

    def mouseMoveEvent(self, event):
        idx = self.indexAt(event.pos())
        if idx.isValid():
            self._set_hovered_row(idx.row())
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        super().mouseMoveEvent(event)

    def wheelEvent(self, event):
        idx = self.indexAt(event.position().toPoint())
        if idx.isValid():
            self._set_hovered_row(idx.row())
        super().wheelEvent(event)

    def leaveEvent(self, event):
        self._set_hovered_row(-1)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        super().leaveEvent(event)
