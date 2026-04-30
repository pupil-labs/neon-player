class Colors:
    PrimaryMain = "#6d7be0"
    PrimaryContrast = "#ededef"
    SecondaryMain = "#2e2e32"
    SecondaryContrast = "#a09fa6"
    TextPrimary = "#ededef"
    TextSecondary = "#a09fa6"
    TextDisabled = "#504f57"


STYLESHEET = f"""
QWidget {{
    font-family: Inter;
    font-size: 11pt;
    color: #a09fa6;
}}

#TimestampLabel {{
    color: {Colors.TextSecondary};
    font-family: ChivoMono;
    font-size: 14px;
}}

QMenuBar, QMenu {{
    color: #ccc;
    background: #1c2021;
}}

QTableWidget, QHeaderView {{
    background: transparent;
    border: none;
}}

QTableWidget::item {{
    border-bottom: 1px solid #292d2d;
    padding: 20px;
    padding-left: 0px;
}}

QTableWidget::item:selected {{
    background: #292d2d;
}}

QHeaderView::section {{
    background-color: transparent;
    border: none;
    color: #a09fa6;
    font-size: 10pt;
    font-weight: normal;
}}

QHeaderView::section:hover {{
    background-color: #292d2d;
}}

QMenuBar::item:selected,
QMenu::item:selected {{
    color: #fff;
    background: #292d2d;
}}

QPushButton {{
    color: #d0cfd6;
}}

QPushButton, QToolButton {{
    background: transparent;
    border: none;
    border-radius: 4px;
}}

QPushButton[style="primary"] {{
    background-color: {Colors.PrimaryMain};
    color: {Colors.PrimaryContrast};
}}

QPushButton[style="secondary"] {{
    background-color: {Colors.SecondaryMain};
    color: {Colors.SecondaryContrast};
}}

QPushButton[variant="M"] {{
    height: 24px;
    font-size: 12.25px;
    min-height: 24px;
    max-height: 24px;
    padding-left: 8px;
    padding-right: 8px;
}}

QToolButton[variant="M"] {{
    height: 24px;
    min-height: 24px;
    max-height: 24px;
    icon-size: 16px;
    padding: 4px;
    width: 24px;
    min-width: 24px;
    max-width: 24px;
}}

#BackButton, #RecentButton {{
    background: transparent;
    border: none;
    color: #a09fa6;
    padding: 5px;
}}

#BackButton:hover, #RecentButton:hover, QPushButton:hover {{
    background: #292d2d;
}}

Expander {{
    border-top: 1px solid #292d2d;
    border-bottom: 1px solid #292d2d;
    padding-top: 10px;
    padding-bottom: 10px;
}}

PluginManagerWidget>QLabel {{
    padding-top: 15px;
    padding-bottom: 15px;
}}

ExpanderList {{
    border: 2px solid #ff0000;
}}

Expander Expander {{
    border: none;
    padding-top: 5px;
    padding-bottom: 5px;
}}

SettingsPanel QLabel#ExpanderName {{
    color: #fff;
    font-size: 12pt;
    font-weight: bold;
}}

Expander Expander QLabel#ExpanderName {{
    font-size: 11pt;
    font-weight: normal;
}}

QToolButton {{
    background-color: transparent;
    border: none;
    border-radius: 4px;
}}

QToolButton:hover {{
    background-color: {Colors.SecondaryMain};
}}

QToolButton#HeaderAction {{
    background-color: #2e2f33;
    padding: 3px;
    border: none;
    border-radius: 4px;
    color: #9e9da1;
}}

QToolButton#PluginManagerHeaderAction {{
    background-color: #6d7be0;
    padding: 3px;
    border: none;
    border-radius: 4px;
    color: #fff;
}}

ConsoleWindow>QTextEdit {{
    font-family: 'ChivoMono', 'Menlo', 'Monico', 'Consolas', 'Lucida Console',
        'monospace', 'Courier New', 'Courier';
}}

TimestampLabel {{
    font-weight: bold;
    font-size: 16pt;
    color: #fff;
}}

BoolWidget>QToolButton {{
    width: 22px;
    height: 20px;
    border-radius: 5px;
    border: 1px solid #555;
    background-color: #111;
    color: #fff;
}}

BoolWidget>QToolButton:checked {{
    background: #6d7be0;
    border: 1px solid #555;
}}

QDockWidget::title {{
    background-color: #0f1314;
    padding: 5px;
}}

TextWidget>QLineEdit {{
    height: 24px;
    border-radius: 5px;
    border: 1px solid #555;
    background-color: #111;
}}

QStatusBar {{
    border-top: 1px solid #333;
}}

QStatusBar > QPushButton {{
    text-align: left;
    padding: 5px 10px;
    font-size: 10pt;
}}

#DeleteButton {{
    border: none;
}}
"""
