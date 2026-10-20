from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QMessageBox

from pupil_labs.neon_player.ui.main_window import MainWindow


def test_main_window_on_update_available(qtbot):
    main_window = MainWindow()
    qtbot.addWidget(main_window)

    assert main_window.statusBar().isHidden()
    assert main_window.update_pill.isHidden()

    with qtbot.waitExposed(main_window):
        def shows_update_pill():
            assert main_window.statusBar().isVisible()
            assert main_window.update_pill.isVisible()

        main_window.show()
        main_window.on_update_available("7.0.0", "url")
        qtbot.waitUntil(shows_update_pill)


def test_main_window_dismissed_update_records_version(qtbot, qapp):
    main_window = MainWindow()
    qtbot.addWidget(main_window)

    qapp.settings.last_displayed_update_version = "6.0.0"
    with qtbot.waitExposed(main_window):
        main_window.show()
        main_window.on_update_available("7.0.0", "url")

    qtbot.mouseClick(main_window.update_pill.dismiss_btn, Qt.MouseButton.LeftButton)

    assert main_window.update_pill.isHidden()
    assert qapp.settings.last_displayed_update_version == "7.0.0"
