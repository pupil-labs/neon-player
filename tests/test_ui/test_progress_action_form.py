from PySide6.QtCore import QObject
from pytestqt.qtbot import QtBot
from qt_property_widgets.utilities import ActionObject

from pupil_labs.neon_player.job_manager import BackgroundJob
from pupil_labs.neon_player.ui.progress_action_form import ProgressActionForm


def test_progress_action_invokes_framework_action_and_tracks_job(qtbot: QtBot) -> None:
    job = BackgroundJob.__new__(BackgroundJob)
    QObject.__init__(job)
    calls: list[bool] = []

    def start_job() -> BackgroundJob:
        calls.append(True)
        return job

    form = ProgressActionForm(ActionObject(start_job, None))
    qtbot.addWidget(form)
    form.action_button.click()
    assert calls == [True]
    assert not form.progress_bar.isHidden()
    job.progress_changed.emit(0.375)
    assert form.progress_bar.value() == 37
    job.finished.emit()
    assert form.progress_bar.isHidden()
    assert not form.action_button.isHidden()
