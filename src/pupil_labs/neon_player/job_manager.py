import contextlib
import logging
import pickle
import subprocess
import sys
import typing as T
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QDataStream, QObject, Signal
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from tqdm import tqdm

from pupil_labs import neon_player
from pupil_labs.neon_recording import NeonRecording


@dataclass
class ProgressUpdate:
    progress: float = 0.0
    datum: T.Any = None


def prepare_command(
    recording_path: Path, 
    action_name: str,
    args: T.Any,
    server_name: str,
    batch_mode_enabled: bool,
    is_frozen: bool,
):
    """
    Prepares the command line call for running a background job in a subprocess.
    """
    
    cmd_args = [
        str(recording_path),
        "--progress-ipc-name",
        server_name
    ]

    # When batch mode is enabled, let the subprocess know that the job
    # needs to be executed with a workspace (i.e., parent folder) in mind
    if batch_mode_enabled:
        cmd_args += ["--workspace"]

    cmd_args += [
        "--job",
        action_name,
    ] + [str(arg) for arg in args]

    if is_frozen:
        return [sys.executable, *cmd_args]
    
    return [sys.executable, "-m", "pupil_labs.neon_player", *cmd_args]


class BaseBackgroundJob(QObject):
    progress_changed = Signal(float)
    finished = Signal()
    canceled = Signal()

    def __init__(self, name: str, job_id: int):
        super().__init__()
        self.name = name
        self.job_id = job_id
        self.progress = -1

    def cancel(self):
        raise NotImplementedError("Subclasses must implement cancel()")


class BackgroundJob(BaseBackgroundJob):
    def __init__(
        self,
        name: str,
        job_id: int,
        recording_path: Path,
        action_name: str,
        *args: T.Any,
    ):
        super().__init__(name, job_id)

        self.socket = None
        self._expected_payload_size = None

        server_name = f"neon-player-job-{job_id}"
        self.server = QLocalServer()
        QLocalServer.removeServer(server_name)  # remove any dangling sockets
        if not self.server.listen(server_name):
            logging.error(f"Could not start IPC server for job {name}")
            return

        self.server.newConnection.connect(self._handle_connection)

        cmd = prepare_command(
            recording_path=recording_path,
            action_name=action_name,
            args=args,
            server_name=server_name,
            batch_mode_enabled=neon_player.instance().batch_mode_enabled,
            is_frozen=neon_player.is_frozen(),
        )
        logging.debug(f"Executing bg job {' '.join(cmd)}")

        self.proc = subprocess.Popen(cmd)  # noqa: S603

        logging.info(f"Background job started: {self.name}")

    def _handle_connection(self):
        self.socket = self.server.nextPendingConnection()
        self.socket.readyRead.connect(self._read_progress_update)
        self.socket.errorOccurred.connect(self._on_socket_error)
        self.socket.disconnected.connect(self._on_socket_disconnected)
        self.server.close()

    def _on_socket_error(self, err):
        if err == QLocalSocket.LocalSocketError.PeerClosedError:
            return

        logging.error("Job %s failed with unexpectedly with error %s", self.name, err)
        self.cancel()

    def _on_socket_disconnected(self):
        err = self.socket.error()
        if err not in (None, QLocalSocket.LocalSocketError.PeerClosedError):
            logging.warning(f"Unexpected job socket disconnection {err} for job {self.name}")

        self.finished.emit()

    def _read_progress_update(self):
        if not self.socket:
            return

        stream = QDataStream(self.socket)
        while self.socket.bytesAvailable() >= 4:
            # Read the length of the data
            if self._expected_payload_size is None:
                self._expected_payload_size = stream.readUInt32()

            # Wait until all data is available
            if self.socket.bytesAvailable() < self._expected_payload_size:
                return

            # Read and unpickle the data
            data = stream.readRawData(self._expected_payload_size)
            self._expected_payload_size = None
            try:
                obj = pickle.loads(data)  # noqa: S301
                self.progress_changed.emit(obj.progress)
                self.progress = obj.progress
            except Exception:
                logging.exception("Could not unpickle progress update")

    def cancel(self):
        self.proc.terminate()
        self.proc.wait()
        self.canceled.emit()


class BatchBackgroundJob(BaseBackgroundJob):
    progress_changed = Signal(float)
    canceled = Signal()
    finished = Signal()

    def __init__(
        self,
        name: str,
        job_id: int,
        action_name: str,
        args_generator: T.Callable[[NeonRecording], T.Any] | None = None,
        recordings: list[NeonRecording] | None = None,
    ):
        super().__init__(name, job_id)

        app = neon_player.instance()
        if not app.batch_mode_enabled:
            logging.warning("Batch mode is not enabled, cannot run batch action.")
            return

        if recordings is None:
            recordings = app.workspace.recordings

        self.action_name = action_name
        self.args_generator = args_generator
        self.recordings = recordings.copy()
        self.size = len(self.recordings)
        self.current_idx = 0
        self.current_job = None

        self._copy_session_settings()
        self._submit_next_job()

    def cancel(self):
        if self.current_job:
            self.current_job.cancel()
        self.canceled.emit()

    def _copy_session_settings(self):
        app = neon_player.instance()
        for rec in self.recordings:
            pass
            # shutil.copy(app.session_settings_path, rec.session_settings_path)

    def _submit_next_job(self):
        current_recording = self.recordings[self.current_idx]
        progress_str = ""
        if self.size > 1:
            progress_str = f" ({self.current_idx + 1}/{self.size})"
        args = self.args_generator(current_recording) if self.args_generator else []

        app = neon_player.instance()
        job = app.job_manager.run_background_action(
            f"{self.name}{progress_str}",
            self.action_name,
            *args,
            recording=current_recording
        )
        job.canceled.connect(lambda: self._on_batch_job_canceled())
        job.finished.connect(lambda: self._on_batch_job_finished())
        self.current_job = job

    def _on_batch_job_canceled(self):
        self.canceled.emit()

    def _on_batch_job_finished(self):
        self.current_idx += 1
        if self.current_idx < len(self.recordings):
            self._submit_next_job()
            self.progress = self.current_idx / self.size
            self.progress_changed.emit(self.progress)
        else:
            self.progress_changed.emit(1.0)
            self.finished.emit()


class JobManager(QObject):
    updated = Signal()
    job_started = Signal(BackgroundJob)
    job_finished = Signal(BackgroundJob)
    job_canceled = Signal(BackgroundJob)

    def __init__(self):
        super().__init__()
        self.current_jobs = []
        self.job_counter = 0

    def work_job(self, job: T.Generator[ProgressUpdate, None, None]) -> None:
        ipc_name = neon_player.instance().progress_ipc_name
        # runs in child process
        if ipc_name:
            socket = QLocalSocket()
            socket.connectToServer(ipc_name)
            if not socket.waitForConnected(1000):
                logging.error(f"Could not connect to IPC server {ipc_name}")
                return

            if not hasattr(job, "__iter__"):
                logging.warning("A background job did not generate progress updates")
            else:
                try:
                    outstream = QDataStream(socket)
                    for update in job:
                        data = pickle.dumps(update)
                        outstream.writeUInt32(len(data))
                        outstream.writeRawData(data)
                        socket.flush()
                except Exception as e:
                    logging.exception(f"IPC communication failed in work_job: {e}")
                    return

            socket.flush()
            socket.close()
            if socket.state() != socket.LocalSocketState.UnconnectedState:
                socket.waitForDisconnected(5000)

        else:
            with tqdm(total=1.0) as pbar:
                for update in job:
                    pbar.n = update.progress
                    pbar.refresh()

    def run_background_action(
        self,
        name: str,
        action_name: str,
        *args: T.Any,
        recording: NeonRecording | None = None
    ) -> BackgroundJob:
        if neon_player.instance().headless:
            logging.warning("Not starting background job in headless mode")
            return

        neon_player.instance().save_settings()

        if recording is None:
            recording = neon_player.instance().recording

        if recording is None:
            rec_dir = None
        else:
            rec_dir = recording._rec_dir

        job = BackgroundJob(
            name,
            self.job_counter,
            rec_dir,
            action_name,
            *args,
        )
        self.job_counter += 1

        job.canceled.connect(lambda: self.on_job_canceled(job))
        job.finished.connect(lambda: self.on_job_finished(job))
        job.progress_changed.connect(lambda _: self.updated.emit())

        self.current_jobs.append(job)
        self.job_started.emit(job)
        self.updated.emit()

        logging.info(f"{job.name} started in the background")

        return job

    def run_background_batch_action(
        self,
        name: str,
        action_name: str,
        args_generator: T.Callable[[NeonRecording], T.Any] | None = None,
        recordings: list[NeonRecording] | None = None
    ):
        neon_player.instance().save_settings()

        job = BatchBackgroundJob(
            name,
            self.job_counter,
            action_name,
            args_generator=args_generator,
            recordings=recordings
        )
        self.job_counter += 1

        job.canceled.connect(lambda: self.on_job_canceled(job))
        job.finished.connect(lambda: self.on_job_finished(job))
        job.progress_changed.connect(lambda _: self.updated.emit())

        self.current_jobs.append(job)
        self.job_started.emit(job)
        self.updated.emit()

        logging.info(f"{job.name} started in the background")

        return job

    def on_job_finished(self, job: BackgroundJob) -> None:
        logging.info(f"{job.name} finished")
        neon_player.instance().show_notification(
            "Job finished", f"Job '{job.name}' has completed"
        )
        self.remove_job(job)
        self.updated.emit()

    def on_job_canceled(self, job: BackgroundJob) -> None:
        self.job_canceled.emit(job)
        self.remove_job(job)
        self.updated.emit()

    def remove_job(self, job: BackgroundJob) -> None:
        self.job_counter -= 1
        with contextlib.suppress(ValueError):
            self.current_jobs.remove(job)
        self.job_finished.emit(job)
