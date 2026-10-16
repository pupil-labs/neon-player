import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import QApplication, QWidget

from pupil_labs.neon_player.managers.playback import PlaybackPosition

if TYPE_CHECKING:
    from pupil_labs.neon_player.plugins import Plugin


def _require_gui_thread() -> None:
    app = QApplication.instance()
    if app is None or QThread.currentThread() != app.thread():
        raise RuntimeError("Panel operations require the GUI thread")


class Panel(QObject):
    """Visualization identity independent of the docking adapter.

    The manager supplies playback on successful registration. Access before
    registration raises RuntimeError; plugin authors need not provide a clock.
    closable governs user requests, not explicit owner removal or shutdown.
    """

    def __init__(
        self,
        panel_id: str,
        title: str,
        widget: QWidget,
        *,
        closable: bool = True,
    ) -> None:
        _require_gui_thread()
        super().__init__()
        self._panel_id = panel_id
        self._title = title
        self._widget = widget
        self._closable = closable
        self._playback: PlaybackPosition | None = None

    @property
    def panel_id(self) -> str:
        return self._panel_id

    @property
    def title(self) -> str:
        return self._title

    @property
    def widget(self) -> QWidget:
        return self._widget

    @property
    def closable(self) -> bool:
        return self._closable

    @property
    def playback(self) -> PlaybackPosition:
        if self._playback is None:
            raise RuntimeError("Panel must be registered before accessing playback")
        return self._playback


class PanelLayoutBackend(QObject):
    """Internal docking seam; concrete dock types stay inside adapters.

    attach and detach must leave ownership and layout unchanged on failure.
    detach releases the visualization widget before disposing its dock wrapper.
    The manager owns this adapter; the container parents its host widget.
    """

    close_requested = Signal(str)

    @property
    def widget(self) -> QWidget:
        raise NotImplementedError

    def attach(self, panel: Panel) -> None:
        raise NotImplementedError

    def detach(self, panel_id: str) -> None:
        raise NotImplementedError


class PanelManager(QObject):
    """Registry owning panels and the injected layout adapter.

    Call shutdown before destroying the container that holds the adapter's host.
    The shared playback object is borrowed and outlives this manager.

    New panels and widgets must be parentless. Successful addition transfers
    ownership; failure preserves it. Duplicate IDs and invalid ownership
    raise ValueError; removal of missing IDs raises KeyError. Lookup returns
    None for missing IDs. Adapter errors propagate.
    All operations run on the GUI thread. Mutation during notifications raises
    RuntimeError, as does mutation after shutdown.

    Optional contribution owners identify plugin instances, independently of Qt
    parenting. remove_owned_panel requires the current panel object and owner;
    stale objects raise ValueError and other owners raise PermissionError.
    remove_panel remains privileged for application cleanup.
    """

    panel_added = Signal(object)
    panel_removed = Signal(object)

    def __init__(
        self,
        backend: PanelLayoutBackend,
        playback: PlaybackPosition,
        parent: QObject | None = None,
    ) -> None:
        _require_gui_thread()
        if backend.parent() is not None:
            raise ValueError("Layout backend is already owned")
        if backend.thread() != QThread.currentThread():
            raise ValueError("Layout backend must live on the GUI thread")
        if playback.thread() != QThread.currentThread():
            raise ValueError("Playback position must live on the GUI thread")
        super().__init__(parent)
        self._backend = backend
        self._playback = playback
        self._panels: dict[str, Panel] = {}
        self._panel_owners: dict[str, Plugin] = {}
        self._busy = False
        self._closed = False
        backend.setParent(self)
        backend.close_requested.connect(self._on_close_requested)

    @property
    def playback(self) -> PlaybackPosition:
        return self._playback

    @property
    def panels(self) -> tuple[Panel, ...]:
        _require_gui_thread()
        return tuple(self._panels.values())

    def get_panel(self, panel_id: str) -> Panel | None:
        """Return the registered panel, or None if its ID is absent."""
        _require_gui_thread()
        return self._panels.get(panel_id)

    def add_panel(self, panel: Panel, *, owner: "Plugin | None" = None) -> None:
        with self._mutation():
            if owner is not None:
                self._validate_owner(owner)
            if panel.panel_id in self._panels:
                raise ValueError(f"Duplicate panel ID: {panel.panel_id}")
            if panel.parent() is not None or panel.widget.parent() is not None:
                raise ValueError("Panel and visualization widget must be parentless")
            if (
                panel.thread() != QThread.currentThread()
                or panel.widget.thread() != QThread.currentThread()
            ):
                raise ValueError("Panel and visualization must live on the GUI thread")
            self._backend.attach(panel)
            panel._playback = self._playback
            panel.setParent(self)
            self._panels[panel.panel_id] = panel
            if owner is not None:
                self._panel_owners[panel.panel_id] = owner
            self.panel_added.emit(panel)

    def remove_owned_panel(self, panel: Panel, owner: "Plugin") -> None:
        """Remove an exact contribution without granting application privileges."""
        with self._mutation():
            current = self._panels[panel.panel_id]
            if current is not panel:
                raise ValueError(f"Stale panel object: {panel.panel_id}")
            self._validate_owner(owner)
            if self._panel_owners.get(panel.panel_id) is not owner:
                raise PermissionError(
                    f"Panel is not owned by this plugin: {panel.panel_id}"
                )
            self._remove_panel(panel.panel_id)

    def remove_panels_owned_by(self, owner: "Plugin") -> None:
        """Remove this instance's remaining contributions in registration order.

        A detach failure stops cleanup. Remaining panels retain registration and
        ownership for retry; successfully removed panels stay removed.
        """
        with self._mutation():
            self._validate_owner(owner)
            panel_ids = tuple(
                panel_id
                for panel_id, panel_owner in self._panel_owners.items()
                if panel_owner is owner
            )
            for panel_id in panel_ids:
                self._remove_panel(panel_id)

    @staticmethod
    def _validate_owner(owner: "Plugin") -> None:
        from pupil_labs.neon_player.plugins import Plugin

        if not isinstance(owner, Plugin):
            raise TypeError("Panel contribution owner must be a Plugin instance")
        if owner.thread() != QThread.currentThread():
            raise ValueError("Panel contribution owner must live on the GUI thread")

    def remove_panel(self, panel_id: str) -> None:
        with self._mutation():
            self._remove_panel(panel_id)

    def _remove_panel(self, panel_id: str) -> None:
        panel = self._panels[panel_id]
        self._backend.detach(panel_id)
        del self._panels[panel_id]
        self._panel_owners.pop(panel_id, None)
        panel.widget.setParent(self._backend.widget)
        self.panel_removed.emit(panel)
        panel.widget.deleteLater()
        panel.deleteLater()

    def shutdown(self) -> None:
        """Remove panels and disconnect close requests before host destruction.

        Repeated successful shutdown is a no-op. Failures propagate and can be
        retried; already removed panels stay removed. This is not recording unload.
        """
        _require_gui_thread()
        if self._closed:
            return
        with self._mutation():
            for panel_id in tuple(self._panels):
                self._remove_panel(panel_id)
            self._backend.close_requested.disconnect(self._on_close_requested)
            self._closed = True

    @contextmanager
    def _mutation(self) -> Iterator[None]:
        _require_gui_thread()
        if self._closed:
            raise RuntimeError("Panel manager is shut down")
        if self._busy:
            raise RuntimeError("Panel mutation is already in progress")
        self._busy = True
        try:
            yield
        finally:
            self._busy = False

    def _on_close_requested(self, panel_id: str) -> None:
        try:
            _require_gui_thread()
            panel = self._panels[panel_id]
            if panel.closable:
                self.remove_panel(panel_id)
        except Exception:
            logging.getLogger(__name__).exception("Failed to close panel %s", panel_id)
