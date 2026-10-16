import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import QWidget
from shiboken6 import isValid

from pupil_labs.neon_player.managers.playback import PlaybackPosition
from pupil_labs.neon_player.overlays import PanelOverlay, PanelOverlay2D
from pupil_labs.neon_player.plugins import Plugin
from pupil_labs.neon_player.ui.panel_widget import PanelWidget, _require_gui_thread
from pupil_labs.neon_player.ui.panel_widget_2d import PanelWidget2D


@dataclass(frozen=True)
class _OverlayContribution:
    panel: "Panel"
    host: "PanelWidget"
    owner: "Plugin"
    destroyed_handler: Callable[..., None]


def _require_overlay_compatibility(
    host: "PanelWidget", overlay: "PanelOverlay"
) -> None:
    if not isinstance(host, PanelWidget2D) or not isinstance(overlay, PanelOverlay2D):
        raise TypeError("Only PanelOverlay2D on PanelWidget2D is currently supported")


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
        self._overlay_owners: dict[PanelOverlay, _OverlayContribution] = {}
        self._overlay_operation: PanelWidget | None = None
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
            hosts = self._panel_overlay_hosts(panel)
            for host in hosts:
                self._validate_host_manager(host)
            self._backend.attach(panel)
            panel._playback = self._playback
            panel.setParent(self)
            self._panels[panel.panel_id] = panel
            if owner is not None:
                self._panel_owners[panel.panel_id] = owner
            for host in hosts:
                self._bind_overlay_host(host)
            self.panel_added.emit(panel)

    @staticmethod
    def _panel_overlay_hosts(panel: Panel) -> list["PanelWidget"]:
        hosts: list[PanelWidget] = [
            child
            for child in panel.widget.findChildren(QWidget)
            if isinstance(child, PanelWidget)
        ]
        if isinstance(panel.widget, PanelWidget):
            hosts.insert(0, panel.widget)
        return hosts

    def _validate_host_manager(self, host: "PanelWidget") -> None:
        if (
            host._contribution_removed is not None
            and host._contribution_removed != self._forget_overlay
        ):
            raise ValueError("Overlay host is already managed by another manager")

    def _bind_overlay_host(self, host: "PanelWidget") -> None:
        self._validate_host_manager(host)
        if host._mutation_guard is None:
            host._mutation_guard = partial(self._guard_host_mutation, host)
            host._contribution_removed = self._forget_overlay

    def _resolve_overlay_host(
        self, panel: Panel, host: "PanelWidget | None"
    ) -> "PanelWidget":
        current = self._panels[panel.panel_id]
        if current is not panel or not isValid(panel):
            raise ValueError("Stale panel object")
        target = panel.widget if host is None else host
        if not isinstance(target, PanelWidget) or not isinstance(target, QWidget):
            raise TypeError("Target must be a PanelWidget and a QWidget")
        if not isValid(target):
            raise ValueError("Overlay host has been destroyed")
        if target is not panel.widget and not panel.widget.isAncestorOf(target):
            raise ValueError("Overlay host does not belong to this panel")
        return target

    def _guard_host_mutation(self, host: "PanelWidget") -> None:
        _require_gui_thread()
        if self._closed:
            raise RuntimeError("Panel manager is shut down")
        if self._busy and self._overlay_operation is not host:
            raise RuntimeError("Panel mutation is already in progress")

    def add_overlay(
        self,
        panel: Panel,
        overlay: "PanelOverlay",
        *,
        owner: "Plugin",
        host: "PanelWidget | None" = None,
        layer: int = 0,
    ) -> None:
        """Contribute a compatible overlay to an exact root or nested host."""
        with self._mutation():
            self._validate_owner(owner)
            if owner._overlay_contributions_closed:
                raise RuntimeError("Plugin overlay contributions are closed")
            target = self._resolve_overlay_host(panel, host)
            _require_overlay_compatibility(target, overlay)
            if overlay in self._overlay_owners:
                raise ValueError("Overlay is already contributed")
            self._bind_overlay_host(target)
            if not isValid(overlay):
                raise ValueError("Overlay has been destroyed")
            destroyed = partial(self._forget_overlay, overlay)
            self._overlay_owners[overlay] = _OverlayContribution(
                panel, target, owner, destroyed
            )
            overlay.destroyed.connect(destroyed)
            self._overlay_operation = target
            try:
                target.add_overlay(overlay, layer=layer)
            except Exception:
                self._forget_overlay(overlay)
                raise
            finally:
                self._overlay_operation = None

    def _forget_overlay(self, overlay: "PanelOverlay") -> None:
        contribution = self._overlay_owners.pop(overlay, None)
        if contribution is not None and isValid(overlay):
            overlay.destroyed.disconnect(contribution.destroyed_handler)

    def remove_owned_overlay(
        self,
        panel: Panel,
        overlay: "PanelOverlay",
        owner: "Plugin",
        *,
        host: "PanelWidget | None" = None,
    ) -> None:
        """Remove only the exact contributor's overlay; direct host removal is privileged."""
        with self._mutation():
            self._validate_owner(owner)
            target = self._resolve_overlay_host(panel, host)
            contribution = self._overlay_owners.get(overlay)
            if contribution is None:
                if overlay in target.overlays:
                    raise PermissionError("Overlay is host-owned")
                raise ValueError("Overlay is not contributed to this host")
            if contribution.panel is not panel or contribution.host is not target:
                raise ValueError("Overlay belongs to a different host")
            if contribution.owner is not owner:
                raise PermissionError("Overlay belongs to another plugin")
            self._remove_overlay(target, overlay)

    def _remove_overlay(
        self, host: "PanelWidget", overlay: "PanelOverlay"
    ) -> None:
        self._overlay_operation = host
        try:
            host.remove_overlay(overlay)
        finally:
            self._overlay_operation = None

    def remove_overlays_owned_by(self, owner: "Plugin") -> None:
        """Drain one instance's remaining overlays before cleaning up its panels."""
        with self._mutation():
            self._validate_owner(owner)
            for overlay, contribution in tuple(self._overlay_owners.items()):
                if contribution.owner is owner:
                    if isValid(overlay) and isValid(contribution.host._widget):
                        self._remove_overlay(contribution.host, overlay)
                    else:
                        self._forget_overlay(overlay)

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
        if not isinstance(owner, Plugin):
            raise TypeError("Panel contribution owner must be a Plugin instance")
        if owner.thread() != QThread.currentThread():
            raise ValueError("Panel contribution owner must live on the GUI thread")

    def remove_panel(self, panel_id: str) -> None:
        with self._mutation():
            self._remove_panel(panel_id)

    def _remove_panel(self, panel_id: str) -> None:
        panel = self._panels[panel_id]
        hosts = self._panel_overlay_hosts(panel)
        for host in hosts:
            if host._painting or host._mutating:
                raise RuntimeError("Cannot remove a panel during overlay painting/removal")
        self._backend.detach(panel_id)
        try:
            for host in hosts:
                self._overlay_operation = host
                host.clear_overlays()
        finally:
            self._overlay_operation = None
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
