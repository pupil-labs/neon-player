import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import partial

from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication, QWidget
from shiboken6 import isValid

from pupil_labs.neon_player.overlays import PanelOverlay


def _require_gui_thread() -> None:
    app = QApplication.instance()
    if app is None or QThread.currentThread() != app.thread():
        raise RuntimeError("Panel operations require the GUI thread")


class PanelWidget:
    """Rendering-independent overlay lifecycle for concrete QWidget hosts.

    Mix this base into a QWidget subclass and call _initialize_overlays after
    initializing the Qt widget. Rendering compatibility belongs to the manager,
    not this base; the initial supported pair is exclusively 2D.
    """

    def _initialize_overlays(self) -> None:
        _require_gui_thread()
        self._overlays: dict[PanelOverlay, int] = {}
        self._painting = False
        self._mutating = False
        self._mutation_guard: Callable[[], None] | None = None
        self._contribution_removed: Callable[[PanelOverlay], None] | None = None
        self._overlay_added_callback: Callable[[PanelOverlay], None] | None = None
        self._overlay_manager: object | None = None
        self._destroy_handlers: dict[PanelOverlay, Callable[..., None]] = {}

    @property
    def _widget(self) -> QWidget:
        if not isinstance(self, QWidget):
            raise TypeError("A PanelWidget host must also be a QWidget")
        return self

    @property
    def overlays(self) -> tuple[PanelOverlay, ...]:
        _require_gui_thread()
        return tuple(sorted(self._overlays, key=self._overlays.__getitem__))

    def _require_mutation(self) -> None:
        _require_gui_thread()
        if self._mutation_guard is not None:
            self._mutation_guard()
        if self._painting or self._mutating:
            raise RuntimeError("Cannot mutate overlays during painting or removal")

    @contextmanager
    def _mutation(self) -> Iterator[None]:
        self._require_mutation()
        self._mutating = True
        try:
            yield
        finally:
            self._mutating = False

    def add_overlay(self, overlay: PanelOverlay, *, layer: int = 0) -> None:
        # The manager imports concrete hosts to validate compatibility.
        from pupil_labs.neon_player.managers.panel import _require_overlay_compatibility

        self._require_mutation()
        _require_overlay_compatibility(self, overlay)
        if not isinstance(layer, int):
            raise TypeError("Overlay layer must be an integer")
        if not isValid(overlay):
            raise ValueError("Overlay has been destroyed")
        if overlay.thread() != self._widget.thread():
            raise ValueError("Overlay must live on the GUI thread")
        if overlay.parent() is not None or overlay._attached_once:
            raise ValueError("Overlay is already owned or was previously attached")
        tooltip = overlay.tooltip
        if tooltip is not None:
            tooltip._require_attachable()
        with self._mutation():
            overlay.changed.connect(self._overlay_changed)
            overlay.setParent(self._widget)
            overlay._attached_once = True
            self._overlays[overlay] = layer
            if tooltip is not None:
                tooltip._attach(self._widget)
            handler = partial(self._overlay_destroyed, overlay)
            self._destroy_handlers[overlay] = handler
            overlay.destroyed.connect(handler)
            self._widget.update()
            self._notify_overlay_added(overlay)

    def remove_overlay(self, overlay: PanelOverlay) -> None:
        self._require_mutation()
        if overlay not in self._overlays or not isValid(overlay):
            raise ValueError("Overlay is not attached to this host")
        with self._mutation():
            try:
                overlay.on_removed()
            except Exception:
                logging.getLogger(__name__).exception(
                    "Overlay cleanup hook failed; disposing overlay"
                )
            tooltip = overlay.tooltip
            if tooltip is not None:
                tooltip._dispose()
            overlay.changed.disconnect(self._overlay_changed)
            overlay.destroyed.disconnect(self._destroy_handlers.pop(overlay))
            del self._overlays[overlay]
            if self._contribution_removed is not None:
                self._contribution_removed(overlay)
            self._notify_overlay_removed(overlay)
            overlay.deleteLater()
            self._widget.update()

    def _notify_overlay_removed(self, overlay: PanelOverlay) -> None:
        """Allow concrete hosts to expose a rendering-specific notification."""

    def _notify_overlay_added(self, overlay: PanelOverlay) -> None:
        if self._overlay_added_callback is not None:
            self._overlay_added_callback(overlay)

    def _overlay_destroyed(self, overlay: PanelOverlay) -> None:
        self._destroy_handlers.pop(overlay, None)
        self._overlays.pop(overlay, None)
        tooltip = overlay.tooltip
        if tooltip is not None:
            tooltip._dispose()
        if isValid(self._widget):
            if self._contribution_removed is not None:
                self._contribution_removed(overlay)
            self._notify_overlay_removed(overlay)

    def clear_overlays(self) -> None:
        """Dispose all overlays; hook failures are logged, not retried."""
        self._require_mutation()
        for overlay in self.overlays:
            self.remove_overlay(overlay)

    def _overlay_changed(self) -> None:
        self._widget.update()
