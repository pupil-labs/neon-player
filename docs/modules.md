::: pupil_labs.neon_player

## Plugin visualization panels

Import `Panel` and `Plugin` from `pupil_labs.neon_player`. A plugin can contribute
any parentless `QWidget`; no rendering base class or overlay support is required:

```python
from PySide6.QtWidgets import QLabel
from pupil_labs.neon_player import Panel, Plugin


class AnalysisPlugin(Plugin):
    def on_recording_loaded(self, recording):
        panel = Panel("analysis.summary", "Summary", QLabel("Static analysis"))
        self.add_panel(panel)
        self._summary_panel = panel
```

`Plugin.add_panel(panel)` registers the contribution under that exact plugin
instance. The manager then owns the panel and widget as Qt objects. Only
successful registration transfers ownership and binds `panel.playback`;
accessing playback beforehand raises `RuntimeError`. Failed attachment leaves
the panel and widget with the caller, without a success notification.

`panel.playback.time` is read-only integer nanoseconds on the recording time axis,
or `None` without a recording. Reading it does not subscribe or schedule work.
Widgets can explicitly connect `panel.playback.changed` to a QObject-bound method
after registration, then initialize from `panel.playback.time`. Every seek and
playback tick notifies, even for repeated timestamps. Static widgets should not
subscribe. Removal schedules widget deletion; QObject-bound subscriptions end
with the widget's lifetime.

### Discovery and removal

`Plugin.panel_manager` exposes shared `playback`, a snapshot tuple `panels`,
`get_panel(panel_id)`, and `panel_added`/`panel_removed` signals. Discovery includes
the required scene panel and all other plugins' panels; it grants no contribution
ownership. The property is excluded from plugin settings.

On the GUI thread, connect both lifecycle handlers and immediately process
`panel_manager.panels`, without yielding to the event loop between connection and
snapshot processing. This discovers earlier and later contributions without a
gap or duplicate delivery. Prefer QObject-bound handlers and disconnect them
when disabling an observer. Addition notifications expose fully registered
panels with bound playback and committed contribution ownership. Removal
notifications expose live panels/widgets already absent from discovery, before
deferred deletion. Handlers may inspect the registry but must defer mutations
until the current operation returns.

`Plugin.remove_panel(panel)` takes the exact panel object, not just an ID:

| Failure                                                                | Error             |
| ---------------------------------------------------------------------- | ----------------- |
| No registered panel with that ID                                       | `KeyError`        |
| ID now names a different panel object                                  | `ValueError`      |
| Current panel belongs to another plugin instance or the application    | `PermissionError` |
| Duplicate ID or invalid Qt ownership on addition                       | `ValueError`      |
| Operation off the GUI thread, during a notification, or after shutdown | `RuntimeError`    |

These checks leave registry and layout unchanged. Backend errors propagate; a
failed removal retains registration and contribution ownership for retry.
Validation first checks thread/mutation/shutdown, then lookup, panel identity,
and contribution ownership. Panels/widgets and contributing plugins must live on
the GUI thread.

User-close destroys a closable panel rather than retaining a hidden instance.
Reopening creates a fresh panel; a reference to the old object cannot remove a
replacement that reuses its ID. The required scene panel is not user-closable
and cannot be removed through a plugin's ownership-checked helper.

Contribution ownership is cooperative lifecycle bookkeeping, not a sandbox for
Python plugins. Direct manager removal by ID remains privileged for application
cleanup. Plugin authors should use the plugin helpers for contributions and use
the shared manager for discovery.

Plugin disable/failed-activation cleanup removes panels owned by that exact
plugin instance. Explicit removal in `on_disabled` is also supported; track
removals so already-closed panels are skipped. Disconnect discovery handlers
when disabling an observer.
Do not retain runtime panel references in serialized plugin settings.

See [the runnable Panels Demo](https://github.com/pupil-labs/neon-player/blob/main/examples/plugins/demo_panels.py)
for static and playback-driven widgets, discovery, close/reopen, and explicit
cleanup, including a cached static image with a playback-dependent overlay.
Layout controls and cross-plugin overlay ownership are separate work.

### Visualization-specific 2D overlays

`PanelWidget2D` is an optional `QOpenGLWidget` host, not a requirement for
`Panel`. Its default coordinates are widget-local pixels. Override
`_paint_content(painter)` to draw prepared content. `ScalingWidget`, available
from `pupil_labs.neon_player.ui.video_render_widget`, derives from it and adds
aspect-fit source coordinates, `fit_rect(QSize(...))`, inverse `map_point`, mouse
signals, and scaled-child positioning. `VideoRenderWidget` derives from
`ScalingWidget` and adds playback-driven frame preparation and FPS display;
existing scene and surface rendering still use their original frame hooks.

Import `PanelOverlay`, `PanelOverlay2D`, and `PanelWidget2D` from
`pupil_labs.neon_player`. Generic `PanelOverlay` has read-only borrowed playback
and a `changed` signal, with no painting method. `PanelOverlay2D` adds
`paint(painter)`. Playback is supplied explicitly, without an automatic
subscription or an independent clock:

```python
from pupil_labs.neon_player import PanelOverlay2D, PanelWidget2D


class TimeLabel(PanelOverlay2D):
    def __init__(self, playback):
        super().__init__(playback)
        self.timestamp = playback.time
        self._subscribed = True
        playback.changed.connect(self.on_position)

    def on_position(self, timestamp):
        self.timestamp = timestamp
        self.changed.emit()

    def paint(self, painter):
        text = "No recording" if self.timestamp is None else str(self.timestamp)
        painter.drawText(8, 24, text)

    def on_removed(self):
        if self._subscribed:
            self.playback.changed.disconnect(self.on_position)
            self._subscribed = False


# Within a plugin, after discovering a panel:
if isinstance(panel.widget, PanelWidget2D):
    overlay = TimeLabel(self.panel_manager.playback)
    panel.widget.add_overlay(overlay, layer=10)
```

The snippet demonstrates host capability checking, not automatic ownership of
contributions to another plugin's panel. This initial interface supports
host-owned overlays; cross-plugin contribution tracking/disable cleanup is
deferred. Prefer attaching overlays to your own panels for now.

The host paints content first and overlays in ascending integer layer order,
using insertion order for ties. `host.overlays` is a snapshot in painting order.
Layers are attachment metadata, not mutable overlay properties. Each overlay
receives its own saved/restored painter state and clipping to the content
rectangle, intersected with any incoming painter clip. Do not end the borrowed
painter. `paint(painter)` also renders a host into an active image-backed painter,
preserving caller state; it does not change the application's video export path.

The unscaled host's `map_point` is identity. The scaling specialization fits the
source rectangle and uses the same transform for content, overlays, and inverse
mapping. Points in letterboxing extrapolate outside the source rectangle;
mapping does not clamp them. Invalid source dimensions raise `ValueError`
before changing geometry; mapping with a zero-sized/unusable target raises
`RuntimeError`.

`changed` schedules a coalescible `update()`. It never calls content preparation.
Prepare static data or a cached image independently, then draw it in
`_paint_content`. Ordinary repainting of that cache is expected; recomputing it
on every playback notification is not. Neither `PanelWidget2D` nor
`ScalingWidget` requires a recording or subscribes to playback.

Attachment requires a live, parentless, never-attached `PanelOverlay2D` on the
GUI thread and transfers ownership only on success. Incompatible types/layers
raise `TypeError`; invalid ownership, identity, or deleted objects raise
`ValueError`. Off-GUI-thread mutation and mutation during painting/removal raise
`RuntimeError`. Rejected operations leave attachment/ownership unchanged.

`remove_overlay(overlay)` calls `on_removed()` before disconnecting host
invalidation and scheduling deletion. Implement the hook to disconnect optional
subscriptions immediately; make it retry-safe. If it raises, the error propagates
and the attachment remains available for retry. Removed overlays cannot be
reattached, even before deferred deletion. Creating a fresh overlay is required.
Do not independently delete or reparent an attached overlay.

When the host is destroyed, Qt destroys its overlays and disconnects
QObject-bound subscriptions; `on_removed` is for explicit removal, not a
destructor notification. Non-QObject-bound subscriptions require explicit
cleanup before host destruction. Recording unload removes recording-scoped
panels before silently clearing shared playback; no synthetic unload timestamp
notification is introduced.

::: pupil_labs.neon_player.managers.panel
