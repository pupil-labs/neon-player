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
Enable Panel Overlays Demo before or after Panels Demo to demonstrate an
independent overlay contributor. Layout controls remain separate work.

### Visualization-specific 2D overlays

`PanelWidget` is the rendering-independent overlay-lifecycle base. It is a
Qt-independent mixin, not itself a QWidget or an OpenGL widget. Concrete hosts
combine it with a QWidget subclass and call `_initialize_overlays()` after
initializing the widget. It supplies generic attachment, ownership, invalidation,
and disposal, with no painter requirement. Import it from
`pupil_labs.neon_player`.

`PanelWidget2D(PanelWidget, QOpenGLWidget)` is an optional 2D host, not a requirement for
`Panel`. Its default coordinates are widget-local pixels. Override
`_paint_content(painter)` to draw prepared content. `ScalingWidget`, available
from `pupil_labs.neon_player.ui.video_render_widget`, derives from it and adds
aspect-fit source coordinates, `fit_rect(QSize(...))`, inverse `map_point`, mouse
signals, and scaled-child positioning. While the pointer is within the mapped
content rectangle, it uses a cross cursor independently of overlay hit-testing;
leaving the content or widget restores the inherited cursor. `VideoRenderWidget` derives from
`ScalingWidget` and adds playback-driven frame preparation and FPS display;
existing scene and surface rendering still use their original frame hooks.

The built-in Gaze Data plugin attaches each `GazeVisualization` to the scene
panel as an `InteractivePanelOverlay2D`. Circle and crosshair visualizations
hit-test their rendered marks, highlight those marks on hover, and show the
overlay's display name in bold with the nearest gaze x/y coordinates in a
structured tooltip. Those coordinates follow pointer movement and update when
playback changes while the tooltip is visible; clicks do not currently trigger
an action. On-screen gaze painting uses the overlay host, while surface
previews retain their data-driven render path. Video export rendering is not
yet migrated to the overlay host.

Import `PanelOverlay`, `PanelOverlay2D`, `PanelOverlayTooltip`, `PanelWidget`,
and `PanelWidget2D` from `pupil_labs.neon_player`. Generic `PanelOverlay` has
read-only borrowed playback, a `changed` signal, and an optional tooltip, with
no painting method. `PanelOverlay2D` adds `paint(painter)`.
`PanelOverlayTooltip(widget)` accepts any parentless QWidget as its content.
`PanelOverlay.tooltip` exposes the optional tooltip. The tooltip adds a
non-transparent frame using the host's application palette; the host owns the
frame and content and disposes them when that overlay is removed. Initially,
`PanelWidget2D` displays tooltips for the topmost hovered
`InteractivePanelOverlay2D`, positions them beside and follows the pointer, and
hides them on hover exit or target change.
`InteractivePanelOverlay2D` also notifies its overlay as the pointer moves
within the hit region. The tooltip module is generic for future hoverable
overlay types. Playback is supplied explicitly, without an independent clock:

```python
from PySide6.QtWidgets import QLabel
from PySide6.QtCore import QRect

from pupil_labs.neon_player import (
    InteractivePanelOverlay2D,
    PanelOverlayTooltip,
    PanelWidget2D,
)


class TimeLabel(InteractivePanelOverlay2D):
    def __init__(self, playback):
        tooltip = PanelOverlayTooltip(QLabel("Structured tooltip content"))
        super().__init__(
            playback,
            "Time label",
            hitbox=QRect(0, 0, 120, 32),
            tooltip=tooltip,
        )
        self.timestamp = playback.time
        self._subscribed = True
        playback.changed.connect(self.on_position)

    def on_click(self, point):
        pass

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
    self.add_overlay(panel, overlay, layer=10)
```

`Plugin.add_overlay(panel, overlay, *, host=None, layer=0)` accepts a generic
`PanelOverlay` and optional generic `PanelWidget` host, and associates the
contribution with that exact plugin instance, independently of the host's Qt
ownership. It works on application-owned panels and panels from other plugins.
Omitting `host` targets the panel's root widget; an unsupported root raises
`TypeError` rather than choosing an arbitrary descendant. For compound widgets,
pass an explicit `PanelWidget` host that is also a QWidget descendant of `panel.widget`:

```python
self.add_overlay(panel, overlay, host=preview_widget, layer=10)
self.remove_overlay(panel, overlay, host=preview_widget)
```

The existing panel manager validates visualization compatibility before invoking
host attachment or committing contribution ownership. The only supported pair
currently is `PanelOverlay2D` on `PanelWidget2D`, including their subclasses.
Other combinations raise `TypeError` with no attachment or ownership change.
The generic interfaces do not imply universal rendering support; future
visualizations need an explicit compatibility rule. Direct host attachment
uses the same manager-defined rule so it cannot bypass the check.

`Plugin.remove_overlay(panel, overlay, *, host=None)` requires the same exact
panel, host, overlay, and contributing plugin instance. Supply the explicit host
again for nested contributions. Another instance of the same plugin class
does not inherit ownership. Missing panels raise `KeyError`; stale panel/host/
overlay identities and unrelated hosts raise `ValueError`; trying to remove
another plugin's or a host-owned overlay raises `PermissionError`. Failed
attachment leaves the overlay with the caller and no contribution record.

Direct `host.add_overlay` creates host-owned content without a plugin
contribution association. Direct `host.remove_overlay` is privileged and clears
any contribution record. `host.overlays` remains the shared snapshot; there is
no second discovery registry.

Plugin disable runs `on_disabled` while contributions remain available, then
removes its overlays before its panels. A surviving visualization keeps other
plugins' and host-owned overlays, and cached content is not recomputed.
Failed activation uses the same cleanup path after successful construction and
state restoration. Once automatic cleanup starts, further contributions from
that plugin instance are rejected with `RuntimeError`.

Closing/removing a panel drains overlays from its root and nested hosts while
they are valid, then notifies removal and schedules deletion. Backend detach
failures happen before this drain and retain the panel and overlays for retry.
Actual backend failures still retain pending plugin cleanup; successful earlier
overlay removals are not undone. If the host closes first, later contributor
cleanup finds no remaining records. Reusing the panel ID does not transfer
contributions to the replacement.

Discovery uses the same GUI-thread connect-then-snapshot pattern as panels.
Panel lifecycle notifications permit reads, not mutations: queue attachment
until notification delivery returns, then revalidate the exact panel and
contributor activity. Cancel pending work and disconnect discovery on disable.
The runnable Panel Overlays Demo uses a child single-shot timer to do this and
logs unexpected attachment errors at the Qt callback seam.

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
subscriptions immediately. If it raises, the host logs the exception with its
traceback and still disconnects invalidation, clears contribution tracking, and
schedules disposal. This log-and-dispose policy also applies to explicit removal
and plugin/panel teardown; cleanup-hook exceptions do not leave a pending
contribution. A broken hook's remaining QObject-bound playback callbacks end
when deferred destruction runs, not necessarily immediately upon removal.
Removed overlays cannot be
reattached, even before deferred deletion. Creating a fresh overlay is required.
Do not independently delete or reparent an attached overlay.

Managed panel removal calls `on_removed` while overlays are valid. Unexpected
host destruction uses Qt ownership and QObject-bound subscription lifetimes,
not Python cleanup hooks on dead wrappers; contribution records are also
cleared. Non-QObject-bound subscriptions require explicit cleanup before host
destruction. Recording unload removes recording-scoped
panels before silently clearing shared playback; no synthetic unload timestamp
notification is introduced.

::: pupil_labs.neon_player.managers.panel
