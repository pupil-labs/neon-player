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

Automatic per-plugin disable/failed-activation cleanup is not implemented yet.
Plugins must explicitly remove their live panels and disconnect discovery
handlers in `on_disabled`; track removals so already-closed panels are skipped.
Do not retain runtime panel references in serialized plugin settings.

See [the runnable Panels Demo](https://github.com/pupil-labs/neon-player/blob/main/examples/plugins/demo_panels.py)
for static and playback-driven widgets, discovery, close/reopen, and explicit
cleanup. Surface Tracking preview migration, layout controls, and overlays are
separate work.

::: pupil_labs.neon_player.panels
