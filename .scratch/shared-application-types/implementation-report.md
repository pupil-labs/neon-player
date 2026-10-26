# Shared application types implementation

Status: ready-for-human

The application-owned migration is implemented on `static_testing_updates`.
Full acceptance remains blocked by authoritative dependency type contracts and
three existing XDF getters that cannot become optional until the property
framework supports restoring union-annotated properties.

## Verification

- Before strict mypy reported 506 errors in 31 files. After it reports 70 errors
  in 13 files. Both runs checked the same configured 43 source files.
- Before pytest passed 100 tests. After pytest passes 145 tests.
- Ruff lint, formatting, and `git diff --check` pass.
- `mypy.ini`, `pyproject.toml`, and `ruff.toml` are unchanged. No new casts,
  application type suppressions, or dependency exceptions were added.
- Real application construction, no-recording playback controls, and unload pass
  with a temporary home and unique IPC socket. The window remains hidden under
  offscreen Qt. This is startup coverage, not a real-recording playback or
  rendered OpenGL equivalence test.

The first extra startup attempt omitted the virtual environment from PATH and
could not launch pyside6-uic. The next attempt reached an IPC startup failure.
The final run used the virtual environment PATH, an isolated socket name, and
sandbox escalation and passed. The existing suite used the supported PATH and
headless Qt configuration throughout.

Logs are [mypy before](mypy-before.txt), [mypy after](mypy-after.txt),
[pytest before](pytest-before.txt), [pytest after](pytest-after.txt), and
[startup smoke](startup-smoke.txt), and [lint/configuration checks](checks-after.txt).

The [independent final review](review.md) found no new actionable issues.

## Contracts and consumers

| Shared contract                  | Provider change                                                                                                                                      | Migrated consumers and evidence                                                                                                                                                     |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Application availability         | `instance()` returns a narrowed player or None. `require_instance()` establishes initialized access.                                                 | Plugin construction, settings defaults, UI callbacks, job manager. Startup and application contract tests cover absent, unrelated, and initialized applications.                    |
| Recording and settings lifecycle | Recording remains optional. Recording settings have an explicit optional type and clear on unload. Required plugin access captures a recording once. | Plugin render/export methods, playback speed changes, cache operations, timeline callbacks. Tests cover absent recording, load, switch, unload, and delayed callbacks after unload. |
| Plugin registration and lookup   | Class registries contain concrete class types. Name lookup remains optional.                                                                         | Gaze/fixation/surface/XDF/export dependencies establish concrete plugin classes. Tests cover missing and available plugins.                                                         |
| Callbacks and descriptors        | Callbacks have explicit call contracts. Action descriptor preserves binding and parameter/result types while the framework owns registration.        | Action invocation, argument conversion, progress form, and timeline callbacks are exercised through real Qt and property-framework objects.                                         |
| Numerical arrays                 | Arrays describe timestamps, indices, float32/float64 coordinates, and image bytes.                                                                   | Gaze/fixation computations, XDF samples, timeline data, and video frame boundaries. Existing result tests and new empty/populated cases pass.                                       |
| Qt objects                       | Missing objects remain optional. Concrete widgets are retained. Fractional geometry uses floating Qt APIs.                                           | Menus, timeline markers, drag state, signals, progress bars, preview surfaces. UI tests cover fractional positions, exact large timestamps, and equal export endpoints.             |
| Serialization                    | JSON remains unstructured until relevant fields or loaded object classes are checked. Opaque plugin state uses object values.                        | Recording/global settings and plugin caches. Populated nested state and nonempty XDF fields restore through the installed framework.                                                |

Model the Domain shaped optional lifecycle states and concrete registries.
The Laziness Protocol kept the existing application structure. Prove It Works
required the full suite and direct Qt/startup checks before handoff.

## Remaining diagnostics

| Owner or limitation                                | Count | Why it remains                                                                                                                                                      |
| -------------------------------------------------- | ----: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Neon Recording                                     |    42 | Field descriptors, timestamps, NumPy indexing, timeseries data, and calibration items have insufficient authoritative annotations. Player still uses library reads. |
| Marker Mapper                                      |    10 | Existing geometry/cropping functions lack annotations. Stronger plugin types expose two previously hidden calls.                                                    |
| Qt Property Widgets widget class parameters        |     4 | The runtime supports widget classes but its annotation permits widget instances.                                                                                    |
| Qt Property Widgets persisted optional XDF getters |     3 | Existing getter annotations remain unchanged because optional annotations break nonempty persisted state.                                                           |
| Qt Property Widgets action return                  |     2 | `ActionObject.__call__` declares None while its implementation returns its function result. Headless job dispatch retains the existing diagnostics.                 |
| pyqtgraph                                          |     4 | Existing subclass bases are untyped. A complete stub package is out of scope.                                                                                       |
| pandas stubs                                       |     1 | The installed stubs omit `pandas.errors.ChainedAssignmentError`. Its runtime use is unchanged.                                                                      |
| Independent IPC logger code                        |     4 | Unchanged payload-length narrowing, unreachable assertion, and exception serialization diagnostics. No application-access or recording contract is involved.        |

See [numerical library proposals](numerical-library-proposals.md) and
[plugin library proposals](plugin-library-proposals.md) for exact symbols,
proposed library changes, and how Neon Player would consume them.

The file-and-error-code comparison exposes five additional diagnostics after
stronger narrowing. They are `Calibration.items`, masked blink selection,
`Surface.rotate`, `normalized_corners`, and the scene timestamp `zip` call.
The last was previously reported as two argument errors. All are reviewed
library contract limitations, not newly introduced application operations.
No caller bypasses the recording or video libraries to remove these diagnostics.

The audio-only WAV cache writer predates this migration and remains unchanged.
Its relation to the video-writer assignment needs a library scope decision;
this work does not assert an unsupported PL Video capability.

## Reproduce

Run these commands from the requested branch worktree.

```sh
.venv/bin/mypy
PATH="$PWD/.venv/bin:$PATH" QT_QPA_PLATFORM=offscreen .venv/bin/pytest -q
.venv/bin/ruff check --no-cache src tests --no-fix
.venv/bin/ruff format --no-cache --check src tests
git diff --check
git diff --exit-code -- mypy.ini pyproject.toml ruff.toml
python3 .scratch/shared-application-types/summarize_diagnostics.py .scratch/shared-application-types/mypy-before.txt .scratch/shared-application-types/mypy-after.txt
```

The final mypy run used an external cache directory to keep verification writes
isolated. That changes caching only, not package selection or diagnostics.

## Per-file diagnostic comparison

| Application file                               | Before | After |
| ---------------------------------------------- | -----: | ----: |
| `__init__.py`                                  |      1 |     0 |
| `app.py`                                       |     26 |     4 |
| `ipc_logger.py`                                |      4 |     4 |
| `job_manager.py`                               |     10 |     0 |
| `plugins/__init__.py`                          |     21 |     0 |
| `plugins/audio.py`                             |     22 |     0 |
| `plugins/blinks.py`                            |      6 |     2 |
| `plugins/events.py`                            |      8 |     1 |
| `plugins/export_all.py`                        |      8 |     1 |
| `plugins/eye_overlay.py`                       |     41 |     0 |
| `plugins/eyestate.py`                          |     20 |    16 |
| `plugins/fixations.py`                         |     37 |     3 |
| `plugins/gaze.py`                              |     26 |     8 |
| `plugins/imu.py`                               |     11 |    10 |
| `plugins/scene_renderer.py`                    |      8 |     0 |
| `plugins/surface_tracking/surface_tracking.py` |     56 |     5 |
| `plugins/surface_tracking/tracked_surface.py`  |     64 |     6 |
| `plugins/surface_tracking/ui.py`               |     21 |     0 |
| `plugins/video_exporter.py`                    |      6 |     0 |
| `plugins/xdf.py`                               |     16 |     6 |
| `settings.py`                                  |      6 |     0 |
| `ui/__init__.py`                               |     16 |     0 |
| `ui/console.py`                                |      3 |     0 |
| `ui/main_window.py`                            |      8 |     0 |
| `ui/plugin_installation_dialog.py`             |      2 |     0 |
| `ui/progress_action_form.py`                   |      2 |     0 |
| `ui/settings_panel.py`                         |      8 |     0 |
| `ui/timeline_dock.py`                          |     22 |     0 |
| `ui/timeline_dock_components.py`               |     11 |     4 |
| `ui/video_render_widget.py`                    |      8 |     0 |
| `utilities.py`                                 |      8 |     0 |
| Total                                          |    506 |    70 |
