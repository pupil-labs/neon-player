# Dependency typing exceptions

Neon Player keeps strict MyPy checks enabled. Code-specific ignores cover verified
limitations in dependency annotations or persisted-property conversion. These
exceptions preserve existing library calls and do not replace library behavior.

This pass starts from `static_testing_updates` at `24f242e`. The baseline reports
70 errors in 13 files, checking 43 source files. Four IPC logger errors are fixed
in Player. The remaining 66 diagnostics receive 64 code-specific ignores. One
`arg-type` ignore covers two timestamp arguments, and one action call needs both
`arg-type` and `func-returns-value`.

## Exceptions and upstream proposals

| Dependency          | Baseline diagnostics | Limitation and proposal                                                                                                                                                                                                                                                                                                                                 |
| ------------------- | -------------------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Neon Recording      |                   42 | Numerical field descriptors, timestamp arrays, NumPy tuple and mask indexing, column selection, `Timeseries.data`, and `Record.items` need accurate types. See the [numerical proposals](../.scratch/shared-application-types/numerical-library-proposals.md) and [plugin proposals](../.scratch/shared-application-types/plugin-library-proposals.md). |
| Marker Mapper       |                   10 | `normalized_corners`, `crop_image`, and `Surface.rotate` need annotations. See the [plugin proposals](../.scratch/shared-application-types/plugin-library-proposals.md#marker-mapper).                                                                                                                                                                  |
| Qt Property Widgets |                    9 | Four widget class arguments and two action-result diagnostics need accurate annotations. Three XDF getter results remain optional at runtime, while optional getter annotations break persisted conversion. See the [widget and persistence proposals](../.scratch/shared-application-types/plugin-library-proposals.md#qt-property-widgets).           |
| pyqtgraph           |                    4 | `ViewBox`, `AxisItem`, `LegendItem`, and `PlotItem` lack typed subclass contracts. Typed upstream declarations or a maintained stub package would remove these exceptions.                                                                                                                                                                              |
| pandas stubs        |                    1 | The installed stubs omit `pandas.errors.ChainedAssignmentError`, although the installed pandas exports it. Correcting the stub declaration would remove this exception.                                                                                                                                                                                 |

Each ignored line names its dependency limitation. Application files reference
this document so the corresponding proposals remain discoverable. Existing
import exceptions and the earlier video-export ignore remain unchanged.

## Verify and retire exceptions

Run the full check from this checkout using its virtual environment.

```sh
.venv/bin/mypy
PATH="$PWD/.venv/bin:$PATH" QT_QPA_PLATFORM=offscreen .venv/bin/pytest -q
.venv/bin/ruff check --no-cache src tests --no-fix
.venv/bin/ruff format --no-cache --check src tests
```

`strict = true` retains `warn_unused_ignores`. After updating a dependency, rerun
MyPy and remove ignores that it reports as unused. Recheck numerical outputs,
exports, persisted XDF fields, and action dispatch after the relevant library
contracts change. These exceptions make the current check pass without claiming
that the dependency contracts are correct.
