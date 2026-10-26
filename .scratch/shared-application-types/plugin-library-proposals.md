# Upstream typing limitations in plugin consumers

These limitations were verified in the installed dependencies during the shared
application types migration. Existing calls stay in Neon Player. No replacement
recording reads, video writes, stubs, casts, or diagnostic suppressions were added.

## Neon Recording

`timeseries/array_record.py` declares `Array` with a record type parameter.
`TimeseriesProps.time` uses that parameter for `np.int64`. Fancy indexing a
recording's `scene.time` and calling `tolist()` therefore infers a scalar `int`.
The actual result is an array of timestamps followed by a list. The marker
visibility timeline consequently retains a `zip` diagnostic.

The proposed change separates an array's record type from its scalar dtype and
provides accurate scalar, slice, and NumPy index overloads. Neon Player keeps its
current marker visibility calculation and consumes the correctly typed timestamp
lists directly.

`Timeseries.__getitem__` accepts integer, slice, and selected string keys in its
annotations. Runtime NumPy boolean masks return an `Array` through `_data[key]`.
The surface export now checks that this external result is a `GazeArray`. A new
regression test verifies populated and empty export intervals. The proposed
library overload accepts boolean masks and returns the generic `ArrayType`.
Neon Player could then remove that boundary check.

`Calibration` inherits an unannotated `Record.items()` method. Exporting the
calibration retains a `no-untyped-call` diagnostic. The proposed change types
`Record.items()` with the record's supported values. Neon Player continues to
iterate those values and convert NumPy values for its existing JSON schema.

## Marker Mapper

`surface.normalized_corners()` returns a float32 NumPy array but has no return
annotation. `Surface.rotate()` has no annotations. `utils.crop_image()` also
lacks parameter and return annotations. Their existing application calls retain
`no-untyped-call` diagnostics.

The proposed change annotates `normalized_corners()` as
`NDArray[np.float32]`, `Surface.rotate()` as returning `None`, and `crop_image()`
with its image array, transformation matrix, optional integer dimensions, and
image return type. Neon Player continues to use those library operations for
surface geometry and previews.

## Qt Property Widgets

`utilities.property_params.widget` is annotated as
`QWidget | Literal["auto"] | None`. The implementation also accepts widget
classes, including `DynamicComboWidget`, and creates their instances. XDF's
three dynamic selectors retain `arg-type` diagnostics for their existing widget
classes.

The proposed change adds the supported widget class type to this annotation.
Neon Player keeps passing `DynamicComboWidget` for its data stream, marker
stream, and sync event selectors.

## Optional persisted getters and action results

`PersistentPropertiesMixin.type_convert()` uses a getter's return annotation to
restore its persisted value. It does not support PEP 604 union annotations.
Annotating XDF's existing `file_path`, `data_stream_uid`, and `marker_stream_uid`
getters as optional makes nonempty persisted values fail with `TypeError` when
the framework tries to construct `types.UnionType`.

The proposed library change handles unions explicitly, preserves `None` for
nullable values, and converts non-null values through the appropriate member.
Neon Player would then annotate these three existing getters as optional without
changing persisted property names. Their original annotations and three original
mypy diagnostics are deliberately retained until that support exists. The new
`test_xdf_plugin_restores_nonempty_persisted_fields` regression covers restoration.

`type_convert()` also tries `isinstance(value, typing.Any)` inside nested generic
dictionaries. Recording settings now use `object` for opaque plugin-state values,
which describes their boundary truthfully and works with current conversion. A
regression restores populated nested plugin state through the real framework.

`ActionObject.__call__()` is annotated as returning `None`, although its body
returns the decorated function's result. The proposed library change preserves
that return contract, including generators used by headless background jobs.
The two existing diagnostics in `NeonPlayerApp.run_jobs` remain visible. Normal
action binding and invocation are tested through real framework action objects.
