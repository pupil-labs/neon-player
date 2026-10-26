# Numerical dependency typing proposals

These findings come from the installed Neon Recording 2.1.6 source and the strict
mypy baseline captured for this implementation. No local recording parser,
replacement timeseries, casts, or dependency suppressions were added.

## Neon Recording array contracts

`pupil_labs.neon_recording.timeseries.array_record.fields.__get__` declares a
scalar result for `Record` and `Array[T]` for array and timeseries owners.
`Array` actually parameterizes its record class, not its numerical dtype.
Multi-column record fields return a NumPy vector at runtime, although their
annotation declares a scalar. Multi-column array fields return an unstructured
NumPy array. This affects gaze points, fixation mean points, IMU vectors, and
3D eye state vectors.

Proposed library change. Split scalar and vector field descriptors, or add
accurate overloads for their owner and field dimensionality. Numerical field
arrays return `numpy.typing.NDArray` with the stored dtype. Vector record fields
return a one-dimensional numerical array. Gaze's stored point fields use float32,
although `GazeProps.point` currently declares float64.

Neon Player use. Continue consuming `recording.gaze.point`, fixation mean points,
IMU rotation and acceleration vectors, and eye state vectors through those
properties. Numerical helpers then accept the library's actual array dtypes.

`Array.__getitem__` omits tuple and NumPy mask indices. `Timeseries.__getitem__`
omits NumPy mask indices and declares `list[str]` selection as the same timeseries
class, although its implementation returns an unstructured numerical array.
The NumPy mask sometimes matches the scalar index overload, so mypy infers a
record where runtime returns an array. `Timeseries.data` also lacks a declared
`ArrayType` return type.

Proposed library change. Type scalar selection as a record, slices as a
corresponding timeseries or array, NumPy masks and integer index arrays as the
concrete array class, and column lists as numerical arrays. Declare `data` as
`ArrayType`. Preserve existing runtime selection behavior.

Neon Player use. Keep boolean selections in gaze, fixation, blink, and saccade
exports and the existing column selections for timeline bars. These operations
remain library calls.

`sample.match_ts` and `Timeseries.sample` require `ArrayLike[int]`, while the
library's own timestamp fields declare NumPy int64 elements. The declared
`Array` iterator further inherits the record iterator contract.

Proposed library change. Accept the actual supported NumPy integer timestamp
arrays as well as Python integer sequences. Keep indices and timestamp values
distinct in return annotations.

Neon Player use. Continue passing sampled gaze timestamps into worn-data matching.

## Existing and newly exposed diagnostics

The baseline already contains the IMU and eye state tuple-index errors, worn
sampling and matching errors, masked gaze and fixation selection errors,
incorrect scalar point errors, and timeline column-selection mismatches.
Those diagnostics remain unresolved because their authoritative contracts are
owned by Neon Recording.

The blink export assignment now reveals a `BlinkRecord` versus `BlinkTimeseries`
error after a stable nonoptional recording reference removes an `Any` branch.
It has the same missing NumPy-mask overload cause. Fixation visualization input
now explicitly uses `FixationArray`, matching runtime masked selection, but the
library still infers `FixationRecord` at its caller.

## Property widget class annotation

`qt_property_widgets.utilities.property_params.widget` declares a widget instance,
`None`, or `"auto"`. Its docstring explicitly supports a `QWidget` subclass.
The baseline already rejects `widget=EventTypeListWidget`.

Proposed library change. Include `type[QWidget]` in the annotation. Neon Player
keeps its current custom widget class registration.

## Existing audio writer integration

`AudioPlugin.extract_audio` already opens an output container with `av.open` and
writes encoded packets directly. The repository now assigns all video writes,
including exports, to PL Video. This implementation preserves that existing
writer rather than introducing another direct writer. Whether an audio-only WAV
cache falls inside the assigned video-write scope requires a library scope
assessment before any writer migration. No PL Video API limitation is asserted
without that assessment.
