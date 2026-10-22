"""Per-frame AOI data: CSV parsing, point-in-shape tests and metrics.

Kept free of Qt so it can be tested and reused on its own.

CSV format (one row per AOI per scene video frame, coordinates in scene
video pixels):

    timestamp [ns],aoi,shape,coords
    1764856738522000000,logo,rectangle,"412,300,180,95"
    1764856738522000000,face,polygon,"803,211 851,198 880,260 841,302"
    1764856738522000000,clock,ellipse,"1210,455,60,42,15"

- `timestamp [ns]` is the scene frame timestamp. A `frame_index` column
  (0-based scene frame index) can be used instead.
- shapes: rectangle "x,y,width,height" (x/y = top left corner),
  polygon "x1,y1 x2,y2 x3,y3 ..." (3 or more points),
  ellipse "center_x,center_y,radius_x,radius_y[,angle_deg]".
- Several rows for the same AOI and frame are combined (union).
- An AOI without a row for a frame is not visible in that frame.
"""

import re
import typing as T
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pandas as pd

ELLIPSE_POINTS = 64
SHAPES = ("polygon", "rectangle", "ellipse")
_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")

# Accepted spellings of the column names (lower case, without spaces)
_COLUMN_ALIASES = {
    "timestamp": ("timestamp[ns]", "timestamp", "timestamp_ns", "time[ns]"),
    "frame": ("frame_index", "frameindex", "frame", "frame_idx", "sceneframeindex"),
    "aoi": ("aoi", "aoi_name", "name", "label"),
    "shape": ("shape", "type", "aoi_type"),
    "coords": ("coords", "coordinates", "points", "geometry"),
}


class AOIImportError(ValueError):
    pass


@dataclass
class AOIData:
    """All imported AOI shapes, stored as polygons.

    Shape i belongs to AOI `names[aoi_index[i]]`, is visible on scene frame
    `frame_index[i]`, and has the vertices `vertices[i, :n_vertices[i]]`.
    Rows are sorted by frame.
    """

    names: list[str]
    aoi_index: npt.NDArray[np.int32]
    frame_index: npt.NDArray[np.int64]
    vertices: npt.NDArray[np.float64]  # (n_shapes, max_vertices, 2), padded
    n_vertices: npt.NDArray[np.int32]
    shape_type: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.aoi_index)

    def shapes_for_frame(self, frame_idx: int) -> range:
        start = np.searchsorted(self.frame_index, frame_idx, side="left")
        stop = np.searchsorted(self.frame_index, frame_idx, side="right")
        return range(int(start), int(stop))

    def polygon(self, shape_idx: int) -> npt.NDArray[np.float64]:
        return self.vertices[shape_idx, : self.n_vertices[shape_idx]]

    def visible_frames(self, aoi_idx: int) -> npt.NDArray[np.int64]:
        return np.unique(self.frame_index[self.aoi_index == aoi_idx])


def _normalize(col: str) -> str:
    return re.sub(r"\s+", "", str(col).strip().lower())


def _find_column(df: pd.DataFrame, key: str) -> str | None:
    normalized = {_normalize(c): c for c in df.columns}
    for alias in _COLUMN_ALIASES[key]:
        if alias in normalized:
            return normalized[alias]
    return None


def shape_to_polygon(shape: str, numbers: list[float]) -> npt.NDArray[np.float64]:
    if shape == "rectangle":
        if len(numbers) != 4:
            raise ValueError("rectangle needs 4 numbers: x,y,width,height")
        x, y, w, h = numbers
        if w <= 0 or h <= 0:
            raise ValueError("rectangle width and height must be positive")
        return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], float)

    if shape == "ellipse":
        if len(numbers) not in (4, 5):
            raise ValueError(
                "ellipse needs 4 or 5 numbers: center_x,center_y,radius_x,radius_y[,angle]"
            )
        cx, cy, rx, ry = numbers[:4]
        if rx <= 0 or ry <= 0:
            raise ValueError("ellipse radii must be positive")
        angle = np.deg2rad(numbers[4]) if len(numbers) == 5 else 0.0
        t = np.linspace(0, 2 * np.pi, ELLIPSE_POINTS, endpoint=False)
        x, y = rx * np.cos(t), ry * np.sin(t)
        ca, sa = np.cos(angle), np.sin(angle)
        return np.column_stack([cx + x * ca - y * sa, cy + x * sa + y * ca])

    if shape == "polygon":
        if len(numbers) < 6 or len(numbers) % 2:
            raise ValueError("polygon needs at least 3 x,y points")
        return np.array(numbers, float).reshape(-1, 2)

    raise ValueError(f"unknown shape '{shape}', use one of {', '.join(SHAPES)}")


def parse_aoi_dataframe(
    df: pd.DataFrame,
    scene_times: npt.NDArray[np.int64],
    max_errors: int = 10,
) -> AOIData:
    """Turn the imported CSV into AOIData.

    Timestamps are matched to the nearest scene frame; rows further than half
    a frame away from any frame are skipped with a warning.
    """
    aoi_col = _find_column(df, "aoi")
    shape_col = _find_column(df, "shape")
    coords_col = _find_column(df, "coords")
    ts_col = _find_column(df, "timestamp")
    frame_col = _find_column(df, "frame")

    missing = [
        name
        for name, col in (("aoi", aoi_col), ("shape", shape_col), ("coords", coords_col))
        if col is None
    ]
    if ts_col is None and frame_col is None:
        missing.append("timestamp [ns] or frame_index")
    if missing:
        raise AOIImportError(
            f"Missing column(s): {', '.join(missing)}. "
            f"Found: {', '.join(map(str, df.columns))}"
        )

    n_frames = len(scene_times)
    warnings: list[str] = []

    # frame index per row
    if ts_col is not None:
        ts = pd.to_numeric(df[ts_col], errors="coerce").to_numpy(dtype=np.float64)
        frame_idx = _match_frames(ts, np.asarray(scene_times))
    else:
        fi = pd.to_numeric(df[frame_col], errors="coerce").to_numpy(dtype=np.float64)
        frame_idx = np.where(
            np.isfinite(fi) & (fi >= 0) & (fi < n_frames) & (fi == np.round(fi)),
            fi,
            -1,
        ).astype(np.int64)

    unmatched = int(np.sum(frame_idx < 0))
    if unmatched:
        warnings.append(
            f"{unmatched} row(s) could not be matched to a scene frame and were skipped"
        )

    names: list[str] = []
    name_lookup: dict[str, int] = {}
    polygons: list[npt.NDArray[np.float64]] = []
    aoi_indices: list[int] = []
    frames: list[int] = []
    shape_types: list[str] = []
    errors: list[str] = []

    aois = df[aoi_col].astype(str).str.strip().to_numpy()
    shapes = df[shape_col].astype(str).str.strip().str.lower().to_numpy()
    coords = df[coords_col].astype(str).to_numpy()

    for row, (frame, name, shape, coord) in enumerate(
        zip(frame_idx, aois, shapes, coords, strict=True)
    ):
        if frame < 0:
            continue
        if shape == "rect":
            shape = "rectangle"
        try:
            if not name or name.lower() == "nan":
                raise ValueError("empty AOI name")
            numbers = [float(v) for v in _NUMBER.findall(coord)]
            polygon = shape_to_polygon(shape, numbers)
        except ValueError as e:
            # +2: header line and 1-based line numbers
            errors.append(f"line {row + 2}: {e}")
            if len(errors) >= max_errors:
                break
            continue

        if name not in name_lookup:
            name_lookup[name] = len(names)
            names.append(name)

        polygons.append(polygon)
        aoi_indices.append(name_lookup[name])
        frames.append(int(frame))
        shape_types.append(shape)

    if errors:
        raise AOIImportError("Invalid rows:\n" + "\n".join(errors))

    if not polygons:
        raise AOIImportError("The file contains no usable AOI rows")

    max_vertices = max(len(p) for p in polygons)
    vertices = np.empty((len(polygons), max_vertices, 2), np.float64)
    n_vertices = np.empty(len(polygons), np.int32)
    for i, p in enumerate(polygons):
        vertices[i, : len(p)] = p
        # pad with the last vertex: zero-length edges don't affect the
        # point-in-polygon test
        vertices[i, len(p) :] = p[-1]
        n_vertices[i] = len(p)

    frames_arr = np.array(frames, np.int64)
    order = np.argsort(frames_arr, kind="stable")

    return AOIData(
        names=names,
        aoi_index=np.array(aoi_indices, np.int32)[order],
        frame_index=frames_arr[order],
        vertices=vertices[order],
        n_vertices=n_vertices[order],
        shape_type=[shape_types[i] for i in order],
        warnings=warnings,
    )


def _match_frames(
    timestamps: npt.NDArray[np.float64], scene_times: npt.NDArray[np.int64]
) -> npt.NDArray[np.int64]:
    """Nearest scene frame per timestamp, -1 if more than half a frame off."""
    result = np.full(len(timestamps), -1, np.int64)
    if len(scene_times) == 0:
        return result

    valid = np.isfinite(timestamps)
    ts = timestamps[valid]
    st = scene_times.astype(np.float64)

    right = np.clip(np.searchsorted(st, ts), 0, len(st) - 1)
    left = np.clip(right - 1, 0, len(st) - 1)
    pick_left = np.abs(ts - st[left]) <= np.abs(ts - st[right])
    nearest = np.where(pick_left, left, right)

    if len(st) > 1:
        tolerance = np.median(np.diff(st)) / 2
    else:
        tolerance = np.inf
    ok = np.abs(ts - st[nearest]) <= tolerance

    matched = np.where(ok, nearest, -1)
    result[valid] = matched
    return result


def points_in_polygons(
    points: npt.NDArray[np.float64], polygons: npt.NDArray[np.float64]
) -> npt.NDArray[np.bool_]:
    """Even-odd point-in-polygon test, point i against polygon i.

    points: (n, 2), polygons: (n, v, 2) (closed implicitly, may be padded by
    repeating the last vertex).
    """
    if len(points) == 0:
        return np.zeros(0, bool)

    px = points[:, 0:1]
    py = points[:, 1:2]
    xi = polygons[:, :, 0]
    yi = polygons[:, :, 1]
    xj = np.roll(xi, 1, axis=1)
    yj = np.roll(yi, 1, axis=1)

    crosses = (yi > py) != (yj > py)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_at_py = (xj - xi) * (py - yi) / (yj - yi) + xi
    hit = crosses & (px < x_at_py)
    return (np.count_nonzero(hit, axis=1) % 2) == 1


def gaze_frame_index(
    gaze_times: npt.NDArray[np.int64], scene_times: npt.NDArray[np.int64]
) -> npt.NDArray[np.int64]:
    """Scene frame shown while each gaze sample was taken (-1 before the first)."""
    return np.searchsorted(scene_times, gaze_times, side="right") - 1


def compute_gaze_hits(
    aois: AOIData,
    gaze_points: npt.NDArray[np.float64],
    gaze_frames: npt.NDArray[np.int64],
    chunk_size: int = 200_000,
) -> npt.NDArray[np.bool_]:
    """(n_gaze, n_aois) bool: gaze sample inside the AOI on its scene frame.

    gaze_frames must be sorted ascending (gaze samples sorted by time).
    """
    n_gaze = len(gaze_points)
    hits = np.zeros((n_gaze, len(aois.names)), bool)
    if n_gaze == 0 or len(aois) == 0:
        return hits

    # gaze sample range for each shape's frame
    starts = np.searchsorted(gaze_frames, aois.frame_index, side="left")
    stops = np.searchsorted(gaze_frames, aois.frame_index, side="right")
    counts = stops - starts

    # all (shape, gaze sample) pairs, built in chunks of shapes to cap memory
    cum = np.cumsum(counts)
    shape_start = 0
    while shape_start < len(aois):
        base = cum[shape_start - 1] if shape_start > 0 else 0
        shape_stop = int(np.searchsorted(cum, base + chunk_size, side="right"))
        shape_stop = max(shape_stop, shape_start + 1)
        sl = slice(shape_start, shape_stop)

        c = counts[sl]
        total = int(c.sum())
        if total:
            pair_shape = np.repeat(np.arange(shape_start, shape_stop), c)
            offsets = np.arange(total) - np.repeat(np.cumsum(c) - c, c)
            pair_gaze = np.repeat(starts[sl], c) + offsets

            inside = points_in_polygons(
                gaze_points[pair_gaze], aois.vertices[pair_shape]
            )
            hits[pair_gaze[inside], aois.aoi_index[pair_shape[inside]]] = True

        shape_start = shape_stop

    return hits


def runs_to_spans(
    mask: npt.NDArray[np.bool_],
    start_times: npt.NDArray[np.int64],
    stop_times: npt.NDArray[np.int64],
) -> list[tuple[int, int]]:
    """Merge consecutive True entries into (start, stop) time spans."""
    if len(mask) == 0 or not mask.any():
        return []

    padded = np.concatenate([[False], mask, [False]])
    diff = np.diff(padded.astype(np.int8))
    run_starts = np.flatnonzero(diff == 1)
    run_stops = np.flatnonzero(diff == -1) - 1
    return list(
        zip(
            start_times[run_starts].tolist(),
            stop_times[run_stops].tolist(),
            strict=True,
        )
    )


def fixation_aoi_shares(
    hits: npt.NDArray[np.bool_],
    gaze_times: npt.NDArray[np.int64],
    fixation_starts: npt.NDArray[np.int64],
    fixation_stops: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]:
    """(n_fixations, n_aois): share of a fixation's gaze samples in each AOI."""
    n_aois = hits.shape[1]
    shares = np.zeros((len(fixation_starts), n_aois))
    if len(fixation_starts) == 0 or len(gaze_times) == 0:
        return shares

    cum = np.vstack([np.zeros((1, n_aois), np.int64), np.cumsum(hits, axis=0)])
    first = np.searchsorted(gaze_times, fixation_starts, side="left")
    last = np.searchsorted(gaze_times, fixation_stops, side="right")
    n = (last - first).astype(np.float64)
    in_aoi = cum[last] - cum[first]
    with np.errstate(divide="ignore", invalid="ignore"):
        shares = np.where(n[:, None] > 0, in_aoi / n[:, None], 0.0)
    return shares


def aoi_metrics(
    names: list[str],
    gaze_hits: npt.NDArray[np.bool_],
    gaze_times: npt.NDArray[np.int64],
    fixation_on_aoi: npt.NDArray[np.bool_],
    fixation_starts: npt.NDArray[np.int64],
    fixation_stops: npt.NDArray[np.int64],
    visible_frames: list[npt.NDArray[np.int64]],
    scene_times: npt.NDArray[np.int64],
    window_start: int,
) -> pd.DataFrame:
    """Summary per AOI. All inputs already limited to the export window."""
    if len(gaze_times) > 1:
        sample_ms = float(np.median(np.diff(gaze_times))) / 1e6
    else:
        sample_ms = 0.0

    if len(scene_times) > 1:
        frame_s = float(np.median(np.diff(scene_times))) / 1e9
    else:
        frame_s = 0.0

    durations_ms = (fixation_stops - fixation_starts) / 1e6
    rows = []
    for i, name in enumerate(names):
        on = fixation_on_aoi[:, i] if len(fixation_on_aoi) else np.zeros(0, bool)
        fix_idx = np.flatnonzero(on)

        # a visit is a run of consecutive fixations on the AOI
        visits = int(np.count_nonzero(np.diff(np.concatenate([[0], on.astype(np.int8)])) == 1))

        first = fix_idx[0] if len(fix_idx) else None
        n_visible = len(visible_frames[i])
        rows.append({
            "aoi": name,
            "visible frames": n_visible,
            "visible duration [s]": n_visible * frame_s,
            "gaze samples in aoi": int(gaze_hits[:, i].sum()) if len(gaze_hits) else 0,
            "dwell time [ms]": (
                float(gaze_hits[:, i].sum()) * sample_ms if len(gaze_hits) else 0.0
            ),
            "fixation count": len(fix_idx),
            "total fixation duration [ms]": float(durations_ms[fix_idx].sum()),
            "visit count": visits,
            "time to first fixation [ms]": (
                (fixation_starts[first] - window_start) / 1e6 if first is not None else np.nan
            ),
            "first fixation duration [ms]": (
                durations_ms[first] if first is not None else np.nan
            ),
        })

    return pd.DataFrame(rows)


def safe_column_name(name: str) -> str:
    return name.replace(",", " ").replace("\n", " ")


def aoi_colors(n: int) -> list[tuple[int, int, int]]:
    """Distinct colors for n AOIs (evenly spaced hues)."""
    import colorsys

    colors = []
    for i in range(n):
        h = (i * 0.618033988749895) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.75, 1.0)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors


__all__: T.Sequence[str] = (
    "AOIData",
    "AOIImportError",
    "aoi_colors",
    "aoi_metrics",
    "compute_gaze_hits",
    "fixation_aoi_shares",
    "gaze_frame_index",
    "parse_aoi_dataframe",
    "points_in_polygons",
    "runs_to_spans",
)
