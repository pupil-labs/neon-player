# AOI import

The **AOI Import** plugin loads areas of interest (AOIs) that were mapped onto
the scene video outside Neon Player, one shape per AOI per video frame. Neon
Player draws them on the video, shows a timeline row per AOI, tests every gaze
sample and fixation against them, and exports the results.

## CSV format

One row per AOI per scene video frame. Coordinates are **pixels in the scene
video as Neon Player shows it** (1600 × 1200, origin top left, *not*
undistorted).

```csv
timestamp [ns],aoi,shape,coords
1764856738522000000,logo,rectangle,"412,300,180,95"
1764856738522000000,face,polygon,"803,211 851,198 880,260 841,302 799,280"
1764856738522000000,clock,ellipse,"1210,455,60,42,15"
```

| Column | Content |
| --- | --- |
| `timestamp [ns]` | Timestamp of the scene video frame (as in `world_timestamps.csv` of a Neon export). Matched to the nearest frame; rows more than half a frame away are skipped with a warning. |
| `frame_index` | Alternative to `timestamp [ns]`: 0-based index of the scene video frame. Use one of the two. |
| `aoi` | AOI name. Rows with the same name belong to the same AOI. |
| `shape` | `rectangle`, `polygon` or `ellipse` |
| `coords` | Numbers for the shape, see below. Separators (comma, space, semicolon) don't matter, only the order of the numbers. |

Shapes:

| `shape` | `coords` |
| --- | --- |
| `rectangle` | `x,y,width,height` (x/y = top left corner) |
| `polygon` | `x1,y1 x2,y2 x3,y3 ...` (at least 3 points, any shape incl. concave) |
| `ellipse` | `center_x,center_y,radius_x,radius_y[,angle]` (angle in degrees, clockwise in image coordinates, optional) |

Rules:

- An AOI that has no row for a frame is **not visible** in that frame.
- Several rows with the same AOI and frame are combined (the AOI is the union
  of the shapes), e.g. for an AOI that is split by an occluding object.
- AOIs may overlap. Each gaze sample / fixation is tested against each AOI
  separately.

Writing the file from Python:

```python
import pandas as pd

rows = []
for frame_ts, aoi_name, polygon in mapped_aois:   # your mapping output
    coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in polygon)
    rows.append((frame_ts, aoi_name, "polygon", coords))

pd.DataFrame(rows, columns=["timestamp [ns]", "aoi", "shape", "coords"]).to_csv(
    "aois.csv", index=False
)
```

## From makesense.ai (flat stimuli)

`tools/makesense_to_aoi.py` creates this CSV directly from a makesense.ai
"VGG JSON" export and the stimulus image(s). It finds the stimulus image in
every video frame by image features (no markers, no training) and maps the
AOIs onto the frame, including the lens distortion:

```bash
.venv/bin/python tools/makesense_to_aoi.py RECORDING_DIR labels.json \
    --images-dir FOLDER_WITH_STIMULUS_IMAGES --out aois.csv --preview 5
```

`--preview 5` writes 5 frames with the AOIs drawn, to check the result.
Frames in which the stimulus isn't found reliably (fewer than
`--min-inliers` feature matches, default 40) get no AOIs.

## Using it in Neon Player

1. Enable **AOI Import** under *Plugins → Add/Remove*.
2. Click **Import csv** and choose the file. The file is copied into the
   recording (`.neon_player/cache/AOIPlugin/`), so it stays with the recording.
   Importing again replaces the AOIs.
3. **Clear aois** removes them.

Settings: show AOIs, show labels, fill opacity, line width. An AOI is filled
more strongly while gaze is inside it.

The gaze offset set in *Gaze Data* is applied to all AOI tests.

## Export

**Export** (or *Export All*) writes three files, limited to the export window:

`aoi_gaze.csv`, one row per gaze sample:

- `timestamp [ns]`, `scene frame index`, `gaze x [px]`, `gaze y [px]`
- `aoi`: names of the AOIs the sample is in, separated by `;`
- `in aoi: <name>`: True/False per AOI

`aoi_fixations.csv`, one row per fixation:

- fixation id, start/end timestamp, duration, position
- `share in aoi: <name>`: share of the fixation's gaze samples inside the AOI
  (AOIs can move during a fixation)
- `on aoi: <name>`: share ≥ 0.5
- `aoi`: the AOI with the highest share, if ≥ 0.5 (with overlapping AOIs see
  the per-AOI columns)

`aoi_metrics.csv`, one row per AOI:

| Column | Meaning |
| --- | --- |
| `visible frames`, `visible duration [s]` | frames in which the AOI exists |
| `gaze samples in aoi`, `dwell time [ms]` | gaze samples inside × sample interval |
| `fixation count`, `total fixation duration [ms]` | fixations on the AOI |
| `visit count` | runs of consecutive fixations on the AOI |
| `time to first fixation [ms]` | from the start of the export window |
| `first fixation duration [ms]` | |
