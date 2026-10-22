"""Map makesense.ai AOIs from stimulus images onto a Neon recording.

Finds each stimulus image in every scene video frame by image features
(SIFT + RANSAC homography, lens distortion handled) and writes the AOI
polygons per frame as a CSV for Neon Player's "AOI Import" plugin.

No markers and no training needed; works for flat stimuli (posters, prints,
screens) that appear in the video like in the image file.

Usage:
    python tools/makesense_to_aoi.py RECORDING_DIR LABELS.json \
        [--images-dir DIR] [--out aois.csv] [--preview 5]

    RECORDING_DIR  Neon recording folder (native format, as opened in Neon Player)
    LABELS.json    makesense.ai export "VGG JSON"
    --images-dir   folder with the stimulus images named in the JSON
                   (default: the JSON's folder)
"""

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import pupil_labs.neon_recording as nr
from pupil_labs.camera import Camera

STIMULUS_WIDTH = 1600  # stimulus images are matched at this width
FRAME_SCALE = 0.6  # video frames are matched at this scale (speed)
EDGE_STEP_PX = 6  # densify polygon edges (stimulus px at match scale)


# ------------------------------------------------------------- makesense JSON


def _ellipse(cx, cy, rx, ry, theta=0.0, n=48):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x, y = rx * np.cos(t), ry * np.sin(t)
    c, s = np.cos(theta), np.sin(theta)
    return np.column_stack([cx + x * c - y * s, cy + x * s + y * c])


def load_vgg_regions(json_path: Path) -> dict[str, list[tuple[str, np.ndarray]]]:
    """{image filename: [(label, polygon in image px), ...]}"""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    result = {}
    for key, entry in data.items():
        filename = entry.get("filename") or key
        regions = entry.get("regions", {})
        if isinstance(regions, dict):
            regions = list(regions.values())

        aois = []
        for i, region in enumerate(regions):
            shape = region.get("shape_attributes", {})
            attrs = region.get("region_attributes", {})
            label = attrs.get("label") or next(
                (str(v) for v in attrs.values() if v), f"region_{i}"
            )
            kind = shape.get("name")
            if kind in ("polygon", "polyline"):
                poly = np.column_stack([shape["all_points_x"], shape["all_points_y"]])
            elif kind == "rect":
                x, y, w, h = (shape[k] for k in ("x", "y", "width", "height"))
                poly = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
            elif kind == "circle":
                poly = _ellipse(shape["cx"], shape["cy"], shape["r"], shape["r"])
            elif kind == "ellipse":
                poly = _ellipse(
                    shape["cx"], shape["cy"], shape["rx"], shape["ry"],
                    shape.get("theta", 0.0),
                )
            else:
                print(f"  skipping '{label}': shape '{kind}' is not an area")
                continue

            poly = np.asarray(poly, np.float64)
            # makesense polygons repeat the first point at the end
            if len(poly) > 3 and np.allclose(poly[0], poly[-1]):
                poly = poly[:-1]
            if len(poly) >= 3:
                aois.append((label, poly))

        if aois:
            result[filename] = aois
    return result


def densify(poly: np.ndarray, step: float) -> np.ndarray:
    out = []
    for a, b in zip(poly, np.roll(poly, -1, axis=0), strict=True):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        t = np.arange(n)[:, None] / n
        out.append(a + (b - a) * t)
    return np.vstack(out)


def clip_to_rect(poly: np.ndarray, w: float, h: float) -> np.ndarray:
    """Sutherland-Hodgman clipping to [0, w] x [0, h]."""
    def clip(points, inside, intersect):
        out = []
        for i, cur in enumerate(points):
            prev = points[i - 1]
            if inside(cur):
                if not inside(prev):
                    out.append(intersect(prev, cur))
                out.append(cur)
            elif inside(prev):
                out.append(intersect(prev, cur))
        return out

    def at_x(x):
        return lambda p, q: p + (q - p) * ((x - p[0]) / (q[0] - p[0]))

    def at_y(y):
        return lambda p, q: p + (q - p) * ((y - p[1]) / (q[1] - p[1]))

    pts = list(poly)
    for inside, inter in (
        (lambda p: p[0] >= 0, at_x(0)),
        (lambda p: p[0] <= w, at_x(w)),
        (lambda p: p[1] >= 0, at_y(0)),
        (lambda p: p[1] <= h, at_y(h)),
    ):
        if not pts:
            break
        pts = clip(pts, inside, inter)
    return np.array(pts).reshape(-1, 2)


# -------------------------------------------------------------------- worker

_W: dict = {}


def _init_worker(rec_dir, stimuli, min_inliers):
    cv2.setNumThreads(1)
    rec = nr.open(rec_dir)
    cal = rec.calibration
    _W["scene"] = rec.scene
    _W["size"] = (rec.scene.width, rec.scene.height)
    _W["camera"] = Camera(
        rec.scene.width, rec.scene.height,
        cal.scene_camera_matrix, cal.scene_distortion_coefficients,
    )
    _W["sift"] = cv2.SIFT_create(nfeatures=4000)
    _W["matcher"] = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    _W["min_inliers"] = min_inliers

    sift_stim = cv2.SIFT_create(nfeatures=6000)
    _W["stimuli"] = []
    for st in stimuli:
        img = cv2.imread(st["path"], cv2.IMREAD_GRAYSCALE)
        scale = STIMULUS_WIDTH / img.shape[1]
        small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        kps, desc = sift_stim.detectAndCompute(small, None)
        pts = np.float64([k.pt for k in kps]) / scale  # back to full image px
        corners = np.float64([
            [0, 0], [img.shape[1], 0], [img.shape[1], img.shape[0]], [0, img.shape[0]]
        ])
        aois = [
            (name, densify(poly, EDGE_STEP_PX / scale)) for name, poly in st["aois"]
        ]
        _W["stimuli"].append((pts, desc, corners, aois))


def _locate(frame_gray):
    """Homographies (stimulus px -> undistorted frame px) per stimulus."""
    small = cv2.resize(
        frame_gray, None, fx=FRAME_SCALE, fy=FRAME_SCALE, interpolation=cv2.INTER_AREA
    )
    kps, desc = _W["sift"].detectAndCompute(small, None)
    if desc is None or len(kps) < 8:
        return [None] * len(_W["stimuli"])

    frame_pts = np.float64([k.pt for k in kps]) / FRAME_SCALE
    frame_und = _W["camera"].undistort_points(frame_pts)[:, :2]

    result = []
    for stim_pts, stim_desc, corners, _ in _W["stimuli"]:
        matches = _W["matcher"].knnMatch(stim_desc, desc, k=2)
        good = [m[0] for m in matches if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
        if len(good) < _W["min_inliers"]:
            result.append(None)
            continue

        src = stim_pts[[m.queryIdx for m in good]]
        dst = frame_und[[m.trainIdx for m in good]]
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        if H is None or int(mask.sum()) < _W["min_inliers"]:
            result.append(None)
            continue

        # reject degenerate / mirrored solutions
        quad = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H).reshape(-1, 2)
        if not cv2.isContourConvex(quad.astype(np.float32)) or cv2.contourArea(
            quad.astype(np.float32), oriented=True
        ) <= 0:
            result.append(None)
            continue

        result.append(H)
    return result


def _project(poly, H):
    """Stimulus polygon -> distorted video px, clipped to the visible frame."""
    camera = _W["camera"]
    w, h = _W["size"]
    und = cv2.perspectiveTransform(poly.reshape(-1, 1, 2), H).reshape(-1, 2)
    dist = camera.distort_points(und)[:, :2]

    # far outside the camera's view the distortion model folds back:
    # keep only points that survive a distort/undistort round trip
    back = camera.undistort_points(dist)[:, :2]
    ok = np.linalg.norm(back - und, axis=1) < 1.0
    if ok.sum() < 3:
        return None
    clipped = clip_to_rect(dist[ok], w - 1, h - 1)
    if len(clipped) < 3 or cv2.contourArea(clipped.astype(np.float32)) < 4:
        return None
    return clipped


def _process(frame_range):
    start, stop = frame_range
    scene = _W["scene"]
    rows = []
    found = 0
    for idx in range(start, stop):
        frame = scene[idx]
        gray = np.ascontiguousarray(frame.gray)
        homographies = _locate(gray)
        for H, (_, _, _, aois) in zip(homographies, _W["stimuli"], strict=True):
            if H is None:
                continue
            found += 1
            for name, poly in aois:
                p = _project(poly, H)
                if p is not None:
                    coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in p)
                    rows.append((int(frame.time), name, "polygon", coords))
    return start, stop, rows, found


# ---------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording", type=Path)
    ap.add_argument("labels", type=Path, help="makesense.ai VGG JSON export")
    ap.add_argument("--images-dir", type=Path, help="folder with the stimulus images")
    ap.add_argument("--out", type=Path, default=Path("aois.csv"))
    ap.add_argument("--min-inliers", type=int, default=40,
                    help="minimum feature matches to accept a frame (default 40)")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--start", type=int, default=0, help="first frame index")
    ap.add_argument("--stop", type=int, default=None, help="last frame index (exclusive)")
    ap.add_argument("--preview", type=int, default=0,
                    help="write N preview images with the AOIs drawn next to --out")
    args = ap.parse_args()

    images_dir = args.images_dir or args.labels.parent
    regions = load_vgg_regions(args.labels)
    stimuli = []
    for filename, aois in regions.items():
        path = images_dir / filename
        if not path.exists():
            sys.exit(f"Stimulus image not found: {path} (use --images-dir)")
        prefix = f"{Path(filename).stem}/" if len(regions) > 1 else ""
        stimuli.append({
            "path": str(path),
            "aois": [(prefix + name, poly) for name, poly in aois],
        })
        print(f"{filename}: {len(aois)} AOI(s): {', '.join(n for n, _ in aois)}")
    if not stimuli:
        sys.exit("No AOIs found in the JSON")

    rec = nr.open(args.recording)
    n_frames = len(rec.scene)
    stop = min(args.stop or n_frames, n_frames)
    chunk = 200
    ranges = [(s, min(s + chunk, stop)) for s in range(args.start, stop, chunk)]

    print(f"Mapping {stop - args.start} frames with {args.workers} workers ...")
    t0 = time.time()
    all_rows, frames_found, done = [], 0, 0
    with mp.get_context("spawn").Pool(
        args.workers, _init_worker, (str(args.recording), stimuli, args.min_inliers)
    ) as pool:
        for s, e, rows, found in pool.imap_unordered(_process, ranges):
            all_rows.extend(rows)
            frames_found += found
            done += e - s
            print(f"\r  {done}/{stop - args.start} frames, "
                  f"{time.time() - t0:.0f} s", end="", flush=True)
    print()

    df = pd.DataFrame(all_rows, columns=["timestamp [ns]", "aoi", "shape", "coords"])
    df = df.sort_values(["timestamp [ns]", "aoi"], kind="stable")
    df.to_csv(args.out, index=False, encoding="utf-8")

    total = (stop - args.start) * len(stimuli)
    print(f"Stimulus found in {frames_found} of {total} frames "
          f"({100 * frames_found / max(total, 1):.0f} %)")
    print(f"Wrote {len(df)} rows to {args.out}")

    if args.preview and len(df):
        _write_previews(rec, df, args.out, args.preview)


def _write_previews(rec, df, out: Path, n: int) -> None:
    times = df["timestamp [ns]"].unique()
    picks = times[np.linspace(0, len(times) - 1, n).astype(int)]
    frame_times = rec.scene.time
    for t in picks:
        idx = int(np.searchsorted(frame_times, t))
        img = cv2.cvtColor(np.ascontiguousarray(rec.scene[idx].gray), cv2.COLOR_GRAY2BGR)
        for _, row in df[df["timestamp [ns]"] == t].iterrows():
            nums = np.array(row["coords"].replace(",", " ").split(), float).reshape(-1, 2)
            cv2.polylines(img, [nums.astype(np.int32)], True, (0, 0, 255), 3)
            cv2.putText(img, row["aoi"], tuple(nums.min(0).astype(int) + [0, -8]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        path = out.with_name(f"{out.stem}_preview_{idx:06d}.jpg")
        cv2.imwrite(str(path), img)
        print(f"Preview: {path}")


if __name__ == "__main__":
    main()
