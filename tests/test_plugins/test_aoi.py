import numpy as np
import pandas as pd
import pytest

from pupil_labs.neon_player.plugins.aoi.aoi_data import (
    AOIImportError,
    aoi_metrics,
    compute_gaze_hits,
    fixation_aoi_shares,
    gaze_frame_index,
    parse_aoi_dataframe,
    points_in_polygons,
    runs_to_spans,
)

SCENE = (np.arange(10) * 50_000_000 + 1_000_000_000).astype(np.int64)  # 20 fps


def _df(rows, ts_col="timestamp [ns]"):
    return pd.DataFrame(rows, columns=[ts_col, "aoi", "shape", "coords"])


def test_parse_all_shapes():
    aois = parse_aoi_dataframe(
        _df([
            (SCENE[0], "box", "rectangle", "10,20,100,50"),
            (SCENE[1], "face", "polygon", "0,0 10,0 10,10 0,10"),
            (SCENE[1], "dial", "ellipse", "50,50,20,10,30"),
        ]),
        SCENE,
    )
    assert aois.names == ["box", "face", "dial"]
    assert aois.frame_index.tolist() == [0, 1, 1]
    np.testing.assert_allclose(
        aois.polygon(0), [[10, 20], [110, 20], [110, 70], [10, 70]]
    )
    assert aois.n_vertices.tolist() == [4, 4, 64]


def test_parse_frame_index_and_column_aliases():
    df = pd.DataFrame({
        "Frame Index": [3, 4],
        "Name": ["a", "a"],
        "Type": ["Rect", "rectangle"],
        "Points": ["0 0 5 5", "1;1;5;5"],
    })
    aois = parse_aoi_dataframe(df, SCENE)
    assert aois.frame_index.tolist() == [3, 4]
    assert aois.names == ["a"]


def test_timestamps_match_nearest_frame_within_half_frame():
    rows = [
        (SCENE[2] + 20_000_000, "a", "rectangle", "0,0,1,1"),  # 20 ms off: ok
        (SCENE[2] + 26_000_000, "b", "rectangle", "0,0,1,1"),  # nearer frame 3
        (SCENE[-1] + 30_000_000, "c", "rectangle", "0,0,1,1"),  # past the end
    ]
    aois = parse_aoi_dataframe(_df(rows), SCENE)
    assert aois.frame_index.tolist() == [2, 3]
    assert "1 row(s)" in aois.warnings[0]


def test_invalid_rows_are_reported_with_line_numbers():
    with pytest.raises(AOIImportError, match="line 3: polygon needs"):
        parse_aoi_dataframe(
            _df([
                (SCENE[0], "a", "rectangle", "0,0,1,1"),
                (SCENE[0], "b", "polygon", "0,0 1,1"),
            ]),
            SCENE,
        )


def test_missing_columns():
    with pytest.raises(AOIImportError, match="Missing column"):
        parse_aoi_dataframe(pd.DataFrame({"aoi": ["a"]}), SCENE)


def test_points_in_concave_polygon():
    # L-shape
    L = np.array([[0, 0], [10, 0], [10, 4], [4, 4], [4, 10], [0, 10]], float)
    pts = np.array([[2, 2], [8, 2], [2, 8], [8, 8], [11, 2]], float)
    inside = points_in_polygons(pts, np.repeat(L[None], len(pts), axis=0))
    assert inside.tolist() == [True, True, True, False, False]


def test_gaze_hits_per_frame_and_union():
    aois = parse_aoi_dataframe(
        _df([
            (SCENE[0], "a", "rectangle", "0,0,10,10"),
            (SCENE[0], "a", "rectangle", "100,100,10,10"),  # 2nd part of a
            (SCENE[1], "a", "ellipse", "50,50,5,5"),
            (SCENE[1], "b", "rectangle", "0,0,10,10"),
        ]),
        SCENE,
    )
    gaze_t = np.array([SCENE[0] + 1, SCENE[0] + 2, SCENE[1] + 1, SCENE[1] + 2, SCENE[2]])
    gaze_p = np.array([[5, 5], [105, 105], [50, 52], [5, 5], [5, 5]], float)
    frames = gaze_frame_index(gaze_t, SCENE)
    hits = compute_gaze_hits(aois, gaze_p, frames, chunk_size=2)
    assert hits.tolist() == [
        [True, False],
        [True, False],
        [True, False],
        [False, True],
        [False, False],  # frame 2 has no AOIs
    ]


def test_fixation_shares_and_metrics():
    gaze_t = np.arange(10, dtype=np.int64) * 5_000_000  # 200 Hz
    hits = np.zeros((10, 2), bool)
    hits[0:4, 0] = True
    hits[6:10, 1] = True
    starts = np.array([0, 25_000_000], np.int64)
    stops = np.array([20_000_000, 45_000_000], np.int64)
    shares = fixation_aoi_shares(hits, gaze_t, starts, stops)
    np.testing.assert_allclose(shares, [[0.8, 0.0], [0.0, 0.8]])

    metrics = aoi_metrics(
        ["a", "b"], hits, gaze_t, shares >= 0.5, starts, stops,
        [np.array([0, 1]), np.array([1])], SCENE, window_start=0,
    )
    a, b = metrics.to_dict("records")
    assert a["fixation count"] == 1 and a["visit count"] == 1
    assert a["dwell time [ms]"] == pytest.approx(20.0)
    assert b["time to first fixation [ms]"] == pytest.approx(25.0)
    assert b["visible duration [s]"] == pytest.approx(0.05)


def test_runs_to_spans():
    mask = np.array([False, True, True, False, True])
    t = np.arange(5) * 10
    assert runs_to_spans(mask, t, t + 10) == [(10, 30), (40, 50)]
