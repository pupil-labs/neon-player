import json
import logging
import shutil
import typing as T
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QFont, QIcon, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QApplication, QMessageBox
from qt_property_widgets.utilities import FilePath, action_params, property_params

from pupil_labs import neon_player
from pupil_labs.neon_player import Plugin, action
from pupil_labs.neon_player.plugins.gaze import apply_offset
from pupil_labs.neon_recording import NeonRecording

from .aoi_data import (
    AOIData,
    AOIImportError,
    aoi_colors,
    aoi_metrics,
    compute_gaze_hits,
    fixation_aoi_shares,
    gaze_frame_index,
    parse_aoi_dataframe,
    runs_to_spans,
    safe_column_name,
)

# A fixation counts as "on" an AOI if at least this share of its gaze
# samples are inside the AOI
FIXATION_SHARE_THRESHOLD = 0.5


class AOIPlugin(Plugin):
    """Areas of interest imported from a per-frame CSV.

    See aoi_data.py for the CSV format.
    """

    label = "AOI Import"

    def __init__(self) -> None:
        super().__init__()
        self.render_layer = 2

        self._show_aois = True
        self._show_labels = True
        self._fill_opacity = 25
        self._line_width = 3

        self.aois: AOIData | None = None
        self.colors: list[tuple[int, int, int]] = []
        self.gaze_frames: npt.NDArray[np.int64] | None = None
        self.gaze_hits: npt.NDArray[np.bool_] | None = None
        self._hits_offset: tuple[float, float] | None = None
        self._timeline_rows: list[str] = []

        # recompute hits a moment after the gaze offset changed
        self._rehit_timer = QTimer()
        self._rehit_timer.setSingleShot(True)
        self._rehit_timer.setInterval(1000)
        self._rehit_timer.timeout.connect(self._refresh_hits)

    # ------------------------------------------------------------------ files

    @property
    def _csv_cache(self) -> Path:
        return self.get_cache_path() / "aois.csv"

    @property
    def _npz_cache(self) -> Path:
        return self.get_cache_path() / "aois.npz"

    @property
    def _hits_cache(self) -> Path:
        return self.get_cache_path() / "gaze_hits.npz"

    def _save_aoi_cache(self, aois: AOIData) -> None:
        self.get_cache_path().mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self._npz_cache,
            names=np.array(json.dumps(aois.names)),
            aoi_index=aois.aoi_index,
            frame_index=aois.frame_index,
            vertices=aois.vertices,
            n_vertices=aois.n_vertices,
        )

    def _load_aoi_cache(self) -> AOIData | None:
        if not self._npz_cache.exists():
            return None
        try:
            with np.load(self._npz_cache) as d:
                return AOIData(
                    names=json.loads(str(d["names"])),
                    aoi_index=d["aoi_index"],
                    frame_index=d["frame_index"],
                    vertices=d["vertices"],
                    n_vertices=d["n_vertices"],
                )
        except Exception:
            logging.exception("Could not read AOI cache")
            return None

    # -------------------------------------------------------------- lifecycle

    def on_recording_loaded(self, recording: NeonRecording) -> None:
        aois = self._load_aoi_cache()
        if aois is None and self._csv_cache.exists():
            # cache from an older version or deleted: parse the CSV again
            try:
                aois = parse_aoi_dataframe(
                    pd.read_csv(self._csv_cache), recording.scene.time
                )
                self._save_aoi_cache(aois)
            except Exception:
                logging.exception("Could not load imported AOIs")

        if aois is not None:
            self._set_aois(aois)

    def on_disabled(self) -> None:
        self._remove_timeline_rows()
        self.aois = None
        self.gaze_hits = None

    def _set_aois(self, aois: AOIData) -> None:
        self.aois = aois
        self.colors = aoi_colors(len(aois.names))
        self.gaze_frames = gaze_frame_index(
            self.recording.gaze.time, self.recording.scene.time
        )
        self._refresh_hits(use_cache=True)

    def _gaze_offset(self) -> tuple[float, float]:
        gaze_plugin = Plugin.get_instance_by_name("GazeDataPlugin")
        if gaze_plugin is None:
            return (0.0, 0.0)
        return tuple(gaze_plugin.offset)

    def _offset_gaze_points(self) -> npt.NDArray[np.float64]:
        return apply_offset(
            self.recording, self.recording.gaze.point, self._gaze_offset()
        )

    def _refresh_hits(self, use_cache: bool = False) -> None:
        if self.aois is None or self.recording is None:
            return

        offset = self._gaze_offset()
        hits = None
        if use_cache and self._hits_cache.exists():
            try:
                with np.load(self._hits_cache) as d:
                    cached_offset = tuple(d["offset"].tolist())
                    if cached_offset == offset and d["hits"].shape == (
                        len(self.recording.gaze),
                        len(self.aois.names),
                    ):
                        hits = d["hits"]
            except Exception:
                logging.exception("Could not read AOI gaze cache")

        if hits is None:
            hits = compute_gaze_hits(
                self.aois, self._offset_gaze_points(), self.gaze_frames
            )
            try:
                self.get_cache_path().mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    self._hits_cache, hits=hits, offset=np.array(offset)
                )
            except Exception:
                logging.exception("Could not write AOI gaze cache")

        self.gaze_hits = hits
        self._hits_offset = offset
        self._update_timeline()
        if not self.headless:
            self.trigger_scene_update()

    # --------------------------------------------------------------- timeline

    def _remove_timeline_rows(self) -> None:
        if self.headless:
            return
        for row in self._timeline_rows:
            self.get_timeline().remove_timeline_plot(row)
        self._timeline_rows = []

    def _update_timeline(self) -> None:
        if self.headless or self.aois is None or self.gaze_hits is None:
            return

        self._remove_timeline_rows()

        scene_times = self.recording.scene.time
        scene_stops = np.append(scene_times[1:], scene_times[-1])
        gaze_times = self.recording.gaze.time
        gaze_stops = np.append(gaze_times[1:], gaze_times[-1])

        timeline = self.get_timeline()
        for i, name in enumerate(self.aois.names):
            row = f"AOI: {name}"
            visible = np.zeros(len(scene_times), bool)
            visible[self.aois.visible_frames(i)] = True

            timeline.add_timeline_broken_bar(
                row,
                runs_to_spans(visible, scene_times, scene_stops),
                color="#5a5a5a",
            )
            r, g, b = self.colors[i]
            timeline.add_timeline_broken_bar(
                row,
                runs_to_spans(self.gaze_hits[:, i], gaze_times, gaze_stops),
                color=f"#{r:02x}{g:02x}{b:02x}",
            )
            self._timeline_rows.append(row)

    # ----------------------------------------------------------------- render

    def render(self, painter: QPainter, time_in_recording: int) -> None:
        if not self._show_aois or self.aois is None or self.recording is None:
            return

        if self._gaze_offset() != self._hits_offset:
            if not self._rehit_timer.isActive():
                self._rehit_timer.start()

        frame_idx = self.get_scene_idx_for_time(time_in_recording)
        if frame_idx < 0:
            return

        shapes = self.aois.shapes_for_frame(frame_idx)
        if len(shapes) == 0:
            return

        looked_at = np.zeros(len(self.aois.names), bool)
        if self.gaze_hits is not None and self.gaze_frames is not None:
            g0 = np.searchsorted(self.gaze_frames, frame_idx, side="left")
            g1 = np.searchsorted(self.gaze_frames, frame_idx, side="right")
            if g1 > g0:
                looked_at = self.gaze_hits[g0:g1].any(axis=0)

        old_pen, old_brush, old_font = painter.pen(), painter.brush(), painter.font()
        font = QFont(painter.font())
        font.setPointSize(24)
        font.setBold(True)
        painter.setFont(font)

        labeled: set[int] = set()
        for s in shapes:
            aoi = int(self.aois.aoi_index[s])
            r, g, b = self.colors[aoi]
            polygon = QPolygonF([QPointF(x, y) for x, y in self.aois.polygon(s)])

            pen = QPen(QColor(r, g, b))
            pen.setWidth(self._line_width * (2 if looked_at[aoi] else 1))
            painter.setPen(pen)

            fill_alpha = int(255 * self._fill_opacity / 100)
            if looked_at[aoi]:
                fill_alpha = min(255, fill_alpha * 2 + 40)
            if fill_alpha > 0:
                painter.setBrush(QBrush(QColor(r, g, b, fill_alpha)))
            else:
                painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPolygon(polygon)

            if self._show_labels and aoi not in labeled:
                labeled.add(aoi)
                top = self.aois.polygon(s)
                x, y = float(top[:, 0].min()), float(top[:, 1].min())
                painter.setPen(QColor(r, g, b))
                painter.drawText(QPointF(x, y - 8), self.aois.names[aoi])

        painter.setPen(old_pen)
        painter.setBrush(old_brush)
        painter.setFont(old_font)

    # --------------------------------------------------------------- settings

    @property
    def show_aois(self) -> bool:
        return self._show_aois

    @show_aois.setter
    def show_aois(self, value: bool) -> None:
        self._show_aois = value
        self.changed.emit()

    @property
    def show_labels(self) -> bool:
        return self._show_labels

    @show_labels.setter
    def show_labels(self, value: bool) -> None:
        self._show_labels = value
        self.changed.emit()

    @property
    @property_params(min=0, max=100)
    def fill_opacity(self) -> int:
        return self._fill_opacity

    @fill_opacity.setter
    def fill_opacity(self, value: int) -> None:
        self._fill_opacity = int(value)
        self.changed.emit()

    @property
    @property_params(min=1, max=20)
    def line_width(self) -> int:
        return self._line_width

    @line_width.setter
    def line_width(self, value: int) -> None:
        self._line_width = int(value)
        self.changed.emit()

    # ---------------------------------------------------------------- actions

    @action
    @action_params(compact=True, icon=QIcon.fromTheme("document-open"))
    def import_csv(self, source: FilePath) -> None:
        """Import AOIs (one row per AOI per scene frame, pixel coordinates)."""
        if self.recording is None:
            return

        if not self.headless:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            df = pd.read_csv(source)
            aois = parse_aoi_dataframe(df, self.recording.scene.time)
        except (AOIImportError, pd.errors.ParserError, OSError) as e:
            self._report_error("AOI import failed", str(e))
            return
        finally:
            if not self.headless:
                QApplication.restoreOverrideCursor()

        self._remove_timeline_rows()
        self.get_cache_path().mkdir(parents=True, exist_ok=True)
        if Path(source).resolve() != self._csv_cache.resolve():
            shutil.copyfile(source, self._csv_cache)
        self._save_aoi_cache(aois)
        if self._hits_cache.exists():
            self._hits_cache.unlink()

        if not self.headless:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self._set_aois(aois)
        finally:
            if not self.headless:
                QApplication.restoreOverrideCursor()

        n_frames = len(np.unique(aois.frame_index))
        message = (
            f"Imported {len(aois.names)} AOI(s) on {n_frames} frame(s) "
            f"from {Path(source).name}: {', '.join(aois.names)}"
        )
        logging.info(message)
        if aois.warnings:
            self._report_error("AOI import warnings", "\n".join(aois.warnings))

    @action
    @action_params(compact=True, icon=QIcon.fromTheme("edit-delete"))
    def clear_aois(self) -> None:
        self._remove_timeline_rows()
        for path in (self._csv_cache, self._npz_cache, self._hits_cache):
            if path.exists():
                path.unlink()
        self.aois = None
        self.gaze_hits = None
        self.trigger_scene_update()

    def _report_error(self, title: str, message: str) -> None:
        logging.warning(f"{title}: {message}")
        if not self.headless:
            QMessageBox.warning(self.app.main_window, title, message)

    # ----------------------------------------------------------------- export

    @action
    @action_params(compact=True, icon=QIcon(str(neon_player.asset_path("export.svg"))))
    def export(self, destination: Path = Path()) -> None:
        if self.aois is None or self.recording is None:
            return

        if self._gaze_offset() != self._hits_offset or self.gaze_hits is None:
            self._refresh_hits()

        for name, df in self.prepare_exports(self.app.get_export_window()).items():
            out = Path(destination) / name
            df.to_csv(out, index=False)
            logging.info(f"Wrote {out}")

    def prepare_exports(
        self, export_window: tuple[int, int]
    ) -> dict[str, pd.DataFrame]:
        rec = self.recording
        aois = self.aois
        start, stop = export_window
        recording_id = rec.info["recording_id"]
        columns = [f"in aoi: {safe_column_name(n)}" for n in aois.names]

        # gaze
        gaze_times = rec.gaze.time
        gmask = (gaze_times >= start) & (gaze_times <= stop)
        points = self._offset_gaze_points()[gmask]
        hits = self.gaze_hits[gmask]
        gaze_df = pd.DataFrame({
            "recording id": recording_id,
            "timestamp [ns]": gaze_times[gmask],
            "scene frame index": self.gaze_frames[gmask],
            "gaze x [px]": points[:, 0],
            "gaze y [px]": points[:, 1],
        })
        names = np.array(aois.names, dtype=object)
        gaze_df["aoi"] = [";".join(names[row]) for row in hits]
        for i, col in enumerate(columns):
            gaze_df[col] = hits[:, i]

        # fixations (share of gaze samples inside each AOI)
        fix = rec.fixations
        fix_ids = np.arange(len(fix)) + 1
        fmask = (fix.stop_time > start) & (fix.start_time < stop)
        f_start = fix.start_time[fmask]
        f_stop = fix.stop_time[fmask]
        shares = fixation_aoi_shares(self.gaze_hits, gaze_times, f_start, f_stop)
        on_aoi = shares >= FIXATION_SHARE_THRESHOLD
        mean_points = apply_offset(rec, fix.mean_gaze_point[fmask], self._gaze_offset())

        best = shares.argmax(axis=1) if len(aois.names) else np.zeros(len(f_start), int)
        best_name = [
            aois.names[b] if len(aois.names) and on_aoi[k, b] else ""
            for k, b in enumerate(best)
        ]
        fix_df = pd.DataFrame({
            "recording id": recording_id,
            "fixation id": fix_ids[fmask],
            "start timestamp [ns]": f_start,
            "end timestamp [ns]": f_stop,
            "duration [ms]": (f_stop - f_start) / 1e6,
            "fixation x [px]": mean_points[:, 0],
            "fixation y [px]": mean_points[:, 1],
            "aoi": best_name,
        })
        for i, n in enumerate(aois.names):
            fix_df[f"on aoi: {safe_column_name(n)}"] = on_aoi[:, i]
            fix_df[f"share in aoi: {safe_column_name(n)}"] = shares[:, i]

        # metrics
        scene_times = rec.scene.time
        smask = (scene_times >= start) & (scene_times <= stop)
        frames_in_window = np.flatnonzero(smask)
        visible = [
            np.intersect1d(aois.visible_frames(i), frames_in_window)
            for i in range(len(aois.names))
        ]
        metrics = aoi_metrics(
            aois.names,
            hits,
            gaze_times[gmask],
            on_aoi,
            f_start,
            f_stop,
            visible,
            scene_times[smask],
            start,
        )
        metrics.insert(0, "recording id", recording_id)

        return {
            "aoi_gaze.csv": gaze_df,
            "aoi_fixations.csv": fix_df,
            "aoi_metrics.csv": metrics,
        }


__all__: T.Sequence[str] = ("AOIPlugin",)
