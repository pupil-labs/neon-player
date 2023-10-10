"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
from plugin_timeline import RangeElementFrameIdx, Row

# TODO: add an observer to refresh the timeline when the selected calibration of a
#  gaze mapping changes. This is currently not easily possible, but will be easy with
#  attribute observers


class GazeMapperTimeline:
    def __init__(
        self,
        gaze_mapper_storage,
        gaze_mapper_controller,
    ):
        self.render_parent_timeline = None

        self._gaze_mapper_storage = gaze_mapper_storage
        self._gaze_mapper_controller = gaze_mapper_controller

        self._gaze_mapper_storage.add_observer("add", self._on_mapper_storage_changed)
        self._gaze_mapper_storage.add_observer(
            "delete", self._on_mapper_storage_changed
        )
        self._gaze_mapper_storage.add_observer(
            "rename", self._on_mapper_storage_changed
        )

        self._gaze_mapper_controller.add_observer(
            "set_mapping_range_from_current_trim_marks", self._on_mapper_ranges_changed
        )
        self._gaze_mapper_controller.add_observer(
            "publish_all_enabled_mappers", self._on_publish_enabled_mappers
        )

    def create_rows(self):
        rows = []
        for gaze_mapper in self._gaze_mapper_storage:
            alpha = 0.9 if gaze_mapper.activate_gaze else 0.4
            elements = [
                self._create_mapping_range(gaze_mapper, alpha),
            ]
            rows.append(Row(label=gaze_mapper.name, elements=elements))
        return rows

    def _create_mapping_range(self, gaze_mapper, alpha):
        from_idx, to_idx = gaze_mapper.mapping_index_range
        # TODO: find some final color scheme
        color = (
            [0.3, 0.5, 0.5, alpha]
            # [136 / 255, 92 / 255, 197 / 255, alpha*1.0]
            if not gaze_mapper.empty()
            else [0.66 * 0.7, 0.86 * 0.7, 0.46 * 0.7, alpha * 0.8]
        )
        return RangeElementFrameIdx(from_idx, to_idx, color_rgba=color, height=10)

    def _on_mapper_storage_changed(self, *args, **kwargs):
        self.render_parent_timeline()

    def _on_mapper_ranges_changed(self, _):
        self.render_parent_timeline()

    def _on_publish_enabled_mappers(self):
        """Triggered when activate_gaze changes and mapping tasks are complete"""
        self.render_parent_timeline()
