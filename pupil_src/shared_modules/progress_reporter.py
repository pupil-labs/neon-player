"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

from rich.progress import Progress

class ProgressReporter(Progress):
    def __init__(self, queue, *args, **kwargs):
        self.queue = queue
        super().__init__(*args, **kwargs)

    def get_renderable(self):
        tasks = self.tasks
        progress = sum([task.percentage/100/len(tasks) for task in tasks])
        self.queue.put(float(progress))

        return super().get_renderable()
