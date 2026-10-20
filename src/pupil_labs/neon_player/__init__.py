import functools
import sys
import typing as T
from importlib import resources
from pathlib import Path

from qt_property_widgets.utilities import action as object_action

from pupil_labs.neon_player.job_manager import ProgressUpdate
from pupil_labs.neon_player.plugins import GlobalPluginProperties, Plugin

if T.TYPE_CHECKING:
    from pupil_labs.neon_player.app import NeonPlayerApp

LOG_FORMAT_STRING = (
    "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)


def instance() -> "NeonPlayerApp":
    """Return the running Neon Player application instance.

    Within the plugin implementation, it is recommended to use the
    [`Plugin.app`][pupil_labs.neon_player.Plugin.app]
    field to access the main application."""
    from pupil_labs.neon_player.app import NeonPlayerApp

    return NeonPlayerApp.instance()


def action(func: T.Callable) -> T.Any:
    """Mark a plugin method as an action that users can run.

    Apply this decorator to methods that should appear as plugin actions. For
    each action, a corresponding button is added to the plugin settings.

    Args:
        func: Plugin method to make available as an action.
    """
    @functools.wraps(func)
    def wrapper(*args: T.Any, **kwargs: T.Any) -> T.Any:
        return func(*args, **kwargs)

    return object_action(wrapper)


def asset_path(resource: str) -> Path:
    with resources.as_file(
        resources.files(__package__).joinpath("assets")
    ) as assets_path:
        return assets_path / resource


def is_frozen() -> bool:
    return getattr(sys, "frozen", False) or "__compiled__" in globals()


__all__ = [
    "BGWorker",
    "GlobalPluginProperties",
    "Plugin",
    "ProgressUpdate",
    "action",
    "instance",
]
