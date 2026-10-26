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


def instance() -> "NeonPlayerApp | None":
    # Defer the app import to avoid a circular import.
    from pupil_labs.neon_player.app import NeonPlayerApp  # noqa: PLC0415

    app = NeonPlayerApp.instance()
    return app if isinstance(app, NeonPlayerApp) else None


def require_instance() -> "NeonPlayerApp":
    app = instance()
    if app is None:
        raise RuntimeError("Neon Player application is not initialized")
    return app


class ActionDescriptor[Owner, **Params, Result](T.Protocol):
    def __set_name__(self, owner: type[Owner], name: str) -> None: ...

    def __get__(
        self, instance: Owner | None, owner: type[Owner] | None = None
    ) -> T.Callable[Params, Result]: ...


def action[Owner, **ActionParams, ActionResult](
    func: T.Callable[T.Concatenate[Owner, ActionParams], ActionResult],
) -> ActionDescriptor[Owner, ActionParams, ActionResult]:
    @functools.wraps(func)
    def wrapper(
        self: Owner, *args: ActionParams.args, **kwargs: ActionParams.kwargs
    ) -> ActionResult:
        return func(self, *args, **kwargs)

    descriptor: ActionDescriptor[Owner, ActionParams, ActionResult] = object_action(
        wrapper
    )
    return descriptor


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
