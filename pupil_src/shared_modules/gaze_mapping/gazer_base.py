"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import abc
import collections
import logging
import typing as T

from plugin import Plugin

logger = logging.getLogger(__name__)


class Model(abc.ABC):
    @property
    @abc.abstractmethod
    def is_fitted(self) -> bool:
        pass

    @abc.abstractmethod
    def fit(self, X, y):
        """Fit model with input `X` to targets `y`

        Arguments:
            X {array-like} -- of shape (n_samples, n_features)
            y {array-like} -- of shape (n_samples, n_targets)
        """
        pass

    @abc.abstractmethod
    def predict(self, X):
        """Predict values based on input `X`

        Arguments:
            X {array-like} -- of shape (n_samples, n_features)

        Returns:
            array-like -- of shape (n_samples, n_outputs)
        """
        pass

    @abc.abstractmethod
    def set_params(self, **params):
        pass

    @abc.abstractmethod
    def get_params(self):
        pass


class GazerBase(abc.ABC, Plugin):
    label: str = ...  # Subclasses should set this to a meaningful name
    uniqueness = "by_base_class"

    @classmethod
    def _gazer_description_text(cls) -> str:
        return ""

    @classmethod
    def should_register(cls) -> bool:
        return True

    @staticmethod
    def registered_gazer_classes() -> T.List[T.Type["GazerBase"]]:
        return list(GazerBase.__registered_gazer_plugins.values())

    __registered_gazer_plugins = collections.OrderedDict()


    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)

        if not cls.should_register():
            # If the gazer class explicitly opted-out of being registered, skip registration
            return

        store = GazerBase.__registered_gazer_plugins

        assert isinstance(
            cls.label, str
        ), f'Gazer plugin subclass {cls.__name__} must overwrite string class property "label"'

        assert (
            cls.label not in store.keys()
        ), f'Gazer plugin already exists for label "{cls.label}"'

        store[cls.label] = cls

    # ------------ Base Implementation

    # -- Plugin Functions

    @classmethod
    def base_class(cls):
        # This ensures that all gazer plugins return the same base class,
        # even gazers that subclass concrete gazer implementations.
        return GazerBase

    def __init__(
        self, g_pool, *, params=None, raise_calibration_error=False
    ):
        super().__init__(g_pool)
        if params is None:
            raise ValueError("Requires `params`")

        self.set_params(params)

        if self.alive:
            # Used by pupil_data_relay for gaze mapping.
            g_pool.active_gaze_mapping_plugin = self

    def get_init_dict(self):
        return {"params": self.get_params()}
