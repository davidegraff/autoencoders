from __future__ import annotations

from abc import ABC, abstractmethod
from functools import wraps
import warnings


class Configurable(ABC):
    @abstractmethod
    def to_config(self) -> dict:
        pass

    @classmethod
    def from_config(cls, config: dict) -> Configurable:
        return cls(**config)


def warn_not_serializable(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        warnings.warn(f"{self.__class__.__name__} can not be serialized!")
        return method(self, *args, **kwargs)

    return wrapper