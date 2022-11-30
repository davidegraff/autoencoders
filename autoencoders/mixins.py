import inspect
from typing import Iterable


class RegistryMixin:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "alias"):
            cls.registry[cls.alias] = cls
        if hasattr(cls, "aliases"):
            cls.registry.update({alias: cls for alias in cls.aliases})


class ReprMixin:
    def __repr__(self) -> str:
        sig = inspect.signature(self.__init__)

        keys = sig.parameters.keys()
        values = self.get_params()
        defaults = [p.default for p in sig.parameters.values()]

        items = [(k, v) for k, v, d in zip(keys, values, defaults) if v != d]
        argspec = ", ".join(f"{k}={repr(v)}" for k, v in items)

        return f"{self.__class__.__name__}({argspec})"

    def get_params(self) -> Iterable:
        return self.__dict__.values()
