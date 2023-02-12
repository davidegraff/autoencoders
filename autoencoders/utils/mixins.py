import inspect
import json
from os import PathLike
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch


class LoggingMixin:
    def _log_split(self, split: str, metrics: Mapping[str, Any], *args, **kwargs):
        self.log_dict({f"{split}/{k}": v for k, v in metrics.items()}, *args, **kwargs)


class SaveAndLoadMixin:
    def save(self, save_dir: PathLike):
        save_dir = Path(save_dir)
        p_state_dict = save_dir / "model.pt"
        p_config = save_dir / "model.json"

        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), p_state_dict)
        p_config.write_text(json.dumps(self.to_config(), indent=2))

    @classmethod
    def load(cls, save_dir: PathLike) -> object:
        save_dir = Path(save_dir)
        p_state_dict = save_dir / "model.pt"
        p_config = save_dir / "model.json"

        o = cls.from_config(json.loads(p_config.read_text()))
        o.load_state_dict(torch.load(p_state_dict))

        return o

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

        keys, values = self.get_params()
        defaults = [sig.parameters[k].default for k in keys]

        items = [(k, v) for k, v, d in zip(keys, values, defaults) if v != d]
        argspec = ", ".join(f"{k}={repr(v)}" for k, v in items)

        return f"{self.__class__.__name__}({argspec})"

    def get_params(self) -> Iterable[tuple[str, Any]]:
        return self.__dict__.values()
