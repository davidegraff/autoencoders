import inspect
import json
from os import PathLike
from pathlib import Path
from typing import Any, Collection, Mapping
from typing_extensions import Self

import torch


class LoggingMixin:
    def _log_split(self, split: str, metrics: Mapping[str, Any], *args, **kwargs):
        self.log_dict({f"{split}/{k}": v for k, v in metrics.items()}, *args, **kwargs)


class SaveAndLoadMixin:
    def save(self, save_dir: PathLike):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        p_state_dict = save_dir / "model.pt"
        p_config = save_dir / "model.json"

        torch.save(self.state_dict(), p_state_dict)
        p_config.write_text(json.dumps(self.to_config(), indent=2))

    @classmethod
    def load(cls, save_dir: PathLike) -> Self:
        save_dir = Path(save_dir)
        p_state_dict = save_dir / "model.pt"
        p_config = save_dir / "model.json"

        o = cls.from_config(json.loads(p_config.read_text()))
        o.load_state_dict(torch.load(p_state_dict))

        return o


class ReprMixin:
    def __repr__(self) -> str:
        items = self.get_params()
        
        if len(items) > 0:
            keys, values = zip(*items)
            sig = inspect.signature(self.__class__)
            defaults = [sig.parameters[k].default for k in keys]
            items = [(k, v) for k, v, d in zip(keys, values, defaults) if v != d]

        argspec = ", ".join(f"{k}={repr(v)}" for k, v in items)

        return f"{self.__class__.__name__}({argspec})"

    def get_params(self) -> Collection[tuple[str, Any]]:
        return self.__dict__.items()
