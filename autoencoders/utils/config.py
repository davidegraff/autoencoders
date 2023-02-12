from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Configurable(Protocol):
    def to_config(self) -> dict:
        pass

    @classmethod
    def from_config(cls, config: dict) -> Configurable:
        pass