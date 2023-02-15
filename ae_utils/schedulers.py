from __future__ import annotations

from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike

from ae_utils.utils import Configurable

__all__ = ["Scheduler", "CyclicalScheduler"]


class Scheduler(Configurable):
    """Step the weight according to the input schedule. After the schedule is exhausted, will
    continue to return the final weight in the schedule.

    Parameters
    ----------
    schedule : ArrayLike
        the schedule
    i : int
        the iteration to start to from
    name : str
        the name of the weight term from this schedule.
    """

    def __init__(self, schedule: ArrayLike, name: str = "v", i: int = 0):
        self.schedule = np.array(schedule)
        self.name = name
        self.i = i

    def __len__(self) -> int:
        return len(self.schedule)

    @property
    def i(self) -> int:
        return self.__i

    @i.setter
    def i(self, i: int) -> int:
        self.__i = self.__i = max(0, i)

    @property
    def v(self) -> float:
        """The current weight"""
        i = min(self.i, len(self) - 1)
        return self.schedule[i]

    @property
    def v_min(self) -> float:
        return self.schedule[0]

    @property
    def v_max(self) -> float:
        return self.schedule[-1]

    def step(self) -> float:
        """Step the scheduler and return the new weight"""
        self.i += 1
        return self.v

    def reset(self):
        self.i = 0

    def to_config(self) -> dict:
        return {"schedule": self.schedule.tolist(), "i": self.i, "name": self.name}

    @classmethod
    def from_steps_and_weights(cls, steps_weights: Iterable[tuple[int, float]]) -> Scheduler:
        """Build a ManualKLScheduler from an implicit schedule defined in terms of the number of steps at a given weight

        Parameters
        ----------
        steps_and_weights : Iterable[tuple[int, float]]
            an iterable containing pairs of the number of steps and the given weight. I.e.,
            steps_weights = [(2, 0.1), (3, 0.2)] would correspond to the schedule
            [0.1, 0.1, 0.2, 0.2, 0.2]
        """
        schedule = np.concatenate([[v] * n for n, v in steps_weights])

        return cls(schedule)


class CyclicalScheduler(Scheduler):
    """A :class:`CyclicalScheduler` cycles back to `v_min` after exhausting the schedule rather than
    continuing to return a weight of `v_max`"""

    @property
    def v(self) -> float:
        return self.schedule[self.i % len(self)]
