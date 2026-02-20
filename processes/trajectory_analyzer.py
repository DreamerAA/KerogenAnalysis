from abc import ABC, abstractmethod
from functools import cached_property
import numpy.typing as npt
import numpy as np

from base.trajectory import Trajectory


class TrajectoryAnalyzer(ABC):
    @abstractmethod
    def run(self, trj: Trajectory) -> npt.NDArray[np.bool_]:
        pass

    @cached_property
    @abstractmethod
    def name(self) -> str:
        pass
