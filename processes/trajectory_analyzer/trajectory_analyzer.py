from abc import ABC, abstractmethod
import numpy.typing as npt
import numpy as np

from base.trajectory import Trajectory


class TrajectoryAnalyzer(ABC):
    @abstractmethod
    def run(self, trj: Trajectory) -> npt.NDArray[np.bool_]:
        pass
