from processes.trajectory_analyzer import TrajectoryAnalizer, AnalizerParams
import numpy as np
import numpy.typing as npt
from base.trajectory import Trajectory
from dataclasses import dataclass
from processes.distribution_fitter import (
    WeibullFitter,
    FittingData,
    GammaFitter,
    GammaCurveFitter,
)
import matplotlib.pyplot as plt


@dataclass
class ExtendedParams(AnalizerParams):
    critical_probability: float = 0.5


class TrajectoryExtendedAnalizer:
    def __init__(
        self,
        params: ExtendedParams,
        pi_l: npt.NDArray[np.float32],
        throat_lengthes: npt.NDArray[np.float32],
    ):
        self.params = params
        self.throat_lengthes = throat_lengthes
        self.pi_l = pi_l

    def run(self, trj: Trajectory) -> None:
        analizer = TrajectoryAnalizer(self.params)
        traps = analizer.run(trj)

        fd_lengthes = FittingData(self.throat_lengthes, np.array([]), None)
        wfitter = WeibullFitter()
        wfitter.run(fd_lengthes)

        fd_pi_l = FittingData(self.pi_l, np.array([]), None)
        gfitter = GammaCurveFitter()
        gfitter.run(fd_pi_l)

        points = trj.points_without_periodic
        distances = self.distances(points)
        pore_probability = gfitter.pdf(distances, fd_pi_l)
        throat_probability = wfitter.pdf(distances, fd_lengthes)

        mask = (
            pore_probability - throat_probability
        ) > self.params.critical_probability

        result = np.zeros(shape=(distances.shape[0],), dtype=np.int32)
        result[mask] = 1
        result[~mask] = traps[1:][~mask]
        trj.traps = result
        print(" --- Finish!")

    @staticmethod
    def distances(points: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        dxyz = points[:-1, :] - points[1:, :]
        return np.sqrt(np.sum(dxyz**2, axis=1))
