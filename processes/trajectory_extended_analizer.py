from trajectory_analyzer import TrajectoryAnalizer, AnalizerParams
import numpy as np
import numpy.typing as npt
from base.trajectory import Trajectory
from dataclasses import dataclass
from distribution_fitter import (
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
        points = trj.points_without_periodic
        distances = self.distances(points)

        fd_lenthes = FittingData(self.throat_lengthes, np.array(), None)
        fd_pi_l = FittingData(self.pi_l[1, :], np.array(), None)

        wfitter = WeibullFitter()
        wfitter.run(fd_lenthes)

        gfitter = GammaCurveFitter()
        gfitter.run(fd_pi_l)

        p, bb = np.histogram(self.throat_lengths, bins=50)
        xdel = bb[1] - bb[0]
        x = bb[:-1] + xdel * 0.5
        pn = p / np.sum(p * xdel)
        plt.figure()
        plt.hist(self.throat_lengths, bins=50)
        plt.plot(x, pn, label='Throat lengths - histogram data')
        plt.xlabel("Segemnt length (nm)")
        plt.title("PDF (Segment length inside pore) - Pi(L)")
        plt.legend()
        plt.show()

    @staticmethod
    def distances(points: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        dxyz = points[:-1, :] - points[1:, :]
        return np.sqrt(np.sum(dxyz**2, axis=1))
