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
from typing import Optional


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

    @staticmethod
    def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
        return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

    def run(
        self,
        trj: Trajectory,
        trap_approx: Optional[npt.NDArray[np.int32]] = None
    ) -> None:
        if trap_approx is None:
            if not TrajectoryExtendedAnalizer.isclose(self.params.critical_probability, 0.):
                analizer = TrajectoryAnalizer(self.params)
                trap_approx = analizer.run(trj)
            else:
                trap_approx = (-1)*np.ones(shape=(trj.count_points,))

        print(" --- Matrix Algorithm finished")

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

        pore_probability /= pore_probability + throat_probability
        throat_probability /= pore_probability + throat_probability

        x = np.linspace(0, distances.max(), 1000)
        yp = gfitter.pdf(x, fd_pi_l)
        yt = wfitter.pdf(x, fd_lengthes)

        pore_mask = (
            pore_probability - throat_probability
        ) > self.params.critical_probability

        throat_mask = (
            throat_probability - pore_probability
        ) > self.params.critical_probability
        btw_mask = (
            np.abs(throat_probability - pore_probability)
            < self.params.critical_probability
        )
        ex_p_mask = distances < x[np.argmax(yp)]
        ex_t_mask = distances > x[np.argmax(yt)]

        result = np.zeros(shape=(distances.shape[0],), dtype=np.int32)
        result[btw_mask] = trap_approx[1:][btw_mask]
        result[pore_mask] = 1
        result[throat_mask] = 0
        result[ex_p_mask] = 1
        result[ex_t_mask] = 0
        trj.traps = result

        print(" --- Probability Algorithm finished")
        return result

    @staticmethod
    def distances(points: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        dxyz = points[:-1, :] - points[1:, :]
        return np.sqrt(np.sum(dxyz**2, axis=1))
