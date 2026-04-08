from dataclasses import dataclass
from functools import cached_property
from typing import Optional, Tuple
import numpy as np


from base.trajectory import Trajectory
from processes.distribution_fitter import (
    GammaFitter,
    WeibullFitter,
)
from processes.trajectory_analyzer import TrajectoryAnalyzer

from utils.utils import pdistances
from utils.types import NPFArray, NPBArray, i32


@dataclass
class NeumannPearsonAnalizerParams:
    error: float = 0.01


class NeumannPearsonTrajectoryAnalizer(TrajectoryAnalyzer):
    def __init__(
        self,
        params: NeumannPearsonAnalizerParams,
        pi_l_gf: GammaFitter,
        throat_lengthes_wf: WeibullFitter,
    ):
        self.params = params
        self.throat_lengthes_wf: WeibullFitter = throat_lengthes_wf
        self.pi_l_gf: GammaFitter = pi_l_gf

        self.transition_step_fitter: Optional[WeibullFitter] = None
        self.trapped_step_fitter: Optional[GammaFitter] = None

        self.threshold = NeumannPearsonTrajectoryAnalizer.calculate_threshold(
            self.pi_l_gf, self.throat_lengthes_wf, self.params.error
        )

    @staticmethod
    def name() -> str:
        return "neumann_pearson"

    def run(
        self,
        trj: Trajectory,
    ) -> NPBArray:
        likelihood = self.analyze(
            trj,
            self.throat_lengthes_wf,
            self.pi_l_gf,
        )
        result = likelihood < self.threshold
        return result

    @staticmethod
    def analyze(
        trj: Trajectory,
        transition_step_fitter: WeibullFitter | GammaFitter,
        trapped_step_fitter: GammaFitter | WeibullFitter,
    ) -> NPFArray:
        points = trj.points_without_periodic
        distances = pdistances(points)

        L_T = trapped_step_fitter.pdf(distances).astype(
            np.float64
        )  # p(x|trap) H_1
        L_C = transition_step_fitter.pdf(distances).astype(
            np.float64
        )  # p(x|~trap) H_2

        return np.divide(L_C, L_T).astype(np.float64)

    @staticmethod
    def calculate_threshold(f_distr, g_distr, epsilon) -> float:
        x = np.linspace(0, 1, 1_000_000)
        f = f_distr.pdf(x)
        g = g_distr.pdf(x)
        # likelihood ratio
        likelihood = np.divide(g, f, out=np.zeros_like(g), where=f > 0)

        # нормируем веса (аппроксимация интеграла)
        weights = f / np.sum(f)

        # сортируем по убыванию l
        idx = np.argsort(likelihood)[::-1]
        l_sorted = likelihood[idx]
        w_sorted = weights[idx]

        # накопленная масса
        cumulative = np.cumsum(w_sorted)

        # находим порог
        mask = cumulative >= epsilon
        threshold = l_sorted[mask][0]
        return float(threshold)
