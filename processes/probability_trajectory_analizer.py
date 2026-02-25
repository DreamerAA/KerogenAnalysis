from abc import abstractmethod
from dataclasses import dataclass
from functools import cached_property
import pickle
from typing import Optional, Tuple
import os
import numpy as np
import numpy.typing as npt
import time
from base.trajectory import Trajectory
from processes.distribution_fitter import (
    GammaFitter,
    WeibullFitter,
)
from processes.trajectory_analyzer import TrajectoryAnalyzer

from utils.utils import kprint
from utils.types import NPFArray, NPIArray, NPBArray, f32


@dataclass
class ProbabilityAnalizerParams:
    critical_probability: float = 0.5


class ProbabilityTrajectoryAnalizer(TrajectoryAnalyzer):
    def __init__(
        self,
        params: ProbabilityAnalizerParams,
        pi_l_gf: GammaFitter,
        throat_lengthes_wf: WeibullFitter,
    ):
        self.params = params
        self.throat_lengthes_wf: WeibullFitter = throat_lengthes_wf
        self.pi_l_gf: GammaFitter = pi_l_gf
        self.trap_approx: Optional[NPIArray] = None

        self.transition_step_fitter: Optional[WeibullFitter] = None
        self.trapped_step_fitter: Optional[GammaFitter] = None

    @cached_property
    @abstractmethod
    def name(self) -> str:
        return "probability"

    def set_trap_approx(self, trap_approx: npt.NDArray[np.int32]):
        self.trap_approx = trap_approx

    def run(
        self,
        trj: Trajectory,
    ) -> NPBArray:
        _, probabilityies = self.analyze2(
            trj,
            self.throat_lengthes_wf,
            self.pi_l_gf,
            self.params.critical_probability,
        )
        result = probabilityies > 0.5
        return result

    @staticmethod
    def load_fitters(path: str) -> tuple[WeibullFitter, GammaFitter]:
        with open(path + "transition_step_fitter.json", "r") as f:
            transition_step_fitter = pickle.load(f)
        with open(path + "trapped_step_fitter.json", "r") as f:
            trapped_step_fitter = pickle.load(f)
        return transition_step_fitter, trapped_step_fitter

    @staticmethod
    def analyze(
        trj: Trajectory,
        transition_step_fitter: WeibullFitter,
        trapped_step_fitter: GammaFitter,
        critical_probability: float,
    ) -> Tuple[float, NPFArray]:
        points = trj.points_without_periodic
        distances = ProbabilityTrajectoryAnalizer.distances(points)

        L_T = trapped_step_fitter.pdf(distances).astype(np.float64)  # p(x|trap)
        L_C = transition_step_fitter.pdf(distances).astype(
            np.float64
        )  # p(x|not_trap)

        eps = 1e-300
        L_T = np.maximum(L_T, eps)
        L_C = np.maximum(L_C, eps)

        p_trap = 0.5
        prev = np.inf
        iterations = 0
        while np.abs(p_trap - prev) > critical_probability:
            prev = p_trap

            denom = p_trap * L_T + (1.0 - p_trap) * L_C
            denom = np.maximum(denom, eps)

            gamma = (p_trap * L_T) / denom  # P(trap | x)
            # EM update: ожидаемая доля
            p_trap = float(gamma.mean())
            # kprint(
            #     f"Iteration: {iterations}, count steps inside traps: {np.sum(gamma > 0.5)}"
            # )
            iterations += 1

        # print(
        #     " --- Analyze finished! "
        #     f"Trap Probability = {p_trap}, "
        #     f"Iterations = {iterations}, "
        #     f"Error = {np.abs(p_trap - prev)}"
        # )
        assert trapped_step_fitter is not None
        assert transition_step_fitter is not None

        return (p_trap, gamma)

    @staticmethod
    def analyze2(
        trj: Trajectory,
        transition_step_fitter: WeibullFitter,
        trapped_step_fitter: GammaFitter,
        critical_probability: float,
    ) -> Tuple[float, NPFArray]:
        points = trj.points_without_periodic
        distances = ProbabilityTrajectoryAnalizer.distances(points)

        L_T = trapped_step_fitter.pdf(distances).astype(f32)  # p(x|trap)
        L_C = transition_step_fitter.pdf(distances).astype(f32)  # p(x|not_trap)

        p_trap = 0.5
        gamma = np.zeros(shape=L_T.shape, dtype=f32)
        gamma[L_T > L_C] = 1.0

        return (p_trap, gamma)

    @staticmethod
    def distances(points: NPFArray) -> NPFArray:
        dxyz = points[:-1, :] - points[1:, :]
        result: NPFArray = np.sqrt(np.sum(dxyz**2, axis=1))
        return result
