from abc import abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import numpy as np
import numpy.typing as npt
import time
from base.trajectory import Trajectory
from processes.distribution_fitter import (
    GammaCurveFitter,
    WeibullFitter,
)
from processes.struct_trajectory_analyzer import (
    StructTrajectoryAnalizer,
    StructAnalizerParams,
)
from processes.probability_trajectory_analizer import (
    ProbabilityTrajectoryAnalizer,
    ProbabilityAnalizerParams,
)
from processes.trajectory_analyzer import TrajectoryAnalyzer
from utils.utils import NPFArray, NPIArray, NPBArray, f32


@dataclass
class HybridAnalizerParams:
    prob_params: ProbabilityAnalizerParams
    struct_params: StructAnalizerParams
    prob_diff: float = 0.1


class HybridTrajectoryAnalizer(TrajectoryAnalyzer):
    def __init__(
        self,
        params: HybridAnalizerParams,
        pi_l: NPFArray,
        throat_lengthes: NPFArray,
    ):
        self.params = params
        self.throat_lengthes = throat_lengthes
        self.pi_l = pi_l
        self.trap_approx: Optional[NPBArray] = None

    @cached_property
    @abstractmethod
    def name(self) -> str:
        return "hybrid"

    def set_trap_approx(self, trap_approx: NPBArray):
        self.trap_approx = trap_approx

    def get_trap_approx(self, trj: Optional[Trajectory] = None) -> NPBArray:
        if self.trap_approx is None:
            assert trj is not None
            analizer = StructTrajectoryAnalizer(self.params.struct_params)
            self.trap_approx = analizer.run(trj)
            print(" --- Matrix Algorithm finished")
        return self.trap_approx

    def set_prob_fitters(
        self,
        transition_step_fitter: WeibullFitter,
        trapped_step_fitter: GammaCurveFitter,
    ):
        self.transition_step_fitter = transition_step_fitter
        self.trapped_step_fitter = trapped_step_fitter

    def run(
        self,
        trj: Trajectory,
    ) -> NPBArray:
        trap_approx = self.get_trap_approx(trj)

        prob_analyzer = ProbabilityTrajectoryAnalizer.analyze(
            trj,
            self.throat_lengthes,
            self.pi_l,
            self.params.prob_params.critical_probability,
            transition_step_fitter=self.transition_step_fitter,
            trapped_step_fitter=self.trapped_step_fitter,
        )
        result, _, p_length_given_trap, p_length_given_not_trap, _, _ = (
            prob_analyzer
        )
        struct_mask = (
            np.abs(p_length_given_trap - p_length_given_not_trap)
            < self.params.prob_diff
        )
        result[struct_mask] = trap_approx[1:][struct_mask]

        print(" --- Probability Algorithm finished")
        assert not np.any(result == -1)
        return result

    @staticmethod
    def distances(points: NPFArray) -> NPFArray:
        dxyz = points[:-1, :] - points[1:, :]
        result: NPFArray = np.sqrt(np.sum(dxyz**2, axis=1)).astype(f32)
        return result
