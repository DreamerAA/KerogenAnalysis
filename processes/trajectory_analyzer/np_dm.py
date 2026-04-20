from dataclasses import dataclass
from functools import cached_property
from typing import Optional, Tuple
import numpy as np


from base.trajectory import Trajectory
from processes.distribution_fitter import (
    GammaFitter,
    WeibullFitter,
)
from processes.trajectory_analyzer.np import (
    NeymanPearsonAnalyzer,
)
from processes.trajectory_analyzer.trajectory_analyzer import TrajectoryAnalyzer

from utils.utils import pdistances
from utils.types import NPFArray, NPBArray, i32


@dataclass
class NeymanPearsonDistanceMatrixParams:
    error: float = 0.01


# mask_size = []


class NeymanPearsonDistanceMatrixAnalyzer(TrajectoryAnalyzer):
    def __init__(
        self,
        params: NeymanPearsonDistanceMatrixParams,
        pi_l_gf: GammaFitter,
        throat_lengthes_wf: WeibullFitter,
    ):
        self.params = params
        self.throat_lengthes_wf: WeibullFitter = throat_lengthes_wf
        self.pi_l_gf: GammaFitter = pi_l_gf

        self.transition_step_fitter: Optional[WeibullFitter] = None
        self.trapped_step_fitter: Optional[GammaFitter] = None

        self.threshold = NeymanPearsonAnalyzer.calculate_threshold(
            self.pi_l_gf, self.throat_lengthes_wf, self.params.error
        )
        self.reverse_threshold = NeymanPearsonAnalyzer.calculate_threshold(
            self.throat_lengthes_wf, self.pi_l_gf, self.params.error
        )

    @staticmethod
    def name() -> str:
        return "np_dm"

    def set_trap_approx(self, trap_approx: NPBArray):
        self.trap_approx = trap_approx

    def run(
        self,
        trj: Trajectory,
    ) -> NPBArray:
        likelihood = NeymanPearsonAnalyzer.analyze(
            trj,
            self.throat_lengthes_wf,
            self.pi_l_gf,
        )
        reverse_likelihood = NeymanPearsonAnalyzer.analyze(
            trj,
            self.pi_l_gf,
            self.throat_lengthes_wf,
        )

        result = likelihood < self.threshold
        reverse_result = ~(reverse_likelihood < self.reverse_threshold)
        mask = result != reverse_result
        # mask_size.append(mask.sum())
        result[mask] = self.trap_approx[mask]
        return result
