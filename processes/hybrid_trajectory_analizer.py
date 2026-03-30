from abc import abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import numpy as np
import numpy.typing as npt
import time
from base.trajectory import Trajectory
from processes.distribution_fitter import (
    GammaFitter,
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
from utils.types import NPFArray, NPBArray, f32


@dataclass
class HybridAnalizerParams:
    prob_params: ProbabilityAnalizerParams
    struct_params: StructAnalizerParams
    prob_diff: float = 0.1


class HybridTrajectoryAnalizer(TrajectoryAnalyzer):
    def __init__(
        self,
        params: HybridAnalizerParams,
        pi_l_gf: GammaFitter,
        throat_lengthes_wf: WeibullFitter,
    ):
        self.params = params
        self.throat_lengthes_wf: WeibullFitter = throat_lengthes_wf
        self.pi_l_gf: GammaFitter = pi_l_gf
        self.trap_approx: Optional[NPBArray] = None

    @staticmethod
    def name() -> str:
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

    def run(
        self,
        trj: Trajectory,
    ) -> NPBArray:
        trap_approx = self.get_trap_approx(trj)

        _, probabilityies = ProbabilityTrajectoryAnalizer.analyze(
            trj,
            self.throat_lengthes_wf,
            self.pi_l_gf,
            self.params.prob_params.critical_probability,
        )
        result = probabilityies > 0.5
        struct_mask = np.abs(probabilityies - 0.5) < self.params.prob_diff
        result[struct_mask] = trap_approx[struct_mask]
        assert not np.any(result == -1)
        return result

    @staticmethod
    def distances(points: NPFArray) -> NPFArray:
        dxyz = points[:-1, :] - points[1:, :]
        result: NPFArray = np.sqrt(np.sum(dxyz**2, axis=1)).astype(f32)
        return result
