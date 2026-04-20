from dataclasses import dataclass
from typing import Optional
import numpy as np

from base.trajectory import Trajectory
from processes.distribution_fitter import (
    GammaFitter,
    WeibullFitter,
)
from processes.trajectory_analyzer.trajectory_analyzer import TrajectoryAnalyzer

from utils.types import NPBArray, i32
from processes.trajectory_analyzer.np import (
    NeymanPearsonAnalyzer,
    NeymanPearsonParams,
)
from processes.trajectory_analyzer.sib import (
    StructureInformedBayesParams,
    StructureInformedBayesAnalyzer,
)


@dataclass
class StructureInformedBayesNeynamPearsonParams(
    NeymanPearsonParams, StructureInformedBayesParams
):
    pass


class StructureInformedBayesNeynamPearsonAnalyzer(TrajectoryAnalyzer):
    def __init__(
        self,
        params: StructureInformedBayesNeynamPearsonParams,
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

    @staticmethod
    def name() -> str:
        return "sib_np"

    def run(
        self,
        trj: Trajectory,
    ) -> NPBArray:
        likelihood = NeymanPearsonAnalyzer.analyze(
            trj,
            self.throat_lengthes_wf,
            self.pi_l_gf,
        )
        result = (likelihood < self.threshold).astype(i32)
        _, probabilityies = StructureInformedBayesAnalyzer.analyze(
            trj,
            self.throat_lengthes_wf,
            self.pi_l_gf,
            self.params.critical_probability,
            p_trap=np.sum(result) / len(result),
        )
        result = probabilityies > 0.5
        return result
