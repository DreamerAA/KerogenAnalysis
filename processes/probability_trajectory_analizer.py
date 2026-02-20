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
    GammaCurveFitter,
    WeibullFitter,
)
from processes.trajectory_analyzer import TrajectoryAnalyzer

from utils.utils import NPFArray, NPIArray, NPBArray


@dataclass
class ProbabilityAnalizerParams:
    critical_probability: float = 0.5


class ProbabilityTrajectoryAnalizer(TrajectoryAnalyzer):
    def __init__(
        self,
        params: ProbabilityAnalizerParams,
        pi_l: NPFArray,
        throat_lengthes: NPFArray,
    ):
        self.params = params
        self.throat_lengthes = throat_lengthes
        self.pi_l = pi_l
        self.trap_approx: Optional[NPIArray] = None

        self.transition_step_fitter: Optional[WeibullFitter] = None
        self.trapped_step_fitter: Optional[GammaCurveFitter] = None

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
        result = self.analyze(
            trj,
            self.throat_lengthes,
            self.pi_l,
            self.params.critical_probability,
            self.transition_step_fitter,
            self.trapped_step_fitter,
        )
        if self.transition_step_fitter is None:
            self.transition_step_fitter = result[4]
        if self.trapped_step_fitter is None:
            self.trapped_step_fitter = result[5]
        return result[0]

    def set_prob_fitters(
        self,
        transition_step_fitter: WeibullFitter,
        trapped_step_fitter: GammaCurveFitter,
    ):
        self.transition_step_fitter = transition_step_fitter
        self.trapped_step_fitter = trapped_step_fitter

    def save_fitters(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path)

        with open(path + "transition_step_fitter.json", "w") as f:
            pickle.dump(self.transition_step_fitter, f)
        with open(path + "trapped_step_fitter.json", "w") as f:
            pickle.dump(self.trapped_step_fitter, f)

    def get_fitters(self) -> tuple[WeibullFitter, GammaCurveFitter]:
        return self.transition_step_fitter, self.trapped_step_fitter

    @staticmethod
    def load_fitters(path: str) -> tuple[WeibullFitter, GammaCurveFitter]:
        with open(path + "transition_step_fitter.json", "r") as f:
            transition_step_fitter = pickle.load(f)
        with open(path + "trapped_step_fitter.json", "r") as f:
            trapped_step_fitter = pickle.load(f)
        return transition_step_fitter, trapped_step_fitter

    @staticmethod
    def analyze(
        trj: Trajectory,
        throat_lengthes: NPFArray,
        pi_l: NPFArray,
        critical_probability: float,
        transition_step_fitter: Optional[WeibullFitter] = None,
        trapped_step_fitter: Optional[GammaCurveFitter] = None,
    ) -> Tuple[
        NPBArray, float, NPFArray, NPFArray, WeibullFitter, GammaCurveFitter
    ]:
        if transition_step_fitter is None:
            transition_step_fitter = WeibullFitter()
            transition_step_fitter.fit(throat_lengthes)

        if trapped_step_fitter is None:
            trapped_step_fitter = GammaCurveFitter()
            trapped_step_fitter.fit(pi_l)

        points = trj.points_without_periodic
        distances = ProbabilityTrajectoryAnalizer.distances(points)
        p_length_given_trap = trapped_step_fitter.pdf(distances)
        p_length_given_not_trap = transition_step_fitter.pdf(distances)

        p_length_given_trap /= p_length_given_trap + p_length_given_not_trap
        p_length_given_not_trap /= p_length_given_trap + p_length_given_not_trap

        # x = np.linspace(0, distances.max(), 1000)
        # yp = trapped_step_fitter.pdf(x, fd_pi_l)
        # yt = transition_step_wfitter.pdf(x, fd_lengthes)

        p_trap = 0.5  # A priori probability of a trap
        prev_p_trap = 1.0
        result = np.zeros(shape=(len(p_length_given_trap),), dtype=np.bool_)
        iterations = 0
        while np.abs(p_trap - prev_p_trap) > critical_probability:
            full_probability = (
                p_length_given_trap * p_trap
                + p_length_given_not_trap * (1 - p_trap)
            )
            p_trap_given_length = (
                p_length_given_trap * p_trap / full_probability
            )
            assert not np.any(p_trap_given_length > 1)

            result[:] = 0
            result[p_trap_given_length > 0.5] = 1

            prev_p_trap = p_trap
            p_trap = np.sum(result) / len(result)
            iterations += 1

        print(
            " --- Probability Algorithm finished! "
            f"Trap Probability = {p_trap}, "
            f"Iterations = {iterations}, "
            f"Error = {np.abs(p_trap - prev_p_trap)}"
        )
        assert trapped_step_fitter is not None
        assert transition_step_fitter is not None
        return (
            result,
            p_trap,
            p_length_given_trap,
            p_length_given_not_trap,
            transition_step_fitter,
            trapped_step_fitter,
        )

    @staticmethod
    def distances(points: NPFArray) -> NPFArray:
        dxyz = points[:-1, :] - points[1:, :]
        result: NPFArray = np.sqrt(np.sum(dxyz**2, axis=1))
        return result
