from dataclasses import dataclass
import numpy.typing as npt
import numpy as np
from enum import Enum
from scipy.stats import weibull_min, exponweib, gamma


class DistributionType(Enum):
    EXPWEIB = 1
    GAMMA = 2


@dataclass
class FittingData:
    data: npt.NDArray[np.float32]
    params: npt.NDArray[np.float32]
    fit_type: DistributionType


class Fitter:
    def run(fdata: FittingData) -> None:
        pass


class WeibullFitter(Fitter):
    def run(self, fdata: FittingData) -> None:
        fdata.params = exponweib.fit(fdata.data)
        fdata.fit_type = self.ftype()

    @staticmethod
    def ftype() -> DistributionType:
        return DistributionType.EXPWEIB


class GammaFitter(Fitter):
    def run(self, fdata: FittingData) -> None:
        fdata.params = gamma.fit(fdata.data)
        fdata.fit_type = self.ftype()

    @staticmethod
    def ftype() -> DistributionType:
        return DistributionType.GAMMA


class GammaCurveFitter(Fitter):
    def run(self, fdata: FittingData) -> None:
        fdata.params = gamma.fit(fdata.data)
        fdata.fit_type = self.ftype()
        x = fdata.data[0, :]
        p = fdata.data[1, :]
        dx = x[1] - x[0]
        csum = np.cumsum(p * dx)
        count_points = 100_000
        unif_p = np.random.uniform(0, 1, size=count_points)
        ndata = [GammaCurveFitter.interp(sp, x, csum) for sp in unif_p]

        fdata.params = gamma.fit(ndata)
        fdata.fit_type = self.ftype()

    @staticmethod
    def interp(val, x, csum):
        i1 = x[np.where(val < csum)[0][-1]]
        i2 = i1 + 1
        return x[i1] + (x[i2] - x[i1])(val - csum[i1]) / (csum[i2] - csum[i1])

    @staticmethod
    def ftype() -> DistributionType:
        return DistributionType.GAMMA
