from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt
from scipy.stats import exponweib, gamma, weibull_min


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
    ftype: DistributionType = DistributionType.EXPWEIB

    def run(self, fdata: FittingData) -> None:
        fdata.params = exponweib.fit(fdata.data)
        fdata.fit_type = self.ftype

    def pdf(
        self, x: npt.NDArray[np.float32], fdata: FittingData
    ) -> npt.NDArray[np.float32]:
        assert fdata.fit_type == self.ftype
        return exponweib.pdf(x, *fdata.params)

    def cdf(
        self, x: npt.NDArray[np.float32], fdata: FittingData
    ) -> npt.NDArray[np.float32]:
        assert fdata.fit_type == self.ftype
        return exponweib.cdf(x, *fdata.params)


class GammaFitter(Fitter):
    ftype: DistributionType = DistributionType.GAMMA

    def run(self, fdata: FittingData) -> None:
        fdata.params = gamma.fit(fdata.data)
        fdata.fit_type = self.ftype


class GammaCurveFitter(Fitter):
    ftype: DistributionType = DistributionType.GAMMA

    def run(self, fdata: FittingData) -> None:
        fdata.params = gamma.fit(fdata.data)
        fdata.fit_type = self.ftype
        x = fdata.data[:, 0]
        p = fdata.data[:, 1]
        dx = x[1] - x[0]
        csum = np.cumsum(p * dx)
        count_points = 100_000
        unif_p = np.random.uniform(0, 1, size=count_points)
        ndata = np.array(
            [GammaCurveFitter.interp(sp, x, csum) for sp in unif_p],
            dtype=np.float32,
        )

        fdata.params = gamma.fit(ndata)
        fdata.fit_type = self.ftype

        # plt.plot(x, gamma.pdf(x,fdata.params[0],loc=fdata.params[1], scale=fdata.params[2]))

    @staticmethod
    def interp(val: float, x: float, csum: npt.NDArray[np.float32]) -> float:
        if val > csum[-1]:
            return x[-1]
        if val < csum[0]:
            return x[0]
        mask = np.where(val > csum)[0]
        i1 = mask[-1]
        i2 = i1 + 1
        return x[i1] + (x[i2] - x[i1]) * (val - csum[i1]) / (
            csum[i2] - csum[i1]
        )

    def pdf(
        self, x: npt.NDArray[np.float32], fdata: FittingData
    ) -> npt.NDArray[np.float32]:
        assert fdata.fit_type == self.ftype
        return gamma.pdf(
            x, fdata.params[0], loc=fdata.params[1], scale=fdata.params[2]
        )

    def cdf(
        self, x: npt.NDArray[np.float32], fdata: FittingData
    ) -> npt.NDArray[np.float32]:
        assert fdata.fit_type == self.ftype
        return gamma.cdf(
            x, fdata.params[0], loc=fdata.params[1], scale=fdata.params[2]
        )
