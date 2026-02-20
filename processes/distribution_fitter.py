from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy.stats import exponweib, gamma, weibull_min

NPFArray = npt.NDArray[np.float64]


class DistributionType(Enum):
    EXPWEIB = 1
    GAMMA = 2


class WeibullFitter:
    ftype: DistributionType = DistributionType.EXPWEIB

    def __init__(self):
        self.params: Optional[NPFArray] = None

    def fit(self, data: NPFArray):
        assert self.params is None
        self.params = exponweib.fit(data, floc=0)

    def pdf(self, x: NPFArray) -> NPFArray:
        assert self.params is not None
        return exponweib.pdf(x, *self.params)

    def cdf(self, x: NPFArray) -> NPFArray:
        assert self.params is not None
        return exponweib.cdf(x, *self.params)


class GammaCurveFitter:
    ftype: DistributionType = DistributionType.GAMMA

    def __init__(self):
        self.params: Optional[NPFArray] = None

    def fit(self, data: NPFArray):
        assert self.params is None
        self.params = gamma.fit(data)
        # x = data[:, 0]
        # p = data[:, 1]
        # dx = x[1] - x[0]
        # csum = np.cumsum(p * dx)
        # count_points = 100_000
        # unif_p = np.random.uniform(0, 1, size=count_points)
        # ndata = np.array(
        #     [GammaCurveFitter.interp(sp, x, csum) for sp in unif_p],
        #     dtype=np.float32,
        # )

        # params = gamma.fit(ndata)
        # plt.plot(x, gamma.pdf(x,fdata.params[0],loc=fdata.params[1], scale=fdata.params[2]))

    @staticmethod
    def interp(val: float, x: float, csum: NPFArray) -> float:
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

    def pdf(self, x: NPFArray) -> NPFArray:
        assert self.params is not None
        return gamma.pdf(x, *self.params)

    def cdf(self, x: NPFArray) -> NPFArray:
        assert self.params is not None
        return gamma.cdf(x, *self.params)
