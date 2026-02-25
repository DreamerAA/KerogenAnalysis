from utils.types import NPFArray, f32
import numpy as np


class DiscreteCDF:
    """
    ps: shape (m, 2), columns: value, CDF(value)
    value может быть float, но обычно int-ступени.
    """

    def __init__(self, ps: NPFArray):
        ps = np.asarray(ps)
        v = ps[:, 0].astype(f32)
        F = ps[:, 1].astype(f32)

        assert np.isclose(F[-1], 1.0), "CDF last value must be 1"
        assert np.all(np.diff(F) >= 0), "CDF must be non-decreasing"

        self.v = v
        self.F = F

    def rvs(self, size: int) -> NPFArray:
        u = np.random.random(size).astype(f32)
        idx = np.searchsorted(self.F, u, side="left")
        idx = np.clip(idx, 0, len(self.v) - 1)
        return self.v[idx]
