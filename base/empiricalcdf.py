from utils.types import NPFArray, f32
import numpy as np


class EmpiricalCDF:
    def __init__(self, cdf: NPFArray):
        cdf = np.asarray(cdf)
        x = cdf[:, 0].astype(f32)
        F = cdf[:, 1].astype(f32)

        # минимальная валидация
        assert np.isclose(F[-1], 1.0), "cdf[-1,1] must be 1"
        assert np.all(np.diff(F) >= 0), "CDF must be non-decreasing"

        self.x = x
        self.F = F

    def rvs(self, size: int) -> NPFArray:
        u = np.random.random(size).astype(f32)  # [0,1)
        F = self.F
        x = self.x

        idx = np.searchsorted(F, u, side="left")  # 0..len(F)
        idx = np.clip(
            idx, 1, len(F) - 1
        )  # зажали, чтобы были соседи (idx-1, idx)

        x1 = x[idx - 1]
        x2 = x[idx]
        y1 = F[idx - 1]
        y2 = F[idx]

        # защита от нулевой "ступеньки" (одинаковые y) или одинаковых x
        denom = y2 - y1
        safe = denom > 0

        out = np.empty(size, dtype=f32)
        out[~safe] = x1[~safe]  # если ступенька, возвращаем левый x

        t = (u[safe] - y1[safe]) / denom[safe]
        out[safe] = x1[safe] + t * (x2[safe] - x1[safe])
        return out
