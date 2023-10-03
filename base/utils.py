import numpy as np
import numpy.typing as npt
from scipy import stats


def point_generation() -> npt.NDArray[np.float32]:
    def gen_point(
        xc: float,
        yc: float,
        zc: float,
        count: int,
        xstd: float = 0.2,
        ystd: float = 0.2,
        zstd: float = 0.2,
    ) -> npt.NDArray[np.float32]:
        points = np.zeros(shape=(count, 3), dtype=np.float32)
        points[:, 0] = stats.norm.rvs(xc, xstd, size=count)
        points[:, 1] = stats.norm.rvs(yc, ystd, size=count)
        points[:, 2] = stats.norm.rvs(zc, zstd, size=count)
        return points
    x1, y1 = 0.5, 0.5
    x2, y2 = 3.5, 3.5
    count = 200
    points1 = gen_point(x1, y1, 1.0, count)
    p12 = np.array([x1, y1 + (y2 - y1) / 4, 1.0, x1, y1 + (y2 - y1) * 2 / 4, 1.0, x1, y1 + (y2 - y1) * 3 / 4, 1.0], dtype=np.float32).reshape(3, 3)
    points2 = gen_point(x1, y2, 1.5, count)
    p23 = np.array([x1 + (x2 - x1) / 4, y2, 1.0, x1 + (x2 - x1) * 2 / 4, y2, 1.0, x1 + (x2 - x1) * 3 / 4, y2, 1.0], dtype=np.float32).reshape(3, 3)
    points3 = gen_point(x2, y2, 2.0, count)
    p34 = np.array([x2, y1 + (y2 - y1) * 3 / 4, 1.0, x2, y1 + (y2 - y1) * 2 / 4, 1.0, x2, y1 + (y2 - y1) / 4, 1.0], dtype=np.float32).reshape(3, 3)
    points4 = gen_point(x2, y1, 2.5, count)
    return np.vstack((points1, p12, points2, p23, points3, p34, points4))
