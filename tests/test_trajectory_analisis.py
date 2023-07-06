from typing import Tuple

import numpy as np
import numpy.typing as npt
import pytest
from scipy import stats

from base.boundingbox import BoundingBox
from base.trajectory import Trajectory
from processes.trajectory_analyzer import SpectralAnalizer


class TestByRealCase:
    @pytest.fixture
    def ishape(self) -> Tuple[int, int, int]:
        return (100, 100, 100)

    @pytest.fixture
    def trajectory(self) -> Trajectory:
        def gen_point(
            xc: float,
            yc: float,
            zc: float,
            count: int,
            xstd: float = 1,
            ystd: float = 1,
            zstd: float = 1,
        ) -> npt.NDArray[np.float64]:
            x = stats.norm.rvs(xc, xstd, size=count)
            y = stats.norm.rvs(yc, ystd, size=count)
            z = stats.norm.rvs(zc, zstd, size=count)
            return np.vstack((x, y, z))

        points1 = gen_point(0.5, 0.5, 1.0, 10)
        points2 = gen_point(0.5, 2.5, 1.0, 10)
        points3 = gen_point(2.5, 2.5, 1.0, 10)
        points4 = gen_point(2.5, 0.5, 1.0, 10)
        fpoints = np.vstack((points1, points2, points3, points4))
        times = np.array(range(40), dtype=np.float64) * 100
        bbox = BoundingBox()
        bbox.update(np.array([0, 0, 0]))
        bbox.update(np.array([4, 4, 4]))
        trj = Trajectory(fpoints, times, bbox)
        return trj

    def test_trajectory_analizer_regression(self, traj: Trajectory) -> None:
        _ = SpectralAnalizer(traj)
        print(traj.traps)
        assert len(traj.traps) > 0
