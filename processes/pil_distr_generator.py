import sys
from os.path import realpath
from pathlib import Path
from typing import Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from joblib import Parallel, delayed
from matplotlib import cm
from scipy.stats import exponweib
from sklearn.metrics import pairwise_distances


class PiLDistrGenerator:
    def __init__(self, count_points: int = 10_000):
        self.count_points = count_points
        pass

    @staticmethod
    def upper_tri_masking(
        A: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        m = A.shape[0]
        r = np.arange(m)
        mask = r[:, None] < r
        return A[mask]

    def run(self, pore_radiuses):
        max_rad = pore_radiuses[-1]
        max_length = np.sqrt(3 * ((1.5 * max_rad) ** 2))

        cl = 100
        nx_len = np.linspace(0, max_length, cl)

        dl = nx_len[1] - nx_len[0]

        count_points = self.count_points
        xyz = np.zeros(shape=(count_points, 3), dtype=np.float32)
        xyz[:, 0] = np.random.uniform(-1, 1, size=count_points)
        xyz[:, 1] = np.random.uniform(-1, 1, size=count_points)
        xyz[:, 2] = np.random.uniform(-1, 1, size=count_points)
        dist = np.sqrt(np.sum(xyz**2, axis=1))
        xyz = xyz[dist < 1, :]

        def sim(num: int, m: int, radius: float) -> npt.NDArray[np.float32]:
            len_distr = np.zeros(shape=(cl - 1,), dtype=np.int32)
            xyz_scaled = xyz * radius
            distances = pairwise_distances(xyz_scaled, xyz_scaled, n_jobs=2)
            distances = PiLDistrGenerator.upper_tri_masking(distances)

            for i in range(1, cl):
                ll, rl = nx_len[(i - 1) : (i + 1)]
                len_distr[i - 1] = np.sum(
                    np.logical_and(distances > ll, distances <= rl)
                )
            print(f" --- Result of {num+1} from {m}")
            return len_distr / np.sum(len_distr * dl)

        np.sort(pore_radiuses)
        sample_rad = pore_radiuses[::30]
        pi_l_d = np.zeros(shape=(sample_rad.shape[0], cl - 1))

        pres = Parallel(n_jobs=8)(
            delayed(sim)(i, len(sample_rad), rad)
            for i, rad in enumerate(sample_rad)
        )

        for i, res in enumerate(pres):
            pi_l_d[i, :] = res

        pi_l = np.mean(pi_l_d, axis=0)
        new_l = nx_len[:-1] + (nx_len[1:] - nx_len[:-1]) * 0.5

        pi_l_save = np.zeros(shape=(pi_l.shape[0], 2), dtype=np.float32)
        pi_l_save[:, 1] = pi_l
        pi_l_save[:, 0] = new_l
        return pi_l_save
