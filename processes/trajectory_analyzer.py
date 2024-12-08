import asyncio
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from joblib import Parallel, delayed
from numba import jit, njit
from scipy import ndimage
from scipy.signal import convolve2d
from skimage import measure

from base.trajectory import Trajectory

list_vert_median = {
    (0, 'Bm'): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    (0, 'fBm'): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    (5, 'Bm'): [1, 1, 1, 5, 8, 10, 13, 17, 21, 26, 31, 36],
    (5, 'fBm'): [1, 1, 4, 8, 15, 24, 35, 50, 68, 90, 113, 142],
    (10, 'Bm'): [1, 1, 3, 6, 9, 12, 16, 20, 25, 31, 37, 44],
    (10, 'fBm'): [1, 1, 5, 10, 18, 29, 44, 63, 86, 112, 142, 179],
    (50, 'Bm'): [1, 1, 6, 10, 16, 23, 30, 39, 49, 60, 72, 85],
    (50, 'fBm'): [1, 2, 9, 20, 37, 62, 93, 132, 181, 238, 302, 376],
}


@njit
def find_min_max_index(
    index: int, arr: npt.NDArray[np.int64]
) -> Tuple[int, int]:
    N = len(arr)
    tpos_up = index
    tpos_down = index
    for v in arr[(index + 1) :]:
        if v == 0 or tpos_up == N - 1:
            break
        tpos_up += 1
    for v in arr[:(index)][::-1]:
        if v == 0 or tpos_down == 0:
            break
        tpos_down -= 1

    return tpos_down, tpos_up


@dataclass
class AnalizerParams:
    """reference motion type for simulations
    can be `fBm` or `Bm`
    fBm - fractional Brownian Motion
    """

    traj_type: str = 'fBm'

    """ Critical value for the invariant (1 is perfect square)
        a parameter that affects the condition, which is a trap. The maximum value for the invariant.
        can be 0.1,0.3,0.5,0.75,0.9,1; (0.75 in the article)
    """
    nu: float = 0.1

    """ Percentile of block time from reference free motion to be used to fill diagonal lines
        can be 0,5,10 or 50 (10 in the article)
    """
    diag_percentile: int = 50

    """ + Kenrel size for convolution of distance matrix
    """
    kernel_size: int = 2

    """ 
        maximum range ([0.5,1,1.5,2,2.5,3])
        Used to normalize the distance matrix. (lambda in article)
        and by default affects the number of filled diagonals in the distance matrix
    """
    list_mu: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.array([1, 1.5, 2])
    )

    """ can be any percentile (minimum 0.01), (0.05 in the article)
        from 0 to 1
        affects the minimum size of the trap, in time, 
        if the size is larger than the critical one (crit),
        such a trap is not accepted by the algorithm
        In this case, the number of points in the trajectory 
        must be greater than the critical value (crit)
    """
    p_value: float = 0.01

    num_jobs: int = 1


class TrajectoryAnalizer:
    def __init__(self, params: AnalizerParams):
        self.params = params

        self.kernel = (
            np.ones(
                shape=(2 * params.kernel_size + 1, 2 * params.kernel_size + 1),
                dtype=np.float32,
            )
            / (2 * params.kernel_size + 1) ** 2
        )

        self.diag_fill_list = list_vert_median[
            (self.params.diag_percentile, params.traj_type)
        ]

    def run(self, trj: Trajectory) -> npt.NDArray[np.int32]:
        count_points = trj.count_points

        with open(
            f"./list_threshold/nuc{int(self.params.nu*100)}diag_perc={self.params.diag_percentile}.pkl",
            'rb',
        ) as handle:
            mat = pickle.load(handle)

        method = self.params.traj_type + "_3D"
        list_threshold = mat["list_threshold"][method]
        list_trapped = np.zeros(shape=(count_points,), dtype=np.bool_)

        points = trj.points_without_periodic

        def analyse(mu: float) -> Tuple[bool, npt.NDArray[np.bool_]]:
            return self.analyse_by_mu(
                points, self.params.p_value, self.params.nu, mu, list_threshold
            )

        if self.params.num_jobs != 1:
            results = Parallel(n_jobs=self.params.num_jobs)(
                delayed(analyse)(mu) for mu in self.params.list_mu
            )
        else:
            results = [analyse(mu) for mu in self.params.list_mu]

        list_trapped = np.zeros((trj.count_points,), dtype=np.bool_)
        for flag, result in results:
            if flag:
                list_trapped = np.logical_or(list_trapped, result > 0)
        return list_trapped

    def analyse_by_mu(
        self,
        points: npt.NDArray[np.float64],
        p_value: float,
        nu: float,
        mu: float,
        list_threshold: npt.NDArray[np.int32],
    ) -> Tuple[bool, npt.NDArray[np.bool_]]:
        count_points = points.shape[0]
        ind1 = int(mu * 2) - 1
        ind2 = int((1.0 - p_value) * 100.0) - 1  # check this shift
        crit = list_threshold[ind1, ind2]
        if count_points < crit:
            return (False, np.zeros(shape=(0, 0), dtype=np.bool_))
        (
            list_vertical_m,
            list_diagonal_m,
            list_parallel_m,
        ) = self.RQA_block_measures(
            points, mu, self.diag_fill_list[int(2 * mu) - 1]
        )

        list_trapped_m = (
            list_vertical_m / (list_parallel_m + list_diagonal_m - 1) > nu
        )

        List_min_max = TrajectoryAnalizer.Make_list_min_max_index_equal(
            list_trapped_m
        )

        if len(List_min_max) == 0:
            return (False, np.zeros(shape=(0, 0), dtype=np.bool_))
        length_traps = List_min_max[:, 1] - List_min_max[:, 0] + 1
        list_index_false_trap = np.where(length_traps <= crit)[0]

        if len(list_index_false_trap) == 0:
            return (False, np.zeros(shape=(0, 0), dtype=np.bool_))

        for ind_false_trap in list_index_false_trap:
            i1, i2 = (
                List_min_max[ind_false_trap, 0],
                List_min_max[ind_false_trap, 1] + 1,
            )
            list_trapped_m[i1:i2] = False

        return (True, list_trapped_m)

    def laplacian_matrix(
        self, points: npt.NDArray[np.float64], mu: float
    ) -> npt.NDArray[np.float64]:
        # UNTITLED3 Summary of this function goes here
        #    Detailed explanation goes here
        No = points.shape[0]
        inc = points[1:,] - points[:-1,]
        std = np.std(inc, 0, ddof=1)
        tmp = np.zeros(shape=(No, 3), dtype=np.float32)
        tmp[1:, :] = inc / std
        normal_inc = np.cumsum(tmp, 0, dtype=np.float32)
        S = np.zeros(shape=(No, No), dtype=np.float32)
        for i in range(No):
            ts = (normal_inc - normal_inc[i, :]) ** 2
            ss = np.sum(ts, 1, dtype=np.float32)
            S[:, i] = ss.copy()
            S[i, :] = ss.copy()

        S *= -0.5 / mu**2
        S = np.exp(S, dtype=np.float32)
        S: npt.NDArray[np.float32] = (
            convolve2d(S, self.kernel, 'same')
            if self.params.kernel_size > 0
            else S
        )
        return S

    def RQA_block_measures(
        self, points: npt.NDArray[np.float64], mu: float, diagonal_max: int
    ) -> Tuple[
        npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]
    ]:
        N = points.shape[0]
        # Laplacian matrix (distance matrix)
        S3 = self.laplacian_matrix(points, mu)
        mat = S3 > np.exp(-1)
        # fill diagonals
        np.fill_diagonal(mat, True)  # first diagonal
        #  other diagonal depending on the radius size
        for num in range(1, diagonal_max + 1):
            ind = np.array(range(N - num))
            mat[ind, ind + num] = True
            mat[ind + num, ind] = True

        #  select matrix part connex to the diagonal line
        L = measure.label(mat, connectivity=2)
        matrix = np.zeros(L.shape)
        matrix[L == 1] = 1
        #  fill holes
        matrix = ndimage.binary_fill_holes(matrix)

        #  init analysis
        list_diagonal = np.zeros(shape=(N,), dtype=np.int64)
        list_vertical = np.zeros(shape=(N,), dtype=np.int64)
        list_parallel = np.zeros(shape=(N,), dtype=np.int64)
        #  analysis
        for n in range(N):
            #  distribution of vertical lines
            if n < N / 2:
                list_diag_ind = np.array(range(2 * (n + 1) - 1))  # + 1
            else:
                list_diag_ind = np.array(range(2 * (n + 1) - N - 1, N))

            list_bin_vert = matrix[list_diag_ind, list_diag_ind[::-1]]
            index = np.where(
                np.logical_and(list_diag_ind == n, list_diag_ind[::-1] == n)
            )[0][0]

            ud_bin, ud_ver = TrajectoryAnalizer.get_up_down(
                index, n, list_bin_vert == 1, matrix[:, n] == 1
            )
            pos_down, pos_up = ud_bin
            pos_down_v, pos_up_v = ud_ver
            list_vertical[n] = pos_up_v - pos_down_v + 1

            #  diagonal
            list_diagonal[n] = pos_up - pos_down + 1
            #  parallel
            pos_down_par, pos_up_par = find_min_max_index(
                n - (list_diagonal[n] - 1) // 2,
                matrix.diagonal(list_diagonal[n] - 1),
            )
            list_parallel[n] = pos_up_par - pos_down_par + 1
        return list_vertical, list_diagonal, list_parallel

    @staticmethod
    def Make_list_min_max_index_equal(
        list_trapped: npt.NDArray[np.bool_],
    ) -> npt.NDArray[np.int64]:
        N_vec = len(list_trapped)
        index = 0
        List_min_max = []
        stop = 0
        while stop == 0:
            if list_trapped[index] == 1:
                pos_down, pos_up = find_min_max_index(index, list_trapped)
                List_min_max.append((pos_down, pos_up))
                index = pos_up + 1
            else:
                index = index + 1

            if index >= N_vec:
                stop = 1
        result = np.zeros(shape=(len(List_min_max), 2), dtype=np.int64)
        result[:, 0] = [a for a, _ in List_min_max]
        result[:, 1] = [a for _, a in List_min_max]
        return result

    def get_up_down(index, n, bin_arr, vert_arr):
        return find_min_max_index(index, bin_arr), find_min_max_index(
            n, vert_arr
        )
