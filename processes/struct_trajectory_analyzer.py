from abc import abstractmethod
from functools import cached_property
import pickle

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from joblib import Parallel, delayed
from numba import njit
from scipy import ndimage
from scipy.signal import convolve2d
from skimage import measure

from utils.types import NPFArray, NPIArray, NPBArray, f32

from base.trajectory import Trajectory
from processes.trajectory_analyzer import TrajectoryAnalyzer

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
def find_min_max_index(index: int, arr: NPBArray) -> Tuple[int, int]:
    N = len(arr)
    tpos_up = index
    tpos_down = index
    for v in arr[(index + 1) :]:
        if not v or tpos_up == N - 1:
            break
        tpos_up += 1
    for v in arr[:(index)][::-1]:
        if not v or tpos_down == 0:
            break
        tpos_down -= 1

    return tpos_down, tpos_up


@dataclass
class StructAnalizerParams:
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
    list_mu: npt.NDArray[f32] = field(
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

    @staticmethod
    def all_params():
        # good params  [154, 162, 186]
        params = {}
        i = 0
        for dp in [0, 10, 50]:  # 0 - bad
            for pv in [0.01, 0.1, 0.9]:  # 0.9 - bad
                for nu in [0.1, 0.5, 0.9]:
                    for tt in ['fBm', 'Bm']:  # fBm - best
                        for ks in [0, 1, 2, 3]:
                            for lm in [[0.5, 1, 1.5, 2, 2.5, 3]]:
                                params[i] = (dp, pv, nu, tt, ks)
                                i = i + 1
        return params

    @staticmethod
    def get_params(
        indexes: Optional[List[int]] = None,
        lmu: Optional[List[int]] = None,
    ) -> List['StructAnalizerParams']:
        list_mu = (
            np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]) if lmu is None else lmu
        )
        atparams = StructAnalizerParams.all_params()
        if indexes is not None:
            tparams = [atparams[i] for i in indexes]
        else:
            tparams = [v for _, v in atparams.items()]

        aparams = [
            StructAnalizerParams(tt, nu, dp, ks, list_mu, pv)
            for dp, pv, nu, tt, ks in tparams
        ]
        return aparams


class StructTrajectoryAnalizer(TrajectoryAnalyzer):
    def __init__(self, params: StructAnalizerParams):
        self.params = params

        self.kernel = (
            np.ones(
                shape=(2 * params.kernel_size + 1, 2 * params.kernel_size + 1),
                dtype=f32,
            )
            / (2 * params.kernel_size + 1) ** 2
        )

        self.diag_fill_list = list_vert_median[
            (self.params.diag_percentile, params.traj_type)
        ]

        with open(
            f"./list_threshold/nuc{int(self.params.nu * 100)}diag_perc={self.params.diag_percentile}.pkl",
            'rb',
        ) as handle:
            mat = pickle.load(handle)

        method = self.params.traj_type + "_3D"
        self.list_threshold = mat["list_threshold"][method]

    @staticmethod
    def name() -> str:
        return "struct"

    def run(self, trj: Trajectory) -> npt.NDArray[np.bool_]:
        points = trj.points_without_periodic
        sq_dist_matrix = self.compute_pairwise_sq_dist(points)

        def analyse(mu: float) -> Tuple[bool, npt.NDArray[np.bool_]]:
            return self.analyse_by_mu(
                sq_dist_matrix,
                self.params.p_value,
                self.params.nu,
                mu,
                self.list_threshold,
            )

        results = [analyse(mu) for mu in self.params.list_mu]

        list_trapped = np.zeros((trj.count_points,), dtype=np.bool_)
        for flag, result in results:
            if flag:
                list_trapped |= result
        return list_trapped[1:]

    def analyse_by_mu(
        self,
        sq_dist_matrix: NPFArray,
        p_value: float,
        nu: float,
        mu: float,
        list_threshold: NPIArray,
    ) -> Tuple[bool, NPBArray]:
        count_points = sq_dist_matrix.shape[0]
        ind1 = int(mu * 2) - 1
        ind2 = int((1.0 - p_value) * 100.0) - 1  # check this shift
        crit = list_threshold[ind1, ind2]
        if count_points < crit:
            return (False, np.zeros(shape=(count_points,), dtype=np.bool_))
        (
            list_vertical_m,
            list_diagonal_m,
            list_parallel_m,
        ) = self.RQA_block_measures(
            sq_dist_matrix, mu, self.diag_fill_list[int(2 * mu) - 1]
        )

        list_trapped_m = (
            list_vertical_m / (list_parallel_m + list_diagonal_m - 1) > nu
        )

        List_min_max = StructTrajectoryAnalizer.Make_list_min_max_index_equal(
            list_trapped_m
        )

        if len(List_min_max) == 0:
            return (True, list_trapped_m)
        length_traps = List_min_max[:, 1] - List_min_max[:, 0] + 1
        list_index_false_trap = np.where(length_traps <= crit)[0]

        if len(list_index_false_trap) == 0:
            return (True, list_trapped_m)

        for ind_false_trap in list_index_false_trap:
            i1, i2 = (
                List_min_max[ind_false_trap, 0],
                List_min_max[ind_false_trap, 1] + 1,
            )
            list_trapped_m[i1:i2] = False

        return (True, list_trapped_m)

    def compute_pairwise_sq_dist(self, points: NPFArray) -> NPFArray:
        # N x 3
        inc = np.asarray(points[1:] - points[:-1], dtype=f32)

        # std per axis (защита от деления на ноль)
        std = inc.std(axis=0, ddof=1).astype(f32)
        std = np.maximum(std, 1e-12).astype(f32)

        # нормализованные приращения -> "normal_inc" как у тебя
        tmp = np.zeros((points.shape[0], 3), dtype=f32)
        tmp[1:] = inc / std
        X = np.cumsum(tmp, axis=0, dtype=f32)  # shape (N,3)

        # Векторизованная матрица ||x_i - x_j||^2:
        # D2 = ||x||^2[:,None] + ||x||^2[None,:] - 2 X X^T
        x2 = np.sum(X * X, axis=1, dtype=f32)  # (N,)
        G = X @ X.T  # (N,N) float32
        sq_dist_matrix = (x2[:, None] + x2[None, :] - 2.0 * G).astype(f32)

        # из-за численных ошибок может появиться -1e-7: клипнем
        np.maximum(sq_dist_matrix, 0.0, out=sq_dist_matrix)
        return sq_dist_matrix

    def bin_laplacian_matrix_fast(
        self, sq_dist_matrix: NPFArray, mu: float
    ) -> NPBArray:
        if self.params.kernel_size > 0:
            inv = f32(-0.5) / f32(mu * mu)
            S = (sq_dist_matrix * inv).astype(f32, copy=False)
            np.exp(S, out=S)  # in-place exp
            # сглаживание ядром (быстрее convolve2d для маленьких ядер)
            size = 2 * self.params.kernel_size + 1
            S = ndimage.uniform_filter(S, size=size, mode="constant", cval=0.0)
            result: NPBArray = S > np.exp(-1)
            return result

        return sq_dist_matrix < 2 * mu * mu

    def RQA_block_measures(
        self, sq_dist_matrix: NPFArray, mu: float, diagonal_max: int
    ) -> Tuple[NPIArray, NPIArray, NPIArray]:
        N = sq_dist_matrix.shape[0]
        # Laplacian matrix (distance matrix)
        mat = self.bin_laplacian_matrix_fast(sq_dist_matrix, mu)

        # fill diagonals
        np.fill_diagonal(mat, True)  # first diagonal
        #  other diagonal depending on the radius size
        base = np.arange(N)
        for num in range(1, diagonal_max + 1):
            ind = base[: N - num]
            mat[ind, ind + num] = True
            mat[ind + num, ind] = True

        #  select matrix part connex to the diagonal line
        L = measure.label(mat, connectivity=2)
        matrix = L == 1
        #  fill holes
        matrix = ndimage.binary_fill_holes(matrix)

        #  init analysis
        list_diagonal = np.zeros(shape=(N,), dtype=np.int64)
        list_vertical = np.zeros(shape=(N,), dtype=np.int64)
        list_parallel = np.zeros(shape=(N,), dtype=np.int64)
        #  analysis
        for n in range(N):
            #  distribution of vertical lines
            if n < N // 2:
                list_diag_ind = np.arange(0, 2 * n + 1, dtype=np.int32)
            else:
                start = 2 * n - N + 1
                list_diag_ind = np.arange(start, N, dtype=np.int32)

            list_bin_vert = matrix[list_diag_ind, list_diag_ind[::-1]]
            if n < N // 2:
                index = n
            else:
                start = 2 * n - N + 1
                index = N - n - 1

            ud_bin, ud_ver = StructTrajectoryAnalizer.get_up_down(
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
    ) -> NPIArray:
        N_vec = len(list_trapped)
        index = 0
        List_min_max = []
        stop = 0
        while stop == 0:
            if list_trapped[index]:
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
