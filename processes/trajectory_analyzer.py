from base.trajectory import Trajectory
import numpy as np
from sklearn.cluster import DBSCAN, AffinityPropagation, KMeans, MeanShift, SpectralClustering, AgglomerativeClustering, BisectingKMeans
from sklearn.mixture import GaussianMixture
from skimage import measure
import numpy.typing as npt
import matplotlib.pyplot as plt
# %matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_score
import scipy.io
from scipy.signal import convolve2d
from scipy import ndimage
import mat73
from typing import Tuple

list_mu = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]

list_vert_median = {(0, 'Bm'): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    (0, 'fBm'): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    (5, 'Bm'): [1, 1, 1, 5, 8, 10, 13, 17, 21, 26, 31, 36],
                    (5, 'fBm'): [1, 1, 4, 8, 15, 24, 35, 50, 68, 90, 113, 142],
                    (10, 'Bm'): [1, 1, 3, 6, 9, 12, 16, 20, 25, 31, 37, 44],
                    (10, 'fBm'): [1, 1, 5, 10, 18, 29, 44, 63, 86, 112, 142, 179],
                    (50, 'Bm'): [1, 1, 6, 10, 16, 23, 30, 39, 49, 60, 72, 85],
                    (50, 'fBm'): [1, 2, 9, 20, 37, 62, 93, 132, 181, 238, 302, 376]}


class SpectralAnalizer:
    def __init__(self, trj: Trajectory):
        self.p_val_traj_type = 'fBm'
        self.diag_percentile = 50  # can be 0,5,10 or 50 (10 in the article)
        self.nu = 0.1  # can be 0.1,0.3,0.5,0.75,0.9,1; (0.75 in the article)
        # can be any percentile (minimum 0.01), (0.05 in the article)
        self.p_value = 0.01
        # maximum range ([0.5,1,1.5,2,2.5,3])
        self.list_mu = np.array([1, 1.5, 2])
        self.T_mean = 2
        self.sig_noise = 0
        self.diag_fill_list = list_vert_median[(
            self.diag_percentile, self.p_val_traj_type)]
        count_points = trj.count_points()

        method = self.p_val_traj_type + "_3D"
        mat = mat73.loadmat(
            f"./list_threshold/nuc{int(self.nu*100)}diag_perc={self.diag_percentile}.mat")
        list_threshold = mat["list_threshold"][method]
        list_trapped = np.zeros(shape=(count_points,), dtype=np.bool_)

        for mu in self.list_mu:
            ind1 = int(mu*2) - 1
            ind2 = int((1. - self.p_value)*100.) - 1
            crit = list_threshold[ind1, ind2]
            if count_points > crit:
                list_vertical_m, list_diagonal_m, list_parallel_m = self.RQA_block_measures(
                    trj, mu)
                list_trapped_m = list_vertical_m / \
                    (list_parallel_m+list_diagonal_m-1) > self.nu
                List_min_max = self.Make_list_min_max_index_equal(
                    list_trapped_m)
                if len(List_min_max) != 0:
                    list_index_false_trap = np.where(
                        (List_min_max[:, 1]-List_min_max[:, 0]+1) <= crit)[0]
                    if len(list_index_false_trap) != 0:
                        for ind_false_trap in list_index_false_trap:
                            list_trapped_m[List_min_max[ind_false_trap, 0]:(
                                List_min_max[ind_false_trap, 1] + 1)] = 0

                list_trapped = np.logical_or(list_trapped, list_trapped_m > 0)
        trj.traps = list_trapped

    def laplacian_matrix(self, trj: Trajectory, T_mean: int, mu: float) -> npt.NDArray[np.float64]:
        # UNTITLED3 Summary of this function goes here
        #    Detailed explanation goes here
        No = trj.count_points()
        points = trj.points_without_periodic()
        inc = points[1:,] - points[:-1,]
        std = np.std(inc, 0, ddof=1)
        tmp = np.zeros(shape=(No, 3), dtype=np.float64)
        tmp[1:, :] = inc/std
        normal_inc = np.cumsum(tmp, 0)
        S = np.zeros(shape=(No, No))
        for i in range(No):
            # ((Xo-Xo(i,1)).^2+(Yo-Yo(i,1)).^2).^0.5;
            ts = (normal_inc-normal_inc[i, :])**2
            S[:, i] = np.sum(ts, 1)
            S[i, :] = S[:, i]

        S2b = np.exp(-0.5*S/mu**2)
        # mfilter = np.ones(shape=(2*T_mean+1, 2*T_mean+1))/(2*T_mean+1)**2
        # S3: npt.NDArray[np.float64] = convolve2d(S2b, mfilter, 'same')
        S3 = S2b
        return S3

    def RQA_block_measures(self, trj: Trajectory, mu: float) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        N = trj.count_points()
        # Laplacian matrix (distance matrix)
        S3 = self.laplacian_matrix(trj, self.T_mean, mu)
        mat = S3 > np.exp(-1)
        # fill diagonals
        np.fill_diagonal(mat, True)  # first diagonal
        #  other diagonal depending on the radius size
        num_max = self.diag_fill_list[int(2*mu) - 1]
        for num in range(1, num_max + 1):
            ind = np.array(range(N-num))
            mat[ind, ind+num] = True
            mat[ind+num, ind] = True

        #  select matrix part connex to the diagonal line
        L = measure.label(mat, connectivity=2)
        matrix = np.zeros(L.shape)
        matrix[L == 1] = 1
        #  fill holes
        matrix = ndimage.binary_fill_holes(matrix)

        #  init analysis
        list_diagonal = np.zeros(shape=(N, ), dtype=np.int64)
        list_vertical = np.zeros(shape=(N, ), dtype=np.int64)
        list_parallel = np.zeros(shape=(N, ), dtype=np.int64)
        #  analysis
        for n in range(N):
            #  distribution of vertical lines
            if n < N/2:
                list_diag_ind = np.array(range(2*(n+1) - 1))  # + 1
            else:
                list_diag_ind = np.array(range(2*(n + 1) - N - 1, N))

            list_bin_vert = matrix[list_diag_ind, list_diag_ind[::-1]]
            index = np.where(np.logical_and(
                list_diag_ind == n, list_diag_ind[::-1] == n))[0][0]
            pos_down, pos_up = self.find_min_max_index_equal(
                index, list_bin_vert == 1, 1)
            # vertical
            pos_down_v, pos_up_v = self.find_min_max_index_equal(
                n, matrix[:, n] == 1)
            list_vertical[n] = pos_up_v - pos_down_v + 1

            #  diagonal
            list_diagonal[n] = pos_up - pos_down + 1
            #  parallel
            pos_down_par, pos_up_par = self.find_min_max_index_equal(
                n - (list_diagonal[n] - 1)//2, matrix.diagonal(list_diagonal[n]-1))
            list_parallel[n] = pos_up_par - pos_down_par + 1
        return list_vertical, list_diagonal, list_parallel

    def Make_list_min_max_index_equal(self, list_trapped_m: npt.NDArray[np.bool_]) -> npt.NDArray[np.int64]:
        N_vec = len(list_trapped_m)
        index = 0
        List_min_max = []
        stop = 0
        while stop == 0:
            if list_trapped_m[index] == 1:
                pos_down, pos_up = self.find_min_max_index_equal(
                    index, list_trapped_m)
                List_min_max.append((pos_down, pos_up))
                index = pos_up+1
            else:
                index = index+1

            if index >= N_vec:
                stop = 1
        result = np.zeros(shape=(len(List_min_max), 2), dtype=np.int64)
        result[:, 0] = [a for a, _ in List_min_max]
        result[:, 1] = [a for _, a in List_min_max]
        return result

    def find_min_max_index_equal(self, index: int, vector_to_test: npt.NDArray[np.int64], quantity: int = 1) -> Tuple[int, int]:
        N = len(vector_to_test)
        stop_up = 0
        stop_down = 0
        pos_up = index
        pos_down = index
        for z in range(N):
            if stop_up == 0:
                if pos_up+1 < N:
                    if vector_to_test[pos_up+1] == quantity:
                        pos_up = pos_up+1
                    else:
                        stop_up = 1
                else:
                    stop_up = 1
            if stop_down == 0:
                if pos_down-1 >= 0:
                    if vector_to_test[pos_down-1] == quantity:
                        pos_down = pos_down-1
                    else:
                        stop_down = 1
                else:
                    stop_down = 1

            if stop_up == 1 and stop_down == 1:
                break
        return pos_down, pos_up
