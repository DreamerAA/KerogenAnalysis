from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.ticker import MaxNLocator
from processes.trajectory_analyzer.dm import DistanceMatrixAnalyzer
from utils.types import NPBArray, NPIArray
from numba import njit

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


def extract_invariants(matrix, N) -> Tuple[NPIArray, NPIArray, NPIArray]:  
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

        ud_bin, ud_ver = DistanceMatrixAnalyzer.get_up_down(
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

N = 18
x = list(range(N))

mat = np.ones(shape=(N, N), dtype=np.bool_)
# mat = np.zeros(shape=(N, N), dtype=np.bool_)
# mat[x, x] = True
# mat[2:7, 2:7] = True
# mat[9:17,9:17] = True


list_vertical, list_diagonal, list_parallel = extract_invariants(mat, N)

plt.plot(x, list_vertical, label=r'$v_{|}$', color='r')
plt.plot(x, list_diagonal, label=r'$v_{\perp}$', color='magenta')
plt.plot(x, list_parallel, label=r'$v_{\parallel }$', color='blue')
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

plt.legend(fontsize=18, frameon=False)

plt.show()
