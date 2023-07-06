from base.trajectory import Trajectory
from typing import List
from scipy.io import savemat
import numpy as np


class Writer:
    @staticmethod
    def trajectory_to_mat(trjs: List[Trajectory]) -> None:
        count_points = trjs[0].count_points()
        shape = (len(trjs), count_points)
        X, Y, Z = np.zeros(shape), np.zeros(shape), np.zeros(shape)
        for i, trj in enumerate(trjs):
            points = trj.points_without_periodic()
            X[i, :] = points[:, 0]
            Y[i, :] = points[:, 1]
            Z[i, :] = points[:, 2]
        mdic = {"X": X, "Y": Y, "Z": Z}
        savemat("trajectories.mat", mdic)
