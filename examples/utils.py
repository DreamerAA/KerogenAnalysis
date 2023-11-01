import numpy as np
from typing import List
from processes.trajectory_analyzer import AnalizerParams
from visualizer.visualizer import Visualizer
from base.trajectory import Trajectory
import numpy.typing as npt


def write_binary_file(array: npt.NDArray[np.int8], file_name: str) -> None:
    with open(file_name, 'wb') as file:
        for i in range(array.shape[2]):
            for j in range(array.shape[1]):
                file.write(bytes(bytearray(array[:, j, i])))


def visualize_trajectory(
    traj: Trajectory, color_type='dist', win_name: str = ""
) -> None:
    Visualizer.draw_trajectoryes(
        [traj], color_type=color_type, plot_box=False, window_name=win_name
    )


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


def get_params(indexes: List[int]) -> List[AnalizerParams]:
    list_mu = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    atparams = all_params()

    tparams = [atparams[i] for i in indexes]
    aparams = [
        AnalizerParams(tt, nu, dp, ks, list_mu, pv, 1)
        for dp, pv, nu, tt, ks in tparams
    ]
    return aparams
