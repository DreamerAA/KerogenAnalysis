from typing import Optional
import numpy as np
from typing import List
from processes.trajectory_analyzer import AnalizerParams
from visualizer.visualizer import Visualizer
from base.trajectory import Trajectory
import numpy.typing as npt
from scipy.stats import poisson


def ps_generate(type: str, max_count_step: int = 100) -> npt.NDArray[np.float32]: 
    steps = np.arange(0, max_count_step)
    if type == 'poisson':
        steps = np.arange(0, max_count_step)
        prob = poisson.cdf(steps, 100, loc=-50)
        prob[:-1] = prob[:-1] - prob[0]
        prob[1:] = prob[1:] + (1 - prob[-1])

        ps = np.zeros(shape=(len(steps) + 1, 2), dtype=np.float32)
        ps[1:, 0] = steps
        ps[1:, 1] = prob
    else:
        steps = np.arange(0, max_count_step)
        prob = (steps.astype(np.float32)) * 0.01
        ps = np.zeros(shape=(len(steps) + 1, 2), dtype=np.float32)
        ps[1:, 0] = steps
        ps[1:, 1] = prob / prob[-1]
    return ps


def create_cdf(vals, n=30):
    p, bb = np.histogram(vals, bins=n)
    xdel = bb[1] - bb[0]
    x = (bb[:-1] + xdel * 0.5).reshape(n, 1)
    pn = (np.cumsum(p) / np.sum(p)).reshape(n, 1)
    pn[0] = 0
    return np.hstack((x, pn))


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


def get_params(
    indexes: Optional[List[int]] = None,
    lmu: Optional[List[int]] = None,
    num_jobs: int = 1,
) -> List[AnalizerParams]:
    list_mu = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]) if lmu is None else lmu
    atparams = all_params()
    if indexes is not None:
        tparams = [atparams[i] for i in indexes]
    else:
        tparams = [v for _, v in atparams.items()]

    aparams = [
        AnalizerParams(tt, nu, dp, ks, list_mu, pv, num_jobs)
        for dp, pv, nu, tt, ks in tparams
    ]
    return aparams
