import os
import pickle
import sys
from os.path import join, realpath
from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np

path = Path(realpath(__file__))
parent_dir = str(path.parent.parent.absolute())
sys.path.append(parent_dir)

from scipy import optimize
from base.trajectory import Trajectory
from examples.utils import get_params
from processes.trajectory_extended_analizer import (
    ExtendedParams,
    TrajectoryAnalizer,
    TrajectoryExtendedAnalizer,
)
from processes.trap_extractor import TrapExtractor


def pltdistr(times, prefix: str, n: int = 50):
    p, bb = np.histogram(times, bins=n, density=True)
    indexes = p > 22000
    p_cut = p[indexes]
    b = (bb[:-1] + bb[1:]) / 2
    b_cut = b[indexes]

    # fitfunc = lambda p, x: p[0] * x + p[1]
    # errfunc = lambda p, x, y: fitfunc(p, x) - y
    # p0 = np.array([-2.0, 2e8], dtype=np.float32)
    # p2, success = optimize.leastsq(errfunc, p0, args=(b_cut, p_cut))
    # print(f"a, b = {p2}")

    # plt.plot(b, fitfunc(p2, b), label="fit " + prefix)
    plt.scatter((bb[:-1] + bb[1:]) / 2, p, s=20, marker='o', label=prefix)


def plot_trap_tim_distr(trap_list, prefix, tmax):
    time_tuple = tuple(trap.times for trap in trap_list)
    time_trapings = np.concatenate(time_tuple)
    non_zero_tt = time_trapings[time_trapings != 0]
    c = np.sum(time_trapings == 0) / len(time_trapings)
    # print(f"All count trapping steps: {len(time_trapings)}")
    print(f"Count zero time trapping: {np.sum(time_trapings == 0)}")
    print(f"Count non zero time trapping: {np.sum(time_trapings != 0)}")
    print(f"Coeff. zero time trapping by {prefix}: {c}")

    non_zero_tt = non_zero_tt[non_zero_tt < tmax]
    pltdistr(non_zero_tt, prefix)
    


def run(
    traj_path: str, tl_path: str, pil_path: str, pts_trapping: str, el_pref: str
):
    params = get_params([154])[0]

    def get_ext_params(prob_win: float) -> ExtendedParams:
        return ExtendedParams(
            params.traj_type,
            params.nu,
            params.diag_percentile,
            params.kernel_size,
            params.list_mu,
            params.p_value,
            1,
            prob_win,
        )

    trajectories = Trajectory.read_trajectoryes(traj_path)

    # print("read_ready")
    throat_lengthes = np.load(tl_path)
    pi_l = np.load(pil_path)

    params = get_ext_params(0.0)
    ext_params = get_ext_params(0.1)
    prob_analizer = TrajectoryExtendedAnalizer(
        params, pi_l, throat_lengthes
    )
    hybrid_analizer = TrajectoryExtendedAnalizer(
        ext_params, pi_l, throat_lengthes
    )
    matrix_analyzer = TrajectoryAnalizer(params)

    for analyzer, prefix, tmax in [
        (matrix_analyzer, "Structural", 1.6e-6),
        (prob_analizer, "Probabilistic", 3.5e-7),
        (hybrid_analizer, "Hybrid", 3.5e-7),
    ]:
        cur_pts = join(pts_trapping, prefix)
        os.makedirs(cur_pts, exist_ok=True)

        extractor = TrapExtractor(analyzer)

        trap_list = []

        for i, trj in enumerate(trajectories):
            
            seq_file = Path(join(cur_pts, f"seq_{i}.pickle"))
            traps_file = Path(join(cur_pts, f"traps_{i}.pickle"))
            if not seq_file.is_file():
                start_time = time.time()
                seq = extractor.run(trj, lambda a, t: a.run(t))
                print(f"Analize trajectory {i} is ready for {prefix}! Time: {time.time() - start_time}")
                with open(seq_file, 'wb') as handle:
                    pickle.dump(seq, handle)
                with open(traps_file, 'wb') as handle:
                    pickle.dump(trj.traps, handle)
            else:
                # print(f"load {i} {prefix}")
                with open(seq_file, 'rb') as fp:
                    seq = pickle.load(fp)
            trap_list.append(seq)
            

        plot_trap_tim_distr(trap_list, el_pref + " " + prefix, tmax)


if __name__ == '__main__':
    path_to_data = "/media/andrey/Samsung_T5/PHD/Kerogen/"
    input_data = [
        (path_to_data + "type1matrix/300K/ch4/", "CH4", 1),
        # (path_to_data + "type1matrix/300K/h2/", "H2", 2, ),
        # (path_to_data + "type1matrix/300K/ch4/", "type1-300K-CH4", 1),
        # (path_to_data + "type1matrix/300K/h2/", "type1-300K-H2", 2),
        # (path_to_data + "type1matrix/400K/ch4/", "type1-400K-CH4", 1),
        # (path_to_data + "type1matrix/400K/h2/", "type1-400K-H2", 2),
        # (path_to_data + "type2matrix/300K/ch4/", "type2-300K-CH4", 1),
        # (path_to_data + "type2matrix/300K/h2/", "type2-300K-H2", 2),
    ]
    for path, prefix, step in input_data:
        traj_path = path + "/trj.gro"
        tl_path = path + "/throat_lengths.npy"
        pil_path = path + "/pi_l.npy"
        pts_trapping = path + "/traps/"
        run(traj_path, tl_path, pil_path, pts_trapping, prefix)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r"$t, sec$", fontsize=16)
    plt.ylabel(r"$P(t)$", fontsize=16)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.legend(frameon=False, prop={'size': 12})
    plt.show()
