import argparse
from pathlib import Path
import sys
import os
from os import listdir
from os.path import isfile, join, dirname, realpath
import numpy as np
import matplotlib.pyplot as plt
import pickle
from typing import List

path = Path(realpath(__file__))
parent_dir = str(path.parent.parent.absolute())
sys.path.append(parent_dir)

from examples.utils import get_params, visualize_trajectory
from base.trap_sequence import TrapSequence
from base.trajectory import Trajectory
from processes.trap_extractor import TrapExtractor
from processes.trajectory_extended_analizer import (
    TrajectoryExtendedAnalizer,
    ExtendedParams,
)
from processes.trajectory_extended_analizer import (
    TrajectoryAnalizer
)


def pltdistr(times, prefix: str, n:int = 50):
    p, bb = np.histogram(times, bins=n, density=True)
    plt.scatter((bb[:-1]+bb[1:])/2, p, s=20, marker='o',label=prefix)

def plot_trap_tim_distr(trap_list, prefix):
    time_tuple = tuple(trap.times for trap in trap_list)
    time_trapings = np.concatenate(time_tuple)
    non_zero_tt = time_trapings[time_trapings != 0] 
    print(f"All count trapping steps: {len(time_trapings)}")
    print(f"Count zero time trapping: {np.sum(time_trapings == 0)}")
    print(f"Count non zero time trapping: {np.sum(time_trapings != 0)}")

    pltdistr(non_zero_tt, prefix)

def run(traj_path:str, distr_prefix:str, pts_trapping:str):
    params = get_params([154])[0]

    def get_ext_params(prob_win:float)->ExtendedParams:
        return ExtendedParams(
                params.traj_type,
                params.nu,
                params.diag_percentile,
                params.kernel_size,
                params.list_mu,
                params.p_value,
                1,
                prob_win
            )
    trajectories: List[Trajectory] = []
    onlyfiles = [f for f in listdir(traj_path) if isfile(join(traj_path, f))]
    for path in onlyfiles:
        trajectories += Trajectory.read_trajectoryes(join(traj_path,path))

    print("read_ready")
    throat_lengthes = np.load(distr_prefix + "_throat_lengths.npy")
    pi_l = np.load(distr_prefix + "_pi_l.npy")

    ext_params = get_ext_params(0.)
    prob_analizer = TrajectoryExtendedAnalizer(ext_params, pi_l, throat_lengthes)
    matrix_analyzer = TrajectoryAnalizer(ext_params)


    plt.figure()
    for analyzer, prefix in [(matrix_analyzer, "matrix"), (prob_analizer, "prob")]:
        cur_pts = join(pts_trapping, prefix)
        os.makedirs(cur_pts, exist_ok=True)
        
        extractor = TrapExtractor(analyzer)

        trap_list = []
        for i, trj in enumerate(trajectories):
            seq_file = Path(join(cur_pts, f"seq_{i}.pickle"))
            traps_file = Path(join(cur_pts, f"traps_{i}.pickle"))
            if not seq_file.is_file():
                seq = extractor.run(trj, lambda a, t: a.run(t))
                with open(seq_file, 'wb') as handle:
                    pickle.dump(seq, handle)
                with open(traps_file, 'wb') as handle:
                    pickle.dump(trj.traps, handle)
            else:
                with open(seq_file, 'rb') as fp:
                    seq = pickle.load(fp)
            trap_list.append(seq)
            return 

        plot_trap_tim_distr(trap_list, prefix)
        

    ## mixed
    ext_params = get_ext_params(0.5)
    mix_analizer = TrajectoryExtendedAnalizer(ext_params, pi_l, throat_lengthes)
    prefix = "mixed"
    matrix_pts = join(pts_trapping, "matrix")
    cur_pts = join(pts_trapping, prefix)
    os.makedirs(cur_pts, exist_ok=True)

    extractor = TrapExtractor(mix_analizer)
    trap_list = []
    for i, trj in enumerate(trajectories):
        seq_file = Path(join(matrix_pts, f"traps_{i}.pickle"))
        with open(seq_file, 'rb') as fp:
            trap_approx = pickle.load(fp)
        seq = extractor.run(trj, lambda a, t: a.run(t, trap_approx))
        trap_list.append(seq)
    plot_trap_tim_distr(trap_list, prefix)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('t (s)')
    plt.ylabel('P(t)')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--traj_folder',
        type=str,
        default="../data/Kerogen/ch4_traj/",
    )
    parser.add_argument(
        '--distr_prefix',
        type=str,
        default="../data/Kerogen/time_trapping_results/ch4/num=1597500000_500_500_500",
    )
    parser.add_argument(
        '--pts_trapping',
        type=str,
        default="../data/Kerogen/time_trapping_results/ch4/",
    )
    args = parser.parse_args()

    run(args.traj_folder, args.distr_prefix, args.pts_trapping )