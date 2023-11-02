import sys
from pathlib import Path
import argparse
import time
from typing import List, Tuple
from skimage import measure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
from pathlib import Path
from os.path import realpath

path = Path(realpath(__file__))
parent_dir = str(path.parent.parent.absolute())
sys.path.append(parent_dir)

from processes.trajectory_extended_analizer import (
    TrajectoryExtendedAnalizer,
    ExtendedParams,
)
from base.trajectory import Trajectory
from processes.trajectory_analyzer import TrajectoryAnalizer, AnalizerParams
from visualizer.visualizer import Visualizer
from examples.utils import get_params, visualize_trajectory


def visualize_dist_trajectory(traj: Trajectory) -> None:
    diff = traj.points[1:] - traj.points[:-1]
    sq_diff = diff * diff
    sq_dist = np.sum(sq_diff, axis=1)
    dist = np.sqrt(sq_dist)
    cdist = np.cumsum(dist)
    plt.subplot(2, 1, 1)
    plt.plot(traj.times[1:], dist)
    plt.subplot(2, 1, 2)
    plt.plot(traj.times[1:], cdist)


def visualize_trajectories(
    trajs: List[Trajectory],
) -> None:
    Visualizer.draw_trajectoryes(trajs)


def animate_trajectoryes(trajs: List[Trajectory]) -> None:
    Visualizer.animate_trajectoryes(trajs)


def analizy_visualize(trj, params, win_name: str):
    analizer = TrajectoryAnalizer(params)
    trj.traps = analizer.run(trj)
    clusters = measure.label(trj.traps, connectivity=1).astype(np.float32)
    print(f" --- Count clusters: {clusters.max()}")
    visualize_trajectory(trj, 'clusters', win_name)


def run_and_plot_trap_time_distribution(
    trajectories: List[Trajectory], aparams: List[Tuple[int, AnalizerParams]]
):
    fix, axs = plt.subplots(len(aparams))
    for i, ind_params in enumerate(aparams):
        for j, trj in enumerate(trajectories):
            Path(f"./output/h2_traj/{i}").mkdir(parents=True, exist_ok=True)
            path_to_result = f"./output/h2_traj/{i}/{ind_params[0]}.npy"
            my_file = Path(path_to_result)
            if my_file.is_file():
                trj.traps = np.load(my_file)
            else:
                analizer = TrajectoryAnalizer(trj, ind_params[1])

            np.save(f"./output/h2_traj/{i}/{ind_params[0]}.npy", trj.traps)
            print(f" --- Trajectory number: {j+1}, Params number: {i+1}")

        results = get_trap_time_distribution(trajectories)
        axs[i].hist(results)
    plt.show()


def get_trap_time_distribution(trajectories: List[Trajectory]):
    result = []
    for trj in trajectories:
        clusters = measure.label(trj.traps, connectivity=1).astype(np.int32)
        result = result + [
            np.sum(clusters == (c + 1)) for c in range(clusters.max())
        ]
    return result


def run_default_analizer(path: str, win_name: str) -> None:
    trajectories = Trajectory.read_trajectoryes(path)
    aparams = get_params([154, 162, 186])
    params = aparams[0]

    # run_and_plot_trap_time_distribution(trajectories, list(zip(indexes, aparams)))
    # analize_script(4)

    for i in [12, 14, 19]:  #
        visualize_trajectory(trajectories[i])

    # params = AnalizerParams(traj_type='fBm', nu=0.9, diag_percentile=50 , kernel_size=1, list_mu=np.array([0.5, 1. , 1.5, 2. , 2.5, 3.]), p_value=0.01)
    # params.list_mu = np.array([1.5, 2])
    trj = trajectories[0]
    # analizy_visualize(trj, params, win_name)

    # visualize_trajectories(trajectories)
    # animate_trajectoryes(trajectories)

    # plt.plot(trj.times, trj.clusters)
    # plt.xlabel('Time', size = 24)
    # plt.ylabel('Pore number', size = 24)
    # plt.xticks(size = 18)
    # plt.yticks(size = 18)
    # plt.show()

    # for t in trajectories:
    #     visualize_dist_trajectory(t)
    #     plt.show()
    Visualizer.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--traj_path',
        type=str,
        # default="../data/methan_traj/meth_1.7_micros.1.gro"
        default="../data/Kerogen/h2_micros/h2_micros.1.gro",
    )
    parser.add_argument(
        '--throat_len_path',
        type=str,
        default="../data/Kerogen/tmp/1_pbc_atom/throat_lengths.npy",
    )
    parser.add_argument(
        '--pil_path',
        type=str,
        default="../data/Kerogen/tmp/1_pbc_atom/pi_l.npy",
    )
    args = parser.parse_args()

    run_default_analizer(args.traj_path, "Article algorithm")
