from typing import List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt
import time
from base.kerogendata import AtomData, KerogenData
from base.periodizer import Periodizer
from processes.segmentaion import Segmentator
from visualizer.visualizer import Visualizer
from dataclasses import dataclass
from base.trajectory import Trajectory
from processes.trajectory_analyzer import TrajectoryAnalizer, SpectralAnalizer
from base.writer import Writer


def visualize_dist_trajectory(traj: Trajectory) -> None:
    diff = traj.points[1:] - traj.points[:-1]
    sq_diff = diff*diff
    sq_dist = np.sum(sq_diff, axis=1)
    dist = np.sqrt(sq_dist)
    сdist = np.cumsum(dist)
    plt.subplot(2, 1, 1)
    plt.plot(traj.times[1:], dist)
    plt.subplot(2, 1, 2)
    plt.plot(traj.times[1:], сdist)


def visualize_trajectory(traj: Trajectory, color_type='dist') -> None:
    Visualizer.draw_trajectoryes(
        [traj],  color_type=color_type, plot_box=False)


def visualize_trajectories(trajs: List[Trajectory]) -> None:
    Visualizer.draw_trajectoryes(trajs)


def animate_trajectoryes(trajs: List[Trajectory]) -> None:
    Visualizer.animate_trajectoryes(trajs)


if __name__ == '__main__':

    path_to_traj = "../data/Kerogen/meth_traj.gro"
    # path_to_traj = "../data/Kerogen/methan_traj/meth_1.7_micros.1.gro"
    # path_to_traj = "../data/Kerogen/h2_micros/h2_micros.1.gro"
    trajectories = Trajectory.read_trajectoryes(path_to_traj)

    # visualize_trajectories(trajectories)
    # animate_trajectoryes(trajectories)

    # for trj in trajectories:
    # visualize_trajectory(trj)

    # visualize_trajectory(trajectories[2])

    start = time.time()
    analizer = SpectralAnalizer(trajectories[2])
    end = time.time()
    print(f"Elapsed time analizer: {end-start}")
    np.save('clusters.npy', trajectories[2].clusters)

    # trajectories[2].traps = np.load("clusters.npy")

    visualize_trajectory(trajectories[2], 'clusters')
    # visualize_trajectory(trajectories[2], 'dist')

    # for method in range(1):
    # analizer = TrajectoryAnalizer(trj, method)
    # Visualizer.draw_trajectory_points(trj)

    # plt.plot(trj.times, trj.clusters)
    # plt.xlabel('Time', size = 24)
    # plt.ylabel('Pore number', size = 24)
    # plt.xticks(size = 18)
    # plt.yticks(size = 18)
    # plt.show()

    # for t in trajectories:
    #     visualize_dist_trajectory(t)
    #     plt.show()
