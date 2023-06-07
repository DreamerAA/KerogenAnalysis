import os
import random
import time
from typing import IO, Tuple, List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt

from kerogen_data import AtomData, KerogenData
from periodizer import Periodizer
from segmentaion import Segmentator
from visualizer import Visualizer
from dataclasses import dataclass
from trajectory import Trajectory
from trajectory_analyzer import TrajectoryAnalizer



def visualize_dist_trajectory(traj:Trajectory)->None:
    diff = traj.points[1:] - traj.points[:-1]
    sq_diff = diff*diff
    sq_dist = np.sum(sq_diff,axis=1)
    dist = np.sqrt(sq_dist)
    сdist = np.cumsum(dist)
    plt.subplot(2, 1, 1)
    plt.plot(traj.times[1:], dist)
    plt.subplot(2, 1, 2)
    plt.plot(traj.times[1:], сdist)


def visualize_trajectory(traj:Trajectory)->None:
    Visualizer.draw_trajectoryes([traj], plot_box=False)

def visualize_trajectories(trajs:List[Trajectory])->None:
    Visualizer.draw_trajectoryes(trajs)

def animate_trajectoryes(trajs:List[Trajectory])->None:
    Visualizer.animate_trajectoryes(trajs)

if __name__ == '__main__':
    path_to_traj = "../data/Kerogen/meth_traj.gro"
    
    trajectories = Trajectory.read_trajectoryes(path_to_traj)

    # visualize_trajectories(trajectories)
    # animate_trajectoryes(trajectories)
    trj = trajectories[2]
    visualize_trajectory(trj)
    analizer = TrajectoryAnalizer(trj)
    Visualizer.draw_trajectory_points(trj)

    # for t in trajectories:
    #     visualize_dist_trajectory(t)
    #     plt.show()
