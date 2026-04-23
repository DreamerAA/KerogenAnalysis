import argparse
import os
import sys
import time
from os.path import realpath
from pathlib import Path
from typing import List, Tuple
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from skimage import measure


from base.trajectory import Trajectory
from visualizer.visualizer import Visualizer, WrapMode


def visualize_dist_trajectory(
    traj_path: str, num: int, traps_path: str = ""
) -> None:
    trajectories = Trajectory.read_trajectoryes(traj_path)

    use_clusters = len(traps_path) != 0

    if use_clusters:
        with open(traps_path, 'rb') as f:
            trajectories[num].traps = pickle.load(f)

    print(trajectories[num].points.shape)
    Visualizer.draw_trajectoryes(
        [trajectories[num]],
        wrap_mode=WrapMode.EMPTY,
        periodic=False,
        radius=0.15,
        with_points=True,
        color_type='clusters' if use_clusters else 'dist',
    )
    Visualizer.show()


if __name__ == '__main__':
    prefix = "/media/andrey/Samsung_T5/PHD/Kerogen/"

    type = "type1matrix"
    temp = "300K"
    el = "h2"
    traps_type = "SIB"
    num = 1

    trj_path = prefix + f"{type}/{temp}/{el}/trj.gro"

    traps_path = (
        prefix + f"{type}/{temp}/{el}/traps/{traps_type}/traps_{num}.pickle"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--trj_path',
        type=str,
        default=trj_path,
    )
    parser.add_argument(
        '--traps_path',
        type=str,
        default=traps_path,
    )
    args = parser.parse_args()

    trj_path = args.trj_path
    traps_path = args.traps_path

    visualize_dist_trajectory(trj_path, num, traps_path)
    # visualize_dist_trajectory(traj_path, num, traps_path)
