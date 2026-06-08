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
    trj = trajectories[num]
    cp = trj.points.shape[0]
    trj.cut(cp // 2)

    use_clusters = len(traps_path) != 0

    if use_clusters:
        with open(traps_path, 'rb') as f:
            trj.traps = pickle.load(f)

    print(trj.points.shape)
    Visualizer.draw_trajectoryes(
        [trj],
        wrap_mode=WrapMode.EMPTY,
        periodic=False,
        radius=0.005,
        with_points=True,
        color_type='clusters' if use_clusters else 'dist',
    )
    Visualizer.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize distance trajectory")
    parser.add_argument("trj", type=Path, help="Trajectory file (.gro)")
    parser.add_argument("num", type=int, help="Molecule index")
    parser.add_argument("--traps", type=Path, help="Traps pickle file (optional)")
    args = parser.parse_args()

    traps_path = str(args.traps) if args.traps else ""
    visualize_dist_trajectory(str(args.trj), args.num, traps_path)
