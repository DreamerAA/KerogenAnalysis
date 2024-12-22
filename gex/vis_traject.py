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

path = Path(realpath(__file__))
parent_dir = str(path.parent.parent.absolute())
sys.path.append(parent_dir)


from base.trajectory import Trajectory
from visualizer.visualizer import Visualizer, WrapMode


def visualize_dist_trajectory(traj_path: str, num: int, traps_path: str) -> None:
    trajectories = Trajectory.read_trajectoryes(traj_path)

    with open(traps_path, 'rb') as f:
        trajectories[num].traps = pickle.load(f)

    print(trajectories[num].points.shape)
    Visualizer.draw_trajectoryes(
        [trajectories[num]],
        wrap_mode=WrapMode.EMPTY,
        periodic=False,
        radius=0.1,
        with_points=True,
        color_type='clusters'
    )
    Visualizer.show()


if __name__ == '__main__':
    prefix = "/media/andrey/Samsung_T5/PHD/Kerogen/"

    type = "type1matrix"
    temp = "300K"
    el = "h2"
    traps_type = "Structural" # "Probabilistic" "Structural"
    
    num = 12
    traj_path = prefix + f"{type}/{temp}/{el}/trj.gro"
    traps_path = prefix + f"{type}/{temp}/{el}/traps/{traps_type}/traps_{num}.pickle"
    visualize_dist_trajectory(traj_path, num, traps_path)
