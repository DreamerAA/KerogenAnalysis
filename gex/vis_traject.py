import argparse
import os
import sys
import time
from os.path import realpath
from pathlib import Path
from typing import List, Tuple

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


def visualize_dist_trajectory(traj_path: str, num: int) -> None:
    trajectories = Trajectory.read_trajectoryes(traj_path)

    Visualizer.draw_trajectoryes(
        [trajectories[num]],
        wrap_mode=WrapMode.AXES,
        periodic=False,
        radius=0.2,
        with_points=True,
    )
    Visualizer.show()


if __name__ == '__main__':
    prefix = "/media/andrey/Samsung_T5/PHD/Kerogen/"

    type = "type1matrix"
    temp = "300K"
    el, step = "h2", 1
    traj_path = prefix + f"{type}/{temp}/{el}/trj.gro"
    num = 12
    visualize_dist_trajectory(traj_path, num)
