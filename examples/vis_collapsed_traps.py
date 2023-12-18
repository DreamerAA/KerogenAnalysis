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
from base.boundingbox import BoundingBox, Range
from processes.trap_extractor import TrapExtractor
from processes.trajectory_extended_analizer import (
    TrajectoryExtendedAnalizer,
    ExtendedParams,
)
from processes.trajectory_extended_analizer import TrajectoryAnalizer
from visualizer.visualizer import Visualizer


def run(path_seq: str, traj_folder: str, num: int):
    trajectories: List[Trajectory] = []
    onlyfiles = [
        f for f in listdir(traj_folder) if isfile(join(traj_folder, f))
    ]
    for path in onlyfiles:
        trajectories += Trajectory.read_trajectoryes(join(traj_folder, path))

    trj = trajectories[num]
    with open(path_seq + f"traps_{num}.pickle", 'rb') as f:
        trj.traps = pickle.load(f)

    seq = TrapExtractor.get_trap_seq(trj)

    trap_trj = Trajectory(
        seq.points, seq.times, trj.box, atom_size=0.12, traps=seq.traps
    )

    Visualizer.draw_trajectoryes(
        [trap_trj], periodic=True, with_points=True, plot_box=False
    )
    Visualizer.show()


if __name__ == '__main__':
    prefix = "../data/Kerogen/"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path_to_seq',
        type=str,
        default=prefix + "time_trapping_results/ch4/prob/",
    )
    parser.add_argument(
        '--traj_folder',
        type=str,
        default=prefix + "ch4_traj/",
    )
    args = parser.parse_args()

    run(args.path_to_seq, args.traj_folder, 0)
