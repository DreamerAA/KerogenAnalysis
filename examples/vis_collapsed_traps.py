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
from base.boundingbox import BoundingBox
from processes.trap_extractor import TrapExtractor
from processes.trajectory_extended_analizer import (
    TrajectoryExtendedAnalizer,
    ExtendedParams,
)
from processes.trajectory_extended_analizer import (
    TrajectoryAnalizer
)
from visualizer.visualizer import Visualizer


def run(path_seq):
    with open(path_seq, 'rb') as handle:
        seq = pickle.load( handle)
    bb = BoundingBox()
    for p in seq.points:
        bb.update(p)

    trj = Trajectory(seq.points, seq.times, bb)
    # Visualizer.draw_trajectory_points(trj)
    Visualizer.draw_trajectoryes([trj])
    Visualizer.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path_to_seq',
        type=str,
        default="../data/Kerogen/time_trapping_results/ch4/prob/seq_0.pickle",
    )
    args = parser.parse_args()

    run(args.path_to_seq)