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
from base.reader import Reader
from base.trajectory import Trajectory
from base.boundingbox import BoundingBox
from processes.KerogenWalkSimulator import KerogenWalkSimulator
from processes.trap_extractor import TrapExtractor
from processes.trajectory_extended_analizer import (
    TrajectoryExtendedAnalizer,
    ExtendedParams,
)
from processes.trajectory_extended_analizer import (
    TrajectoryAnalizer
)
from visualizer.visualizer import Visualizer





def run(path_to_pnm):
    radiuses, throat_lengths = Reader.read_pnm_data(
        path_to_pnm, scale=1e10, border=0.015
    )

    steps = np.array([s for s in range(1,101)], dtype=np.int32)
    prob = (steps.astype(np.float32))*0.01
    ps = np.vstack((steps, prob))

    simulator = KerogenWalkSimulator(ppl, ps, ptl, 0.5, 0.5)
    traj = simulator.run(5000)
    Visualizer.draw_trajectoryes([traj])
    Visualizer.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path_to_data',
        type=str,
        default="../data/Kerogen/time_trapping_results/ch4/num=1597500000_500_500_500",
    )
    args = parser.parse_args()

    run(args.path_to_data)