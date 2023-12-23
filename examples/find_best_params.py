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
from joblib import Parallel, delayed
import time

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
    TrajectoryAnalizer,
    AnalizerParams,
)
from visualizer.visualizer import Visualizer
from examples.utils import create_cdf, get_params


def run(prefix):
    radiuses, throat_lengths = Reader.read_pnm_data(
        prefix, scale=1e10, border=0.015
    )
    radiuses = radiuses * 3

    steps = np.array([s for s in range(1, 101)], dtype=np.int32).reshape(100, 1)
    prob = ((steps.astype(np.float32)) * 0.01).reshape(100, 1)
    ps = np.hstack((steps, prob))

    ppl = create_cdf(radiuses)
    ptl = create_cdf(throat_lengths)

    simulator = KerogenWalkSimulator(ppl, ps, ptl, 0.5, 0.0)
    trjs = [simulator.run(2000) for i in range(5)]
    l_real_traps = [trj.traps.copy() for trj in trjs]

    llmu = [
        [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        [1.5, 2.0, 2.5],
        [0.5, 1],
        [2.5, 3],
        [0.5, 3.0],
        [1.5, 2.0],
        [0.5],
        [1],
        [1.5],
        [2],
        [2.5],
        [3],
    ]
    ind = 0
    errors = []
    start_time = time.time()
    for j, lmu in enumerate(llmu):
        params = get_params(lmu=lmu, num_jobs=5)

        def wrap(i, param):
            matrix_analyzer = TrajectoryAnalizer(param)
            me = 0

            for trj, real_traps in zip(trjs, l_real_traps):
                matrix_traps_result = matrix_analyzer.run(trj).astype(np.int32)
                me += np.sum(np.abs(real_traps - matrix_traps_result))
            print(
                f"Ready {(i + 1) * (j+1)} from {len(params) * len(llmu)} time = {time.time() - start_time}s "
            )
            return (me, param)

        res = Parallel(n_jobs=6)(
            delayed(wrap)(i, param) for i, param in enumerate(params)
        )
        errors += res

        # for param in params:
        #     matrix_analyzer = TrajectoryAnalizer(param)
        #     me = 0
        #     for trj, real_traps in zip(trjs, l_real_traps):
        #         matrix_traps_result = matrix_analyzer.run(trj).astype(np.int32)
        #         me += np.sum(np.abs(real_traps - matrix_traps_result))

        #     errors.append((me, param))

        ind += 1
        print(f"Checked {ind} from {len(params) * len(llmu)}")

        min_ind = np.array([e for e, _ in errors], dtype=np.float32).argmin()
        print(errors[min_ind][1])

    print('final!')
    min_ind = np.array([e for e, _ in errors], dtype=np.float32).argmin()
    print(errors[min_ind][1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path_to_data',
        type=str,
        default="../data/Kerogen/time_trapping_results/ch4/num=1597500000_500_500_500",
    )
    args = parser.parse_args()

    run(args.path_to_data)