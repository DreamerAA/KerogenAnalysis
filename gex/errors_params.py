import argparse
import sys
import time
import os
from os.path import realpath
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
import pickle

path = Path(realpath(__file__))
parent_dir = str(path.parent.parent.absolute())
sys.path.append(parent_dir)

from base.reader import Reader
from examples.utils import create_cdf, get_params, ps_generate
from processes.kerogen_walk_simulator import KerogenWalkSimulator
from processes.trajectory_extended_analizer import TrajectoryAnalizer


def run(prefix, path_to_save):
    radiuses, throat_lengths = Reader.read_pnm_data(
        prefix, scale=1e10, border=0.015
    )

    ps_type = 'uniform'  # poisson uniform
    ps = ps_generate(ps_type, max_count_step=50)

    ppl = create_cdf(radiuses)
    ptl = create_cdf(throat_lengths)

    count = 10.0
    for k in [0.0, 0.5, 1.0]:
        for p in [0.0, 0.5, 1.0]:
            result_name = f"{path_to_save}/{k}_{p}.pickle"
            if Path(result_name).is_file():
                continue

            simulator = KerogenWalkSimulator(ppl, ps, ptl, k, p)
            trjs = [simulator.run(1000) for i in range(int(count))]
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
            errors = []

            for j, lmu in enumerate(llmu):
                start_time = time.time()
                params = get_params(lmu=lmu, num_jobs=6)

                def wrap(param):
                    matrix_analyzer = TrajectoryAnalizer(param)
                    me = 0.0

                    for trj, real_traps in zip(trjs, l_real_traps):
                        matrix_traps_result = matrix_analyzer.run(trj).astype(
                            np.int32
                        )
                        me += min(
                            np.sum(
                                np.abs(real_traps - matrix_traps_result[1:])
                            ),
                            np.sum(
                                np.abs(real_traps - matrix_traps_result[:-1])
                            ),
                        )
                    me /= count
                    return (me, param)

                for param in params:
                    errors.append(wrap(param))

                print(
                    f"Ready {len(params) * (j + 1)} from {len(params) * len(llmu)} time = {time.time() - start_time}s"
                )

            with open(result_name, 'wb') as handle:
                pickle.dump(errors, handle)
            assert Path(result_name).is_file()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path_to_data',
        type=str,
        default="/media/andrey/Samsung_T5/PHD/Kerogen/type1matrix/300K/ch4/pnm/num=1640025000_500_500_500",
    )
    parser.add_argument(
        '--path_to_save',
        type=str,
        default="/media/andrey/Samsung_T5/PHD/Kerogen/type1matrix/300K/ch4/errors",
    )
    args = parser.parse_args()

    run(args.path_to_data, args.path_to_save)
