import argparse
import sys
import time
from os.path import realpath
import os
from pathlib import Path
import matplotlib.pyplot as plt

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


def run(path):
    lm = []
    pv = []
    mus = []
    for k in [0.0, 0.5, 1.0]:
        for p in [0.0, 0.5, 1.0]:
            result_name = f"{path}/{k}_{p}.pickle"
            assert Path(result_name).is_file()
            with open(result_name, 'rb') as f:
                errors = pickle.load(f)
            min_ind = np.array(
                [e for e, _ in errors], dtype=np.float32
            ).argmin()
            min_error = errors[min_ind][0]
            indexes = np.where(errors == min_error)[0]

            print(f"K = {k}, p = {p}")
            for i in indexes:
                print(i, ": ", errors[i])

            if k == 1.0:
                for i in indexes:
                    params = errors[i][1]
                    tmu = tuple(params.list_mu)
                    if tmu in mus:
                        index = mus.index(tmu)
                    else:
                        index = len(mus) 
                        mus.append(tmu)
                    lm.append(index)
                    pv.append(params.p_value)
        print(mus)


    plt.figure()
    plt.hist(lm)
    plt.title("list mu")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path_to_data',
        type=str,
        default="/media/andrey/Samsung_T5/PHD/Kerogen/type1matrix/300K/ch4/errors/",
    )
    args = parser.parse_args()

    run(args.path_to_data)
