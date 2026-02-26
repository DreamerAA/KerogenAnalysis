import argparse
import sys
import os
from os.path import realpath, exists, join
from pathlib import Path
import matplotlib.pyplot as plt
from processes.struct_trajectory_analyzer import StructAnalizerParams
from utils.types import NPFArray, f32
import numpy as np
import pickle

path = Path(realpath(__file__))
parent_dir = str(path.parent.parent.absolute())
sys.path.append(parent_dir)


def run(path):
    if not exists(path):
        os.mkdir(path)

    full_errors = {}
    for k in [0.1, 0.5, 0.9]:
        lerrors = []
        for p in [0.0, 0.5, 1.0]:
            result_name = join(path, f"k={k}_p={p}.pickle")
            assert Path(result_name).is_file()
            with open(result_name, 'rb') as f:
                errors: dict[
                    tuple[float, float, StructAnalizerParams], float
                ] = pickle.load(f)
            full_errors.update(errors)

    lk = [0.1, 0.5, 0.9]
    lp = [0.0, 0.5, 1.0]
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
    fparams = {}

    for ik, k in enumerate(lk):
        for ip, p in enumerate(lp):
            for imu, lmu in enumerate(llmu):
                params = StructAnalizerParams.get_params(lmu=lmu)
                for idx, param in enumerate(params):
                    fparams[(ik, ip, imu, idx)] = param

    all_idx = set([idx for ik, ip, imu, idx in full_errors.keys()])
    res_by_k = np.zeros(shape=(len(lk), len(llmu), len(all_idx)), dtype=f32)
    for (ik, _, imu, idx), er in full_errors.items():
        res_by_k[ik, imu, idx] += er

    for ik, k in enumerate(lk):
        imu, idx = np.unravel_index(res_by_k[ik].argmin(), res_by_k[ik].shape)
        print(f"for k={k}, optimal params: {fparams[(ik, 0.0, imu, idx)]}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path_to_data',
        type=str,
        default="/media/andrey/Samsung_T5/PHD/Kerogen/type1matrix/300K/ch4/errors/find_best_params",
    )
    args = parser.parse_args()

    run(args.path_to_data)
