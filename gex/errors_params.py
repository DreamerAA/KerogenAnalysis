import argparse
import sys
import time
import os
from os.path import realpath, exists, join, isfile
from pathlib import Path

from joblib import Parallel, delayed
import numpy as np
import pickle
from base.bufferedsampler import BufferedSampler
from base.discretecdf import DiscreteCDF
from base.empiricalcdf import EmpiricalCDF
from utils.utils import kprint, ps_generate, create_empirical_cdf
from base.reader import Reader
from processes.kerogen_walk_simulator import KerogenWalkSimulator
from processes.struct_trajectory_analyzer import (
    StructTrajectoryAnalizer,
    StructAnalizerParams,
)


def run(path_to_main: str):
    path_to_errors = join(path_to_main, "errors", "find_best_params")
    if not exists(path_to_errors):
        os.mkdir(path_to_errors)

    path_to_tl_wf: str = join(path_to_main, "throat_lengths_weibull_fitter.pkl")
    path_to_radiuses: str = join(path_to_main, "radiuses.npy")

    if isfile(path_to_radiuses):
        radiuses = np.load(path_to_radiuses)
    else:
        raise RuntimeError("radiuses not found")

    if isfile(path_to_tl_wf):
        with open(path_to_tl_wf, "rb") as f:
            throat_lengths_weibull_fitter = pickle.load(f)
    else:
        raise RuntimeError("throat_lengths not found")

    psd = create_empirical_cdf(radiuses)
    bs_psd = BufferedSampler(EmpiricalCDF(psd), "psd", size=10_000)
    bs_ptl = BufferedSampler(throat_lengths_weibull_fitter, "ptl", size=10_000)
    ps = ps_generate("uniform", mean_count=100)
    bs_ps = BufferedSampler(DiscreteCDF(ps), "ps", size=10_000)

    count = 10
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

    count_lkmu_jobs = len(lk) * len(lp) * len(llmu)
    index = 0
    for ik, k in enumerate(lk):
        for ip, p in enumerate(lp):
            result_name = join(path_to_errors, f"k={k}_p={p}.pickle")
            if Path(result_name).is_file():
                continue

            simulator = KerogenWalkSimulator(bs_psd, bs_ps, bs_ptl, k, p)
            trjs = [simulator.run(1000) for i in range(int(count))]
            l_real_traps = [trj.traps.copy() for trj in trjs]

            errors = {}

            for imu, lmu in enumerate(llmu):
                start_time = time.time()
                params = StructAnalizerParams.get_params(lmu=lmu)

                def wrap(idx):
                    param = params[idx]
                    matrix_analyzer = StructTrajectoryAnalizer(param)
                    me = 0.0

                    for trj, real_traps in zip(trjs, l_real_traps):
                        matrix_traps_result = matrix_analyzer.run(trj).astype(
                            np.int32
                        )
                        me += np.sum(real_traps != matrix_traps_result[1:])

                    me /= count
                    return me, idx

                results = Parallel(n_jobs=12)(
                    delayed(wrap)(idx) for idx in range(len(params))
                )
                for er, idx in results:
                    errors[(ik, ip, imu, idx)] = er

                index += len(params)
                kprint(
                    f"Ready {index} from {len(params) * count_lkmu_jobs} time = {time.time() - start_time}s"
                )

            with open(result_name, 'wb') as handle:
                pickle.dump(errors, handle)
            assert Path(result_name).is_file()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path_to_main',
        type=str,
        default="/media/andrey/Samsung_T5/PHD/Kerogen/type1matrix/300K/ch4/",
    )
    args = parser.parse_args()

    run(args.path_to_main)
