import argparse
import os
import pickle
import sys
import time
from os.path import isfile, join, realpath
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

from base.discretecdf import DiscreteCDF
from base.empiricalcdf import EmpiricalCDF

path = Path(realpath(__file__))
parent_dir = str(path.parent.parent.absolute())
sys.path.append(parent_dir)

from utils.utils import create_empirical_cdf, kprint, ps_generate
from base.bufferedsampler import BufferedSampler
from processes.kerogen_walk_simulator import KerogenWalkSimulator
from processes.hybrid_trajectory_analizer import (
    HybridAnalizerParams,
    HybridTrajectoryAnalizer,
)
from processes.struct_trajectory_analyzer import (
    StructTrajectoryAnalizer,
    StructAnalizerParams,
)
from processes.probability_trajectory_analizer import (
    ProbabilityTrajectoryAnalizer,
    ProbabilityAnalizerParams,
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path_to_main',
        type=str,
        default="/media/andrey/Samsung_T5/PHD/Kerogen/type1matrix/400K/h2/",
    )
    args = parser.parse_args()
    path_to_main = args.path_to_main

    path_to_pil_gf: str = join(path_to_main, "pi_l_gamma_fitter.pkl")
    path_to_tl_wf: str = join(path_to_main, "throat_lengths_weibull_fitter.pkl")
    path_to_radiuses: str = join(path_to_main, "radiuses.npy")

    print("Start read pnm")

    print("Start generate ps")
    ps_type = 'uniform'  # poisson uniform
    ps = ps_generate(ps_type)

    ind = 0

    if isfile(path_to_pil_gf):
        with open(path_to_pil_gf, "rb") as f:
            pil_gamma_fitter = pickle.load(f)
    else:
        raise RuntimeError("pi_l data not found")

    if isfile(path_to_radiuses):
        radiuses = np.load(path_to_radiuses)
    else:
        raise RuntimeError("radiuses not found")

    if isfile(path_to_tl_wf):
        with open(path_to_tl_wf, "rb") as f:
            throat_lengths_weibull_fitter = pickle.load(f)
    else:
        raise RuntimeError("throat_lengths not found")

    print("Start estimation")

    def get_struct_params():
        return StructAnalizerParams(
            traj_type='fBm',
            nu=0.1,
            diag_percentile=10,
            kernel_size=2,
            list_mu=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            p_value=0.9,
        )

    def get_prob_params():
        return ProbabilityAnalizerParams(
            critical_probability=1e-3,
        )

    def get_struct_analizer():
        return StructTrajectoryAnalizer(get_struct_params())

    def get_prob_analizer():
        return ProbabilityTrajectoryAnalizer(
            get_prob_params(), pil_gamma_fitter, throat_lengths_weibull_fitter
        )

    def get_hybrid_analizer():
        return HybridTrajectoryAnalizer(
            HybridAnalizerParams(
                get_prob_params(),
                get_struct_params(),
                0.1,
            ),
            pil_gamma_fitter,
            throat_lengths_weibull_fitter,
        )

    psd = create_empirical_cdf(radiuses)
    ps = ps_generate(ps_type, mean_count=50)
    bs_ps = BufferedSampler(DiscreteCDF(ps), "ps", size=10_000)
    bs_psd = BufferedSampler(EmpiricalCDF(psd), "psd", size=10_000)
    bs_ptl = BufferedSampler(throat_lengths_weibull_fitter, "ptl", size=10_000)

    count = 5
    trj_lens = np.arange(500, 8000, 500)
    simulator = KerogenWalkSimulator(bs_psd, bs_ps, bs_ptl, 0.5, 0.5)
    trj = simulator.run(1000)

    steps = [
        (get_struct_analizer, 'r', "Structural", 0, 2),
        (get_prob_analizer, 'g', "Probabilistic", 1, 1),
        (get_hybrid_analizer, 'b', "Hybrid", 2, 2),
    ]

    times_path = (
        "/media/andrey/Samsung_T5/PHD/Kerogen/type1matrix/400K/h2/times.npy"
    )
    if os.path.isfile(times_path):
        times = np.load(times_path)
    else:
        times = np.zeros(shape=(len(trj_lens), 3), dtype=np.float32)
        for j, trj_len in enumerate(trj_lens):
            for getter_analyzer, _, _, k, _ in steps:
                for i in range(count):
                    trajectory = simulator.run(trj_len)
                    start_time = time.time()
                    analyzer = getter_analyzer()
                    analyzer.run(trajectory)
                    times[j, k] += time.time() - start_time

            kprint("End estimation for trj_len = ", trj_len)

        np.save(times_path, times)

    times /= float(count)

    times = times[1:, :]
    trj_lens = trj_lens[1:]

    def add_plot(atime, color, name, fit_degree: int = 2):
        errfunc = lambda p, x, y: fitfunc(p, x) - y
        if fit_degree == 2:
            fitfunc = lambda p, x: p[0] * x**2 + p[1] * x + p[2]
            p0 = np.array([1.0, 1.0, 0.0], dtype=np.float32)
        else:
            fitfunc = lambda p, x: p[0] * x + p[1]
            p0 = np.array([1.0, 0.0], dtype=np.float32)

        p2, success = optimize.leastsq(errfunc, p0[:], args=(trj_lens, atime))
        print(f"p = {p2}")
        plt.scatter(trj_lens, atime, s=20, marker='o', c=color)
        plt.plot(
            trj_lens,
            fitfunc(p2, trj_lens),
            color=color,
            label=name,
        )

    for _, color, name, ind, fd in steps:
        add_plot(times[:, ind], color, name, fd)

    plt.xlabel('Trajectory length', fontsize=12)
    plt.ylabel('Execution Time, sec', fontsize=12)
    plt.yscale('log')
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.legend(frameon=False, fontsize=12)
    plt.show()
