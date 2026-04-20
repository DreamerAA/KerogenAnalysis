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


from utils.utils import create_empirical_cdf, kprint, ps_generate
from base.bufferedsampler import BufferedSampler
from processes.kerogen_walk_simulator import KerogenWalkSimulator
from processes.trajectory_analyzer.hybrid import (
    HybridParams,
    HybridAnalyzer,
)
from processes.trajectory_analyzer.dm import (
    DistanceMatrixAnalyzer,
    DistanceMatrixParams,
)
from processes.trajectory_analyzer.sib import (
    StructureInformedBayesParams,
    StructureInformedBayesAnalyzer,
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
        return DistanceMatrixParams(
            traj_type='fBm',
            nu=0.1,
            diag_percentile=10,
            kernel_size=2,
            list_mu=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            p_value=0.9,
        )

    def get_prob_params():
        return StructureInformedBayesParams(
            critical_probability=1e-3,
        )

    def get_dm_analyzer():
        return DistanceMatrixAnalyzer(get_struct_params())

    def get_sib_analyzer():
        return StructureInformedBayesAnalyzer(
            get_prob_params(), pil_gamma_fitter, throat_lengths_weibull_fitter
        )

    def get_hybrid_analyzer():
        return HybridAnalyzer(
            HybridParams(
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
        (get_dm_analyzer, "Distance-matrix", 0),
        (get_sib_analyzer, "SIB", 1),
        (get_hybrid_analyzer, "Hybrid", 2),
    ]

    times_path = (
        "/media/andrey/Samsung_T5/PHD/Kerogen/type1matrix/400K/h2/times.npy"
    )
    if os.path.isfile(times_path):
        times = np.load(times_path)
    else:
        times = np.zeros(shape=(len(trj_lens), 3), dtype=np.float32)
        for j, trj_len in enumerate(trj_lens):
            for getter_analyzer, _, k in steps:
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

    def add_plot(atime, name):
        logx = np.log(trj_lens)
        logy = np.log(atime)
        p = np.polyfit(logx, logy, deg=1)

        print(f"p = {p}")
        points = plt.scatter(trj_lens, atime, s=30, marker='o', alpha=0.35)
        color = points.get_facecolor()[0]
        plt.plot(
            trj_lens,
            np.exp(p[1]) * trj_lens ** p[0],
            label=name,
            color=color,
        )

    for _, name, ind in steps:
        add_plot(times[:, ind], name)

    plt.xlabel('Trajectory length', fontsize=14)
    plt.ylabel('Execution Time, sec', fontsize=14)
    plt.xscale('log')
    plt.yscale('log')
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.legend(frameon=False, fontsize=14)
    plt.savefig(
        "/media/andrey/Samsung_T5/PHD/Kerogen/complexity.pdf",
        bbox_inches="tight",
        pad_inches=0,
    )
    # plt.show()
