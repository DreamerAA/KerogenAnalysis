import argparse
import os
import sys
import time
from os.path import realpath
from pathlib import Path
from copy import deepcopy
import inspect
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from scipy.stats import poisson

path = Path(realpath(__file__))
parent_dir = str(path.parent.parent.absolute())
sys.path.append(parent_dir)

from base.reader import Reader
from examples.utils import create_cdf, get_params, ps_generate
from processes.kerogen_walk_simulator import KerogenWalkSimulator
from processes.pil_distr_generator import PiLDistrGenerator
from processes.trajectory_extended_analizer import (
    ExtendedParams,
    TrajectoryAnalizer,
    TrajectoryExtendedAnalizer,
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path_to_pnm',
        type=str,
        default="/media/andrey/Samsung_T5/PHD/Kerogen/type1matrix/400K/h2/pnm/num=235002500_250_250_250",
    )
    parser.add_argument(
        '--path_to_pi_l',
        type=str,
        default="/media/andrey/Samsung_T5/PHD/Kerogen/type1matrix/400K/h2/pi_l.npy",
    )
    args = parser.parse_args()

    print("Start read pnm")

    radiuses, throat_lengths = Reader.read_pnm_data(
        args.path_to_pnm, scale=1e10, border=0.015
    )

    params = ExtendedParams(
        traj_type='fBm',
        nu=0.1,
        diag_percentile=10,
        kernel_size=2,
        list_mu=[0.5, 1., 1.5, 2., 2.5, 3.],
        p_value=0.9,
        num_jobs=1,
        critical_probability=0.0,
    )

    print("Start generate ps")
    ps_type = 'uniform'  # poisson uniform
    ps = ps_generate(ps_type)

    print("Start create cdf")

    ind = 0
    ppl = create_cdf(radiuses)
    ptl = create_cdf(throat_lengths)

    print("start read Pi(L)")

    pi_l_file_name = args.path_to_pi_l
    if os.path.isfile(pi_l_file_name):
        pi_l = np.load(pi_l_file_name)
    else:
        generator = PiLDistrGenerator()
        pi_l = generator.run(radiuses)
        # np.save(pi_l_file_name, pi_l)

    print("Start estimation")

    eparams = deepcopy(params)
    eparams.critical_probability = 0.1

    def get_matrix():
        return TrajectoryAnalizer(params)
    def get_prob():
        return TrajectoryExtendedAnalizer(params, pi_l, throat_lengths)
    def get_hybrid():
        return TrajectoryExtendedAnalizer(eparams, pi_l, throat_lengths)

    count = 5
    trj_lens = np.arange(500, 6000, 500)
    simulator = KerogenWalkSimulator(ppl, ps, ptl, 0.5, 0.5)
    trj = simulator.run(1000)

    steps = [
        (get_matrix, 'r', "Structural", 0, 2),
        (get_prob, 'g', "Probabilistic", 1, 1),
        (get_hybrid, 'b', "Hybrid", 2, 2)
    ]

    times_path = "/media/andrey/Samsung_T5/PHD/Kerogen/type1matrix/400K/h2/times.npy"
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
                    if k == 1:
                        times[j, k] += analyzer.inside_time
                    else:
                        times[j, k] += time.time() - start_time
                    print(f"Current ex time {times[j, k]/(i+1)}")

            print("End estimation for trj_len = ", trj_len)

        np.save(times_path, times)

    times /= float(count)
    
    times = times[1:,:]
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
        plt.scatter(
            trj_lens, atime, s=20, marker='o', c=color, label=name
        )
        plt.plot(trj_lens, fitfunc(p2, trj_lens), color=color, label=name + " fit by " + ("ax^2 + bx + c" if fit_degree == 2 else "ax + b"))

    for _, color, name, ind, fd in steps:
        add_plot(times[:, ind], color, name, fd)
        
    plt.xlabel('Trajectory length', fontsize=12)
    plt.ylabel('Execution Time, sec', fontsize=12)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.legend(frameon=False, fontsize=12)
    plt.show()
