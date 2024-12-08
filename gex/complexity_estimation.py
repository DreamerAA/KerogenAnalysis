import argparse
from pathlib import Path
import sys
from os.path import realpath
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from scipy.stats import poisson
from scipy import optimize
path = Path(realpath(__file__))
parent_dir = str(path.parent.parent.absolute())
sys.path.append(parent_dir)

from examples.utils import get_params
from base.reader import Reader
from processes.pil_distr_generator import PiLDistrGenerator
from processes.kerogen_walk_simulator import KerogenWalkSimulator
from processes.trajectory_extended_analizer import (
    TrajectoryExtendedAnalizer,
    ExtendedParams,
)
from processes.trajectory_extended_analizer import TrajectoryAnalizer
from examples.utils import create_cdf, get_params, ps_generate


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
                    kernel_size=1,
                    list_mu=[0.5],
                    p_value=0.9,
                    num_jobs=3,
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

    matrix_analyzer = TrajectoryAnalizer(params)
    simulator = KerogenWalkSimulator(ppl, ps, ptl, 0.5, 0.5)
    matrix_analyzer.run(simulator.run(100))

    count = 10
    trj_lens = np.arange(100, 1000, 100)
    exp_time = np.zeros(len(trj_lens), dtype=np.float32)
    for j, trj_len in enumerate(trj_lens):
        times = np.zeros(count, dtype=np.float32)
        for i in range(count):
            trj = simulator.run(trj_len)
            start_time = time.time()
            matrix_analyzer.run(trj)
            times[i] = time.time() - start_time
        print("End estimation for trj_len = ", trj_len)
        print("Times = ", times)
        exp_time[j] = np.sum(times)/float(times.size)
    
    trj_lens = trj_lens[1:]
    exp_time = exp_time[1:]

    # print(trj_lens)
    # print(exp_time)

    fitfunc = lambda p, x: p[0]*x**2 + p[1]*x + p[2]
    errfunc = lambda p, x, y: fitfunc(p, x) - y
    p0 = np.array([1.,1.,0.],dtype=np.float32)
    p2,success = optimize.leastsq(errfunc, p0[:], args=(trj_lens, exp_time)) 
    print(f"p = {p2}")
    plt.scatter(trj_lens, exp_time, s=20, marker='o', c='r', label='experimental')
    plt.plot(trj_lens, fitfunc(p2, trj_lens), label='fit by a*x^2 + b*x + c')
    plt.xlabel('Trajectory length', size = 24)
    plt.ylabel('Execution Time', size = 24)
    plt.legend()
    plt.show()