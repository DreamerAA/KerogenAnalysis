import argparse
import os
import sys
import time
from os.path import realpath
from pathlib import Path
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
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


def run(
    prefix: str,
    path_to_save: str,
    path_to_pil: str,
    count_trj=10,
    count_steps=2000,
):
    radiuses, throat_lengths = Reader.read_pnm_data(
        prefix, scale=1e10, border=0.015
    )

    pset = {
        0.0: ExtendedParams(
            traj_type='Bm',
            nu=0.9,
            diag_percentile=0,
            kernel_size=1,
            list_mu=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            p_value=0.9,
            num_jobs=3,
            critical_probability=0.0,
        ),
        0.5: ExtendedParams(
            traj_type='Bm',
            nu=0.1,
            diag_percentile=0,
            kernel_size=1,
            list_mu=[1.5],
            p_value=0.9,
            num_jobs=3,
            critical_probability=0.0,
        ),
        1.0: ExtendedParams(
            traj_type='fBm',
            nu=0.1,
            diag_percentile=0,
            kernel_size=0,
            list_mu=[1.],
            p_value=0.01,
            num_jobs=3,
            critical_probability=0.0,
        ),
    }

    ps_type = 'poisson'  # poisson uniform
    ps = ps_generate(ps_type)

    ind = 0
    ppl = create_cdf(radiuses)
    ptl = create_cdf(throat_lengths)

    if os.path.isfile(path_to_pil):
        pi_l = np.load(path_to_pil)
    else:
        generator = PiLDistrGenerator()
        pi_l = generator.run(radiuses)

    for k, params in pset.items():
        
        eparams = deepcopy(params)
        eparams.critical_probability = 0.1
        matrix_analyzer = TrajectoryAnalizer(params)
        prob_analizer = TrajectoryExtendedAnalizer(params, pi_l, throat_lengths)
        hybrid_analizer = TrajectoryExtendedAnalizer(
            eparams, pi_l, throat_lengths
        )

        prob = np.arange(1.1, step=0.1)
        r_matrix_error = np.zeros(shape=(len(prob),))
        r_prob_error = np.zeros(shape=(len(prob),))
        r_hybrid_error = np.zeros(shape=(len(prob),))

        
        matrix_er_fn = path_to_save + f"/matrix_k={k}.npy"
        prob_er_fn = path_to_save + f"/prob_k={k}.npy"
        hybrid_er_fn = path_to_save + f"/hybrid_k={k}.npy"
        if (
            not Path(prob_er_fn).is_file()
            or not Path(matrix_er_fn).is_file()
            or not Path(hybrid_er_fn).is_file()
        ):
            for j, p in enumerate(prob):
                simulator = KerogenWalkSimulator(ppl, ps, ptl, k, p)

                for i in range(count_trj):
                    start_time = time.time()
                    traj = simulator.run(count_steps)
                    real_traps = traj.traps.copy().astype(np.int32)

                    matrix_traps_result = matrix_analyzer.run(traj).astype(
                        np.int32
                    )
                    prob_traps_result = prob_analizer.run(traj).astype(
                        np.int32
                    )
                    hybrid_traps_result = hybrid_analizer.run(
                        traj, matrix_traps_result
                    ).astype(np.int32)

                    matrix_error = min(
                        np.sum(
                            np.abs(real_traps - matrix_traps_result[:-1])
                        ),
                        np.sum(
                            np.abs(real_traps - matrix_traps_result[1:])
                        ),
                    )
                    prob_error = np.sum(
                        np.abs(real_traps - prob_traps_result)
                    )

                    hybrid_error = np.sum(
                        np.abs(real_traps - hybrid_traps_result)
                    )

                    r_prob_error[j] += prob_error
                    r_matrix_error[j] += matrix_error
                    r_hybrid_error[j] += hybrid_error

                    ind += 1
                    print(
                        f"Ready {ind} from {count_trj*len(prob)*3*2}, trajectory num={i+1}, prob={p}, time = {time.time() - start_time}s "
                    )

            delim = float(count_trj * count_steps)
            r_prob_error /= delim
            r_matrix_error /= delim
            r_hybrid_error /= delim
            np.save(matrix_er_fn, r_matrix_error)
            np.save(prob_er_fn, r_prob_error)
            np.save(hybrid_er_fn, r_hybrid_error)

        r_matrix_error = np.load(matrix_er_fn)
        r_prob_error = np.load(prob_er_fn)
        r_hybrid_error = np.load(hybrid_er_fn)
        
        plt.figure()
        plt.plot(prob, r_prob_error, label="prob")
        plt.plot(prob, r_hybrid_error, label="hybrid")
        plt.plot(prob, r_matrix_error, label="matrix")

        plt.xlabel("Probability move to next trap")
        plt.ylabel("Avarage error as (Count Error)/(Count Points)")
        plt.title(
            f"Errors for k = {k}, trajectory count steps {count_steps}, count trajectories {count_trj}"
        )
        plt.legend()

    plt.show()


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
        default="/media/andrey/Samsung_T5/PHD/Kerogen/type1matrix/300K/ch4/errors/",
    )
    parser.add_argument(
        '--path_to_pil',
        type=str,
        default="/media/andrey/Samsung_T5/PHD/Kerogen/type1matrix/300K/ch4/pi_l.npy",
    )
    args = parser.parse_args()

    run(args.path_to_data, args.path_to_save, args.path_to_pil)
