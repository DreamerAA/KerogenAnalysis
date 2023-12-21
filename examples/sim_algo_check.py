import argparse
from pathlib import Path
import sys
from os.path import realpath
import numpy as np
import matplotlib.pyplot as plt
import time
import os

path = Path(realpath(__file__))
parent_dir = str(path.parent.parent.absolute())
sys.path.append(parent_dir)

from examples.utils import get_params
from base.reader import Reader
from processes.pil_distr_generator import PiLDistrGenerator
from processes.KerogenWalkSimulator import KerogenWalkSimulator
from processes.trajectory_extended_analizer import (
    TrajectoryExtendedAnalizer,
    ExtendedParams,
)
from processes.trajectory_extended_analizer import TrajectoryAnalizer
from examples.utils import create_cdf, get_params


def run(prefix, count_trj=10, count_steps=3000):
    rradiuses, throat_lengths = Reader.read_pnm_data(
        prefix, scale=1e10, border=0.015
    )

    pset = [
        (
            ExtendedParams(
                traj_type='fBm',
                nu=0.5,
                diag_percentile=50,
                kernel_size=2,
                list_mu=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
                p_value=0.01,
                num_jobs=6,
                critical_probability=0.0,
            ),
            'matrix optimal for p = 1.0',
        ),
        (
            ExtendedParams(
                traj_type='fBm',
                nu=0.1,
                diag_percentile=0,
                kernel_size=0,
                list_mu=[1.5, 2.0, 2.5],
                p_value=0.9,
                num_jobs=5,
                critical_probability=0.0,
            ),
            'matrix optimal for p = 0.0',
        ),
    ]
    ind = 0
    for rad_scale in [1, 2, 4]:
        radiuses = rradiuses * rad_scale
        steps = np.array([s for s in range(1, 101)], dtype=np.int32).reshape(
            100, 1
        )
        prob = ((steps.astype(np.float32)) * 0.01).reshape(100, 1)
        ps = np.hstack((steps, prob))

        ppl = create_cdf(radiuses)
        ptl = create_cdf(throat_lengths)

        pi_l_file_name = prefix + f"_scale={rad_scale}_pi_l.npy"
        if os.path.isfile(pi_l_file_name):
            pi_l = np.load(pi_l_file_name)
        else:
            generator = PiLDistrGenerator()
            pi_l = generator.run(radiuses)
            np.save(pi_l_file_name, pi_l)

        for params, msuffix in pset:
            plt.figure()

            prob_analizer = TrajectoryExtendedAnalizer(
                params, pi_l, throat_lengths
            )
            matrix_analyzer = TrajectoryAnalizer(params)

            prob = np.arange(1.1, step=0.1)
            r_matrix_error = np.zeros(shape=(len(prob),))
            r_prob_error = np.zeros(shape=(len(prob),))

            for style, k in zip(['solid', 'dotted', 'dashed'], [0.1, 0.5, 0.9]):
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

                        prob_v1 = np.hstack((real_traps[0], prob_traps_result))
                        prob_v2 = np.hstack((prob_traps_result, real_traps[-1]))

                        matrix_error = np.sum(
                            np.abs(real_traps - matrix_traps_result)
                        )
                        prob_error = min(
                            np.sum(np.abs(real_traps - prob_v1)),
                            np.sum(np.abs(real_traps - prob_v2)),
                        )
                        r_prob_error[j] += prob_error
                        r_matrix_error[j] += matrix_error

                        ind += 1
                        print(
                            f"Ready {ind} from {count_trj*len(prob)*3*3*2} time = {time.time() - start_time}s "
                        )
                        print(f"Trajectory num={i+1}, prob={p}")

                delim = count_trj * count_steps
                r_prob_error /= delim
                r_matrix_error /= delim

                plt.plot(
                    prob, r_prob_error, label=f"prob k={k}", linestyle=style
                )
                plt.plot(
                    prob, r_matrix_error, label=f"matrix k={k}", linestyle=style
                )

            plt.xlabel("Probability return to previous trap")
            plt.ylabel("Avarage error as (Count Error)/(Count Points)")
            plt.title(
                f"Trajectory count steps {count_steps}, count trajectories {count_trj}, {msuffix}, radiuses *= {rad_scale}"
            )
            plt.legend()

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path_to_data',
        type=str,
        default="../data/Kerogen/time_trapping_results/ch4/num=1597500000_500_500_500",
    )
    args = parser.parse_args()

    run(args.path_to_data)
