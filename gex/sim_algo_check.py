import argparse
import os
import sys
import time
from os.path import realpath, join
from pathlib import Path
from copy import deepcopy
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

path = Path(realpath(__file__))
parent_dir = str(path.parent.parent.absolute())
sys.path.append(parent_dir)

from base.reader import Reader
from examples.utils import create_cdf, ps_generate
from processes.kerogen_walk_simulator import KerogenWalkSimulator
from processes.pil_distr_generator import PiLDistrGenerator

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

np.random.default_rng(1111)


def run(
    path_to_main: str,
    pnm_name: str,
    count_trj=2,
    count_steps=1000,
):
    path_to_save: str = join(path_to_main, "errors")
    path_to_pil: str = join(path_to_main, "pi_l.npy")
    path_to_prob_fitters = join(path_to_main, "fitters", pnm_name)
    path_to_pnm: str = join(path_to_main, "pnm", pnm_name)

    radiuses, throat_lengths = Reader.read_pnm_data(
        path_to_pnm, scale=1e10, border=0.015
    )

    params_struct_set = {
        0.1: StructAnalizerParams(
            traj_type='Bm',
            nu=0.9,
            diag_percentile=0,
            kernel_size=1,
            list_mu=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            p_value=0.9,
            num_jobs=1,
        ),
        0.5: StructAnalizerParams(
            traj_type='Bm',
            nu=0.1,
            diag_percentile=0,
            kernel_size=1,
            list_mu=[1.5],
            p_value=0.9,
            num_jobs=1,
        ),
        0.9: StructAnalizerParams(
            traj_type='fBm',
            nu=0.1,
            diag_percentile=0,
            kernel_size=0,
            list_mu=[1.0],
            p_value=0.01,
            num_jobs=1,
        ),
    }

    params_prob_set = {
        k: ProbabilityAnalizerParams(
            critical_probability=1e-3,
        )
        for k in [0.1, 0.5, 0.9]
    }
    params_hybrid_set = {
        k: HybridAnalizerParams(
            params_prob_set[k],
            params_struct_set[k],
            0.1,
        )
        for k in [0.1, 0.5, 0.9]
    }
    pset = {
        k: {
            "hybrid": params_hybrid_set[k],
            "struct": params_struct_set[k],
            "prob": params_prob_set[k],
        }
        for k in [0.1, 0.5, 0.9]
    }

    ps_type = 'uniform'  # poisson uniform
    ps = ps_generate(ps_type)

    ind = 0
    ppl = create_cdf(radiuses)
    ptl = create_cdf(throat_lengths)

    if os.path.isfile(path_to_pil):
        pi_l = np.load(path_to_pil)
    else:
        generator = PiLDistrGenerator()
        pi_l = generator.run(radiuses)
        np.save(path_to_pil, pi_l)

    header_trajs = [
        f"Trajectory_{i+1}" for i in range(count_trj)
    ]  # Названия траектори

    for k, params in pset.items():
        matrix_analyzer = StructTrajectoryAnalizer(params["struct"])
        prob_analizer = ProbabilityTrajectoryAnalizer(
            params["prob"], pi_l, throat_lengths
        )

        hybrid_analizer = HybridTrajectoryAnalizer(
            params["hybrid"], pi_l, throat_lengths
        )

        prob = np.arange(0.0, 1.05, 0.05)

        ar_matrix_error = np.zeros(shape=(len(prob), count_trj))
        ar_prob_error = np.zeros(shape=(len(prob), count_trj))
        ar_hybrid_error = np.zeros(shape=(len(prob), count_trj))

        header = f"_k={k}_countSteps={count_steps}_countTrj={count_trj}.npy"
        matrix_er_fn = path_to_save + "/matrix" + header
        prob_er_fn = path_to_save + "/prob" + header
        hybrid_er_fn = path_to_save + "/hybrid" + header
        if (
            # True
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

                    prob_traps_result = prob_analizer.run(traj).astype(np.int32)
                    fitters = prob_analizer.get_fitters()
                    hybrid_analizer.set_prob_fitters(*fitters)

                    hybrid_analizer.set_trap_approx(matrix_traps_result)
                    hybrid_traps_result = hybrid_analizer.run(traj).astype(
                        np.int32
                    )

                    matrix_error = min(
                        np.sum(np.abs(real_traps - matrix_traps_result[:-1])),
                        np.sum(np.abs(real_traps - matrix_traps_result[1:])),
                    )
                    prob_error = np.sum(np.abs(real_traps - prob_traps_result))

                    hybrid_error = np.sum(
                        np.abs(real_traps - hybrid_traps_result)
                    )

                    ar_prob_error[j, i] = prob_error
                    ar_matrix_error[j, i] = matrix_error
                    ar_hybrid_error[j, i] = hybrid_error

                    ind += 1
                    print(
                        f"Ready {ind} from {count_trj*len(prob)*len(pset.items())}, trajectory num={i+1}, prob={p}, time = {time.time() - start_time}s "
                    )

            np.save(matrix_er_fn, ar_matrix_error)
            np.save(prob_er_fn, ar_prob_error)
            np.save(hybrid_er_fn, ar_hybrid_error)
        else:
            ind += count_trj * len(prob)

        ar_matrix_error = np.load(matrix_er_fn)
        ar_prob_error = np.load(prob_er_fn)
        ar_hybrid_error = np.load(hybrid_er_fn)

        ar_matrix_error /= count_steps
        ar_prob_error /= count_steps
        ar_hybrid_error /= count_steps

        df_matrix = pd.DataFrame(ar_matrix_error, columns=header_trajs)
        df_prob = pd.DataFrame(ar_prob_error, columns=header_trajs)
        df_hybrid = pd.DataFrame(ar_hybrid_error, columns=header_trajs)

        df_matrix["Probability"] = prob
        df_prob["Probability"] = prob
        df_hybrid["Probability"] = prob

        df_matrix_long = df_matrix.melt(
            id_vars=["Probability"], var_name="Trajectory", value_name="Error"
        )
        df_prob_long = df_prob.melt(
            id_vars=["Probability"], var_name="Trajectory", value_name="Error"
        )
        df_hybrid_long = df_hybrid.melt(
            id_vars=["Probability"], var_name="Trajectory", value_name="Error"
        )

        # df_hybrid_long["Erorr"] = df_hybrid_long["Error"] / count_steps
        # df_prob_long["Erorr"] = df_prob_long["Error"] / count_steps
        # df_matrix_long["Erorr"] = df_matrix_long["Error"] / count_steps

        plt.figure()
        sns.lineplot(
            data=df_prob_long,
            x="Probability",
            y="Error",
            label="Probabilistic",
            legend=False,
        )
        sns.lineplot(
            data=df_hybrid_long,
            x="Probability",
            y="Error",
            label="Hybrid",
            legend=False,
        )
        sns.lineplot(
            data=df_matrix_long,
            x="Probability",
            y="Error",
            label="Structural",
            legend=False,
        )

        plt.xlabel("Probability move to next trap", fontsize=12)
        plt.ylabel("Avarage error as (Count Error)/(Count Steps)", fontsize=12)
        plt.title(
            f"Errors for k = {k}, count trajectories {count_trj},\n trajectory count steps {count_steps}",
            fontsize=14,
        )
        # plt.yticks([0.009, 0.013, 0.017],fontsize=24)
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        plt.legend(frameon=False, prop={'size': 12})

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path_to_main',
        type=str,
        default="/media/andrey/Samsung_T5/PHD/Kerogen/type1matrix/300K/ch4/",
    )
    parser.add_argument(
        '--pnm_name',
        type=str,
        default="num=1640025000_500_500_500",
    )

    # parser.add_argument(
    #     '--path_to_save',
    #     type=str,
    #     default="/media/andrey/Samsung_T5/PHD/Kerogen/type1matrix/300K/ch4/errors/",
    # )
    # parser.add_argument(
    #     '--path_to_pil',
    #     type=str,
    #     default="/media/andrey/Samsung_T5/PHD/Kerogen/type1matrix/300K/ch4/pi_l.npy",
    # )
    args = parser.parse_args()

    run(args.path_to_main, args.pnm_name)
