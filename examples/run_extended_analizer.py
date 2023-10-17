import argparse
import matplotlib.pyplot as plt
import numpy as np
from processes.trajectory_extended_analizer import (
    TrajectoryExtendedAnalizer,
    ExtendedParams,
)
import pandas as pd
import seaborn as sns
from base.trajectory import Trajectory
from examples.utils import get_params


def run_extended_analizer(
    traj_path: str, throat_len_path: str, pil_path: str, win_name: str
) -> None:
    trajectories = Trajectory.read_trajectoryes(traj_path)

    count_dp = 11
    ldp = np.linspace(0, 0.5, count_dp)

    wc_data = []
    full_neq_data = []
    full_mp_data = []
    full_pm_data = []
    cross_ma_1_data = []
    cross_ma_0_data = []
    for j, trj in enumerate(trajectories):
        aparams = get_params([154, 162, 186])
        params = aparams[0]
        ext_params = ExtendedParams(
            params.traj_type,
            params.nu,
            params.diag_percentile,
            params.kernel_size,
            params.list_mu,
            params.p_value,
            0,
        )

        throat_lengthes = np.load(throat_len_path)
        pi_l = np.load(pil_path)

        analizer = TrajectoryExtendedAnalizer(ext_params, pi_l, throat_lengthes)
        (
            traps_result,
            traps_approx,
            pore_probability,
            throat_probability,
            ex_p_mask,
            ex_t_mask,
        ) = analizer.run(trj)

        not_eq = traps_approx[1:] != traps_result

        m_p = traps_approx[1:] - traps_result
        p_m = traps_result - traps_approx[1:]
        m_p[m_p < 0] = 0
        p_m[p_m < 0] = 0

        full_neq_data.append((f"trj_{j}", np.sum(not_eq)))
        full_mp_data.append((f"trj_{j}", np.sum(m_p)))
        full_pm_data.append((f"trj_{j}", np.sum(p_m)))

        for i, dp in enumerate(ldp):
            btw_mask = np.abs(throat_probability - pore_probability) < dp

            btw_mask = np.logical_and(btw_mask, ~ex_p_mask)
            btw_mask = np.logical_and(btw_mask, ~ex_t_mask)

            matrix_res = traps_approx[1:][btw_mask]

            cross_ma_1_data.append(
                (
                    f"trj_{j}",
                    dp,
                    np.sum(matrix_res),
                )
            )
            cross_ma_0_data.append(
                (
                    f"trj_{j}",
                    dp,
                    np.sum(matrix_res == 0),
                )
            )

            wc_data.append(
                (
                    f"trj_{j}",
                    dp,
                    np.sum(btw_mask, dtype=np.float32) / len(btw_mask),
                )
            )
        print(f" --- Trajectory {j+1} finished")

    df_count_wc = pd.DataFrame(
        wc_data,
        columns=[
            "trajectory",
            "delta_probability",
            "count_without_classification",
        ],
    )

    df_cross_ma_1 = pd.DataFrame(
        cross_ma_1_data,
        columns=[
            "trajectory",
            "delta_probability",
            "matrix=1 in not classification area",
        ],
    )
    df_cross_ma_0 = pd.DataFrame(
        cross_ma_0_data,
        columns=[
            "trajectory",
            "delta_probability",
            "matrix=0 in not classification area",
        ],
    )

    df_full_neq_distr = pd.DataFrame(
        full_neq_data,
        columns=["trajectory", "number of different classifications"],
    )
    df_full_mp_distr = pd.DataFrame(
        full_mp_data,
        columns=[
            "trajectory",
            "matrix=1_prob=0",
        ],
    )
    df_full_pm_distr = pd.DataFrame(
        full_pm_data,
        columns=[
            "trajectory",
            "matrix=0_prob=1",
        ],
    )

    fig, axs = plt.subplots(ncols=2, nrows=3)

    sns.lineplot(
        data=df_count_wc,
        x="delta_probability",
        y="count_without_classification",
        ax=axs[0][0],
    )

    sns.lineplot(
        data=df_cross_ma_0,
        x="delta_probability",
        y="matrix=0 in not classification area",
        ax=axs[1][0],
    )

    sns.lineplot(
        data=df_cross_ma_1,
        x="delta_probability",
        y="matrix=1 in not classification area",
        ax=axs[2][0],
    )

    sns.histplot(
        data=df_full_neq_distr,
        x="number of different classifications",
        ax=axs[0][1],
    )
    sns.histplot(data=df_full_mp_distr, x="matrix=1_prob=0", ax=axs[1][1])
    sns.histplot(data=df_full_pm_distr, x="matrix=0_prob=1", ax=axs[2][1])

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--traj_path',
        type=str,
        # default="../data/methan_traj/meth_1.7_micros.1.gro"
        default="../data/Kerogen/h2_micros/h2_micros.1.gro",
    )
    parser.add_argument(
        '--throat_len_path',
        type=str,
        default="../data/Kerogen/tmp/1_pbc_atom/throat_lengths.npy",
    )
    parser.add_argument(
        '--pil_path',
        type=str,
        default="../data/Kerogen/tmp/1_pbc_atom/pi_l.npy",
    )
    args = parser.parse_args()

    run_extended_analizer(
        args.traj_path,
        args.throat_len_path,
        args.pil_path,
        "Article algorithm + PSD",
    )
