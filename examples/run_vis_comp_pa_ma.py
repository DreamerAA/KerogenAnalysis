from base.trajectory import Trajectory
import numpy as np
import argparse
from examples.utils import get_params, visualize_trajectory
from processes.trajectory_extended_analizer import (
    TrajectoryExtendedAnalizer,
    ExtendedParams,
)
from processes.trajectory_analyzer import TrajectoryAnalizer
from visualizer.visualizer import Visualizer
import os


def run_vis_comp_pa_ma(
    traj_path: str, throat_len_path: str, pil_path: str, win_name: str
) -> None:
    trajectories = Trajectory.read_trajectoryes(traj_path)
    trj = trajectories[3]
    count_dp = 6
    ldp = np.linspace(0.5, 0.3, count_dp)

    aparams = get_params([154, 162, 186])
    params = aparams[0]

    path_tmp_traps = "../data/Kerogen/h2_micros/matrix_traps.npy"
    if os.path.exists(path_tmp_traps):
        trap_approx = np.load(path_tmp_traps)
    else:
        analizer = TrajectoryAnalizer(params)
        trap_approx = analizer.run(trj)
        np.save(path_tmp_traps, trap_approx)
    trj.traps = trap_approx
    visualize_trajectory(trj, 'clusters', 'Matrix Algorithm')

    for j, prob_win in enumerate(ldp):
        ext_params = ExtendedParams(
            params.traj_type,
            params.nu,
            params.diag_percentile,
            params.kernel_size,
            params.list_mu,
            params.p_value,
            prob_win,
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
        ) = analizer.run(trj, trap_approx)

        trj.traps = traps_result
        visualize_trajectory(
            trj, 'clusters', f"Probabilistic algorithm with window {prob_win}"
        )

    Visualizer.show()


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

    run_vis_comp_pa_ma(
        args.traj_path,
        args.throat_len_path,
        args.pil_path,
        "Article algorithm + PSD",
    )
