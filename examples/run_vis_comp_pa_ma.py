
import numpy as np
import argparse
import sys
import os
from pathlib import Path
from os.path import realpath

path = Path(realpath(__file__))
parent_dir = str(path.parent.parent.absolute())
sys.path.append(parent_dir)

from base.trajectory import Trajectory
from examples.utils import get_params, visualize_trajectory
from processes.trajectory_extended_analizer import (
    TrajectoryExtendedAnalizer,
    ExtendedParams,
)
from processes.trajectory_analyzer import TrajectoryAnalizer
from visualizer.visualizer import Visualizer


def run_vis_comp_pa_ma(
    traj_path: str, throat_len_path: str, pil_path: str, trj_t: str, trj_num:int
) -> None:
    trajectories = Trajectory.read_trajectoryes(traj_path)
    trj = trajectories[trj_num]
    print(f" --- Count points in trajectory: {trj.count_points()}")
    count_dp = 2
    # ldp = np.linspace(0.5, 0.3, count_dp)
    # ldp = np.array([0.0, 0.3, 0.5])
    ldp = np.array([0.0, 0.5])

    aparams = get_params([154, 162, 186])
    params = aparams[0]
    params.num_jobs = 1
    
    path_tmp_traps = f"../data/Kerogen/tmp/{trj_t}_{trj_num}_matrix_traps.npy"
    if os.path.exists(path_tmp_traps):
        trap_approx = np.load(path_tmp_traps)
    else:
        analizer = TrajectoryAnalizer(params)
        trap_approx = analizer.run(trj)
        np.save(path_tmp_traps, trap_approx)
    trj.traps = trap_approx
    visualize_trajectory(trj, 'clusters', 'Matrix Algorithm')

    test_results = []
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
            trj, 'clusters', f"Probabilistic algorithm with window {prob_win} ({trj_t})"
        )
        test_results.append(traps_result.copy())
    
    # for i in range(1, len(test_results)):
    #     for i in range(i+1, test_results):
    print(np.sum(test_results[0] - test_results[1] != 0))


def run_comp_pa(
    traj_path: str, throat_len_path: str, pil_path: str, trj_t: str, trj_num:int
) -> None:
    trajectories = Trajectory.read_trajectoryes(traj_path)
    trj = trajectories[trj_num]
    print(f" --- Count points in trajectory: {trj.count_points()}")
    count_dp = 2
    # ldp = np.linspace(0.5, 0.3, count_dp)
    # ldp = np.array([0.0, 0.3, 0.5])
    ldp = np.array([0.0, 0.5])

    aparams = get_params([154, 162, 186])
    params = aparams[0]
    params.num_jobs = 1
    
    path_tmp_traps = f"../data/Kerogen/tmp/{trj_t}_{trj_num}_matrix_traps.npy"
    if os.path.exists(path_tmp_traps):
        trap_approx = np.load(path_tmp_traps)
    else:
        analizer = TrajectoryAnalizer(params)
        trap_approx = analizer.run(trj)
        np.save(path_tmp_traps, trap_approx)
    trj.traps = trap_approx
    visualize_trajectory(trj, 'clusters', 'Matrix Algorithm')

    test_results = []
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
            trj, 'clusters', f"Probabilistic algorithm with window {prob_win} ({trj_t})"
        )
        test_results.append(traps_result.copy())
    
    # for i in range(1, len(test_results)):
    #     for i in range(i+1, test_results):
    print(np.sum(test_results[0] - test_results[1] != 0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--traj_path',
        type=str,
        # default="../data/Kerogen/methan_traj/meth_1.7_micros.1.gro"
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

    run_comp_pa()

    # run_vis_comp_pa_ma(
    #     args.traj_path,
    #     args.throat_len_path,
    #     args.pil_path,
    #     "(meth) Article algorithm + PSD",
    #     # "(h2) Article algorithm + PSD",
    #     5
    # )

    # run_vis_comp_pa_ma(
    #     args.traj_path,
    #     args.throat_len_path,
    #     args.pil_path,
    #     "(meth) Article algorithm + PSD",
    #     # "(h2) Article algorithm + PSD",
    #     14
    # )

    # run_vis_comp_pa_ma(
    #     args.traj_path,
    #     args.throat_len_path,
    #     args.pil_path,
    #     "(meth) Article algorithm + PSD",
    #     # "(h2) Article algorithm + PSD",
    #     19
    # )
    Visualizer.show()