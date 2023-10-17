from base.trajectory import Trajectory
import numpy as np
from examples.utils import get_params, visualize_trajectory
from processes.trajectory_extended_analizer import (
    TrajectoryExtendedAnalizer,
    ExtendedParams,
)
from processes.trajectory_analyzer import TrajectoryAnalizer


def run_extended_analizer(
    traj_path: str, throat_len_path: str, pil_path: str, win_name: str
) -> None:
    trajectories = Trajectory.read_trajectoryes(traj_path)
    trj = trajectories[3]
    count_dp = 6
    ldp = np.linspace(0.5, 0.3, count_dp)

    aparams = get_params([154, 162, 186])
    params = aparams[0]
    analizer = TrajectoryAnalizer(params)
    trap_approx = analizer.run(trj)
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

        visualize_trajectory(
            trj, 'clusters', f"Probabilistic algorithm with window {prob_win}"
        )
