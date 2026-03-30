import os
import pickle
from os.path import join, isfile
from pathlib import Path
import time
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from base.trajectory import Trajectory
from base.trap_sequence import TrapSequence
from utils.types import f32
from processes.hybrid_trajectory_analizer import (
    HybridAnalizerParams,
    HybridTrajectoryAnalizer,
)
from processes.struct_trajectory_analyzer import StructTrajectoryAnalizer
from processes.probability_trajectory_analizer import (
    ProbabilityTrajectoryAnalizer,
    ProbabilityAnalizerParams,
)
from processes.struct_trajectory_analyzer import StructAnalizerParams
from processes.trap_extractor import TrapExtractor
from utils.utils import kprint
from scipy.stats import linregress


def compute_survival_unique(times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(times, dtype=float)
    times = times[np.isfinite(times)]
    times = times[times > 0]
    if times.size == 0:
        return np.array([]), np.array([])

    # counts per unique time
    t_unique, counts = np.unique(times, return_counts=True)
    # survival: P(T > t)  (strictly greater)
    n = counts.sum()
    # number of samples with T > t_k equals cumulative counts after k
    n_greater = np.cumsum(counts[::-1])[::-1] - counts  # after current bin
    S = n_greater / n

    # часто удобнее P(T >= t): тогда не будет нулей на последней точке
    # n_ge = np.cumsum(counts[::-1])[::-1]
    # S = n_ge / n

    # убираем нули, чтобы log работал
    mask = S > 0
    return t_unique[mask], S[mask]


def estimate_alpha_range(
    times: np.ndarray,
    t_min: float,
    t_max: float,
) -> tuple[
    float, float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    t, S = compute_survival_unique(times)
    if t.size < 5:
        return np.nan, np.nan, np.nan, np.nan, t, S, np.array([]), np.array([])

    mask = (t >= t_min) & (t <= t_max)
    t_fit = t[mask]
    S_fit = S[mask]

    if t_fit.size < 5:
        return np.nan, np.nan, np.nan, np.nan, t, S, t_fit, S_fit

    log_t = np.log(t_fit)
    log_S = np.log(S_fit)

    slope, intercept, r_value, p_value, std_err = linregress(log_t, log_S)
    alpha = -slope
    r2 = r_value**2
    return alpha, slope, intercept, r2, t, S, t_fit, S_fit


def plot_trapping_on_axis(
    ax,
    times: np.ndarray,
    t_min: float,
    t_max: float,
    label: str,
):
    alpha, slope, intercept, r2, t, S, t_fit, S_fit = estimate_alpha_range(
        times, t_min=t_min, t_max=t_max
    )

    # empirical
    ax.loglog(
        t,
        S,
        marker="o",
        linestyle="none",
        alpha=0.35,
        label=f"{label}",
    )

    # fit
    if np.isfinite(alpha) and t_fit.size > 0:
        t_line = np.logspace(np.log10(t_fit.min()), np.log10(t_fit.max()), 200)
        S_line = np.exp(intercept) * (t_line**slope)

        ax.loglog(
            t_line,
            S_line,
            linewidth=2,
            label=rf"{label} fit: $\alpha={alpha:.3f}$",
        )

    return alpha, r2


def plot_trap_tim_distr(
    trap_list: list[TrapSequence], prefix, t_min, t_max, ax1, ax2
):
    time_tuple = tuple(trap.times for trap in trap_list)
    time_trapings = np.concatenate(time_tuple)
    non_zero_tt = time_trapings[time_trapings != 0]

    zero_time_traps = np.array([seq.get_zero_trap_count() for seq in trap_list])
    non_zero_time_traps = np.array(
        [seq.get_non_zero_trap_count() for seq in trap_list]
    )
    kprint(f"{prefix} - Count zero time steps: {zero_time_traps.mean():.3f}")
    kprint(
        f"{prefix} - Count non zero time steps: {non_zero_time_traps.mean():.3f}"
    )
    zero_probability = np.array(
        [seq.get_zero_trap_probability() for seq in trap_list], dtype=f32
    )
    kprint(f"{prefix} - Zero Trap probability: {zero_probability.mean():.3f}")

    plot_trapping_on_axis(ax1, non_zero_tt, t_min, t_max, prefix)

    print('')


def get_struct_params(gas: str) -> StructAnalizerParams:
    lmu = np.array(
        [0.5, 1.0, 1.5, 2.0, 2.5, 3.0] if gas == "CH4" else [1.5, 2.0],
        dtype=f32,
    )
    return StructAnalizerParams(
        traj_type='fBm',
        nu=0.1,
        diag_percentile=0,
        kernel_size=0,
        list_mu=lmu,
        p_value=0.9,
    )


def run(path_to_main: str, gas: str, step: int, t_min_max, ax1, ax2):
    traj_path = join(path_to_main, "trj.gro")
    pts_trapping = join(path_to_main, "traps")
    path_to_pil_gf: str = join(path_to_main, "pi_l_gamma_fitter.pkl")
    path_to_tl_wf: str = join(path_to_main, "throat_lengths_weibull_fitter.pkl")

    os.makedirs(pts_trapping, exist_ok=True)

    if isfile(path_to_pil_gf):
        with open(path_to_pil_gf, "rb") as f:
            pil_gamma_fitter = pickle.load(f)
    else:
        raise RuntimeError("pi_l data not found")

    if isfile(path_to_tl_wf):
        with open(path_to_tl_wf, "rb") as f:
            throat_lengths_weibull_fitter = pickle.load(f)
    else:
        raise RuntimeError("throat_lengths not found")

    trajectories = Trajectory.read_trajectoryes(traj_path)
    trajectories = trajectories[::step]

    struct_params = get_struct_params(gas)

    pap = ProbabilityAnalizerParams(1e-3)
    prob_analizer = ProbabilityTrajectoryAnalizer(
        pap,
        pil_gamma_fitter,
        throat_lengths_weibull_fitter,
    )
    hybrid_analizer = HybridTrajectoryAnalizer(
        HybridAnalizerParams(pap, struct_params, 0.1),
        pil_gamma_fitter,
        throat_lengths_weibull_fitter,
    )
    matrix_analyzer = StructTrajectoryAnalizer(struct_params)

    results: Dict[Tuple[str, int], npt.NDArray[np.bool_]] = {}
    for analyzer, prefix in [
        (matrix_analyzer, "Structural"),
        (prob_analizer, "Probabilistic"),
        (hybrid_analizer, "Hybrid"),
    ]:
        cur_pts = join(pts_trapping, prefix)
        os.makedirs(cur_pts, exist_ok=True)

        trap_list = []

        for i, trj in enumerate(trajectories):
            if prefix == "Hybrid":
                approx_traps = results[("Structural", i)]
                assert hasattr(analyzer, "set_trap_approx")
                analyzer.set_trap_approx(approx_traps)

            seq_file = Path(join(cur_pts, f"seq_{step * i}.pickle"))
            traps_file = Path(join(cur_pts, f"traps_{step * i}.pickle"))

            # if not seq_file.is_file():
            start_time = time.time()
            traps = analyzer.run(trj)
            seq = TrapExtractor.get_trap_seq(traps, trj.delta_time_sec)
            print(
                f" --- Analize trajectory {i} is ready for {prefix}! Time: {time.time() - start_time}"
            )
            with open(seq_file, 'wb') as handle:
                pickle.dump(seq, handle)
            with open(traps_file, 'wb') as handle:
                pickle.dump(traps, handle)
            # else:
            #     # print(f"load {i} {prefix}")
            #     with open(seq_file, 'rb') as fp:
            #         seq = pickle.load(fp)
            #     with open(traps_file, 'rb') as fp:
            #         traps = pickle.load(fp)
            results[(prefix, i)] = np.copy(traps)

            trap_list.append(seq)
        t_min, t_max = t_min_max[prefix]
        plot_trap_tim_distr(
            trap_list, gas + " " + prefix, t_min, t_max, ax1, ax2
        )


if __name__ == '__main__':
    path_to_data = "/media/andrey/Samsung_T5/PHD/Kerogen/"
    input_data = [
        (
            path_to_data + "type1matrix/300K/ch4/",
            "CH4",
            1,
            {
                "Probabilistic": (5e-8, 2e-7),
                "Hybrid": (5e-8, 2e-7),
                "Structural": (5e-8, 5e-7),
            },
        ),
        (
            path_to_data + "type1matrix/300K/h2/",
            "H2",
            2,
            {
                "Probabilistic": (5e-9, 3e-8),
                "Hybrid": (5e-9, 3e-8),
                "Structural": (1.5e-8, 8e-8),
            },
        ),
        # (path_to_data + "type1matrix/300K/ch4/", "type1-300K-CH4", 1),
        # (path_to_data + "type1matrix/300K/h2/", "type1-300K-H2", 2),
        # (path_to_data + "type1matrix/400K/ch4/", "type1-400K-CH4", 1),
        # (path_to_data + "type1matrix/400K/h2/", "type1-400K-H2", 2),
        # (path_to_data + "type2matrix/300K/ch4/", "type2-300K-CH4", 1),
        # (path_to_data + "type2matrix/300K/h2/", "type2-300K-H2", 2),
    ]
    for input in input_data:
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        # fig2, ax2 = plt.subplots(figsize=(6, 5))
        run(*input, ax1, None)

        # ======================
        # Figure 1 — Survival
        # ======================
        ax1.set_xscale("log")
        ax1.set_yscale("log")

        ax1.set_xlabel(r"$t, sec$", fontsize=16)
        ax1.set_ylabel(r"$S(t) = P(\tau > t)$", fontsize=16)

        ax1.tick_params(axis="both", labelsize=14)
        # fig1.subplots_adjust(right=0.72)

        # легенда вне графика справа
        ax1.legend(
            frameon=False,
            fontsize=13,
            # loc="center left",
            # bbox_to_anchor=(1.02, 0.5),
        )

        fig1.tight_layout()

        # output_path = join(input[0], "traps", f"{input[1]}.png")
        # fig1.savefig(output_path, dpi=300, bbox_inches="tight")
        # print(f"Saved to {output_path}")

        # # ======================
        # # Figure 2 — Local alpha
        # # ======================
        # ax2.set_xscale("log")  # semilog-x уже используется, но явно задаём
        # ax2.set_xlabel(r"$t, sec$", fontsize=16)
        # ax2.set_ylabel(
        #     r"$\alpha(t) = - \frac{d \log S}{d \log t}$", fontsize=16
        # )

        # ax2.tick_params(axis="both", labelsize=14)
        # ax2.legend(frameon=False, fontsize=14)

        # fig2.tight_layout()

    plt.show()
