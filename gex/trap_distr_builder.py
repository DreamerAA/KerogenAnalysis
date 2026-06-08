import argparse
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
from processes.trajectory_analyzer.hybrid import (
    HybridParams,
    HybridAnalyzer,
)
from processes.trajectory_analyzer.dm import (
    DistanceMatrixAnalyzer,
)
from processes.trajectory_analyzer.sib import (
    StructureInformedBayesAnalyzer,
    StructureInformedBayesParams,
)
from processes.trajectory_analyzer.sib_np import (
    StructureInformedBayesNeynamPearsonAnalyzer,
    StructureInformedBayesNeynamPearsonParams,
)
from processes.trajectory_analyzer.dm import (
    DistanceMatrixParams,
)
from processes.trap_extractor import TrapExtractor
from utils.utils import kprint
from scipy.stats import linregress


def plot_trapping_on_axis(
    ax,
    times: np.ndarray,  # sec
    t_min: float,
    t_max: float,
    label: str,
):
    t_min *= 1e6  # to us
    t_max *= 1e6  # to us
    times_us = times * 1e6  # to us
    Pt, bin_edges = np.histogram(times_us, bins=50, density=True)
    t = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    mask = (t >= t_min) & (t <= t_max) & (Pt > 0)
    t_part = t[mask]
    Pt_part = Pt[mask]

    log_t = np.log(t_part)
    log_S = np.log(Pt_part)

    slope, intercept, r_value, p_value, std_err = linregress(log_t, log_S)

    # alpha = -(slope + 1)
    alpha = slope

    kprint(f"{label} - alpha: {alpha:.3f}")

    # empirical
    points = ax.loglog(
        t,
        Pt,
        marker="o",
        linestyle="none",
        alpha=0.35,
    )
    color = points[0].get_color()

    # fit
    if np.isfinite(alpha) and t_part.size > 0:
        t_line = np.logspace(
            np.log10(t_part.min()), np.log10(t_part.max()), 200
        )
        S_line = np.exp(intercept) * (t_line**slope)

        ax.loglog(
            t_line,
            S_line,
            linewidth=2,
            color=color,
            label=label,
        )
        # Подписываем только SIB и Distance-matrix
        if "SIB" in label:
            idx = int(0.72 * (len(t_line) - 1))
            x_txt = t_line[idx]
            y_txt = S_line[idx]

            # текст слева от линии
            ax.annotate(
                rf"$\sim t^{{{slope:.2f}}}$",
                xy=(x_txt, y_txt),
                xytext=(15, -75),  # <-- сдвиг: влево и чуть вверх
                textcoords="offset points",
                color=color,
                fontsize=30,
                ha="right",
                va="bottom",
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.7,
                    pad=1.5,
                ),
            )

        elif "Distance-matrix" in label:
            idx = int(0.62 * (len(t_line) - 1))
            x_txt = t_line[idx]
            y_txt = S_line[idx]

            # текст справа от линии
            ax.annotate(
                rf"$\sim t^{{{slope:.2f}}}$",
                xy=(x_txt, y_txt),
                xytext=(50, -20),  # <-- сдвиг: вправо и чуть вверх
                textcoords="offset points",
                color=color,
                fontsize=30,
                ha="left",
                va="bottom",
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.7,
                    pad=1.5,
                ),
            )


def plot_trap_tim_distr(
    trap_list: list[TrapSequence], prefix, t_min, t_max, ax1
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


def get_struct_params(gas: str) -> DistanceMatrixParams:
    lmu = np.array(
        [0.5, 1.0, 1.5, 2.0, 2.5, 3.0] if gas == "CH4" else [1.5, 2.0],
        dtype=f32,
    )
    return DistanceMatrixParams(
        traj_type='fBm',
        nu=0.1,
        diag_percentile=0,
        kernel_size=0,
        list_mu=lmu,
        p_value=0.9,
    )


def run(path_to_main: str, gas: str, step: int, t_min_max, ax1):
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

    pap = StructureInformedBayesParams(1e-3)
    sib_analyzer = StructureInformedBayesAnalyzer(
        pap,
        pil_gamma_fitter,
        throat_lengths_weibull_fitter,
    )
    hybrid_analyzer = HybridAnalyzer(
        HybridParams(pap, struct_params, 0.1),
        pil_gamma_fitter,
        throat_lengths_weibull_fitter,
    )
    dm_analyzer = DistanceMatrixAnalyzer(struct_params)
    prob_np_params = StructureInformedBayesNeynamPearsonParams(1e-3, 1e-2)
    sib_np_analyzer = StructureInformedBayesNeynamPearsonAnalyzer(
        prob_np_params,
        pil_gamma_fitter,
        throat_lengths_weibull_fitter,
    )

    results: Dict[Tuple[str, int], npt.NDArray[np.bool_]] = {}
    for analyzer, prefix in [
        (dm_analyzer, "Distance-matrix"),
        (sib_np_analyzer, "SIB"),
        # (prob_analyzer, "Probabilistic"),
        (hybrid_analyzer, "Hybrid"),
    ]:
        cur_pts = join(pts_trapping, prefix)
        os.makedirs(cur_pts, exist_ok=True)

        trap_list = []

        for i, trj in enumerate(trajectories):
            if prefix == "Hybrid":
                approx_traps = results[("Distance-matrix", i)]
                assert hasattr(analyzer, "set_trap_approx")
                analyzer.set_trap_approx(approx_traps)

            seq_file = Path(join(cur_pts, f"seq_{step * i}.pickle"))
            traps_file = Path(join(cur_pts, f"traps_{step * i}.pickle"))

            if not seq_file.is_file():
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
            else:
                # print(f"load {i} {prefix}")
                with open(seq_file, 'rb') as fp:
                    seq = pickle.load(fp)
                with open(traps_file, 'rb') as fp:
                    traps = pickle.load(fp)
            results[(prefix, i)] = np.copy(traps)

            trap_list.append(seq)
        t_min, t_max = t_min_max[prefix]
        plot_trap_tim_distr(trap_list, prefix, t_min, t_max, ax1)


_DEFAULT_T_MIN_MAX: Dict[str, Dict[str, Tuple[float, float]]] = {
    "CH4": {
        "SIB": (1e-12, 5e-7),
        "Hybrid": (1e-12, 5e-7),
        "Distance-matrix": (1e-12, 8e-7),
        "SIB_Neamann-Pearson": (1e-12, 5e-7),
    },
    "H2": {
        "SIB": (1e-12, 5e-8),
        "Hybrid": (1e-12, 5e-8),
        "Distance-matrix": (1e-12, 1.5e-7),
        "SIB_Neamann-Pearson": (1e-12, 5e-8),
    },
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Trap time distribution builder")
    parser.add_argument("path", type=Path, help="Data directory")
    parser.add_argument("--label", type=str, required=True, help="Gas label (e.g. CH4, H2)")
    parser.add_argument("--num", type=int, default=1, help="Molecule step (default: 1)")
    args = parser.parse_args()

    t_min_max = _DEFAULT_T_MIN_MAX.get(args.label, _DEFAULT_T_MIN_MAX["CH4"])
    fig, ax = plt.subplots(figsize=(7, 5))
    run(str(args.path), args.label, args.num, t_min_max, ax)

        # ======================
        # Figure 1 — Survival
        # ======================
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$t, \mu s$", fontsize=20)
    ax.set_ylabel(r"$P(t)$", fontsize=20)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(frameon=False, fontsize=20)
    fig.tight_layout()
    fig.savefig(
        join(str(args.path), "traps", "P(t)_loglog.svg"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
    plt.show()
