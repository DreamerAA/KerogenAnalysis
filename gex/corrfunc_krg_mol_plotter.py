import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
import pickle

from itertools import repeat

from pathlib import Path
import time

from matplotlib import pyplot as plt
import numpy as np
from base.trajectory import Trajectory
from utils.utils import kprint
from scipy.stats import linregress


@dataclass
class RMSDResult:
    rmsd: np.ndarray
    t: np.ndarray


def extract_mean_displacement(trajectories, traj_stride: int = 1) -> RMSDResult:
    a_msd = []
    a_t = []

    for i, trj in enumerate(trajectories[::traj_stride]):
        msd = trj.msd_average_time()
        t = np.asarray(trj.times, dtype=float)

        a_t.append((t * 1e-6).astype(float))
        a_msd.append(msd)

    msd_mean = np.mean(np.stack(a_msd, axis=0), axis=0)
    rmsd = np.sqrt(msd_mean)
    # rmsd = msd_mean

    return RMSDResult(rmsd, a_t[0])


def plot_corrfunc_and_md(
    trj_msd: RMSDResult,
    save_path: Path,
    max_t: float = 2.8,
    fit_t_range: tuple[float, float] | None = None,
) -> None:
    fig, ax1 = plt.subplots(figsize=(9, 5))

    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_msd = default_colors[3]

    r = trj_msd.rmsd
    t = trj_msd.t
    mask = t <= max_t
    mask[0] = False
    t = t[mask]
    r = r[mask]

    plt.plot(
        t,
        r,
        color=color_msd,
        linewidth=3.0,
    )

    plt.yscale("log")
    plt.xscale("log")

    positive_r = r[np.isfinite(r) & (r > 0)]
    plt.ylim(bottom=min(positive_r) * 0.8, top=max(positive_r) * 5)

    ax = plt.gca()

    fit_mask = np.isfinite(t) & np.isfinite(r) & (t > 0) & (r > 0)
    if fit_t_range is not None:
        fit_mask &= (t >= fit_t_range[0]) & (t <= fit_t_range[1])
    if fit_mask.sum() >= 2:
        slope, intercept, *_ = linregress(
            np.log(t[fit_mask]), np.log(r[fit_mask])
        )
        t_line = np.logspace(
            np.log10(t[fit_mask].min()), np.log10(t[fit_mask].max()), 200
        )
        count = t_line.shape[0]
        r_line = np.exp(intercept) * t_line**slope
        start = count // 2 - 5
        t_line = t_line[start:]
        r_line = r_line[start:]
        ax.plot(t_line, r_line, '--', linewidth=1.5, color=color_msd)
        ai = int(0.55 * (len(t_line) - 1))
        ax.annotate(
            rf"$\sim t^{{{slope:.2f}}}$",
            xy=(t_line[ai], r_line[ai]),
            xytext=(0, -40),
            textcoords="offset points",
            color=color_msd,
            fontsize=24,
            ha="center",
            va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.5),
        )

    # Оставляем рамку целиком
    for spine in ax.spines.values():
        spine.set_visible(True)

    # Название оси Y переносим вправо
    ax.yaxis.set_label_position("right")

    plt.xlabel(r"Time delay, $\mu$s", fontsize=16)
    plt.ylabel(r"$\mathrm{RMSD}(t)$, nm", fontsize=16)

    # Убираем тики и подписи слева
    ax.tick_params(
        axis="y",
        which="both",
        left=False,
        labelleft=False,
        right=True,
        labelright=True,
        labelsize=12,
    )

    plt.legend(fontsize=16, frameon=False)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_kerogen_molecula_coorfunc(input: Path, output: Path, msd_path: Path):
    trajectories = Trajectory.read_trajectoryes(input)

    msd_path.parent.mkdir(parents=True, exist_ok=True)

    if msd_path.exists():
        with open(msd_path, "rb") as f:
            trj_msd = pickle.load(f)
    else:
        trj_msd = extract_mean_displacement(trajectories, traj_stride=1)
        with open(msd_path, "wb") as f:
            pickle.dump(trj_msd, f)

    plot_corrfunc_and_md(
        trj_msd,
        save_path=output,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=Path, help="Input trajectory file")
    parser.add_argument("output", type=Path, help="Output figure path")
    parser.add_argument("msd", type=Path, help="Output MSD pickle path")

    args = parser.parse_args()

    plot_kerogen_molecula_coorfunc(
        args.input,
        args.output,
        args.msd,
    )
