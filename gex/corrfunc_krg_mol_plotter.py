import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os

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
    save_path: str | Path | None = None,
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
        slope, intercept, *_ = linregress(np.log(t[fit_mask]), np.log(r[fit_mask]))
        t_line = np.logspace(
            np.log10(t[fit_mask].min()), np.log10(t[fit_mask].max()), 200
        )
        r_line = np.exp(intercept) * t_line**slope
        ax.plot(t_line, r_line, '--', linewidth=1.5, color=color_msd)
        ai = int(0.55 * (len(t_line) - 1))
        ax.annotate(
            rf"$\sim t^{{{slope:.2f}}}$",
            xy=(t_line[ai], r_line[ai]),
            xytext=(0, -22),
            textcoords="offset points",
            color=color_msd,
            fontsize=16,
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

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def correlation_average_time(
    image_infos, load_img, ct_save_path: str | Path, num_workers: int = 4
):
    """
    Time-averaged autocorrelation function:

        C(tau_k) = mean_i [ sum_x I_i(x) I_{i+k}(x) / V ]

    image_infos:
        list of tuples (time_ps, img_file_name)

    load_img:
        function that loads image by filename

    save_path:
        path where ct, dt array will be saved
    """
    image_infos = sorted(image_infos, key=lambda x: x[0])

    times_ps = np.array([info[0] for info in image_infos], dtype=np.float64)
    img_files = [info[1] for info in image_infos]

    n = len(image_infos)
    dt = (times_ps - times_ps[0]) * 1e-6
    print("start dt = ", dt[:5])
    print("end dt = ", dt[-5:])

    if os.path.exists(ct_save_path):
        C_t = np.load(ct_save_path)
    else:
        C_t = np.full(n, np.nan, dtype=np.float64)

    def process(chunk_i, lag):
        result = np.full(len(chunk_i), np.nan, dtype=np.float64)
        for local_idx, i in enumerate(chunk_i):
            img_i = load_img(img_files[i])
            image_size = float(np.sum(img_i))
            img_j = load_img(img_files[i + lag])
            result[local_idx] = np.sum(img_i * img_j) / image_size
        return result

    for lag in range(n):
        start_time = time.time()
        if not np.isnan(C_t[lag]):
            kprint(f"Skip lag: {lag} from {n}, C: {C_t[lag]}")
            continue
        xdata = np.array(list(range(n - lag)), dtype=np.int32)
        values = np.zeros(shape=(n - lag), dtype=np.float64)

        if num_workers == 0:
            values = process(xdata, lag)
        else:
            chunk_size = ((n - lag) // num_workers) + 1
            chunks = [
                xdata[i : min(i + chunk_size, len(xdata))]
                for i in range(0, len(xdata), chunk_size)
            ]
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                chunk_results = list(executor.map(process, chunks, repeat(lag)))
                values = np.concatenate(chunk_results)

        C_t[lag] = np.mean(values)

        # Сохраняем промежуточное состояние после каждого lag
        np.save(ct_save_path, C_t)

        kprint(
            f"Ready lag: {lag} from {n}, C: {C_t[lag]} in {time.time() - start_time:.2f} sec"
        )

    return dt, C_t


def plot_kerogen_molecula_coorfunc(input: Path, output: Path):
    trajectories = Trajectory.read_trajectoryes(input)

    trj_msd = extract_mean_displacement(trajectories, traj_stride=1)

    plot_corrfunc_and_md(
        trj_msd,
        save_path=output,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=Path, help="Input trajectory file")
    parser.add_argument("output", type=Path, help="Output figure path")

    args = parser.parse_args()

    plot_kerogen_molecula_coorfunc(
        args.input,
        args.output,
    )
