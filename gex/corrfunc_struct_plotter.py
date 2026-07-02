import argparse
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import repeat
from pathlib import Path
from typing import List

from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from scipy.stats import linregress

from base.trajectory import Trajectory
from utils.utils import kprint

_IMAGE_PATTERN = re.compile(
    r"result-img-num=\d+_time-ps=(?P<time_ps>\d+(?:\.\d+)?).*\.npy$"
)


@dataclass
class RMSDResult:
    rmsd: np.ndarray
    t: np.ndarray


def parse_trj(value: str) -> tuple[Path, str]:
    """Parse 'path/to/trj.gro:LABEL' into (Path, label)."""
    parts = value.rsplit(":", 1)
    if len(parts) != 2 or not parts[1]:
        raise argparse.ArgumentTypeError(f"Expected path:label, got '{value}'")
    return Path(parts[0]), parts[1]


def scan_image_infos(images_dir: Path) -> list[tuple[float, str]]:
    """Scan directory for binarized .npy image files, extract time stamps from names."""
    infos = []
    for path in images_dir.iterdir():
        match = _IMAGE_PATTERN.match(path.name)
        if match is None:
            continue
        time_ps = float(match.group("time_ps"))
        infos.append((time_ps, str(path)))
    if not infos:
        raise RuntimeError(f"No matching image files found in {images_dir}")
    return sorted(infos, key=lambda x: x[0])


def extract_mean_displacement(trajectories, traj_stride: int = 1) -> RMSDResult:
    a_msd = []
    a_t = []

    for _, trj in enumerate(trajectories[::traj_stride]):
        msd = trj.msd_average_time()
        t = np.asarray(trj.times, dtype=float)
        a_t.append((t * 1e-6).astype(float))
        a_msd.append(msd)

    msd_mean = np.mean(np.stack(a_msd, axis=0), axis=0)
    rmsd = np.sqrt(msd_mean)
    return RMSDResult(rmsd, a_t[0])


def plot_corrfunc_and_md(
    dt,
    C_t,
    trj_msd_list: List[tuple[RMSDResult, str]],
    save_path: Path | None = None,
    pore_mode: bool = False,
    max_t: float = 2.8,
    fit_t_range: tuple[float, float] | None = None,
    x_max: float | None = None,
) -> None:
    dt = np.asarray(dt, dtype=float)
    C_t = np.asarray(C_t, dtype=float)
    mask = dt <= max_t
    mask[0] = False
    dt = dt[mask]
    C_t = C_t[mask]

    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_ct = default_colors[0]
    color_md = default_colors[1]

    fig, ax1 = plt.subplots(figsize=(9, 5))

    line1 = ax1.plot(
        dt,
        C_t,
        linewidth=3.0,
        color=color_ct,
        label=r"$C(t)$",
    )
    ax1.set_xlabel(r"Time delay, $\mu$s", fontsize=20)
    
    ax1.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax1.tick_params(axis="both", labelsize=16)
    ax1.tick_params(axis="y", labelcolor=color_ct)

    lines = line1
    styles = ['-', ':', '--', '-.', '.']
    _ann_fracs = [0.30, 0.55, 0.75]
    ax2 = ax1.twinx()
    for curve_idx, (res, prefix) in enumerate(trj_msd_list):
        label = prefix + " " + r"$RMSD(t)$"

        r = res.rmsd
        t = res.t

        mask = t <= max_t
        mask[0] = False
        t = t[mask]
        r = r[mask]

        line = ax2.plot(
            t,
            r,
            linewidth=3.0,
            color=color_md,
            label=label,
            linestyle=styles.pop(0),
        )
        lines += line

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
            r_line = np.exp(intercept) * t_line**slope

            count = len(t_line) // 2 - 50
            t_line = t_line[count:]
            r_line = r_line[count:]

            ax2.plot(t_line, r_line, '--', linewidth=1.5, color=color_md)
            frac = _ann_fracs[min(curve_idx, len(_ann_fracs) - 1)]
            ai = int(frac * (len(t_line) - 1))
            ax2.annotate(
                rf"$\sim t^{{{slope:.2f}}}$",
                xy=(t_line[ai], r_line[ai]),
                xytext=(20, -40),
                textcoords="offset points",
                color=color_md,
                fontsize=24,
                ha="center",
                va="bottom",
                bbox=dict(
                    facecolor="white", edgecolor="none", alpha=0.7, pad=1.5
                ),
            )

    ax2.set_yscale("log")
    ax2.set_xscale("log")
    ax1.set_xscale("log")
    if x_max is not None:
        ax1.set_xlim(right=x_max)
        ax2.set_xlim(right=x_max)
    ax2.set_ylabel(r"$\mathrm{RMSD}(t)$, nm", fontsize=20, color=color_md)
    ax2.tick_params(axis="both", labelsize=16)
    ax2.tick_params(axis="y", labelcolor=color_md)

    positive_r = []
    for res, _ in trj_msd_list:
        r = np.asarray(res.rmsd, dtype=float)
        positive_r.extend(r[np.isfinite(r) & (r > 0)])

    if positive_r:
        ax2.set_ylim(bottom=0.09)
        ax2.set_ylim(top=max(positive_r) * 10)

    labels = [line.get_label() for line in lines]
    ax1.legend(
        lines,
        labels,
        loc="upper right",
        borderaxespad=0.0,
        fontsize=18,
        frameon=False,
    )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def correlation_average_time(
    image_infos, load_img, ct_save_path: Path, num_workers: int = 4
):
    """
    Time-averaged autocorrelation function:

        C(tau_k) = mean_i [ sum_x I_i(x) I_{i+k}(x) / V ]
    """
    image_infos = sorted(image_infos, key=lambda x: x[0])

    times_ps = np.array([info[0] for info in image_infos], dtype=np.float64)
    img_files = [info[1] for info in image_infos]

    n = len(image_infos)
    dt = (times_ps - times_ps[0]) * 1e-6  # ps → μs
    kprint(f"dt range: {dt[1]:.6f} … {dt[-1]:.3f} μs  ({n} frames)")

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
        np.save(ct_save_path, C_t)
        kprint(
            f"Ready lag: {lag} from {n}, C: {C_t[lag]} in {time.time() - start_time:.2f} sec"
        )

    return dt, C_t


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute time-averaged autocorrelation C(t) from binarized images "
            "and plot together with RMSD(t) trajectories."
        )
    )
    parser.add_argument(
        "images_dir",
        type=Path,
        help="Directory containing binarized .npy images (output of binarization_structs).",
    )
    parser.add_argument(
        "ct_file",
        type=Path,
        help="Path to save/load the C(t) cache (.npy). Computed incrementally.",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output figure path (e.g. figs/corrfunc.svg).",
    )
    parser.add_argument(
        "--trj",
        action="append",
        type=parse_trj,
        required=True,
        metavar="PATH:LABEL",
        help=(
            "Trajectory .gro file with a display label, e.g. trj.gro:CH4. "
            "Can be repeated for multiple gases."
        ),
    )
    parser.add_argument(
        "--pore",
        action="store_true",
        help="Pore mode: invert images (1-img) before computing C(t).",
    )
    parser.add_argument(
        "--max-t",
        type=float,
        default=2.8,
        metavar="US",
        help="Maximum time in μs shown on the plot (default: 2.8).",
    )
    parser.add_argument(
        "--x-max",
        type=float,
        default=None,
        metavar="US",
        help="Right X-axis display limit in μs (default: auto from data).",
    )
    parser.add_argument(
        "--fit-t-range",
        type=float,
        nargs=2,
        metavar=("T_MIN", "T_MAX"),
        help="Time range in μs for power-law regression (default: full range).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of threads for C(t) computation (default: 4).",
    )

    args = parser.parse_args()

    images_dir: Path = args.images_dir
    ct_file: Path = args.ct_file
    output: Path = args.output
    pore_mode: bool = args.pore

    image_infos = scan_image_infos(images_dir)
    kprint(f"Found {len(image_infos)} image files in {images_dir}")

    def load_img(file_name: str) -> np.ndarray:
        img = np.load(file_name, mmap_mode="r")
        if pore_mode:
            img = 1 - img
        return img.astype(np.int8)

    ct_file.parent.mkdir(parents=True, exist_ok=True)
    dt, C_t = correlation_average_time(
        image_infos, load_img, ct_file, args.num_workers
    )
    kprint(f"C(t) time range: {dt[0]:.4f} … {dt[-1]:.4f} μs")

    trj_msd_list = []
    for trj_path, label in args.trj:
        trajectories = Trajectory.read_trajectoryes(trj_path)
        res = extract_mean_displacement(trajectories)
        trj_msd_list.append((res, label))

    fit_t_range = tuple(args.fit_t_range) if args.fit_t_range else None

    output.parent.mkdir(parents=True, exist_ok=True)
    plot_corrfunc_and_md(
        dt=dt,
        C_t=C_t,
        trj_msd_list=trj_msd_list,
        save_path=output,
        pore_mode=pore_mode,
        max_t=args.max_t,
        fit_t_range=fit_t_range,
        x_max=args.x_max,
    )


if __name__ == "__main__":
    main()
