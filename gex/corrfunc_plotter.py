import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os

from itertools import repeat
from os.path import join

from pathlib import Path
import time
from typing import List

from matplotlib import pyplot as plt
import numpy as np
from base.trajectory import Trajectory
from utils.utils import kprint

additional_radius = 0.0

atom_real_sizes = {
    i: s for i, s in enumerate([0.17, 0.152, 0.155, 0.109, 0.18])
}

ext_radius = {
    i: s
    for i, s in enumerate(
        [
            additional_radius,
            additional_radius,
            additional_radius,
            0.0,
            additional_radius,
        ]
    )
}


@dataclass
class RMSDResult:
    rmsd: np.ndarray
    t: np.ndarray


def get_size(type_id: int) -> float:
    return atom_real_sizes[type_id]


def get_ext_size(type_id: int) -> float:
    return ext_radius[type_id]


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
    dt,
    C_t,
    trj_msd_list: List[tuple[RMSDResult, str]],
    save_path: str | None = None,
    pore_mode: bool = False,
    max_t: float = 1.5,
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

    fig, ax1 = plt.subplots(figsize=(8.5, 5.5))

    line1 = ax1.plot(
        dt,
        C_t,
        linewidth=3.0,
        color=color_ct,
        label=r"$C(t)$",
    )
    y_name = r"Autocorrelation $C(t)$ of "
    y_name += " pore" if pore_mode else " solid"
    ax1.set_xlabel(r"Time delay, $\mu$s", fontsize=16)
    ax1.set_ylabel(y_name, fontsize=16)
    ax1.tick_params(axis="both", labelsize=12)
    ax1.tick_params(axis="y", labelcolor=color_ct)
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)
    # ax1.grid(True, linestyle="--", alpha=0.4)

    lines = line1
    styles = ['-', ':', '--', '-.', '.']
    ax2 = ax1.twinx()
    for res, prefix in trj_msd_list:
        label = prefix + " "
        label += r"$RMSD(t)$"

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

    ax2.set_yscale("log")
    ax2.set_ylabel(r"$\mathrm{RMSD}(t)$, $\AA$", fontsize=16, color=color_md)
    ax2.tick_params(axis="both", labelsize=12)
    ax2.tick_params(axis="y", labelcolor=color_md)
    ax2.set_ylim(bottom=0)

    labels = [line.get_label() for line in lines]
    ax1.legend(
        lines,
        labels,
        loc="center right",
        # bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=16,
        frameon=False,
    )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

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


def extanded_struct_extr(
    path_to_data: str,
    path_to_trjs: list[tuple[str, str]],
    template: str,
    indexes: list[int],
    times: list[int],
    ref_size: int,
    prefix: str,
) -> None:
    path_to_img = join(path_to_data, "images")
    path_to_save_fig = join(path_to_data, "figs", "corrfunc+msd.svg")

    filenames = [
        join(path_to_img, template.format(num=n, time=t))
        for n, t in zip(indexes, times)
    ]
    image_infos = list(zip(times, filenames))

    pore_mode = True

    def load_img(file_name: str) -> np.ndarray:
        if not os.path.isfile(file_name):
            raise FileNotFoundError(file_name)
        img = np.load(file_name, mmap_mode="r")  # type: ignore
        if pore_mode:
            img = 1 - img
        if ref_size not in img.shape:
            raise ValueError(f"ref_size {ref_size} not in {img.shape}")
        return img.astype(np.int8)

    c_t_save_path = join(path_to_data, prefix + "_ct.npy")
    dt, C_t = correlation_average_time(image_infos, load_img, c_t_save_path, 8)
    kprint(f"C(t) min time {dt[0]}, max time {dt[-1]}")

    trj_msd_list: list[RMSDResult, str] = []
    for ptrj, gas in path_to_trjs:
        trajectories = Trajectory.read_trajectoryes(ptrj)
        res = extract_mean_displacement(trajectories, traj_stride=1)
        trj_msd_list.append((res, gas))

    plot_corrfunc_and_md(
        dt=dt,
        C_t=C_t,
        trj_msd_list=trj_msd_list,
        save_path=path_to_save_fig,
        pore_mode=pore_mode,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--def_path',
        type=str,
        default="/media/andrey/Samsung_T5/PHD/Kerogen/type1matrix/300K/h2/",
    )
    def_template = (
        "result-img-num={num}_time-ps={time}"
        "_bbox=(x=(0.000-6.231)_y=(0.590-6.821)_z=(3.392-9.623))"
        "_resolution=0.024923800.npy"
    )

    parser.add_argument(
        '--path_temp_case',
        type=str,
        default="/media/andrey/Samsung_T5/PHD/Kerogen/type1matrix/300K/",
    )

    parser.add_argument(
        '--template',
        type=str,
        default=def_template,
    )

    args = parser.parse_args()

    full_count_steps = 6612

    start_time = 50
    step_time_size = 500

    start_step = 25000
    full_count_steps = 6612
    step_size = 250000

    last_step = full_count_steps * step_size + start_step
    last_time_step = full_count_steps * step_time_size + start_time

    count_slices = 500

    def gen_list(start, step_size, step, count):
        return [start + step_size * i * step for i in range(count)]

    mode = "all"
    # mode = "part"
    if mode == "all":
        step = int(full_count_steps / count_slices)
        indexes = gen_list(start_step, step_size, step, count_slices)
        times = gen_list(start_time, step_time_size, step, count_slices)
        if last_step not in indexes:
            indexes.append(last_step)
            times.append(last_time_step)
    else:
        step = 1
        indexes = gen_list(start_step, step_size, step, count_slices)
        times = gen_list(start_time, step_time_size, step, count_slices)

    trj_paths = [
        (
            join(args.path_temp_case, gas, "trj.gro"),
            sgas,
        )
        for gas, sgas in [("ch4", r"$CH_4$"), ("h2", r"$H_2$")]
    ]

    extanded_struct_extr(
        args.def_path,
        trj_paths,
        args.template,
        indexes,
        times,
        250,
        f"mode={mode}_count-slices={count_slices}",
    )
