import argparse
import os

import time
from os.path import join

from typing import List

from matplotlib import pyplot as plt
import numpy as np
from base.kerogendata import KerogenData
from base.periodizer import Periodizer
from base.reader import Reader
from base.trajectory import Trajectory
from utils.utils import kprint
from processes.segmentaion import Segmentator

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


def get_size(type_id: int) -> float:
    return atom_real_sizes[type_id]


def get_ext_size(type_id: int) -> float:
    return ext_radius[type_id]


def extract_mean_displacement(
    trajectories, traj_stride: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    a_msd = []
    a_t = []

    for i, trj in enumerate(trajectories[::traj_stride]):
        msd = trj.msd_average_time()
        t = np.asarray(trj.times, dtype=float)

        a_t.append((t * 1e-6).astype(float))
        a_msd.append(msd)

    msd_mean = np.mean(np.stack(a_msd, axis=0), axis=0)
    rmsd = np.sqrt(msd_mean)

    return rmsd, a_t[0]


def plot_corrfunc_and_md(
    dt,
    C_t,
    ddata: tuple[np.ndarray, np.ndarray],
    save_path: str | None = None,
    pore_mode: bool = False,
) -> None:
    dt = np.asarray(dt, dtype=float)
    C_t = np.asarray(C_t, dtype=float)

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

    ax2 = ax1.twinx()
    line2 = ax2.plot(
        ddata[1],
        ddata[0],
        linewidth=3.0,
        color=color_md,
        label=r"$\sqrt{MSD(t)}$",
    )
    ax2.set_ylabel(
        r"$\sqrt{\mathrm{MSD}(t)}$, $\AA$", fontsize=16, color=color_md
    )
    ax2.tick_params(axis="both", labelsize=12)
    ax2.tick_params(axis="y", labelcolor=color_md)
    ax2.set_ylim(bottom=0)

    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax1.legend(
        lines,
        labels,
        loc="lower right",
        # bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=16,
        frameon=False,
    )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def extanded_struct_extr(
    path_to_data: str,
    struct_file_name: str,
    indexes: List[int],
    ref_size: int,
) -> None:
    path_to_structure = join(path_to_data, struct_file_name)
    path_to_save_img = join(path_to_data, "images")
    path_to_save_fig = join(path_to_data, "figs", "corrfunc+msd.svg")
    path_to_trj = join(path_to_data, "trj.gro")

    start_time = time.time()
    structures = Reader.read_structures_by_num(path_to_structure, indexes)
    print(f" -- Count structures: {len(structures)}")
    print(f" -- Reading finished! Elapsed time: {time.time() - start_time}s")

    image_infos: tuple[int, str] = []
    for num, time_ps, atoms, size in structures:
        start_time = time.time()

        bbox = Segmentator.cut_cell(size, 2)
        resolution = np.array([s for s in size]).min() / ref_size
        img_size = Segmentator.calc_image_size(
            bbox.size(), reference_size=ref_size, by_min=True
        )
        binarized_file_name = join(
            path_to_save_img,
            f"result-img-num={num}_time-ps={time_ps}_bbox={bbox._short_str()}_resolution={resolution:.9f}.npy",
        )
        image_infos.append((time_ps, binarized_file_name))

        # print(f" --- Current num: {num}")
        # print(f" --- Box size: {bbox.size()}")

        if os.path.isfile(binarized_file_name):
            kprint(f"Skip and load image with num={num}")
            with open(binarized_file_name, 'rb') as f:  # type: ignore
                img = np.load(f)  # type: ignore
        else:
            kprint(f"Run segmentation for num={num}")
            start_time = time.time()
            kerogen_data = KerogenData(None, atoms, bbox)  # type: ignore
            if not kerogen_data.checkPeriodization():
                Periodizer.periodize(kerogen_data)

            segmentator = Segmentator(
                kerogen_data,
                img_size,
                size_data=get_size,
                radius_extention=get_ext_size,
                partitioning=2,
            )
            img = 1 - segmentator.binarize()
            np.save(binarized_file_name, img)  # type: ignore

            kprint(
                f"Binarization struct {num} is finished! Elapsed time: {time.time() - start_time}s"
            )

    pore_mode = True

    def load_img(file_name: str) -> np.ndarray:

        with open(file_name, 'rb') as f:  # type: ignore
            img = np.load(f)  # type: ignore
        if pore_mode:
            img = 1 - img
        return img.astype(np.int8)

    start_info = image_infos[0]
    start_time_ps = start_info[0]
    first_step_img = load_img(start_info[1])
    image_size = float(np.sum(first_step_img, dtype=np.float32))

    assert np.all(first_step_img <= 1)

    dt = []
    C_t = []
    for info in image_infos:
        time_ps, img_file_name = info
        img_t = load_img(img_file_name)
        assert np.all(img_t <= 1)
        C_ti = np.sum(first_step_img * img_t) / image_size
        C_t.append(C_ti)
        dt.append((time_ps - start_time_ps) * 1e-6)
    kprint(f"C(t) min time {dt[0]}, max time {dt[-1]}")

    trajectories = Trajectory.read_trajectoryes(path_to_trj)
    ddata = extract_mean_displacement(trajectories, traj_stride=2)

    plot_corrfunc_and_md(
        dt=dt,
        C_t=C_t,
        ddata=ddata,
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

    parser.add_argument(
        '--structure_file_name',
        type=str,
        default="type1.h2.300.gro",
    )

    args = parser.parse_args()

    start_step = 25000
    full_count_steps = 6612
    step_size = 250000
    last_step = full_count_steps * step_size + start_step

    count_slices = 200

    mode = "all"
    if mode == "all":
        step = int(full_count_steps / count_slices)
        indexes = [
            start_step + step_size * i * step for i in range(count_slices)
        ]
        if last_step not in indexes:
            indexes.append(last_step)
    else:
        step = 1
        indexes = [
            start_step + step_size * i * step for i in range(count_slices)
        ]

    extanded_struct_extr(
        args.def_path,
        args.structure_file_name,
        indexes,
        250,
    )
