import argparse
import json
import subprocess
import sys
import time
from os import listdir
from os.path import dirname, isfile, join, realpath
from pathlib import Path
from typing import Any, List, Tuple
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import exponweib, weibull_min

from processes.pil_distr_generator import PiLDistrGenerator

from base.reader import Reader
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def compute_density(
    data: np.ndarray,
    bins: int = 50,
    smooth: bool = True,
    window_length: int = 25,
    polyorder: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Строит нормированную гистограмму-плотность.
    Возвращает:
        x  - центры бинов
        pn - значения плотности
    """
    p, bb = np.histogram(data, bins=bins, density=False)
    xdel = bb[1] - bb[0]
    x = bb[:-1] + 0.5 * xdel
    pn = p / np.sum(p * xdel)

    if smooth and len(pn) >= 7:
        # window_length должен быть нечётным и <= len(pn)
        wl = min(window_length, len(pn) if len(pn) % 2 == 1 else len(pn) - 1)
        if wl >= polyorder + 2 and wl % 2 == 1:
            pn = savgol_filter(pn, 31, 3)  # wl, polyorder)
            pn[pn < 0] = 0.0

    x = x[:-10]
    pn = pn[:-10]
    mask = x < 1.2
    x = x[mask]
    pn = pn[mask]

    return x, pn


def plot_fill(x, p, time_us, color):
    verts_l = [(x, time_us, z) for x, z in zip(x, p)]
    verts_l += [(x[-1], time_us, 0.0), (x[0], time_us, 0.0)]

    return Poly3DCollection(
        [verts_l],
        facecolors=color,
        alpha=0.12,
        edgecolors="none",
    )


def get_sorted_pnm_files(path_to_pnms: str) -> list[tuple[int, str]]:
    onlyfiles = [
        f for f in listdir(path_to_pnms) if isfile(join(path_to_pnms, f))
    ]
    onlyfiles = [
        file
        for file in onlyfiles
        if "_link1" in file and file.startswith("num")
    ]

    steps = [int((file.split("=")[1]).split("_")[0]) for file in onlyfiles]
    return sorted(zip(steps, onlyfiles), key=lambda x: x[0])


def convert_step_to_time_us(step: int) -> float:
    ps = int(50 + (step - 25000.0) * 500.0 / 250000.0)

    return ps * 1e-6


def build_3d_distributions(
    paths: Tuple[str, str, str],
    pnm_step: int = 10,
    bins: int = 50,
    smooth: bool = True,
    border: float = 0.015,
    scale: float = 1e10,
) -> None:
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Фиксированные цвета по типу распределения
    color_throat = "tab:blue"  # P(l)
    color_pore = "tab:orange"  # P(r)

    is_fill = True

    label_added_pl = False
    label_added_pr = False

    path_to_pnms, dataset_label, path_to_save = paths
    sorted_files = get_sorted_pnm_files(path_to_pnms)

    for step, file in sorted_files[::pnm_step]:
        base_path = join(path_to_pnms, file[:-10])
        radiuses, throat_lengths = Reader.read_pnm_data(
            base_path,
            scale=scale,
            border=border,
        )

        time_us = convert_step_to_time_us(step)
        print(f"{dataset_label}: step={step}, time={time_us:.2f} ps")

        # P(l)
        x_l, p_l = compute_density(
            throat_lengths,
            bins=bins,
            smooth=smooth,
        )
        y_l = np.full_like(x_l, time_us, dtype=float)
        ax.plot(
            x_l,
            y_l,
            p_l,
            color=color_throat,
            label=r"$P(h)$" if not label_added_pl else None,
            linewidth=2.5,
        )
        ax.text(
            x_l[-1] - 0.15,  # небольшой сдвиг вправо
            y_l[-1],  # то же время
            p_l[-1] + 0.5,  # конец кривой по z
            rf"{time_us:.2f}",
            color=color_throat,
            fontsize=12,
            ha="left",
            va="center",
        )
        label_added_pl = True

        # P(r)
        x_r, p_r = compute_density(
            radiuses,
            bins=bins,
            smooth=smooth,
        )
        p_r = p_r * 0.5
        y_r = np.full_like(x_r, time_us, dtype=float)
        ax.plot(
            x_r,
            y_r,
            p_r,
            color=color_pore,
            label=r"$P(r)$" if not label_added_pr else None,
            linewidth=2.5,
        )
        label_added_pr = True

        if is_fill:
            collection_pl_fill = plot_fill(x_l, p_l, time_us, color_throat)
            collection_pr_fill = plot_fill(x_r, p_r, time_us, color_pore)
            ax.add_collection3d(collection_pl_fill)
            ax.add_collection3d(collection_pr_fill)
        else:
            ax.plot(
                x_l,
                y_l,
                np.zeros_like(p_l),
                color=color_throat,
                alpha=0.22,
                linewidth=1.2,
            )
            ax.plot(
                x_r,
                y_r,
                np.zeros_like(p_r),
                color=color_pore,
                alpha=0.22,
                linewidth=1.2,
            )

    if not is_fill:
        y_back = ax.get_ylim()[1]
        ax.plot(
            x_l,
            np.full_like(x_l, y_back),
            p_l,
            color=color_throat,
            alpha=0.15,
            linewidth=1.0,
        )

    ax.set_xlabel(r"$r, h (\AA)$", fontsize=16, labelpad=-8)
    ax.set_ylabel(r"Time ($\mu$s)", fontsize=16, labelpad=-10)
    ax.set_zlabel("Density", fontsize=16, labelpad=-6)

    xlabels = ax.get_xticklabels()
    res_xlabels = []
    for i, x in enumerate(xlabels):
        if i % 2 != 1:
            x._text = ""
        res_xlabels.append(x)

    ax.set_xticklabels(xlabels)
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.tick_params(axis="z", labelsize=11)

    ax.view_init(elev=15, azim=-86)
    ax.legend(frameon=False, fontsize=12)

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # белые плоскости
    ax.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))

    # можно ещё убрать серые границы плоскостей
    ax.xaxis.pane.set_edgecolor("white")
    ax.yaxis.pane.set_edgecolor("white")
    ax.zaxis.pane.set_edgecolor("white")
    ax.grid(False)

    # ax.xaxis._axinfo["juggled"] = (1, 0, 2)
    # ax.yaxis._axinfo["juggled"] = (0, 1, 2)
    # ax.zaxis._axinfo["juggled"] = (0, 2, 1)

    ax.xaxis._axinfo["juggled"] = (1, 0, 2)
    ax.yaxis._axinfo["juggled"] = (0, 1, 2)
    ax.zaxis._axinfo["juggled"] = (1, 2, 0)
    ax.set_proj_type("ortho")

    for e, a in [(22, -72), (28, -65), (30, -58), (20, -78)]:
        ax.view_init(elev=e, azim=a)
        f_name = "fill" if is_fill else "proj"
        fig.savefig(
            path_to_save + f"{f_name}_elev_{e}_azim_{a}.svg",
            bbox_inches="tight",
        )

    # fig.savefig("all_distributions_3d.svg", bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    build_3d_distributions(
        (
            "/media/andrey/Samsung_T5/PHD/Kerogen/type1matrix/300K/ch4/pnm/",
            r"$CH_4$",
            "/media/andrey/Samsung_T5/PHD/Kerogen/type1matrix/300K/ch4/figs/",
        ),
        # (
        #     "/media/andrey/Samsung_T5/PHD/Kerogen/type1matrix/300K/h2/pnm/",
        #     r"$H_2$",
        # ),
        # (
        #     "/media/andrey/Samsung_T5/PHD/Kerogen/type1matrix/300K/ch4/pnm/",
        #     "type1-300K-CH4",
        # ),
        # (
        #     "/media/andrey/Samsung_T5/PHD/Kerogen/type1matrix/300K/h2/pnm/",
        #     "type1-300K-H2",
        # ),
        # (
        #     "/media/andrey/Samsung_T5/PHD/Kerogen/type1matrix/400K/ch4/pnm/",
        #     "type1-400K-CH4",
        # ),
        # (
        #     "/media/andrey/Samsung_T5/PHD/Kerogen/type1matrix/400K/h2/pnm/",
        #     "type1-400K-H2",
        # ),
        # (
        #     "/media/andrey/Samsung_T5/PHD/Kerogen/type2matrix/300K/ch4/pnm/",
        #     "type2-300K-CH4",
        # ),
        # (
        #     "/media/andrey/Samsung_T5/PHD/Kerogen/type2matrix/300K/h2/pnm/",
        #     "type2-300K-H2",
        # ),
        pnm_step=30,
        bins=50,
        smooth=True,
    )
