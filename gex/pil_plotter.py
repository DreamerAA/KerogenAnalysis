import pickle
import sys
from os.path import join, realpath
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import exponweib

path = Path(realpath(__file__))
parent_dir = str(path.parent.parent.absolute())
sys.path.append(parent_dir)

from processes.pil_distr_generator import PiLDistrGenerator


def plot_distributions(path_to_save: str):
    radiuses = np.load(join(path_to_save, "radiuses.npy"))
    radiuses = radiuses[radiuses < 0.22]

    generator = PiLDistrGenerator()
    sample_rad, l_vals, pi_cond = generator.get_conditional_curves(
        radiuses, step=150
    )

    # --- fit P(r) ---
    params = exponweib.fit(radiuses[radiuses > 0], floc=0)
    r_fit = np.linspace(float(sample_rad.min()), float(sample_rad.max()), 300)
    pr_fit = exponweib.pdf(r_fit, *params)

    # --- mask zeros in Π(l|r) so they are transparent ---
    pi_masked = np.ma.masked_where(pi_cond == 0, pi_cond)

    fig, ax = plt.subplots(figsize=(9, 7), facecolor="white")
    ax.set_facecolor("white")

    # --- upper panel in the same axes: heatmap Π(l|r), y >= 0 ---
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(alpha=0)

    mesh = ax.pcolormesh(
        sample_rad,
        l_vals,
        pi_masked.T,
        cmap=cmap,
        shading="auto",
    )

    # --- lower schematic part: scaled P(r), y < 0 ---
    pr_height = 0.25 * float(l_vals.max())
    y_pr = -(pr_fit / pr_fit.max()) * pr_height

    ax.fill_between(r_fit, y_pr, 0.0, color="orange", alpha=0.35, linewidth=0)
    ax.plot(r_fit, y_pr, color="darkorange", linewidth=1.5)

    # divider / x-axis at y = 0
    ax.axhline(0.0, color="black", linewidth=1.2)
    y_tick_text = -0.025 * float(l_vals.max())  # чуть ниже линии y=0
    tick_len = 0.01 * float(l_vals.max())
    x_center = 0.5 * (float(sample_rad.min()) + float(sample_rad.max()))
    y_xlabel = -0.10 * float(l_vals.max())

    # --- limits ---
    x_start = 0.03
    ax.set_xlim(x_start, float(sample_rad.max()))
    ax.set_ylim(float(y_pr.min()) * 1.15, float(l_vals.max()) * 1.03)

    # --- x axis ---
    xticks = np.linspace(float(sample_rad.min()), float(sample_rad.max()), 5)
    ax.set_xticks(xticks)
    ax.set_xticklabels([])
    ax.set_xlabel("")
    ax.tick_params(axis="x", length=0)

    ax.text(
        float(sample_rad.max()),
        y_xlabel,
        r"$r (\AA)$",
        fontsize=24,
        ha="right",
        va="top",
    )

    for i, x in enumerate(xticks):
        shift = (xticks.max() - xticks.min()) / len(xticks) / 3
        if i == 0:
            x += shift
        elif i + 1 == len(xticks):
            x -= shift
        ax.text(
            x,
            y_tick_text,
            f"{x:.2f}",
            fontsize=16,
            ha="center",
            va="top",
        )
        ax.plot(
            [x, x],
            [0.0, -tick_len],
            color="black",
            linewidth=1.0,
        )

    # --- y axis: show ticks only for l > 0 ---
    yticks_top = np.linspace(0.1, float(l_vals.max()), 5)
    ax.set_yticks(yticks_top)
    ax.set_yticklabels([f"{y:.1f}" for y in yticks_top], fontsize=16)
    ax.set_ylabel(r"$l (\AA)$", fontsize=24)

    # --- clean spines ---
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # --- manual label for lower schematic P(r) part ---
    x_text = x_start - 0.035 * (
        float(sample_rad.max()) - float(sample_rad.min())
    )
    y_text = 0.55 * float(y_pr.min())

    ax.text(
        x_text,
        y_text,
        r"$P(r)$",
        fontsize=24,
        rotation=90,
        va="center",
        ha="center",
    )

    # --- colorbar ---
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.06)
    cbar.set_label(r"$\Pi(l \mid r)$", fontsize=20)
    cbar.ax.tick_params(labelsize=14)

    fig.savefig(
        join(path_to_save, "figs", "pilr_2d.svg"),
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        join(path_to_save, "figs", "pilr_2d.png"),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.figure()
    with open(join(path_to_save, "pi_l_gamma_fitter.pkl"), "rb") as f:
        gfitter = pickle.load(f)
    x = np.linspace(0, 0.32, 300)
    y = gfitter.pdf(x)

    plt.plot(x, y, label=r"$\Pi(l)$", linewidth=2.5, color="tab:blue")
    plt.fill_between(x, y, 0, color="tab:blue", alpha=0.25)

    plt.tick_params(axis="both", labelsize=16, length=6, width=1.2)

    plt.xlim(left=0)
    plt.ylim(bottom=0)

    # plt.legend(fontsize=16, frameon=False)

    ax = plt.gca()

    ax.set_xlabel(r"$l$ ($\AA$)", fontsize=20)
    ax.set_ylabel(r"$\Pi(l)$", fontsize=20)

    ax.xaxis.set_label_coords(1.07, -0.05)
    ax.yaxis.set_label_coords(-0.05, 0.98)

    plt.subplots_adjust(left=0.16, bottom=0.16)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.savefig(
        join(path_to_save, "figs", "pil_2d.svg"),
        # bbox_inches="tight",
        facecolor="white",
    )
    plt.savefig(
        join(path_to_save, "figs", "pil_2d.png"),
        dpi=300,
        # bbox_inches="tight",
        facecolor="white",
    )

    plt.show()


if __name__ == "__main__":
    path_to_save = join(
        "/media/andrey/Samsung_T5/PHD/Kerogen/", "type1matrix", "300K", "ch4"
    )
    plot_distributions(path_to_save)
