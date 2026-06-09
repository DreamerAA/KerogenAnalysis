import argparse
import json
import pickle
import sys
from os import listdir
from os.path import dirname, isfile, join, realpath, exists
from pathlib import Path

from typing import Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from joblib import Parallel, delayed
from matplotlib import cm
from scipy.stats import exponweib
from sklearn.metrics import pairwise_distances
from scipy.signal import savgol_filter
from utils.types import NPFArray
from processes.distribution_fitter import (
    GammaFitter,
    WeibullFitter,
)
from utils.utils import kprint
from utils.timer import Timer

path = Path(realpath(__file__))
parent_dir = str(path.parent.parent.absolute())
sys.path.append(parent_dir)

from base.reader import Reader
from processes.pil_distr_generator import PiLDistrGenerator


def plot_ar(ar, maxes, title, xlabel):
    maxes.plot(ar[:, 0], ar[:, 1], label=xlabel)
    maxes.set_title(title, fontsize=12)
    maxes.set_xlabel(xlabel, fontsize=12)
    maxes.tick_params(axis='x', labelsize=12)
    maxes.tick_params(axis='y', labelsize=12)


def plot_hist(data, maxes, title, xlabel, n=50, smooth: bool = True):
    p, bb = np.histogram(data, bins=n)
    xdel = bb[1] - bb[0]
    x = bb[:-1] + xdel * 0.5
    pn = p / np.sum(p * xdel)
    if smooth:
        pn = savgol_filter(pn, 25, 3)
        pn[pn < 0] = 0
    maxes.plot(x, pn, label=xlabel)
    maxes.set_title(title, fontsize=12)
    maxes.set_xlabel(xlabel, fontsize=12)
    maxes.tick_params(axis='x', labelsize=12)
    maxes.tick_params(axis='y', labelsize=12)


def plot_distributions(path_to_pnms: str, path_to_save_pil: str):

    radiuses, throat_lengths = get_radiuses_lengthes(path_to_pnms)

    pi_l = np.load(path_to_save_pil + "pi_l.npy")

    fig, axs = plt.subplots(1, 3)
    plot_ar(pi_l, axs[2], "PDF(Step size in the trap)", "Step size (nm)")
    plot_hist(
        radiuses, axs[0], "PDF(Pore size distribution)", "Pore diameter (nm)"
    )
    plot_hist(
        throat_lengths,
        axs[1],
        "PDF(Throat length distribution)",
        "Throat length (nm)",
    )
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.legend(frameon=False, prop={'size': 12})


def get_radiuses_lengthes(path_to_pnms: str):
    onlyfiles = [
        f for f in listdir(path_to_pnms) if isfile(join(path_to_pnms, f))
    ]
    onlyfiles = [
        join(path_to_pnms, file[:-10]) for file in onlyfiles if "_link1" in file
    ]
    radiuses, throat_lengths = [], []
    for f in onlyfiles:
        r, t = Reader.read_pnm_data(f, border=0.003)
        radiuses = np.concatenate((radiuses, r))
        throat_lengths = np.concatenate((throat_lengths, t))

    return radiuses, throat_lengths


def save_distribution_figures(
    path_to_save: str, radiuses: npt.NDArray, throat_lengths: npt.NDArray
):
    figs_dir = Path(path_to_save) / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    path_pi_l_gf = Path(join(path_to_save, "pi_l_gamma_fitter.pkl"))
    path_tl_wf = Path(join(path_to_save, "throat_lengths_weibull_fitter.pkl"))
    path_pi_l_data = Path(join(path_to_save, "pi_l_data.npy"))

    with open(path_pi_l_gf, "rb") as f:
        gfitter = pickle.load(f)
    with open(path_tl_wf, "rb") as f:
        wfitter = pickle.load(f)
    pi_l_data = np.load(path_pi_l_data)

    def _plot_hist_and_fit(ax, data, fitter, xlabel):
        p, bb = np.histogram(data, bins=60)
        xdel = bb[1] - bb[0]
        x = bb[:-1] + xdel * 0.5
        pn = p / np.sum(p * xdel)
        ax.bar(x, pn, width=xdel, alpha=0.5, color="tab:blue", label="Data")
        x_fit = np.linspace(data.min(), data.max(), 300)
        ax.plot(x_fit, fitter.pdf(x_fit), color="tab:orange", lw=2, label="Fit")
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("PDF", fontsize=12)
        ax.tick_params(labelsize=12)
        ax.legend(frameon=False, fontsize=11)

    # Figure 1: throat length distribution
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    _plot_hist_and_fit(ax1, throat_lengths, wfitter, "Throat length (nm)")
    ax1.set_title("Throat length distribution", fontsize=13)
    fig1.tight_layout()
    for ext in ("svg", "png"):
        fig1.savefig(figs_dir / f"throat_length_distribution.{ext}", dpi=200)
    plt.close(fig1)

    # Figure 2: pi(l) step-size distribution + pore radii rug
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    _plot_hist_and_fit(ax2, pi_l_data, gfitter, "Step size (nm)")
    unique_r = np.unique(radiuses)
    ax2.plot(
        unique_r,
        np.zeros_like(unique_r),
        "|",
        ms=15,
        color="tab:green",
        alpha=0.7,
        label="Pore radii (nm)",
    )
    ax2.set_title("π(l) step-size distribution", fontsize=13)
    ax2.legend(frameon=False, fontsize=11)
    fig2.tight_layout()
    for ext in ("svg", "png"):
        fig2.savefig(figs_dir / f"pil_distribution.{ext}", dpi=200)
    plt.close(fig2)


def generate_pil_distribution(
    path_to_pnms: str, path_to_save: str, radius_min: float
):
    path_rads = join(path_to_save, "radiuses.npy")
    path_lens = join(path_to_save, "throat_lengths.npy")
    path_units = join(path_to_save, "pnm_distribution_units.json")

    use_cached_samples = isfile(path_lens) and isfile(path_rads)
    if use_cached_samples:
        radiuses = np.load(path_rads)
        throat_lengths = np.load(path_lens)
    else:
        radiuses, throat_lengths = get_radiuses_lengthes(path_to_pnms)
        np.save(path_rads, radiuses)
        np.save(path_lens, throat_lengths)
        with open(path_units, "w") as f:
            json.dump({"length_unit": "nm"}, f)

    radiuses = radiuses[radiuses > radius_min]

    generator = PiLDistrGenerator()

    timer = Timer()
    timer.start()
    path_pi_l_gf = Path(join(path_to_save, "pi_l_gamma_fitter.pkl"))
    path_pi_l_data = Path(join(path_to_save, "pi_l_data.npy"))
    if not path_pi_l_gf.exists():
        if not path_pi_l_data.exists():
            data = generator.gen_set(radiuses)
            np.save(path_pi_l_data, data)
        else:
            data = np.load(path_pi_l_data)
        gfitter = GammaFitter()
        gfitter.fit(data)
        with open(path_pi_l_gf, "wb") as f:
            pickle.dump(gfitter, f)
    else:
        kprint("Using cached gamma fitter")
    timer.stop("Gamma fitter")

    timer.start()
    path_tl_wf = Path(join(path_to_save, "throat_lengths_weibull_fitter.pkl"))
    if not path_tl_wf.exists():
        wfitter = WeibullFitter()
        wfitter.fit(throat_lengths)
        with open(path_tl_wf, "wb") as f:
            pickle.dump(wfitter, f)
    else:
        kprint("Using cached Weibull fitter")
    timer.stop("Weibull fitter")

    save_distribution_figures(path_to_save, radiuses, throat_lengths)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate PIL distributions from PNM data"
    )
    parser.add_argument("pnm_dir", type=Path, help="PNM directory (input)")
    parser.add_argument(
        "output_dir", type=Path, help="Output directory for fitter pickles"
    )
    parser.add_argument("--x-min", type=float, default=0.025)
    args = parser.parse_args()

    generate_pil_distribution(
        str(args.pnm_dir), str(args.output_dir), args.x_min
    )
