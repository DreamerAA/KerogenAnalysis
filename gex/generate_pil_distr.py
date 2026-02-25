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
    plot_ar(pi_l, axs[2], "PDF(Step size in the trap)", "Step size (A)")
    plot_hist(
        radiuses, axs[0], "PDF(Pore size distribution)", "Pore diameter (A)"
    )
    plot_hist(
        throat_lengths,
        axs[1],
        "PDF(Throat length distribution)",
        "Throat length (A)",
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
        r, t = Reader.read_pnm_data(
            f, scale=1e10, border=0.03
        )  # scale=1e10 - to A
        radiuses = np.concatenate((radiuses, r))
        throat_lengths = np.concatenate((throat_lengths, t))

    return radiuses, throat_lengths


def generate_pil_distribution(path_to_pnms: str, path_to_save: str):
    path_rads = join(path_to_save, "radiuses.npy")
    path_lens = join(path_to_save, "throat_lengths.npy")
    if isfile(path_lens) and isfile(path_rads):
        radiuses = np.load(path_rads)
        throat_lengths = np.load(path_lens)
    else:
        radiuses, throat_lengths = get_radiuses_lengthes(path_to_pnms)
        np.save(path_rads, radiuses)
        np.save(path_lens, throat_lengths)

    generator = PiLDistrGenerator()

    paht_pi_l = join(path_to_save, "pi_l_curve.npy")
    if isfile(paht_pi_l):
        pi_l = np.load(paht_pi_l)
    else:
        pi_l = generator.get_curve(radiuses)
        np.save(paht_pi_l, pi_l)

    timer = Timer()
    timer.start()
    paht_pi_l_gf = join(path_to_save, "pi_l_gamma_fitter.pkl")
    data = generator.gen_set(radiuses)
    gfitter = GammaFitter()
    gfitter.fit(data)
    with open(paht_pi_l_gf, "wb") as f:
        pickle.dump(gfitter, f)
    timer.stop("Gamma fitter")

    timer.start()
    path_tl_wf = join(path_to_save, "throat_lengths_weibull_fitter.pkl")
    wfitter = WeibullFitter()
    wfitter.fit(throat_lengths)
    with open(path_tl_wf, "wb") as f:
        pickle.dump(wfitter, f)
    timer.stop("Weibull fitter")


if __name__ == '__main__':

    for mtype in ["type1matrix", "type2matrix"]:
        # for mtype in ["type1matrix"]:
        mpath = join("/media/andrey/Samsung_T5/PHD/Kerogen/", mtype)
        for tem in ["300K", "400K"]:
            if not exists(join(mpath, tem)):
                continue
            # for tem in ["300K"]:
            for el in ["h2", "ch4"]:
                # for el in ["ch4"]:
                path_to_save = join(mpath, tem, el)
                path_to_pnm = join(path_to_save, "pnm")

                generate_pil_distribution(path_to_pnm, path_to_save)

                # plot_distributions(path_to_pnm, path_to_save)

    # plt.show()
