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

path = Path(realpath(__file__))
parent_dir = str(path.parent.parent.absolute())
sys.path.append(parent_dir)

from base.reader import Reader


def extract_weibull_psd(
    path_to_pnm: str, scale: float, border: float = 0.02
) -> Tuple[Any]:
    radiuses, throat_lengths = Reader.read_pnm_data(
        path_to_pnm, scale=scale, border=border
    )
    psd_params = exponweib.fit(radiuses)
    tld_params = exponweib.fit(throat_lengths)
    return psd_params, tld_params, radiuses, throat_lengths


def plot_ar(x, y, maxes, title, plabel, xlabel):
    maxes.plot(x, y, label=plabel)
    maxes.set_title(title, fontsize=12)
    maxes.set_xlabel(xlabel, fontsize=12)
    maxes.tick_params(axis='x', labelsize=12)
    maxes.tick_params(axis='y', labelsize=12)


def drawDistr(axs, radiuses, throat_lengths, prefix, smooth: bool = True):
    n = 50

    def plot_hist(data, maxes, title, xlabel):
        p, bb = np.histogram(data, bins=n)
        xdel = bb[1] - bb[0]
        x = bb[:-1] + xdel * 0.5
        pn = p / np.sum(p * xdel)
        if smooth:
            pn = savgol_filter(pn, 25, 3)
            pn[pn < 0] = 0
        plot_ar(x, pn, maxes, title, prefix, xlabel)

    plot_hist(
        radiuses,
        axs[1],
        "Pore size distribution - $P(r)$",
        "Pore radius ($\AA$)",
    )
    plot_hist(
        throat_lengths,
        axs[0],
        "Throat length distribution - $P(l)$",
        "Throat length ($\AA$)",
    )


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


def generate_pil_distribution(
    path_to_pnms: str, path_to_save: str
) -> np.ndarray:
    if isfile(path_to_save + "pi_l.npy") and isfile(
        path_to_save + "throat_lengths.npy"
    ):
        return np.load(path_to_save + "pi_l.npy")

    radiuses, throat_lengths = get_radiuses_lengthes(path_to_pnms)

    generator = PiLDistrGenerator()
    pi_l = generator.run(radiuses)

    np.save(path_to_save + "pi_l.npy", pi_l)
    np.save(path_to_save + "throat_lengths.npy", throat_lengths)
    return pi_l


def build_distributions(
    paths: List[Tuple[str, str, str]],
    pnm_step=10,
    avarage: bool = False,
    smooth: bool = True,
) -> None:
    fig, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(15, 4))
    for path_to_pnms, hist_prefix, path_pil in paths:
        onlyfiles = [
            f for f in listdir(path_to_pnms) if isfile(join(path_to_pnms, f))
        ]
        onlyfiles = [
            file
            for file in onlyfiles
            if "_link1" in file and file.startswith("num")
        ]
        steps = [int((file.split("=")[1]).split("_")[0]) for file in onlyfiles]
        sorted_lfiles = list(zip(steps, onlyfiles))
        sorted_lfiles = sorted(sorted_lfiles, key=lambda x: x[0])
        if avarage:
            ar_radiuses = None
            ar_throat_lengths = None
            for step, file in sorted_lfiles[::pnm_step]:
                radiuses, throat_lengths = Reader.read_pnm_data(
                    join(path_to_pnms, file[:-10]), scale=1e10, border=0.015
                )
                if ar_radiuses is None:
                    ar_radiuses = radiuses
                else:
                    ar_radiuses = np.concatenate((ar_radiuses, radiuses))
                if ar_throat_lengths is None:
                    ar_throat_lengths = throat_lengths
                else:
                    ar_throat_lengths = np.concatenate(
                        (ar_throat_lengths, throat_lengths)
                    )
            drawDistr(axs, ar_radiuses, ar_throat_lengths, hist_prefix, smooth)
        else:
            for step, file in sorted_lfiles[::pnm_step]:
                radiuses, throat_lengths = Reader.read_pnm_data(
                    join(path_to_pnms, file[:-10]), scale=1e10, border=0.015
                )
                time = int(50 + (step - 25000.0) * 500.0 / 250000.0)
                print(f"Step: {step} Time: {time}")
                drawDistr(
                    axs,
                    radiuses,
                    throat_lengths,
                    hist_prefix + ", $t = " + str(time) + "$ psec",
                    smooth,
                )
        pi_l = generate_pil_distribution(path_to_pnms, path_pil)
        max_l = pi_l[-1, 0] * 0.6

        plot_ar(
            pi_l[:, 0],
            pi_l[:, 1],
            axs[2],
            "Step size in the trap distribution - $\Pi(l)$",
            hist_prefix,
            "Step size ($\AA$)",
        )
        axs[1].set_xlim(0, max_l)
        axs[2].set_xlim(0, max_l)

    for ax in axs:
        ax.tick_params(axis="both", labelsize=12)

    axs[1].legend(
        frameon=False,
        prop={"size": 12},
        loc="upper right",
        # bbox_to_anchor=(1.02, 1.0),
    )
    axs[2].legend(
        frameon=False,
        prop={"size": 12},
        loc="upper right",
        # bbox_to_anchor=(1.02, 1.0),
    )

    fig.savefig("all_distributions.svg", bbox_inches="tight")

    plt.show()


if __name__ == '__main__':
    build_distributions(
        [
            (
                "/media/andrey/Samsung_T5/PHD/Kerogen/type1matrix/300K/ch4/pnm/",
                "$CH_4$",
                "/media/andrey/Samsung_T5/PHD/Kerogen/type1matrix/300K/ch4/",
            ),
            (
                "/media/andrey/Samsung_T5/PHD/Kerogen/type1matrix/300K/h2/pnm/",
                "$H_2$",
                "/media/andrey/Samsung_T5/PHD/Kerogen/type1matrix/300K/h2/",
            ),
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
        ],
        pnm_step=30,
        avarage=False,
        smooth=True,
    )
