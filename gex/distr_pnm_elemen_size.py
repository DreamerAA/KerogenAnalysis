import sys
import argparse
import json
import time
from pathlib import Path
from typing import Tuple, Any, List
from os import listdir
from os.path import isfile, join, dirname, realpath
import subprocess
from scipy.stats import weibull_min, exponweib
import numpy as np
import matplotlib.pyplot as plt

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


def drawDistr(axs, radiuses, throat_lengths, label):
    n = 50

    def plot_hist(data, maxes, title, xlabel):
        p, bb = np.histogram(data, bins=n)
        xdel = bb[1] - bb[0]
        x = bb[:-1] + xdel * 0.5
        pn = p / np.sum(p * xdel)
        maxes.plot(x, pn, label=label)
        maxes.set_title(title)
        maxes.set_xlabel(xlabel)

    plot_hist(
        radiuses, axs[0], "PDF(Pore size distribution)", "Pore diameter (A)"
    )
    plot_hist(
        throat_lengths,
        axs[1],
        "PDF(Throat length distribution)",
        "Throat length (A)",
    )


def build_distributions(paths: List[Tuple[str, str]], pnm_step=10) -> None:
    fig, axs = plt.subplots(1, 2)
    for path_to_pnms, hist_prefix in paths:
        onlyfiles = [
            f for f in listdir(path_to_pnms) if isfile(join(path_to_pnms, f))
        ]
        onlyfiles = [file for file in onlyfiles if "_link1" in file]
        steps = [int((file.split("=")[1]).split("_")[0]) for file in onlyfiles]
        sorted_lfiles = list(zip(steps, onlyfiles))
        sorted_lfiles = sorted(sorted_lfiles, key=lambda x: x[0])
        for step, file in sorted_lfiles[::pnm_step]:
            radiuses, throat_lengths = Reader.read_pnm_data(
                join(path_to_pnms, file[:-10]), scale=1e10, border=0.015
            )
            drawDistr(axs, radiuses, throat_lengths, hist_prefix + str(step))
    plt.legend()
    plt.show()


if __name__ == '__main__':
    build_distributions(
        [
            (
                "/media/andrey/Samsung_T5/PHD/Kerogen/ch4/pnm/",
                "CH4, num=",
            ),
            (
                "/media/andrey/Samsung_T5/PHD/Kerogen/h2/pnm/",
                "H2, num=",
            ),
        ]
    )
