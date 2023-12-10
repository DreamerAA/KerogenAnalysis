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
       


def extract_weibull_psd(path_to_pnm: str, scale: float, border: float = 0.02)->Tuple[Any]:
    radiuses, throat_lengths = Reader.read_pnm_data(
        path_to_pnm, scale=scale, border=border
    )
    psd_params = exponweib.fit(radiuses)
    tld_params = exponweib.fit(throat_lengths)
    return psd_params, tld_params, radiuses, throat_lengths


def build_distributions(path_to_pnms:str)->None:
    fig, axs = plt.subplots(1, 2)

    onlyfiles = [f for f in listdir(path_to_pnms) if isfile(join(path_to_pnms, f))]    
    onlyfiles = [file for file in onlyfiles if "_link1" in file]
    steps = [int((file.split("_")[0]).split("=")[1]) for file in onlyfiles]
    sorted_lfiles = list(zip(steps, onlyfiles))
    sorted_lfiles = sorted(sorted_lfiles, key=lambda x: x[0])
    
    for step, file in sorted_lfiles[::2]:
        psd_params, tld_params, radiuses, throat_lengths = extract_weibull_psd(
            join(path_to_pnms, file[:-10]), 1e10, 0.015
        )

        nx_rad = np.linspace(radiuses[0], radiuses[-1], 1000)        
        nx_len = np.linspace(throat_lengths[0], throat_lengths[-1], 1000)
        

        n = 50

        # axs[0].hist(radiuses,bins=n)
        # axs[1].hist(throat_lengths,bins=n)
        
        p, bb = np.histogram(radiuses, bins=n)
        xdel = bb[1] - bb[0]
        x = bb[:-1] + xdel * 0.5
        pn = p / np.sum(p * xdel)
        axs[0].plot(x, pn, label=f"Histogram data step={step}")
        # axs[0].plot(nx_rad, fit_y_rad, label='Fitting')
        axs[0].set_title("PDF(Pore size distribution)")
        axs[0].set_xlabel("Pore diameter (A)")

        p, bb = np.histogram(throat_lengths, bins=n)
        xdel = bb[1] - bb[0]
        x = bb[:-1] + xdel * 0.5
        pn = p / np.sum(p * xdel)
        axs[1].plot(x, pn, label=f"Histogram data step={step}")
        # axs[1].plot(nx_len, fit_y_len, label='Fitting')
        axs[1].set_title("PDF(Throat length distribution)")
        axs[1].set_xlabel("Throat length (A)")

    plt.legend()
    plt.show()

def drawDistr(axs, radiuses, throat_lengths, label):
    n = 50

    # axs[0].hist(radiuses,bins=n)
    # axs[1].hist(throat_lengths,bins=n)
    
    p, bb = np.histogram(radiuses, bins=n)
    xdel = bb[1] - bb[0]
    x = bb[:-1] + xdel * 0.5
    pn = p / np.sum(p * xdel)
    axs[0].plot(x, pn, label=label)
    # axs[0].plot(nx_rad, fit_y_rad, label='Fitting')
    axs[0].set_title("PDF(Pore size distribution)")
    axs[0].set_xlabel("Pore diameter (A)")

    p, bb = np.histogram(throat_lengths, bins=n)
    xdel = bb[1] - bb[0]
    x = bb[:-1] + xdel * 0.5
    pn = p / np.sum(p * xdel)
    axs[1].plot(x, pn, label=label)
    # axs[1].plot(nx_len, fit_y_len, label='Fitting')
    axs[1].set_title("PDF(Throat length distribution)")
    axs[1].set_xlabel("Throat length (A)")

def build_distributions_compare_bbb(path_to_pnms:str, path_to_bb_pnm:str)->None:
    fig, axs = plt.subplots(1, 2)

    onlyfiles = [f for f in listdir(path_to_pnms) if isfile(join(path_to_pnms, f))]    
    onlyfiles = [file for file in onlyfiles if "_link1" in file]
    steps = [int((file.split("_")[0]).split("=")[1]) for file in onlyfiles]
    sorted_lfiles = list(zip(steps, onlyfiles))
    sorted_lfiles = sorted(sorted_lfiles, key=lambda x: x[0])
    
    for step, file in sorted_lfiles[::2]:
        psd_params, tld_params, radiuses, throat_lengths = extract_weibull_psd(
            join(path_to_pnms, file[:-10]), 1e10, 0.015
        )

        nx_rad = np.linspace(radiuses[0], radiuses[-1], 1000)        
        nx_len = np.linspace(throat_lengths[0], throat_lengths[-1], 1000)

        drawDistr(axs, radiuses, throat_lengths, f"Histogram data step={step}")

    psd_params, tld_params, radiuses, throat_lengths = extract_weibull_psd(path_to_bb_pnm, 1e10, 0.015)
    drawDistr(axs, radiuses, throat_lengths, f"Full cell, Last step")

    plt.legend()
    plt.show()

def build_distributions(paths:List[Tuple[str,str]])->None:
    fig, axs = plt.subplots(1, 2)
    for path_to_pnms,hist_prefix in paths:
        onlyfiles = [f for f in listdir(path_to_pnms) if isfile(join(path_to_pnms, f))]    
        onlyfiles = [file for file in onlyfiles if "_link1" in file]
        steps = [int((file.split("=")[1]).split("_")[0]) for file in onlyfiles]
        sorted_lfiles = list(zip(steps, onlyfiles))
        sorted_lfiles = sorted(sorted_lfiles, key=lambda x: x[0])
        for step, file in sorted_lfiles:
            radiuses, throat_lengths = Reader.read_pnm_data(
                join(path_to_pnms, file[:-10]), scale=1e10, border= 0.015
            )
            drawDistr(axs, radiuses, throat_lengths, hist_prefix + str(step))
    plt.legend()
    plt.show()

if __name__ == '__main__':
    build_distributions([("/home/andrey/PHD/Kerogen/data/Kerogen/tmp/result_time_depend_struct/pnm/full/", "Hist data for FULL cell num="),
                         ("/home/andrey/PHD/Kerogen/data/Kerogen/tmp/result_time_depend_struct/pnm/part/", "Hist data for PART cell num=")])
