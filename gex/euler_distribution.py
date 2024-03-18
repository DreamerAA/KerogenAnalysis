import argparse
from pathlib import Path
import sys
import os
from os import listdir
from os.path import isfile, join, dirname, realpath
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import pickle
from typing import List, Tuple
from joblib import Parallel, delayed
import time
from scipy.stats import poisson
import json
import subprocess

path = Path(realpath(__file__))
parent_dir = str(path.parent.parent.absolute())
sys.path.append(parent_dir)

from base.reader import Reader


def calculateEulerImage(
    path_to_img: str, path_to_config: str
) -> npt.NDArray[np.int64]:
    path_to_exe = "/home/andrey/DigitalCore/PNE/pore-network-extraction/build/bin/extractor_example"

    # Opening JSON file
    with open(path_to_config) as f:
        # returns JSON object as
        # a dictionary
        jconfig = json.load(f)

    path_to_pnm = ''

    onlyfiles = [
        join(path_to_img, f)
        for f in listdir(path_to_img)
        if isfile(join(path_to_img, f)) and '.raw' in f
    ]

    eulers = np.zeros(shape=(len(onlyfiles)), dtype=np.int64)

    for i, file in enumerate(onlyfiles):
        lfile = file.split('_')
        parts = [
            l.split('=')[1]
            for l in lfile
            if "num" in l or "resolution" in l or "is=" in l
        ]
        # print(parts)
        num = int(parts[0])
        xs, ys, zs = [int(e) for e in parts[1][1:-1].split(',')]
        resolution = float(parts[2][:-5])

        pnm_pref = join(path_to_pnm, f"num={num}_{xs}_{ys}_{zs}")
        jconfig["input_data"]["filename"] = file
        jconfig["input_data"]["size"]["x"] = xs
        jconfig["input_data"]["size"]["y"] = ys
        jconfig["input_data"]["size"]["z"] = zs
        jconfig["output_data"]["statoil_prefix"] = pnm_pref
        jconfig["output_data"]["filename"] = pnm_pref
        jconfig["extraction_parameters"]["resolution"] = resolution * 0.1

        with open(path_to_config, 'w') as f:
            json.dump(jconfig, f)
            f.close()

        process = subprocess.Popen(
            [
                path_to_exe,
                path_to_config,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # wait for the process to terminate
        out, err = process.communicate()
        errcode = process.returncode
        if errcode != 0:
            print("Error!!!")
        s = str(out)
        s = s[2:-1]
        str_e = (s.split('\n')[-1]).split(':')[1]
        e = int(str_e[:-2])
        # print(e, str(out))
        eulers[i] = e

        print(f"Ready {i+1} from {len(onlyfiles)}")
    return eulers


def calculateEulerPNM(path_to_pnm: str):
    onlyfiles = [
        join(path_to_pnm, f)
        for f in listdir(path_to_pnm)
        if isfile(join(path_to_pnm, f))
        if "_link1.dat" in f
    ]

    nums = []
    for i, img_file in enumerate(onlyfiles):
        num = int(((img_file.split('/')[-1]).split('_')[0]).split('=')[1])
        nums.append(num)

    eulers = np.zeros(shape=(len(onlyfiles)), dtype=np.int64)
    for j, n in enumerate(nums):
        fn = path_to_pnm + f"num={n}_500_500_500"
        radiuses, throat_lengths = Reader.read_pnm_data(fn, 1, 0)
        ll, _ = Reader.read_pnm_linklist(fn + "_link1.dat")

        V = len(radiuses)
        E = len(throat_lengths)
        ne = V - E
        eulers[j] = ne
    return eulers


def plotEulerDistributions(
    pnm: npt.NDArray[np.int64], images: npt.NDArray[np.int64], prefix, axs
):
    n = 20

    def plot_hist(data, maxes):
        p, bb = np.histogram(data, bins=n)
        xdel = bb[1] - bb[0]
        x = bb[:-1] + xdel * 0.5
        maxes.plot(x, p, label=prefix)

    plot_hist(pnm, axs[0])
    plot_hist(images, axs[1])


def calculateEulers(
    path_to_euler: str, path_to_img: str, path_to_conf: str, path_to_pnm: str
) -> Tuple[List[int], List[int]]:
    img_eulers_file = path_to_euler + "img_euler.pkl"
    img_eulers_file_j = path_to_euler + "img_euler.json"
    if isfile(img_eulers_file):
        with open(img_eulers_file, 'rb') as f:
            img_eulers = pickle.load(f)
    elif isfile(img_eulers_file_j):
        with open(img_eulers_file_j, 'rb') as f:
            j_eulers = json.load(f)
            img_eulers = [v for k, v in j_eulers.items()]
    else:
        img_eulers = calculateEulerImage(path_to_img, path_to_conf)
        with open(img_eulers_file, 'wb') as f:
            pickle.dump(img_eulers, f)

    pnm_eulers_file = path_to_euler + "pnm_euler.pkl"
    if isfile(pnm_eulers_file):
        with open(pnm_eulers_file, 'rb') as f:
            pnm_eulers = pickle.load(f)
    else:
        pnm_eulers = calculateEulerPNM(path_to_pnm)
        with open(pnm_eulers_file, 'wb') as f:
            pickle.dump(pnm_eulers, f)
    return pnm_eulers, img_eulers


if __name__ == '__main__':
    fig, axs = plt.subplots(1, 2)
    for i, a in enumerate(axs):
        a.set_xlabel("Euler number")
    axs[0].set_title("Eulers by Pore Network")
    axs[1].set_title("Eulers by Image")

    main_path = "/media/andrey/Samsung_T5/PHD/Kerogen/"
    path_to_conf = main_path + "ExtractorExampleConfig.json"

    for tem in ["300K", "400K"]:
        for el in ["h2", "ch4"]:
            c_path = main_path + f"{tem}/{el}/"

            path_to_euler = c_path
            path_to_img = c_path + "images/"
            path_to_pnm = c_path + "pnm/"

            pnm_eulers_h2, img_eulers_h2 = calculateEulers(
                path_to_euler, path_to_img, path_to_conf, path_to_pnm
            )
            plotEulerDistributions(
                pnm_eulers_h2, img_eulers_h2, el.upper() + " - " + tem, axs
            )

    plt.legend()
    plt.show()
