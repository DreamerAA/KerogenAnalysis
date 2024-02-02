import argparse
from pathlib import Path
import sys
import os
from os import listdir
from os.path import isfile, join, dirname, realpath
import numpy as np
import matplotlib.pyplot as plt
import pickle
from typing import List
from joblib import Parallel, delayed
import time
from scipy.stats import poisson
import json
import subprocess

path = Path(realpath(__file__))
parent_dir = str(path.parent.parent.absolute())
sys.path.append(parent_dir)

from base.reader import Reader


def filesPath(path: str):
    return [join(path, f) for f in listdir(path) if isfile(join(path, f))]


def data_by_img():
    data = [
        -4139,
        -4159,
        -4155,
        -9493,
        -9429,
        -9707,
        -9326,
        -4133,
        -9435,
        -9293,
        -4211,
        -9568,
        -4048,
        -9121,
        -4390,
        -4150,
        -9566,
        -9823,
        -9954,
        -4135,
        -9384,
        -4156,
        -4177,
        -9458,
        -4208,
        -9371,
        -4357,
        -9518,
        -9342,
        -9716,
        -4009,
        -4124,
        -4162,
        -4190,
        -9367,
        -4186,
        -9837,
        -4217,
        -9154,
        -9511,
        -9249,
        -4204,
        -9219,
        -4212,
        -9411,
        -9061,
        -9991,
        -4287,
        -9814,
        -9004,
        -14979,
        -4208,
        -4597,
        -4159,
        -4195,
        -8080,
        -4143,
        -9305,
        -4019,
        -9622,
        -4224,
        -4321,
        -9297,
        -4179,
    ]
    return data


def calculateEuler(path_to_img: str, path_to_config: str) -> None:
    path_to_exe = "/home/andrey/DigitalCore/PNE/pore-network-extraction/build/bin/extractor_example"

    onlyfiles = filesPath(path_to_img)
    for i, img_file in enumerate(onlyfiles):
        if "(500, 500, 500)" not in img_file:
            continue
        # Opening JSON file
        with open(path_to_config, 'r') as f:
            data = json.load(f)
            f.close()

        data["input_data"]["filename"] = img_file
        data["output_data"]["filename"] = "./tmp"

        with open(path_to_config, 'w') as f:
            json.dump(data, f)
            f.close()

        subprocess.run([path_to_exe, path_to_config])
        print(f"Ready {i+1} from {len(onlyfiles)}")


def plotEulerDistributions(path_to_pnm: str):
    onlyfiles = filesPath(path_to_pnm)
    pnm = [
        Reader.read_pnm_data(path[:-10], scale=1e10, border=-1)
        for path in onlyfiles
    ]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path_to_img',
        type=str,
        default="../data/Kerogen/tmp/result_time_depend_struct/images",
    )
    parser.add_argument(
        '--path_to_pnm',
        type=str,
        default="../data/Kerogen/tmp/result_time_depend_struct/pnm/part/",
    )
    parser.add_argument(
        '--path_to_conf',
        type=str,
        default="../data/Kerogen/tmp/result_time_depend_struct/extract_config.json",
    )
    args = parser.parse_args()

    data_by_img()

    # calculateEuler(args.path_to_img, args.path_to_conf)
    # plotEulerDistributions(args.path_to_pnm)
