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


def data_by_pnm():
    return [
        2973,
        2712,
        3235,
        2875,
        3197,
        2748,
        2875,
        2572,
        2536,
        2839,
        2839,
        2892,
        2981,
        2812,
        3043,
        3138,
        2869,
        2748,
        2812,
        3118,
        2892,
        2868,
        3046,
        2712,
        2918,
        2946,
        2918,
        2836,
        2868,
        2889,
        3075,
        3075,
        3043,
        2946,
        3043,
        2712,
        2925,
        2973,
        2918,
        3118,
        2572,
        2925,
        2847,
        2661,
        2661,
        2973,
        3235,
        2712,
        2918,
        2869,
        2712,
        3138,
        2875,
        2805,
        2925,
        2839,
        2833,
        2836,
        3197,
        2869,
        3210,
        2805,
        2833,
        3043,
        2836,
        2875,
        2589,
        3235,
        2661,
        3197,
        3197,
        2889,
        3035,
        2812,
        3118,
        2847,
        3210,
        2981,
        2661,
        3235,
        2589,
        2536,
        2847,
        2892,
        2661,
        3043,
        2946,
        2889,
        2826,
        2748,
        2572,
        2839,
        2847,
        3197,
        3035,
        2973,
        2918,
        2869,
        2826,
        2805,
        2805,
        2946,
        2536,
        2869,
        2892,
        2836,
        2868,
        2868,
        2973,
        2748,
        2973,
        2981,
        2869,
        2812,
        2812,
        2661,
        2826,
        3197,
        2589,
        2536,
        2946,
        3138,
        3046,
        3046,
        3210,
        3138,
        3046,
        2833,
        2536,
        2925,
        2748,
        2875,
        2536,
        2868,
        2572,
        2833,
        3138,
        2839,
        3035,
        2918,
        3138,
        2925,
        2946,
        2981,
        2805,
        2875,
        2833,
        3075,
        3235,
        2589,
        2892,
        2925,
        2892,
        2981,
        2826,
        2836,
        3210,
        3043,
        3075,
        2836,
        3210,
        2572,
        2826,
        3035,
        3075,
        2589,
        2572,
        3075,
        2889,
        2748,
        3118,
        2805,
        2981,
        2712,
        3235,
        2847,
        2839,
        3035,
        3118,
        3046,
        2812,
        2833,
        3035,
        3046,
        2889,
        2826,
        2868,
        2889,
        3210,
        2589,
        3118,
        2847,
    ]


def calculateEulerImage(path_to_img: str, path_to_config: str) -> None:
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


def calculateEulerPNM(path_to_pnm: str):
    onlyfiles = filesPath(path_to_pnm)
    nums = []
    eulers = []
    for i, img_file in enumerate(onlyfiles):
        num = int((img_file.split('_')[3]).split('=')[1])
        nums.append(num)
    for n in nums:
        fn = path_to_pnm + f"num={n}_500_500_500"
        radiuses, throat_lengths = Reader.read_pnm_data(fn, 1, 0)
        ll, _ = Reader.read_pnm_linklist(fn + "_link1.dat")

        adj = {}
        for n1, n2 in ll:
            if n1 < 0 or n2 < 0:
                continue
            if n1 not in adj:
                adj[n1] = [n2]
            else:
                adj[n1].append(n2)

            if n2 not in adj:
                adj[n2] = [n1]
            else:
                adj[n2].append(n1)
        resf = set()
        for k, lv in adj.items():
            if len(lv) == 1:
                continue
            for i, v1 in enumerate(lv[:-1]):
                for _, v2 in enumerate(lv[(i + 1) :]):
                    if (
                        v2 in adj[v1]
                        and k in adj[v1]
                        and v1 in adj[v2]
                        and k in adj[v2]
                    ):
                        lll = tuple(set({k, v1, v2}))
                        resf.add(lll)
        F = len(resf)
        V = len(radiuses)
        E = len(throat_lengths)
        ne = V - E + F
        eulers.append(ne)
        print(ne)
    print(eulers)


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

    # calculateEulerImage(args.path_to_img, args.path_to_conf)
    # result of calculateEulerImage saved to data_by_img()

    calculateEulerPNM(args.path_to_pnm)
    # result of calculateEulerPNM saved to data_by_pnm()

    # data_by_img()

    # plotEulerDistributions(args.path_to_pnm)
