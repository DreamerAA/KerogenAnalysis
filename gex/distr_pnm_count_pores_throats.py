import argparse
import json
import subprocess
import sys
import time
from os import listdir
from os.path import dirname, isfile, join, realpath
from pathlib import Path
from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import exponweib, weibull_min

path = Path(realpath(__file__))
parent_dir = str(path.parent.parent.absolute())
sys.path.append(parent_dir)

from base.reader import Reader


def build_distributions(paths: List[Tuple[str, str]]) -> None:
    result = {}
    for path_to_pnms, hist_prefix in paths:
        onlyfiles = [
            f for f in listdir(path_to_pnms) if isfile(join(path_to_pnms, f))
        ]
        onlyfiles = [file for file in onlyfiles if "_link1" in file]
        steps = [int((file.split("=")[1]).split("_")[0]) for file in onlyfiles]
        sorted_lfiles = list(zip(steps, onlyfiles))
        sorted_lfiles = sorted(sorted_lfiles, key=lambda x: x[0])

        count_pores = []
        count_throats = []
        for _, file in sorted_lfiles:
            r, tl = Reader.read_pnm_data(join(path_to_pnms, file)[:-10])
            count_pores.append(len(r))
            count_throats.append(len(tl))

        result[hist_prefix] = (count_pores, count_throats)

    cp = []
    ct = []
    names = []
    for _, hist_prefix in paths:
        r = result[hist_prefix]
        names = names + [hist_prefix] * len(r[0])
        cp = cp + r[0]
        ct = ct + r[1]

    dcp = pd.DataFrame(
        list(zip(cp, names)),
        columns=['Count pores', 'Prefix'],
    )
    plt.figure()
    sns.histplot(data=dcp, x="Count pores", hue="Prefix")
    # plt.legend()

    dct = pd.DataFrame(
        list(zip(ct, names)),
        columns=['Count throats', 'Prefix'],
    )
    plt.figure()
    sns.histplot(data=dct, x="Count throats", hue="Prefix").set(
        title="Count throat distribution in the porenetwork"
    )
    # plt.legend()
    plt.show()


def _parse_trj(s: str) -> Tuple[str, str]:
    parts = s.split(":", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Expected PATH:LABEL, got: {s!r}")
    return parts[0], parts[1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PNM pore/throat count distributions")
    parser.add_argument(
        "--trj",
        action="append",
        type=_parse_trj,
        required=True,
        metavar="PATH:LABEL",
        help="PNM directory and label (repeatable)",
    )
    args = parser.parse_args()

    build_distributions(args.trj)
