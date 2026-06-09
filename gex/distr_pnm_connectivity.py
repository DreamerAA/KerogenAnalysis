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
    for path_to_pnms, hist_prefix in paths:
        onlyfiles = [
            f for f in listdir(path_to_pnms) if isfile(join(path_to_pnms, f))
        ]
        onlyfiles = [file for file in onlyfiles if "_link1" in file]
        steps = [int((file.split("=")[1]).split("_")[0]) for file in onlyfiles]
        sorted_lfiles = list(zip(steps, onlyfiles))
        sorted_lfiles = sorted(sorted_lfiles, key=lambda x: x[0])

        mresult = {}
        for step, file in sorted_lfiles:
            res = Reader.read_pnm_linklist(join(path_to_pnms, file))
            mask = np.logical_and(res[:, 0] > 0, res[:, 1] > 0)
            res = res[mask, :]
            graph = nx.Graph()
            graph.add_edges_from(
                [(n1, n2) for n1, n2 in zip(res[:, 0], res[:, 1])]
            )
            degrees = [val for (node, val) in graph.degree()]
            degree_count = {}
            for d in degrees:
                if d in degree_count:
                    degree_count[d] = degree_count[d] + 1
                else:
                    degree_count[d] = 1
            mresult[step] = degree_count
        steps = []
        degrees = []
        count = []
        for s, res in mresult.items():
            steps = steps + [s] * len(res)
            degrees = degrees + [k for k, _ in res.items()]
            count = count + [v for _, v in res.items()]
        ddata = pd.DataFrame(
            list(zip(degrees, count, steps)),
            columns=['Degree', 'Count', 'Number simulation'],
        )
        sns.lineplot(data=ddata, x="Degree", y="Count", label=hist_prefix).set(
            title="Сonnectivity distribution in the porenetwork model"
        )

    plt.legend()
    plt.show()


def _parse_trj(s: str) -> Tuple[str, str]:
    parts = s.split(":", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Expected PATH:LABEL, got: {s!r}")
    return parts[0], parts[1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="PNM connectivity distributions"
    )
    parser.add_argument(
        "--pnm",
        action="append",
        type=_parse_trj,
        required=True,
        metavar="PATH:LABEL",
        help="PNM directory and label (repeatable)",
    )
    args = parser.parse_args()

    build_distributions(args.trj)
