import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os

from itertools import repeat

from pathlib import Path
import re
import time

from matplotlib import pyplot as plt
import numpy as np
from base.trajectory import Trajectory
from utils.utils import kprint


def plot_kerogen_struct_slice(file_name: str | Path, output: Path):
    file_path = Path(file_name)
    pattern = re.compile(
        r"result-img-num=(?P<step>\d+)"
        r"_time-ps=(?P<time_ps>\d+(?:\.\d+)?)"
        r"_bbox=\(x=\((?P<x_min>-?\d+(?:\.\d+)?)-(?P<x_max>-?\d+(?:\.\d+)?)\)"
        r"_y=\((?P<y_min>-?\d+(?:\.\d+)?)-(?P<y_max>-?\d+(?:\.\d+)?)\)"
        r"_z=\((?P<z_min>-?\d+(?:\.\d+)?)-(?P<z_max>-?\d+(?:\.\d+)?)\)\)"
        r"_resolution=(?P<resolution>\d+(?:\.\d+)?)"
    )

    match = pattern.match(file_path.name)
    if not match:
        kprint("No match")
        return
    data = match.groupdict()
    num = data["step"]
    x_min = data["x_min"]
    x_max = data["x_max"]
    y_min = data["y_min"]
    y_max = data["y_max"]
    z_min = data["z_min"]
    z_max = data["z_max"]
    resolution = float(data["resolution"])

    img = np.load(file_path, mmap_mode="r")

    plt.imshow(img[img.shape[0] // 2, :, :])
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=Path, help="Input trajectory file")
    parser.add_argument("output", type=Path, help="Output figure path")

    args = parser.parse_args()

    plot_kerogen_struct_slice(
        args.input,
        args.output,
    )
