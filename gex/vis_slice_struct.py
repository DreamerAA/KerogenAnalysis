import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os

from itertools import repeat

from pathlib import Path
import re
import time
from typing import Optional

from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib import pyplot as plt
import numpy as np
from base.trajectory import Trajectory
from utils.utils import kprint


def plot_kerogen_struct_slice(
    file_name: str | Path, output: Path, size: float, ref_size: Optional[float]
):
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
    nsize = int(size // resolution)
    if nsize > img.shape[0] or nsize > img.shape[1] or nsize > img.shape[2]:
        kprint("Size too big")
        return

    nimg = img[:nsize, :nsize, :nsize]
    if ref_size:
        kprint("Count pixels in ref_size: ", int(ref_size // resolution))

    slice_img = nimg[nimg.shape[0] // 2, :, :]
    slice_mask = (slice_img > 0).astype(np.uint8)
    kprint(np.min(nimg), np.max(nimg))
    cmap = ListedColormap(
        [
            (1.0, 1.0, 1.0, 1.0),
            (0.75, 0.64, 0.48, 1.0),
        ]
    )
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    fig, ax = plt.subplots()
    ax.imshow(slice_mask, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)

    fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=Path, help="Input trajectory file")
    parser.add_argument("output", type=Path, help="Output figure path")
    parser.add_argument("--size", type=float, default=0.4)
    parser.add_argument(
        "--ref-size", "--ref_size", dest="ref_size", type=float, default=None
    )

    args = parser.parse_args()

    plot_kerogen_struct_slice(args.input, args.output, args.size, args.ref_size)
