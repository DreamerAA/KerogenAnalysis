import argparse
import os
import sys
import time
from os.path import join, realpath
from pathlib import Path
from typing import List, Tuple
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
import seaborn as sns
from skimage import measure


from base.boundingbox import BoundingBox, Range
from base.trajectory import Trajectory
from utils.utils import get_pattern_bbox, kprint
from visualizer.visualizer import Visualizer, WrapMode


def visualize_dist_trajectory(
    main_path: str, path_to_img: str, start_cut: int, stop_cut: int, num: int
) -> None:

    pattern = get_pattern_bbox()
    match = pattern.search(path_to_img)

    if not match:
        raise ValueError("Не удалось распарсить имя файла")

    data = {k: float(v) for k, v in match.groupdict().items()}
    resolution = data["resolution"]

    bbox_full = BoundingBox(
        Range(data["x_min"], data["x_max"]),
        Range(data["y_min"], data["y_max"]),
        Range(data["z_min"], data["z_max"]),
    )

    float_img = np.load(path_to_img)
    sx, sy, sz = float_img.shape
    ix0 = sx // 5
    iy0 = sy // 5
    iz0 = sz // 5
    spacing = np.asarray(bbox_full.size(), dtype=float) / (
        np.asarray(float_img.shape, dtype=float)
    )

    float_img = float_img[:ix0, :iy0, :iz0]

    cut_shift = spacing * np.array([ix0, iy0, iz0], dtype=float)
    add_shift = cut_shift / 3

    # bbox = bbox_full
    cut_bbox = BoundingBox(
        Range(
            bbox_full.xb_.min_ - add_shift[0],
            bbox_full.xb_.max_ - cut_shift[0] - add_shift[0],
        ),
        Range(
            bbox_full.yb_.min_ - add_shift[1],
            bbox_full.yb_.max_ - cut_shift[1] - add_shift[1],
        ),
        Range(
            bbox_full.zb_.min_ - add_shift[2],
            bbox_full.zb_.max_ - cut_shift[2] - add_shift[2],
        ),
    )
    dev = 2.0
    diff = [b.diff() / dev for b in [cut_bbox.xb_, cut_bbox.yb_, cut_bbox.zb_]]
    cut_bbox = BoundingBox(
        Range(cut_bbox.xb_.min_, cut_bbox.xb_.min_ + diff[0]),
        Range(cut_bbox.yb_.min_, cut_bbox.yb_.min_ + diff[1]),
        Range(cut_bbox.zb_.min_, cut_bbox.zb_.min_ + diff[2]),
    )

    pad = 2
    dpad = (pad, pad)
    ldpad = [dpad] * 3

    float_img = ndimage.gaussian_filter(float_img, 4)
    float_img = np.pad(float_img, ldpad, 'constant', constant_values=1)

    pad_shift = pad * resolution
    bbox = BoundingBox(
        Range(cut_bbox.xb_.min_ - pad_shift, cut_bbox.xb_.max_ + pad_shift),
        Range(cut_bbox.yb_.min_ - pad_shift, cut_bbox.yb_.max_ + pad_shift),
        Range(cut_bbox.zb_.min_ - pad_shift, cut_bbox.zb_.max_ + pad_shift),
    )

    traj_path = join(main_path, "trj.gro")
    trajectories = Trajectory.read_trajectoryes(traj_path)
    trj = trajectories[num]
    trj.cut(start=start_cut, stop=stop_cut)
    res = bbox.inside(trj.points[:, 0], trj.points[:, 1], trj.points[:, 2])
    kprint("Inside: ", np.any(res))
    kprint("Trj shape:", trj.points.shape)
    kprint("trj min: ", trj.points.min(axis=0))
    kprint("trj max: ", trj.points.max(axis=0))
    kprint("new img shape:", float_img.shape)
    kprint("new bbox min: ", bbox.min())
    kprint("new bbox max: ", bbox.max())
    kprint("new bbox size:", bbox.size())
    kprint("spacing:", spacing)

    mult = 2
    rad = 0.01 * mult

    Visualizer.draw_img_trj(
        float_img,
        bbox,
        trj,
        wrap_mode=WrapMode.EMPTY,
        isovalue=0.11,
        volume_mode=False,
        radius=rad,
        color_type='dist',
        line_width=rad * 0.3,
        periodic=True,
        with_points=True,
        img_opacity=1.0,
    )
    Visualizer.show()


if __name__ == '__main__':
    main_path = "/media/andrey/Samsung_T5/PHD/Kerogen/type1matrix/300K/ch4/"
    img_name = "result-img-num=551025000_time-ps=1102050_bbox=(x=(1.489-4.742)_y=(2.078-5.332)_z=(4.881-8.134))_resolution=0.013015000.npy"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--main_path',
        type=str,
        default=main_path,
    )
    parser.add_argument(
        '--path_to_img',
        type=str,
        default=join(main_path, "float_images", img_name),
    )
    parser.add_argument(
        '--num',
        type=int,
        default=2,  # 14 17
    )
    args = parser.parse_args()

    visualize_dist_trajectory(
        args.main_path, args.path_to_img, 4500, 6700, args.num
    )
