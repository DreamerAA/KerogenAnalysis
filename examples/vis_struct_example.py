import sys
import os
from pathlib import Path
from os.path import realpath
import time
from typing import List, Tuple
import numpy as np
import argparse
import networkx as nx
from scipy import ndimage

path = Path(realpath(__file__))
parent_dir = str(path.parent.parent.absolute())
sys.path.append(parent_dir)

from base.boundingbox import BoundingBox
from base.kerogendata import AtomData, KerogenData
from base.periodizer import Periodizer
from base.reader import Reader
from base.trajectory import Trajectory
from base.utils import create_box_mask
from processes.segmentaion import Segmentator
from visualizer.visualizer import Visualizer
from examples.utils import write_binary_file


additional_radius = 0.0

atom_real_sizes = {
    i: s for i, s in enumerate([0.17, 0.152, 0.155, 0.109, 0.18])
}

ext_radius = {
    i: s
    for i, s in enumerate(
        [
            additional_radius,
            additional_radius,
            additional_radius,
            0.0,
            additional_radius,
        ]
    )
}


def get_size(type_id: int) -> float:
    return atom_real_sizes[type_id]


def get_ext_size(type_id: int) -> float:
    return ext_radius[type_id]


def extanded_struct_extr(
    path_to_structure: str,
    path_to_save: str,
    cut_cell: bool,
    ref_size: int,
) -> None:

    with open(path_to_structure, 'r') as f:
        atoms, size, num = Reader.read_raw_struct_ff(f)

    bbox = Segmentator.cut_cell(size, 9) if cut_cell else Segmentator.full_cell(size)  # type: ignore

    print(f" --- Current num: {num}")

    print(f" --- Box size: {bbox.size()}")

    kerogen_data = KerogenData(None, atoms, bbox)  # type: ignore
    if not kerogen_data.checkPeriodization():
        print("Periodization!")
        Periodizer.periodize(kerogen_data)
        # Periodizer.rm_long_edges(kerogen_data)

    print("Periodization is end!")

    resolution = np.array([s for s in size]).min() / ref_size
    str_resolution = "{:.9f}".format(resolution)

    img_size = Segmentator.calc_image_size(
        kerogen_data.box.size(), reference_size=ref_size, by_min=True
    )
    prf_cell = "partcell" if cut_cell else "fullcell"
    binarized_file_name = (
        # f"../data/Kerogen/result_img_{file_name}_rs={ref_size}_mr={methan_radius}_div={div}.npy"
        path_to_save
        + f"./float_img_num={num}_cell={prf_cell}_is={img_size}_ar={additional_radius}_resolution={str_resolution}.npy"
    )

    if os.path.isfile(binarized_file_name):
        with open(binarized_file_name, 'rb') as f:  # type: ignore
            img = np.load(f)  # type: ignore
    else:
        print(f" --- Image size for calculating: {img_size}")
        segmentator = Segmentator(
            kerogen_data,
            img_size,
            size_data=get_size,
            radius_extention=get_ext_size,
            partitioning=1,
        )
        img = segmentator.dist_map()
        np.save(binarized_file_name, img)  # type: ignore

    import matplotlib.pyplot as plt

    img = ndimage.gaussian_filter(img, 4)

    img = np.pad(img, [(1, 1), (1, 1), (1, 1)], 'maximum')
    Visualizer.draw_float_img(img, 0.11, kerogen_data.box)
    Visualizer.show()


if __name__ == '__main__':
    prefix = "/media/andrey/Samsung_T5/PHD/Kerogen/"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--structure_path',
        type=str,
        default=prefix + "type1matrix/300K/ch4/type1.ch4.1.gro",
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default=prefix,
    )

    args = parser.parse_args()

    extanded_struct_extr(
        args.structure_path,
        args.save_path,
        True,
        200,
    )
