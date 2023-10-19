import sys
import os
import random
import time
from typing import IO, List, Tuple
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.stats as stats
from scipy.interpolate import UnivariateSpline
from scipy.stats import weibull_min, exponweib
from joblib import Parallel, delayed
from sklearn.metrics import pairwise_distances
from matplotlib.collections import PolyCollection
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from base.boundingbox import BoundingBox
from base.kerogendata import AtomData, KerogenData
from base.periodizer import Periodizer
from base.reader import Reader
from base.trajectory import Trajectory
from processes.segmentaion import Segmentator
from visualizer.visualizer import Visualizer
import argparse

# from examples.utils import
from base.utils import create_box_mask
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


def read_structures(
    path_to_structure: str,
) -> List[Tuple[List[AtomData], Tuple[float, float, float]]]:
    structures = []
    num = 0
    count_to_skip = 10
    with open(path_to_structure) as f:
        atoms, size = Reader.read_raw_struct_ff(f)
        is_end = False
        while not is_end:
            structures.append((num, np.array(atoms), size))
            atoms, size = Reader.read_raw_struct_ff(f)
            print(" -- Reading struct is ended!")
            for _ in range(count_to_skip):
                is_end = Reader.skip_struct(f)
                if is_end:
                    break
            num += count_to_skip

    return structures


def dynamic_struct_extr(path_to_structure: str) -> None:
    start_time = time.time()
    structures = read_structures(path_to_structure)
    print(f" -- Count structures: {len(structures)}")

    print(f" -- Reading finished! Elapsed time: {time.time() - start_time}s")
    for num, atoms, size in structures:
        start_time = time.time()

        div = 2
        bbox = Segmentator.cut_cell(size, div)  # type: ignore

        print(f" --- Box size: {bbox.size()}")

        kerogen_data = KerogenData(None, atoms, bbox)  # type: ignore
        # Periodizer.rm_long_edges(kerogen_data)
        # Periodizer.periodize(kerogen_data)

        ref_size = 500
        resolution = bbox.size()[0] / ref_size
        str_resolution = "{:.9f}".format(resolution)
        binarized_file_name = (
            ""
            # f"../data/Kerogen/result_img_{file_name}_rs={ref_size}_mr={methan_radius}_div={div}.npy"
            f"../data/Kerogen/tmp/result_img_num={num}_rs={ref_size}_ar={additional_radius}_resolution={str_resolution}.npy"
        )

        if os.path.isfile(binarized_file_name):
            with open(binarized_file_name, 'rb') as f:  # type: ignore
                img = np.load(f)  # type: ignore
        else:
            img_size = Segmentator.calc_image_size(
                kerogen_data.box.size(), reference_size=ref_size, by_min=True
            )
            segmentator = Segmentator(
                kerogen_data,
                img_size,
                size_data=get_size,
                radius_extention=get_ext_size,
                partitioning=1,
            )
            img = segmentator.binarize()
            np.save(binarized_file_name, img)  # type: ignore

        print(f" --- Image size: {img.shape}")
        str_resolution = "{:.9f}".format(resolution)
        raw_file_name = (
            # f"../data/Kerogen/result_raw_img_{file_name}_is={img.shape}_mr={methan_radius}_div={div}.raw"
            f"../data/Kerogen/tmp/result_raw_img_num={num}_is={img.shape}_mr={additional_radius}_resolution={str_resolution}nm.raw"
        )

        img = 1 - img
        write_binary_file(img, raw_file_name)

        print(
            f" -- Binarization struct {num} is finished! Elapsed time: {time.time() - start_time}s"
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--structure_path',
        type=str,
        # default="../data/methan_traj/meth_1.7_micros.1.gro"
        default="../data/Kerogen/traj.gro",
    )
    args = parser.parse_args()

    dynamic_struct_extr(args.structure_path)
