import sys
import os
from pathlib import Path
from os.path import realpath
import time
from typing import List, Tuple
import numpy as np
import argparse


path = Path(realpath(__file__))
parent_dir = str(path.parent.parent.absolute())
sys.path.append(parent_dir)

from base.boundingbox import BoundingBox
from base.kerogendata import AtomData, KerogenData
from base.periodizer import Periodizer
from base.reader import Reader, StepsInfo
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


def read_structures(
    path_to_structure: str,
) -> List[Tuple[List[AtomData], Tuple[float, float, float]]]:
    structures = []

    count_to_skip = 65
    with open(path_to_structure) as f:
        atoms, size, n = Reader.read_raw_struct_ff(f)
        is_end = False
        while not is_end:
            structures.append((n, np.array(atoms), size))
            atoms, size, n = Reader.read_raw_struct_ff(f)
            print(" -- Reading struct is ended!")
            for _ in range(count_to_skip):
                is_end = Reader.skip_struct(f)
                if is_end:
                    break

    return structures


def read_structures_by_num(
    path_to_structure: str, indexes: List[int]
) -> List[Tuple[List[AtomData], Tuple[float, float, float]]]:
    structures = []
    info = StepsInfo()
    with open(path_to_structure) as f:
        is_end = False
        while not is_end:
            num = Reader.read_head_struct(f, info)
            if num == -1:
                is_end = True
                continue

            if num not in indexes:
                is_end = Reader.skip_struct_main_part(f)
                if is_end:
                    break
            else:
                atoms, size = Reader.read_raw_struct_ff_main(f)
                structures.append((num, np.array(atoms), size))
                print(" -- Reading struct is ended!")

    return structures


def dynamic_struct_extr(path_to_structure: str) -> None:
    start_time = time.time()
    structures = read_structures(path_to_structure)[:1]
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


def extanded_struct_extr(
    path_to_structure: str,
    path_to_save: str,
    indexes: List[int],
    cut_cell: bool,
    ref_size: int,
) -> None:
    start_time = time.time()
    structures = read_structures_by_num(path_to_structure, indexes)

    print(f" -- Count structures: {len(structures)}")
    print(f" -- Count structures after filter: {len(structures)}")

    print(f" -- Reading finished! Elapsed time: {time.time() - start_time}s")
    for num, atoms, size in structures:
        start_time = time.time()

        bbox = Segmentator.cut_cell(size, 2) if cut_cell else Segmentator.full_cell(size)  # type: ignore

        print(f" --- Current num: {num}")

        print(f" --- Box size: {bbox.size()}")

        kerogen_data = KerogenData(None, atoms, bbox)  # type: ignore
        if not kerogen_data.checkPeriodization():
            Periodizer.periodize(kerogen_data)
            # Periodizer.rm_long_edges(kerogen_data)

        resolution = np.array([s for s in size]).min() / ref_size
        str_resolution = "{:.9f}".format(resolution)

        img_size = Segmentator.calc_image_size(
            kerogen_data.box.size(), reference_size=ref_size, by_min=True
        )
        prf_cell = "partcell" if cut_cell else "fullcell"
        binarized_file_name = (
            # f"../data/Kerogen/result_img_{file_name}_rs={ref_size}_mr={methan_radius}_div={div}.npy"
            path_to_save
            + f"./result_img_num={num}_cell={prf_cell}_is={img_size}_ar={additional_radius}_resolution={str_resolution}.npy"
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
            img = segmentator.binarize()
            np.save(binarized_file_name, img)  # type: ignore

        print(f" --- Image size: {img.shape}")
        str_resolution = "{:.9f}".format(resolution)
        raw_file_name = (
            # f"../data/Kerogen/result_raw_img_{file_name}_is={img.shape}_mr={methan_radius}_div={div}.raw"
            path_to_save
            + f"./result_raw_img_num={num}_cell={prf_cell}_is={img.shape}_mr={additional_radius}_resolution={str_resolution}a.raw"
        )

        img = 1 - img
        write_binary_file(img, raw_file_name)

        print(
            f" -- Binarization struct {num} is finished! Elapsed time: {time.time() - start_time}s"
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--def_path',
        type=str,
        default="/media/andrey/Samsung_T5/PHD/Kerogen/type2matrix/300K/h2/",
    )

    parser.add_argument(
        '--structure_path',
        type=str,
        default="./type2.h2.300.gro",
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default="./images/",
    )

    args = parser.parse_args()

    indexes = [2500 + 25000 * i * 200 for i in range(50)] + [249977500]

    extanded_struct_extr(
        args.def_path + args.structure_path,
        args.def_path + args.save_path,
        indexes, True, 250
    )
