import argparse
import os
import time
from os.path import join

from typing import List

import numpy as np
from base.kerogendata import KerogenData
from base.periodizer import Periodizer
from base.reader import Reader
from utils.utils import kprint
from utils.utils import write_binary_file
from processes.segmentaion import Segmentator

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
    path_to_data: str,
    struct_file_name: str,
    indexes: List[int],
    ref_size: int = 500,
    dev: float = 2.0,
) -> None:
    path_to_structure = join(path_to_data, struct_file_name)
    path_to_save_fimg = join(path_to_data, "float_images")
    path_to_save_img = join(path_to_data, "images")
    path_to_save_rimg = join(path_to_data, "raw_images")
    if not os.path.exists(path_to_save_img):
        os.makedirs(path_to_save_img, exist_ok=True)
    if not os.path.exists(path_to_save_fimg):
        os.makedirs(path_to_save_fimg, exist_ok=True)
    if not os.path.exists(path_to_save_rimg):
        os.makedirs(path_to_save_rimg, exist_ok=True)

    start_time = time.time()
    structures = Reader.read_structures_by_num(path_to_structure, indexes)
    kprint(f"Count structures: {len(structures)}")
    kprint(f"Reading finished! Elapsed time: {time.time() - start_time}s")

    for num, time_ps, atoms, size in structures:
        bbox = Segmentator.cut_cell(size, dev)
        resolution = (
            np.array([bbox.xb_.diff(), bbox.yb_.diff(), bbox.zb_.diff()]).mean()
            / ref_size
        )
        img_size = Segmentator.calc_image_size(
            bbox.size(), reference_size=ref_size, by_min=True
        )
        main_name = f"result-img-num={num}_time-ps={time_ps}_bbox={bbox._short_str()}_resolution={resolution:.9f}"
        float_file_name = join(path_to_save_fimg, main_name + ".npy")
        binarized_file_name = join(path_to_save_img, main_name + ".npy")
        raw_file_name = join(path_to_save_rimg, main_name + ".raw")

        if (
            os.path.isfile(float_file_name)
            and os.path.isfile(binarized_file_name)
            and os.path.isfile(raw_file_name)
        ):
            kprint(f"Skip and load image with num={num}")
        else:
            kprint(f"Run segmentation for num={num}")
            start_time = time.time()
            kerogen_data = KerogenData(None, atoms, bbox)  # type: ignore
            if not kerogen_data.checkPeriodization():
                Periodizer.periodize(kerogen_data)
            segmentator = Segmentator(
                kerogen_data,
                img_size,
                size_data=get_size,
                radius_extention=get_ext_size,
                partitioning=2,
            )
            # kprint("Start dist map")
            if not os.path.isfile(float_file_name):
                float_img = segmentator.dist_map()
                np.save(float_file_name, float_img)
            if not os.path.isfile(binarized_file_name) or not os.path.isfile(
                raw_file_name
            ):
                img = 1 - segmentator.binarize(num_workers=6)
                np.save(binarized_file_name, img)
                write_binary_file(img, raw_file_name)
            kprint(
                f"Binarization struct {num} is finished! Elapsed time: {time.time() - start_time}s"
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--def_path',
        type=str,
        default="/media/andrey/Samsung_T5/PHD/Kerogen/type1matrix/300K/ch4/",
    )

    parser.add_argument(
        '--structure_file_name',
        type=str,
        default="type1.ch4.300.gro",
    )

    args = parser.parse_args()

    start_step = 25000
    full_count_steps = 6612
    step_size = 250000
    last_step = full_count_steps * step_size + start_step

    count_slices = 3

    mode = "all"
    if mode == "all":
        step = int(full_count_steps / count_slices)
        indexes = [
            start_step + step_size * i * step for i in range(count_slices)
        ]
        if last_step not in indexes:
            indexes.append(last_step)
    else:
        step = 1
        indexes = [
            start_step + step_size * i * step for i in range(count_slices)
        ]

    extanded_struct_extr(
        args.def_path,
        args.structure_file_name,
        indexes,
        ref_size=250,
        dev=4.0,
    )
