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
    ref_size: int,
) -> None:
    path_to_structure = join(path_to_data, struct_file_name)
    path_to_save_img = join(path_to_data, "images")

    start_time = time.time()
    structures = Reader.read_structures_by_num(path_to_structure, indexes)
    print(f" -- Count structures: {len(structures)}")
    print(f" -- Reading finished! Elapsed time: {time.time() - start_time}s")

    image_infos: tuple[int, str] = []
    for num, time_ps, atoms, size in structures:
        start_time = time.time()

        bbox = Segmentator.cut_cell(size, 2)
        resolution = np.array([s for s in size]).min() / ref_size
        img_size = Segmentator.calc_image_size(
            bbox.size(), reference_size=ref_size, by_min=True
        )
        binarized_file_name = join(
            path_to_save_img,
            f"result-img-num={num}_time-ps={time_ps}_bbox={bbox._short_str()}_resolution={resolution:.9f}.npy",
        )
        raw_file_name = join(
            path_to_save_img,
            f"result-img-num={num}_time-ps={time_ps}_bbox={bbox._short_str()}_resolution={resolution:.9f}.raw",
        )
        image_infos.append((time_ps, binarized_file_name))

        if os.path.isfile(binarized_file_name):
            kprint(f"Skip and load image with num={num}")
            with open(binarized_file_name, 'rb') as f:  # type: ignore
                img = np.load(f)  # type: ignore
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
            img = 1 - segmentator.binarize()
            np.save(binarized_file_name, img)  # type: ignore
            write_binary_file(img, raw_file_name)

            kprint(
                f"Binarization struct {num} is finished! Elapsed time: {time.time() - start_time}s"
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--def_path',
        type=str,
        default="/media/andrey/Samsung_T5/PHD/Kerogen/type1matrix/300K/h2/",
    )

    parser.add_argument(
        '--structure_file_name',
        type=str,
        default="type1.h2.300.gro",
    )

    args = parser.parse_args()

    start_step = 25000
    full_count_steps = 6612
    step_size = 250000
    last_step = full_count_steps * step_size + start_step

    count_slices = 200

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
        250,
    )
