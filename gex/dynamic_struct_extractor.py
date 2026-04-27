import argparse
import os
import pickle
import re
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
    times: List[int],
    ref_size: int,
    slice_len: int = 100,
) -> None:
    path_to_structure = join(path_to_data, struct_file_name)
    path_to_save_img = join(path_to_data, "images")
    path_to_save_rimg = join(path_to_data, "raw_images")
    path_to_save_structs = join(path_to_data, "structures")
    if not os.path.exists(path_to_save_img):
        os.makedirs(path_to_save_img, exist_ok=True)
    if not os.path.exists(path_to_save_rimg):
        os.makedirs(path_to_save_rimg, exist_ok=True)
    if not os.path.exists(path_to_save_structs):
        os.makedirs(path_to_save_structs, exist_ok=True)

    atimes = np.asarray(times, dtype=np.int32)
    aindexes = np.asarray(indexes, dtype=np.int32)

    mask = np.asarray(
        [
            os.path.isfile(
                join(
                    path_to_save_structs,
                    f"struct-num={num}_time-ps={time}.pickle",
                )
            )
            for num, time in zip(aindexes, atimes)
        ],
        dtype=bool,
    )
    aindexes = aindexes[~mask]
    atimes = atimes[~mask]

    count_steps = len(aindexes) // slice_len + 1
    for i in range(count_steps):
        start_time = time.time()
        start = i * slice_len
        stop = min((i + 1) * slice_len, len(aindexes))
        cur_indexes = aindexes[start:stop]
        if len(cur_indexes) == 0:
            continue
        structures = Reader.read_structures_by_num(
            path_to_structure, cur_indexes
        )
        for struct in structures:
            num, time_ps, _, _ = struct
            save_file_name = join(
                path_to_save_structs,
                f"struct-num={num}_time-ps={time_ps}.pickle",
            )
            with open(save_file_name, 'wb') as f:
                pickle.dump(struct, f)

        print(f" -- Count structures step: {i+1} from {count_steps}")
        print(
            f" -- Reading finished! Elapsed time: {time.time() - start_time}s"
        )

    pattern = re.compile(
        r"struct-num=(?P<step>\d+)"
        r"_time-ps=(?P<time_ps>\d+(?:\.\d+)?).pickle"
    )
    filenames = []
    for file in os.listdir(path_to_save_structs):
        match = pattern.match(file)
        if not match:
            raise Exception(f"Bad file name: {file}")
        data = match.groupdict()
        if int(data["step"]) in indexes:
            assert int(data["time_ps"]) in times
            filenames.append(join(path_to_save_structs, file))

    for i, filename in enumerate(filenames):
        with open(filename, 'rb') as f:
            num, time_ps, atoms, size = pickle.load(f)

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
            path_to_save_rimg,
            f"result-img-num={num}_time-ps={time_ps}_bbox={bbox._short_str()}_resolution={resolution:.9f}.raw",
        )

        if os.path.isfile(binarized_file_name):
            kprint(
                f"Skip and load image with num={num}. Its {i + 1} from {len(filenames)}"
            )
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
            img = 1 - segmentator.binarize(num_workers=4)
            np.save(binarized_file_name, img)  # type: ignore
            write_binary_file(img, raw_file_name)

            kprint(
                f"Binarization struct {num} is finished! Elapsed time: {time.time() - start_time}s. Its {i + 1} from {len(filenames)}"
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

    full_count_steps = 6612

    start_time = 50
    step_time_size = 500

    start_step = 25000
    step_size = 250000

    last_step = full_count_steps * step_size + start_step
    last_time_step = full_count_steps * step_time_size + start_time

    count_slices = 500

    def gen_list(start, step_size, step, count):
        return [start + step_size * i * step for i in range(count)]

    mode = "all"
    # mode = "part"
    if mode == "all":
        step = int(full_count_steps / count_slices)
        indexes = gen_list(start_step, step_size, step, count_slices)
        times = gen_list(start_time, step_time_size, step, count_slices)
        if last_step not in indexes:
            indexes.append(last_step)
            times.append(last_time_step)
    else:
        step = 1
        indexes = gen_list(start_step, step_size, step, count_slices)
        times = gen_list(start_time, step_time_size, step, count_slices)

    extanded_struct_extr(
        args.def_path,
        args.structure_file_name,
        indexes,
        times,
        250,
    )
