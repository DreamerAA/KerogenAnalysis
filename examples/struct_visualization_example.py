import sys
import os
from pathlib import Path
from os.path import realpath
import time
from typing import List, Tuple
import numpy as np
import argparse
import networkx as nx

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


def read_structures(
    path_to_structure: str,
) -> List[Tuple[List[AtomData], Tuple[float, float, float]]]:
    structures = []
    
    count_to_skip = 10
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



def extanded_struct_extr(path_to_structure: str, path_to_structure_links: str, path_to_save:str, mfilter, cut_cell: bool, ref_size: int) -> None:
    start_time = time.time()
    # structures = Reader.read_struct_and_linked_list(path_to_structure,)
    structures = read_structures(path_to_structure)
    linked_list = Reader.read_linked_list(path_to_structure_links)
    
    print(f" -- Count structures: {len(structures)}")
    structures = mfilter(structures)
    print(f" -- Count structures after filter: {len(structures)}")

    print(f" -- Reading finished! Elapsed time: {time.time() - start_time}s")
    for num, atoms, size in structures:
        start_time = time.time()

        bbox = Segmentator.cut_cell(size, 2) if cut_cell else Segmentator.full_cell(size)  # type: ignore

        print(f" --- Current num: {num}")

        print(f" --- Box size: {bbox.size()}")

        graph = nx.Graph()
        graph.add_edges_from(linked_list)
        kerogen_data = KerogenData(graph, atoms, bbox)  # type: ignore
        if not kerogen_data.checkPeriodization():
            print("Periodization!")
            Periodizer.periodize(kerogen_data)
            # Periodizer.rm_long_edges(kerogen_data)

        print("Periodization is end!")

        resolution = np.array([s for s in size]).min()/ ref_size
        str_resolution = "{:.9f}".format(resolution)

        img_size = Segmentator.calc_image_size(
            kerogen_data.box.size(), reference_size=ref_size, by_min=True
        )
        prf_cell = "partcell" if cut_cell else "fullcell"
        binarized_file_name = (
            # f"../data/Kerogen/result_img_{file_name}_rs={ref_size}_mr={methan_radius}_div={div}.npy"
            path_to_save + f"./float_img_num={num}_cell={prf_cell}_is={img_size}_ar={additional_radius}_resolution={str_resolution}.npy"
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
        plt.imshow(img[250,:,:])
        plt.show()
           
        img = np.pad(img, [(1,1),(1,1),(1,1)], 'maximum')
        Visualizer.draw_float_img(img, 0.08, kerogen_data.box)
    Visualizer.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--structure_path',
        type=str,
        default="../data/Kerogen/traj_ch4.gro",
    )
    parser.add_argument(
        '--save_path',
        type=str,
        # default="../data/Kerogen/methan_traj/meth_0.5_micros.gro"
        default="../data/Kerogen/time_trapping_results/ch4/",
    )
    parser.add_argument(
        '--linked_list',
        type=str,
        # default="../data/Kerogen/methan_traj/meth_0.5_micros.gro"
        default="../data/Kerogen/ker.pdb",
    )


    args = parser.parse_args()

    def cfilter(s):
        return [s[-1]]

    extanded_struct_extr(args.structure_path, args.linked_list, args.save_path, cfilter, True, 500)