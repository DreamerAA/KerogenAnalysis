import sys
import os 
from pathlib import Path
from os import listdir
from os.path import realpath, join, isfile
import time
from typing import List, Tuple
import numpy as np
import argparse
import networkx as nx
import json
from scipy import ndimage
import subprocess

path = Path(realpath(__file__))
parent_dir = str(path.parent.parent.absolute())
sys.path.append(parent_dir)

from base.boundingbox import BoundingBox, Range
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
    indexes: np.ndarray,
    cut_cell: bool,
    ref_size: int,
) -> None:
    
    structures = Reader.read_structures_by_num(path_to_structure, indexes)
    for num, atoms, size in structures:
        bbox = Segmentator.cut_cell(size, 4) if cut_cell else Segmentator.full_cell(size)  # type: ignore

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
        float_file_name = (
            # f"../data/Kerogen/result_img_{file_name}_rs={ref_size}_mr={methan_radius}_div={div}.npy"
            path_to_save
            + f"./float_img_num={num}_cell={prf_cell}_is={img_size}_ar={additional_radius}_resolution={str_resolution}.npy"
        )
        binarized_file_name = (
            # f"../data/Kerogen/result_img_{file_name}_rs={ref_size}_mr={methan_radius}_div={div}.npy"
            path_to_save
            + f"./bin_img_num={num}_cell={prf_cell}_is={img_size}_ar={additional_radius}_resolution={str_resolution}.raw"
        )

        if os.path.isfile(float_file_name) and os.path.isfile(binarized_file_name):
            with open(float_file_name, 'rb') as f:  # type: ignore
                float_img = np.load(f)  # type: ignore
            with open(binarized_file_name, 'rb') as f:
                bin_img = np.fromfile(f, dtype=np.uint8)   # type: ignore
        else:
            print(f" --- Image size for calculating: {img_size}")
            segmentator = Segmentator(
                kerogen_data,
                img_size,
                size_data=get_size,
                radius_extention=get_ext_size,
                partitioning=1,
            )
            float_img = segmentator.dist_map()
            bin_img = segmentator.binarize()
            np.save(float_file_name, float_img)  # type: ignore
            bin_img.astype('uint8').tofile(binarized_file_name)
        
        float_img = ndimage.gaussian_filter(float_img, 4)

        # float_img = np.pad(float_img, [(1, 1), (1, 1), (1, 1)], 'maximum')
        # Visualizer.draw_float_img(float_img, 0.11, kerogen_data.box)

    # Visualizer.show()

def run_extractor(prefix: str, bin_path: str, config_path: str):
    img_path = join(prefix, "images")
    for file in listdir(img_path):
        if not file.startswith("bin"):
            continue
        str_num = (file.split("_")[2]).split('=')[1]
        num = int(str_num)
        str_size = (file.split("_")[4]).split('=')[1]
        size = tuple([int(s) for s in (str_size[1:-1]).split(',')])
        str_resolution = ((file.split("_")[6]).split('=')[1])[:-4]

        config = json.load(open(config_path))
        config["input_data"]["filename"] = join(img_path, file)
        config["input_data"]["size"]["x"] = size[0]
        config["input_data"]["size"]["y"] = size[1]
        config["input_data"]["size"]["z"] = size[2]

        config["output_data"]["statoil_prefix"] = join(prefix, "pnm", f"bin_num={num}_size=({size[0]},{size[1]},{size[2]})_resolution={str_resolution}")
        with open(config_path, "w") as file:
            json.dump(config, file)

        process = subprocess.Popen(
            [
                bin_path,
                config_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # wait for the process to terminate
        out, err = process.communicate()
        errcode = process.returncode
        if errcode != 0:
            print("Error!!!")
        print(f"File {file} is calculated")
                

def read_and_draw_pnm(pnm_path: str, path_to_img):
    r, tl, ll, positions = Reader.read_pnm_ext_data(pnm_path)

    colors_data = {
        0: (1., 0.0, 0.0, 1.0),
    }
    scales_data = {
        i: rad for i, rad in enumerate(r)
    }

    node_pos = {idx: pos for idx, pos in enumerate(positions)}

    bord = positions.max(axis=0)
    bbox = BoundingBox(
        Range(0, bord[0]),
        Range(0, bord[1]),
        Range(0, bord[2]),
    )

    graph = nx.Graph()
    graph.add_nodes_from(
        [
            (i, {"color_id": 0, "scale_id": i})
            for i, _ in enumerate(r)
        ]
    )
    graph.add_weighted_edges_from([(l[0], l[1], tl[i, 0]) for i, l in enumerate(ll)])

    float_img = np.load(path_to_img)
    float_img = ndimage.gaussian_filter(float_img, 4)
    float_img = np.pad(float_img, [(1, 1), (1, 1), (1, 1)], 'maximum')
    float_img[:, (float_img.shape[1]//2):, :] = 10.*float_img.max()

    Visualizer.draw_pnm_and_img(  # type: ignore
        graph,
        float_img,
        node_pos,
        0.11,
        bbox,
        size_node=0.8,
        size_edge=0.000003,
        colors_data=colors_data,
        scales_data=scales_data,
        scale='non',
    )
    Visualizer.show()

if "__main__" == __name__:
    indexes = [16275000, 796275000, 1640025000]
    ker_prefix = "/media/andrey/Samsung_T5/PHD/Kerogen/type1matrix/300K/ch4/"
    pnm_prefix = "/home/andrey/DigitalCore/PNE/pore-network-extraction/build/clang-15-release-cpu/"

    print(f"indexes: {indexes}")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--structure_path',
        type=str,
        default=ker_prefix + "type1.ch4.1.gro",
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default=ker_prefix + "/images/",
    )
    parser.add_argument(
        '--extractor_path',
        type=str,
        default=pnm_prefix + "/bin/extractor_example",
    )
    parser.add_argument(
        '--config_extractor_path',
        type=str,
        default=pnm_prefix + "/example/config/ExtractorExampleConfig.json",
    )

    parser.add_argument(
        '--path_to_pnm',
        type=str,
        default=pnm_prefix + "/example/config/ExtractorExampleConfig.json",
    )

    args = parser.parse_args()

    # extanded_struct_extr(
    #     args.structure_path,
    #     args.save_path,
    #     indexes,
    #     True,
    #     200,
    # )

    # run_extractor(ker_prefix, args.extractor_path, args.config_extractor_path)

    read_and_draw_pnm(
        join(ker_prefix, "pnm", "bin_num=1640025000_size=(200,200,200)_resolution=0.031154750.raw"),
        join(ker_prefix, "images", "float_img_num=1640025000_cell=partcell_is=(200, 200, 200)_ar=0.0_resolution=0.031154750.npy")
    )