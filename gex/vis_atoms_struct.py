import pickle
import sys
import os
from pathlib import Path
from os import listdir
from os.path import realpath, join, isfile
import time
import re
from typing import List, Tuple
import numpy as np
import argparse
import networkx as nx
import json
from scipy import ndimage
import subprocess

from base.boundingbox import BoundingBox, Range
from base.reader import Reader
from visualizer.visualizer import Visualizer

pattern = re.compile(
    r"bbox=\("
    r"x=\((?P<x_min>[\d.]+)-(?P<x_max>[\d.]+)\)_"
    r"y=\((?P<y_min>[\d.]+)-(?P<y_max>[\d.]+)\)_"
    r"z=\((?P<z_min>[\d.]+)-(?P<z_max>[\d.]+)\)"
    r"\)_resolution=(?P<resolution>[\d.]+).npy"
)

atom_real_sizes = {
    i: s * 0.5 for i, s in enumerate([0.17, 0.152, 0.155, 0.109, 0.18])
}
atom_colors = {
    i: color
    for i, color in enumerate(
        [
            (0.4, 0.4, 0.4),  # C — темно-серый
            (0.85, 0.1, 0.1),  # O — мягкий красный
            (0.2, 0.3, 0.9),  # N — насыщенный синий
            (0.95, 0.95, 0.95),  # H — не чисто белый
            (0.9, 0.8, 0.1),  # S — теплый желтый
        ]
    )
}


def read_and_draw_atoms_struct(
    path_to_data: str,
    path_to_img: str,
    index: int,
    time_ps: int,
):
    path_to_structure = join(path_to_data, "type1.ch4.300.gro")
    path_to_linked_list = join(path_to_data, "ker.pdb")
    path_to_save_structs = join(path_to_data, "structures")

    if not os.path.exists(path_to_save_structs):
        os.makedirs(path_to_save_structs, exist_ok=True)

    save_struct_name = join(
        path_to_save_structs,
        f"struct-num={index}_time-ps={time_ps}.pickle",
    )

    match = pattern.search(path_to_img)

    if not match:
        raise ValueError("Не удалось распарсить имя файла")

    data = {k: float(v) for k, v in match.groupdict().items()}
    if not isfile(save_struct_name):
        structure = Reader.read_structures_by_num(path_to_structure, [index])[0]
        with open(save_struct_name, 'wb') as f:
            pickle.dump(structure, f)
    else:
        with open(save_struct_name, 'rb') as f:
            structure = pickle.load(f)

    atoms = structure[2]

    linked_list = Reader.read_linked_list(path_to_linked_list)

    bbox = BoundingBox(
        Range(data["x_min"], data["x_max"]),
        Range(data["y_min"], data["y_max"]),
        Range(data["z_min"], data["z_max"]),
    )

    # 1. Фильтруем атомы, но сохраняем старые индексы
    old_to_new = {}
    filtered_atoms = []

    for old_idx, atom in enumerate(atoms):
        if bbox.is_inside(atom.pos):
            new_idx = len(filtered_atoms)
            old_to_new[old_idx] = new_idx
            filtered_atoms.append(atom)

    # 2. Приводим список связей к новым индексам
    new_links = []
    for i, j in linked_list:
        if i in old_to_new and j in old_to_new:
            new_links.append([old_to_new[i], old_to_new[j]])

    atoms = filtered_atoms
    linked_list = np.asarray(new_links, dtype=np.int64)

    node_pos = {idx: atom.pos for idx, atom in enumerate(atoms)}

    min_connection_size = np.array(list(atom_real_sizes.values())).min() / 2

    graph = nx.Graph()
    graph.add_nodes_from(
        [
            (i, {"color_id": atom.type_id, "scale_id": atom.type_id})
            for i, atom in enumerate(atoms)
        ]
    )
    graph.add_weighted_edges_from(
        [(l[0], l[1], min_connection_size) for i, l in enumerate(linked_list)]
    )

    float_img = np.load(path_to_img)
    float_img = ndimage.gaussian_filter(float_img, 4)
    float_img = np.pad(float_img, [(1, 1), (1, 1), (1, 1)], 'maximum')
    # float_img[:, (float_img.shape[1] // 2) :, :] = 10.0 * float_img.max()

    Visualizer.draw_graph_and_img(  # type: ignore
        graph,
        float_img,
        node_pos,
        bbox,
        size_node=0.8,
        size_edge=min_connection_size,
        colors_data=atom_colors,
        scales_data=atom_real_sizes,
        scale='non',
        isovalue=0.11,
        img_opacity=0.8,
    )
    Visualizer.show()


if "__main__" == __name__:
    main_path = "/media/andrey/Samsung_T5/PHD/Kerogen/type1matrix/300K/ch4/"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path_to_data',
        type=str,
        default=main_path,
    )
    parser.add_argument(
        '--path_to_img',
        type=str,
        default=join(
            main_path,
            "float_images",
            "result-img-num=551025000_time-ps=1102050_bbox=(x=(1.489-4.742)_y=(2.078-5.332)_z=(4.881-8.134))_resolution=0.013015000.npy",
        ),
    )
    parser.add_argument(
        '--index',
        type=int,
        default=551025000,
    )
    parser.add_argument(
        '--time_ps',
        type=int,
        default=551025000,
    )

    args = parser.parse_args()

    read_and_draw_atoms_struct(
        args.path_to_data,
        args.path_to_img,
        args.index,
        args.time_ps,
    )
