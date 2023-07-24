import os
import random
import time
from typing import IO, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt

from base.boundingbox import BoundingBox, Range
from base.kerogendata import AtomData, KerogenData
from base.periodizer import Periodizer
from base.reader import Reader
from base.trajectory import Trajectory
from processes.segmentaion import Segmentator
from visualizer.visualizer import Visualizer

methan_radius = 0.0

atom_real_sizes = {
    i: s for i, s in enumerate([0.17, 0.152, 0.155, 0.109, 0.18])
}

ext_radius = {
    i: s
    for i, s in enumerate(
        [methan_radius, methan_radius, methan_radius, 0.0, methan_radius]
    )
}


def random_color() -> Tuple[float, float, float, float]:
    return (
        random.randint(0, 255) / 255.0,
        random.randint(0, 255) / 255.0,
        random.randint(0, 255) / 255.0,
        1.0,
    )


def draw_kerogen_data(kerogen: KerogenData, scale: str = 'physical') -> None:
    colors_data = {
        0: (0.25, 0.25, 0.25, 1.0),
        1: (1.0, 0.0, 0.0, 1.0),
        2: (0.0, 0, 1.0, 1.0),
        3: (0.75, 0.75, 0.75, 1.0),
        4: (1.0, 1.0, 0.0, 1.0),
    }
    node_pos = kerogen.positionsAsDict()
    if scale == "physical":
        scales_data = atom_real_sizes
    else:
        scales_data = {i: 0.05 for i in range(5)}

    Visualizer.draw_nxvtk(  # type: ignore
        kerogen.graph,
        node_pos,
        size_node=1,
        size_edge=0.02,
        colors_data=colors_data,
        scales_data=scales_data,
        scale='non',
    )


def get_size(type_id: int) -> float:
    return atom_real_sizes[type_id]


def get_ext_size(type_id: int) -> float:
    return ext_radius[type_id]


def test_kerogen_data() -> KerogenData:
    atoms = [
        AtomData(i, "KRG", 'c', 0, np.array([0.25, 0.25, 0.25])),
        AtomData(i, "KRG", 'c', 0, np.array([0.75, 0.25, 0.25])),
        AtomData(i, "KRG", 'c', 0, np.array([0.25, 0.75, 0.25])),
        AtomData(i, "KRG", 'c', 0, np.array([0.75, 0.75, 0.25])),
        AtomData(i, "KRG", 'o', 1, np.array([0.25, 0.25, 0.75])),
        AtomData(i, "KRG", 'o', 1, np.array([0.75, 0.25, 0.75])),
        AtomData(i, "KRG", 'o', 1, np.array([0.25, 0.75, 0.75])),
        AtomData(i, "KRG", 'o', 1, np.array([0.75, 0.75, 0.75])),
    ]
    g = nx.Graph()
    g.add_nodes_from(
        [
            (i, {"color_id": atom.type_id, "scale_id": atom.type_id})
            for i, atom in enumerate(atoms)
        ]
    )
    kerogen_data = KerogenData(g, atoms, BoundingBox(Range(0, 1), Range(0, 1), Range(0, 1)))
    return kerogen_data


def write_binary_file(array: npt.NDArray[np.int8], file_name: str) -> None:
    with open(file_name, 'wb') as file:
        for i in range(array.shape[2]):
            for j in range(array.shape[1]):
                file.write(bytes(bytearray(array[:, j, i])))


def filter_linked_list(link_list, atom_to_rm) -> List[tuple[int, int]]:
    return [
        ll
        for ll in link_list
        if ll[0] not in atom_to_rm and ll[1] not in atom_to_rm
    ]


def reset_atom_id(link_list, old_to_new) -> List[tuple[int, int]]:
    return [(old_to_new[l1], old_to_new[l2]) for l1, l2 in link_list]


def create_box_mask(atoms: List[AtomData], box: BoundingBox):
    removed_atoms = set()
    rm_mask = np.array(range(len(atoms)), dtype=np.bool_)
    for i, a in enumerate(atoms):
        rm_mask[i] = box.is_inside(a.pos)
        if ~rm_mask[i]:
            removed_atoms.add(i)
    return removed_atoms, rm_mask


if __name__ == '__main__':
    start_time = time.time()
    # path_to_structure = "../data/Kerogen/confout.gro"
    path_to_structure = "../data/Kerogen/1_pbc_atom.gro"
    # path_to_structure = "../data/Kerogen/ker_wrapped.gro"
    path_to_linklist = "../data/Kerogen/ker.pdb"
    path_to_traj = "../data/Kerogen/meth_0.5_micros.gro"

    atoms, size, linked_list = Reader.read_struct_and_linked_list(
        path_to_structure, path_to_linklist
    )

    # div = 4
    # box = Segmentator.cut_cell(size, div)# type: ignore
    trajectories = Trajectory.read_trajectoryes(path_to_traj)
    start = 2
    num_traj = -1
    for i, trj in enumerate(trajectories[start:]):
        if not trj.is_intersect_borders():
            num_traj = i + start
            bbox = trj.trjbox()
            print(f" --- Choose i:{i+start} trajectory")
            break

    print(f" --- Box size: {bbox.size()}")

    removed_atoms, rm_mask = create_box_mask(atoms, bbox)
    atoms = atoms[rm_mask]
    old_to_new = np.cumsum(rm_mask.astype(np.int32)) - 1
    linked_list = filter_linked_list(linked_list, removed_atoms)
    linked_list = reset_atom_id(linked_list, old_to_new)

    graph = nx.Graph()
    graph.add_nodes_from(
        [
            (i, {"color_id": atom.type_id, "scale_id": atom.type_id})
            for i, atom in enumerate(atoms)
        ]
    )
    graph.add_edges_from(linked_list)

    kerogen_data = KerogenData(graph, atoms, bbox)  # type: ignore
    Periodizer.rm_long_edges(kerogen_data)
    # # Periodizer.periodize(kerogen_data)

    ref_size = 150
    file_name = (path_to_structure.split("/")[-1]).split(".")[0]
    binarized_file_name = (
        ""
        # f"../data/Kerogen/result_img_{file_name}_rs={ref_size}_mr={methan_radius}_div={div}.npy"
        f"../data/Kerogen/result_img_{file_name}_rs={ref_size}_mr={methan_radius}_num_traj={num_traj}.npy"
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
        print(
            f" -- Calculation finished! Elapsed time: {time.time() - start_time}s"
        )
        with open(binarized_file_name, 'wb') as f:  # type: ignore
            np.save(f, img)  # type: ignore

    print(f" --- Image size: {img.shape}")
    raw_file_name = (
        # f"../data/Kerogen/result_raw_img_{file_name}_is={img.shape}_mr={methan_radius}_div={div}.raw"
        f"../data/Kerogen/result_raw_img_{file_name}_is={img.shape}_mr={methan_radius}_num_traj={num_traj}.raw"
    )
    # write_binary_file(1 - img, raw_file_name)

    # Visualizer.draw_img(1-img, True)
    Visualizer.draw_img_trj(img, bbox, trajectories[num_traj], True)

    # plt.imshow(img[100,:,:])
    # plt.show()

    # plt.imshow(img[:,100,:])
    # plt.show()

    # plt.imshow(img[:,:,100])
    # plt.show()

    # if not read_error:
    #     # draw_kerogen_data(kerogen_data,scale="")
    #     draw_kerogen_data(kerogen_data)
