import re
import sys
import os
from pathlib import Path
from os.path import join, realpath
import random
import time
from typing import IO, List, Tuple, Any
from scipy import ndimage
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

path = Path(realpath(__file__))
parent_dir = str(path.parent.parent.absolute())
sys.path.append(parent_dir)

from base.boundingbox import BoundingBox, KerogenBox, Range
from base.kerogendata import AtomData, KerogenData
from base.periodizer import Periodizer
from base.reader import Reader
from base.trajectory import Trajectory
from processes.segmentaion import Segmentator
from visualizer.visualizer import Visualizer


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
        size_node=0.5,
        size_edge=0.00,
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
    kerogen_data = KerogenData(
        g, atoms, BoundingBox(Range(0, 1), Range(0, 1), Range(0, 1))
    )
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


def main_part_struct(path_to_structure: str, path_to_linklist: str):
    start_time = time.time()

    atoms, size, linked_list = Reader.read_struct_and_linked_list(
        path_to_structure, path_to_linklist
    )

    div = 2
    bbox = Segmentator.cut_cell(size, div)  # type: ignore

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
    # Periodizer.periodize(kerogen_data)

    ref_size = 250
    resolution = bbox.size()[0] / ref_size
    str_resolution = "{:.9f}".format(resolution)
    file_name = (path_to_structure.split("/")[-1]).split(".")[0]
    binarized_file_name = (
        ""
        # f"../data/Kerogen/result_img_{file_name}_rs={ref_size}_mr={methan_radius}_div={div}.npy"
        f"../data/result_img_{file_name}_rs={ref_size}_mr={additional_radius}_resolution={str_resolution}.npy"
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
    str_resolution = "{:.9f}".format(resolution)
    raw_file_name = (
        # f"../data/Kerogen/result_raw_img_{file_name}_is={img.shape}_mr={methan_radius}_div={div}.raw"
        f"../data/result_raw_img_{file_name}_is={img.shape}_mr={additional_radius}_resolution={str_resolution}nm.raw"
    )

    img = 1 - img
    write_binary_file(img, raw_file_name)

    Visualizer.draw_img(img, True, bbox)
    # Visualizer.draw_img_trj(img, bbox, trajectories[num_traj], True)

    plt.imshow(img[100, :, :])
    plt.show()

    # plt.imshow(img[:,100,:])
    # plt.show()

    # plt.imshow(img[:,:,100])
    # plt.show()

    # if not read_error:
    #     # draw_kerogen_data(kerogen_data,scale="")
    #     draw_kerogen_data(kerogen_data)


def kerogen_struct_vis(
    path_to_structure: str, path_to_linklist: str, struct_number: int
):
    atoms, size, linked_list = Reader.read_struct_and_linked_list(
        path_to_structure, path_to_linklist
    )
    bbox = Segmentator.full_cell(size)  # type: ignore

    print(f" --- Box size: {bbox.size()}")

    cut_atoms = []
    num_to_nn = {}
    for i, a in enumerate(atoms):
        if a.struct_number == struct_number:
            cut_atoms.append(a)
            num_to_nn[i + 1] = len(num_to_nn)

    graph = nx.Graph()
    graph.add_nodes_from(
        [
            (i, {"color_id": atom.type_id, "scale_id": atom.type_id})
            for i, atom in enumerate(cut_atoms)
        ]
    )
    graph.add_edges_from(
        [
            (num_to_nn[n1], num_to_nn[n2])
            for n1, n2 in linked_list
            if (
                atoms[n1 - 1].struct_number == struct_number
                and atoms[n2 - 1].struct_number == struct_number
            )
        ]
    )

    kerogen_data = KerogenData(graph, cut_atoms, bbox)  # type: ignore
    # Periodizer.rm_long_edges(kerogen_data)
    # Periodizer.periodize(kerogen_data)

    draw_kerogen_data(kerogen_data)


def cut_image_by_image_size(
    image: np.ndarray,
    bbox: BoundingBox,
    cut_params: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
) -> tuple[np.ndarray, BoundingBox]:
    cut_img = image[
        cut_params[0][0] : cut_params[0][1],
        cut_params[1][0] : cut_params[1][1],
        cut_params[2][0] : cut_params[2][1],
    ]
    image_size = image.shape
    bbox_size = np.array(bbox.size())
    voxel_size = np.divide(bbox_size, image_size)

    bbox_axes = [bbox.xb_, bbox.yb_, bbox.zb_]

    cut_bbox = BoundingBox(
        *(
            Range(
                *[ba.min_ + cut_params[i][j] * voxel_size[i] for j in range(2)]
            )
            for i, ba in enumerate(bbox_axes)
        )
    )

    return cut_img, cut_bbox


def voxelized_struct_and_traj_vis(
    path_to_data: str,
    num_trj: int,
    resolution: float = 0.031154750,
    real_size_full_cell: tuple[float, float, float] = (
        6.23095,
        7.41057,
        13.01500,
    ),
    use_floating=False,
):
    prefix = path_to_data
    path_to_traj: str = join(prefix, "trj.gro")
    path_to_images: str = join(prefix, "images")
    trajectories = Trajectory.read_trajectoryes(path_to_traj)

    trj = trajectories[num_trj]

    if use_floating:
        for file in os.listdir(path_to_images):
            if not file.endswith(".npy") or not file.startswith("float"):
                continue

            with open(join(path_to_images, file), 'rb') as f:  # type: ignore
                start_img = np.load(f)  # type: ignore

                pattern = re.compile(
                    r"float_img_num=(?P<step>\d+)_cell=.*?_is=\((?P<nx>\d+),\s*(?P<ny>\d+),\s*(?P<nz>\d+)\)_ar=.*?_resolution=(?P<resolution>\d+(?:\.\d+)?)"
                )

                match = pattern.search(file)
                if match is None:
                    raise ValueError(
                        "Строка не соответствует ожидаемому формату"
                    )

                step = int(match.group("step"))
                resolution = float(match.group("resolution"))
                print(step, resolution)

                img_shape = start_img.shape
                img_real_size = np.array(img_shape) * resolution
                center_img = np.array(real_size_full_cell) * 0.5
                xyz_img_min = center_img - img_real_size * 0.5
                xyz_img_max = center_img + img_real_size * 0.5
                real_img_box = BoundingBox(
                    *(
                        Range(mmin, mmax)
                        for mmin, mmax in zip(xyz_img_min, xyz_img_max)
                    )
                )
                break

        float_img = ndimage.gaussian_filter(start_img, 4)
        float_img = np.pad(float_img, [(1, 1), (1, 1), (1, 1)], 'maximum')
        float_img[:, (float_img.shape[1] // 2) :, :] = 10.0 * float_img.max()
        Visualizer.draw_img_trj(
            float_img, real_img_box, trj, False, isovalue=0.11
        )
    else:
        for file in os.listdir(path_to_images):
            if (
                not file.endswith(".npy")
                or not file.startswith("result")
                or "16275000" not in file
            ):
                continue

            with open(join(path_to_images, file), 'rb') as f:  # type: ignore
                start_img = np.load(f)  # type: ignore

                pattern = re.compile(
                    r"result_img_num=(?P<step>\d+)_cell=.*?_is=\((?P<nx>\d+),\s*(?P<ny>\d+),\s*(?P<nz>\d+)\)_ar=.*?_resolution=(?P<resolution>\d+(?:\.\d+)?)"
                )

                match = pattern.search(file)
                if match is None:
                    raise ValueError(
                        "Строка не соответствует ожидаемому формату"
                    )

                step = int(match.group("step"))
                resolution = float(match.group("resolution"))
                print(step, resolution)

                img_shape = start_img.shape
                img_real_size = np.array(img_shape) * 4 * resolution
                center_img = np.array(real_size_full_cell) * 0.5
                xyz_img_min = center_img - img_real_size * 0.5
                xyz_img_max = center_img + img_real_size * 0.5
                real_img_box = BoundingBox(
                    *(
                        Range(mmin, mmax)
                        for mmin, mmax in zip(xyz_img_min, xyz_img_max)
                    )
                )
                break

        img, bbox = cut_image_by_image_size(
            start_img, real_img_box, tuple((200, 300) for i in range(3))
        )
        Visualizer.draw_img_trj(img, bbox, trj, True)


if __name__ == '__main__':
    # prefix = "/media/andrey/Samsung_T5/PHD/Kerogen/type1matrix/300K/ch4/"
    # main_part_struct(prefix + "type1.ch4.1.gro", prefix + "ker.pdb")
    # kerogen_struct_vis(prefix + "kerogen_moleculas.gro", prefix + "ker.pdb")

    # voxelized_struct_and_traj_vis(
    #     prefix + "kerogen_moleculas.gro",
    #     prefix + "ker.pdb",
    #     prefix + "trj.gro",
    #     4
    # )

    path_data = "/media/andrey/Samsung_T5/PHD/Kerogen/type1matrix/300K/ch4/"

    voxelized_struct_and_traj_vis(
        path_data,
        17,
    )
