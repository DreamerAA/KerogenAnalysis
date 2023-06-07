import os
import random
import time
from typing import IO, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt

from kerogen_data import AtomData, KerogenData
from periodizer import Periodizer
from segmentaion import Segmentator
from visualizer import Visualizer

methan_radius = 0.0

atom_real_sizes = {
    i: s for i, s in enumerate([0.17, 0.152, 0.155, 0.109, 0.18])
}

ext_radius = {
    i: s for i, s in enumerate([methan_radius, methan_radius, methan_radius, 0., methan_radius])
}

def skip_line(file: IO, count: int = 1) -> None: # type: ignore
    for _ in range(count):
        next(file)

def read_atom_info() -> None:
    pass

def random_color() -> Tuple[float, float, float, float]:
    return (
        random.randint(0, 255) / 255.0,
        random.randint(0, 255) / 255.0,
        random.randint(0, 255) / 255.0,
        1.0,
    )

def type_to_type_id(type: str) -> int:
    if type[0] == 'c':
        return 0
    elif type[0] == 'o':
        return 1
    elif type[0] == 'n':
        return 2
    elif type[0] == 'h':
        return 3
    elif type[0] == 's':
        return 4
    return -1

def draw_kerogen_data(kerogen: KerogenData, scale:str='physical')->None:
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

    Visualizer.draw_nxvtk(# type: ignore
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

def test_kerogen_data()->KerogenData:
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

    kerogen_data = KerogenData(g, atoms, (1.0, 1.0, 1.0))
    return kerogen_data

def write_binary_file(array: npt.NDArray[np.int8], file_name:str) -> None:
    with open(file_name, 'wb') as file:
        for i in range(array.shape[2]):
            for j in range(array.shape[1]):
                file.write(bytes(bytearray(array[:, j, i])))


if __name__ == '__main__':
    start_time = time.time()
    # path_to_structure = "../data/Kerogen/confout.gro"
    path_to_structure = "../data/Kerogen/1_pbc_atom.gro"
    # path_to_structure = "../data/Kerogen/ker_wrapped.gro"
    path_to_linklist = "../data/Kerogen/ker.pdb"

    graph = nx.Graph()
    atoms = []
    methane = []
    atom_nums = set()
    with open(path_to_structure) as f:
        skip_line(f, 1)
        count_atoms = int(next(f))
        print(f" --- Count atoms: {count_atoms}")
        read_error = False
        for i in range(count_atoms):
            line = next(f)
            try:
                struct_number = int(line[0:5])
                struct_type = line[5:8]

                number = int(line[15:20]) - 1

                atom_id: str = str(line[8:15])
                atom_id = atom_id.replace(" ", "")
                type_id = type_to_type_id(atom_id)

                x = float(line[20:28])
                y = float(line[28:36])
                z = float(line[36:44])

                data = AtomData(
                    struct_number, struct_type, atom_id, type_id, np.array([x, y, z])
                )

                atom_nums.add(number)

                if "CH4" in struct_type:
                    methane.append(data)
                else:
                    atoms.append(data)
            except Exception as e:
                read_error = True
                print(i, line)
        cell_sizes = next(f)

        str_size = list(filter(lambda x: x != '', cell_sizes.split(' ')))
        size = tuple([float(e) for e in str_size])
    atoms = np.array(atoms)
    div = 4
    box = Segmentator.cut_cell(size, div)# type: ignore
    print(f" --- Box size: {box.size()}")

    removed_atoms = set()
    rm_mask = np.array(range(len(atoms)),dtype=np.bool_)
    cur_ind_atom = 0
    for i,a in enumerate(atoms):
        need_remove = ~box.is_inside(a.pos)
        if need_remove:
            removed_atoms.add(i)
        else:
            a.pos -= box.min()
        rm_mask[i] = ~need_remove
    
    atoms = atoms[rm_mask]

    old_to_new = np.cumsum(rm_mask.astype(np.int32))  

    linked_list = []
    with open(path_to_linklist) as f:
        for line in f:
            if "CONECT" in line:
                n1, n2 = int(line[6:11]) - 1, int(line[11:16]) - 1
                if n1 in removed_atoms or n2 in removed_atoms:
                    continue
                linked_list.append((old_to_new[n1], old_to_new[n2]))

    graph.add_nodes_from(
        [
            (i, {"color_id": atom.type_id, "scale_id": atom.type_id})
            for i, atom in enumerate(atoms)
        ]
    )
    graph.add_edges_from(linked_list)

    kerogen_data = KerogenData(graph, atoms, box) # type: ignore
    Periodizer.rm_long_edges(kerogen_data)
    # # Periodizer.periodize(kerogen_data)

    ref_size = 250
    file_name = (path_to_structure.split("/")[-1]).split(".")[0]
    binarized_file_name = (
        f"../data/Kerogen/result_img_{file_name}_rs={ref_size}_mr={methan_radius}_div={div}.npy"
    )

    if os.path.isfile(binarized_file_name):
        with open(binarized_file_name, 'rb') as f: # type: ignore
            img = np.load(f) # type: ignore
    else:
        img_size = Segmentator.calc_image_size(
            kerogen_data.box.size(), reference_size=ref_size, by_min=True
        )
        print(f" --- Image size: {img_size}")
        segmentator = Segmentator(
            kerogen_data,
            img_size,
            size_data=get_size,
            radius_extention=get_ext_size,
            partitioning=5,
        )
        img = segmentator.binarize()
        print(
            f" -- Calculation finished! Elapsed time: {time.time() - start_time}s"
        )
        with open(binarized_file_name, 'wb') as f: # type: ignore
            np.save(f, img) # type: ignore

    raw_file_name = (
        f"../data/Kerogen/result_raw_img_{file_name}_is={img.shape}_mr={methan_radius}_div={div}.raw"
    )
    # write_binary_file(1 - img, raw_file_name)

    Visualizer.draw_img(img, True)

    plt.imshow(img[100,:,:])
    plt.show()

    plt.imshow(img[:,100,:])
    plt.show()

    plt.imshow(img[:,:,100])
    plt.show()

    # if not read_error:
    #     # draw_kerogen_data(kerogen_data,scale="")
    #     draw_kerogen_data(kerogen_data)
