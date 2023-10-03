import os
import random
import time
from typing import IO, List, Tuple
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd
from base.boundingbox import BoundingBox
from base.kerogendata import AtomData, KerogenData
from base.periodizer import Periodizer
from base.reader import Reader
from base.trajectory import Trajectory
from processes.segmentaion import Segmentator
from visualizer.visualizer import Visualizer
import scipy.stats as stats
from scipy.interpolate import UnivariateSpline
from scipy.stats import weibull_min, exponweib
from joblib import Parallel, delayed

from matplotlib.collections import PolyCollection
from matplotlib import cm
from matplotlib.ticker import LinearLocator

additional_radius = 0.0

atom_real_sizes = {
    i: s for i, s in enumerate([0.17, 0.152, 0.155, 0.109, 0.18])
}

ext_radius = {
    i: s
    for i, s in enumerate(
        [additional_radius, additional_radius, additional_radius, 0.0, additional_radius]
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

    kerogen_data = KerogenData(g, atoms, (1.0, 1.0, 1.0))
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


def main_part_struct():
    start_time = time.time()
    # path_to_structure = "../data/confout.gro"
    path_to_structure = "../data/1_pbc_atom.gro"
    # path_to_structure = "../data/ker_wrapped.gro"
    path_to_linklist = "../data/ker.pdb"
    path_to_traj = "../data/meth_0.5_micros.gro"

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
    # # Periodizer.periodize(kerogen_data)

    ref_size = 500
    resolution = bbox.size()[0] / ref_size
    str_resolution = "{:.9f}".format(resolution)
    file_name = (path_to_structure.split("/")[-1]).split(".")[0]
    binarized_file_name = (
        ""
        # f"../data/Kerogen/result_img_{file_name}_rs={ref_size}_mr={methan_radius}_div={div}.npy"
        f"../data/tmp/result_img_{file_name}_rs={ref_size}_mr={additional_radius}_resolution={str_resolution}.npy"
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
        f"../data/tmp/result_raw_img_{file_name}_is={img.shape}_mr={additional_radius}_resolution={str_resolution}nm.raw"
    )

    img = 1 - img
    write_binary_file(img, raw_file_name)

    Visualizer.draw_img(img, True, bbox)
    # Visualizer.draw_img_trj(img, bbox, trajectories[num_traj], True)

    # plt.imshow(img[100, :, :])
    # plt.show()

    # plt.imshow(img[:,100,:])
    # plt.show()

    # plt.imshow(img[:,:,100])
    # plt.show()

    # if not read_error:
    #     # draw_kerogen_data(kerogen_data,scale="")
    #     draw_kerogen_data(kerogen_data)


def extract_weibull_psd(path_to_pnm: str, scale: float, border: float = 0.02):
    path_to_node_2 = path_to_pnm + "_node2.dat"
    path_to_link_2 = path_to_pnm + "_link2.dat"

    radiuses = Reader.read_psd(path_to_node_2)
    radiuses *= scale

    linked_list, t_throat_lengths = Reader.read_pnm_linklist(path_to_link_2)
    mask0 = linked_list[:, 0] < 0
    mask1 = linked_list[:, 1] < 0
    nn1 = linked_list[mask0, 1] - 1
    nn0 = linked_list[mask1, 0] - 1

    node_mask = np.ones(shape=(len(radiuses),), dtype=np.bool_)
    node_mask[nn0] = False
    node_mask[nn1] = False

    radiuses = radiuses[node_mask]
    radiuses.sort()
    radiuses = radiuses[radiuses > border]

    t_throat_lengths = t_throat_lengths[~np.logical_or(mask0, mask1), :]
    throat_lengths = t_throat_lengths[:, 2]
    throat_lengths *= 1e9
    throat_lengths.sort()

    psd_params = exponweib.fit(radiuses)
    tld_params = exponweib.fit(throat_lengths)
    return psd_params, tld_params, radiuses, throat_lengths


def main_pnm_psd_analizer() -> None:
    psd_params, tld_params, radiuses, throat_lengths = extract_weibull_psd("../data/tmp/1_pbc_atom/result", 1e9, 0.02)

    params = exponweib.fit(radiuses)
    nx_rad = np.linspace(radiuses[0], radiuses[-1], 1000)
    fit_y_rad = exponweib.pdf(nx_rad, *psd_params)

    params = exponweib.fit(throat_lengths)
    nx_len = np.linspace(throat_lengths[0], throat_lengths[-1], 1000)
    fit_y_len = exponweib.pdf(nx_len, *tld_params)

    n = 50
    fig, axs = plt.subplots(1, 2)

    p, bb = np.histogram(radiuses, bins=n)
    xdel = bb[1] - bb[0]
    x = bb[:-1] + xdel * 0.5
    pn = p / np.sum(p * xdel)
    axs[0].plot(x, pn, label='Histogram data')
    axs[0].plot(nx_rad, fit_y_rad, label='Fitting')
    axs[0].set_title("PDF(Pore size distribution)")
    axs[0].set_xlabel("Pore diameter (nm)")

    p, bb = np.histogram(throat_lengths, bins=n)
    xdel = bb[1] - bb[0]
    x = bb[:-1] + xdel * 0.5
    pn = p / np.sum(p * xdel)
    axs[1].plot(x, pn, label='Histogram data')
    axs[1].plot(nx_len, fit_y_len, label='Fitting')
    axs[1].set_title("PDF(Throat length distribution)")
    axs[1].set_xlabel("Throat length (nm)")

    plt.legend()
    plt.show()


def generate_distribution() -> None:
    psd_params, tld_params, radiuses, throat_lengths = extract_weibull_psd("../data/tmp/1_pbc_atom/result", 1e9, 0.02)

    max_rad = radiuses[-1]
    max_diam = (2 * max_rad)
    max_length = 1.5 * max_diam
    nx_rad = np.linspace(0, max_rad, 50)
    nx_lan = np.linspace(0, max_length, 50)

    rng = np.random.default_rng()
    count_segm = 100_000_000
    count_points = 1_000_000

    xyz = np.zeros(shape=(count_points, 3), dtype=np.float32)
    xyz[:, 0] = np.random.uniform(-1.5 * max_rad, 1.5 * max_rad, size=count_points)
    xyz[:, 1] = np.random.uniform(-1.5 * max_rad, 1.5 * max_rad, size=count_points)
    xyz[:, 2] = np.random.uniform(-1.5 * max_rad, 1.5 * max_rad, size=count_points)
    dist = np.sqrt(np.sum(xyz**2, axis=1))

    def sim(ind: int, sl: float, el: float) -> npt.NDArray[np.int32]:
        len_distr = np.zeros(shape=nx_rad.shape, dtype=np.int32)
        for i, rad in enumerate(nx_rad):

            inside = dist < rad

            indexes = rng.integers(0, count_points, size=2 * count_segm, dtype=np.int32)
            indexes = indexes.reshape(count_segm, 2)

            out_ind_1 = inside[indexes[:, 0]]
            out_ind_2 = inside[indexes[:, 1]]

            ind_mask = np.logical_and(out_ind_1, out_ind_2)
            indexes = indexes[ind_mask, :]

            p1xyz = xyz[indexes[:, 0], :]
            p2xyz = xyz[indexes[:, 1], :]
            dxyz = p1xyz - p2xyz
            lengthes = np.sqrt(np.sum(dxyz**2, axis=1))
            len_distr[i] = np.sum(np.logical_and(lengthes >= sl, lengthes <= el))
        print(f" -- Finish {ind}")
        return len_distr

    pi_l_d = Parallel(n_jobs=10)(
        delayed(sim)(i, nx_lan[i - 1], nx_lan[i]) for i in range(1, len(nx_lan))
    )
    # plt.hist(sim_results)
    # n = 50
    # p, bb = np.histogram(pi_l, bins=n)
    # xdel = bb[1] - bb[0]
    # x = bb[:-1] + xdel * 0.5
    # pn = p / np.sum(p * xdel)
    # plt.plot(x, pn, label=f"Diameter = {2*max_rad}")
    # print(len(pi_l) , pi_l)
    # print(len(nx_lan[1:]))
    # plt.plot(nx_lan[:-1] + (nx_lan[1:] - nx_lan[:-1]) * 0.5, pi_l, label="Histogram of Pi(L)")
    # plt.xlabel("Length (nm)")
    # plt.ylabel("Count")
    # plt.legend()
    # plt.show()
    dr = nx_rad[1] - nx_rad[0]
    dl = nx_lan[1] - nx_lan[0]
    new_l = nx_lan[:-1] + (nx_lan[1:] - nx_lan[:-1]) * 0.5

    X = nx_lan[:-1] + (nx_lan[1:] - nx_lan[:-1]) * 0.5
    Y = nx_rad
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros(shape=(len(nx_rad), len(nx_lan) - 1), dtype=np.float32)
    for i, pi in enumerate(pi_l_d):
        Z[:, i] = pi

    print(f" --- Z shape = {Z.shape}")
    print(f" --- Len nx_rad = {nx_rad.shape}")

    for i, r in enumerate(nx_rad):
        s = np.sum(Z[i, :] * dl)
        Z[i, :] /= s if s != 0 else 1

    fit_prob_psd = exponweib.pdf(nx_rad, *psd_params)

    pi_l = np.zeros(shape=(len(nx_rad) - 1,), dtype=np.float32)
    for i, l in enumerate(nx_lan[1:]):
        pi_l[i] = np.sum(Z[:, i] * fit_prob_psd * dr)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    # ax.set_zlim(-1.01, 1.01)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    # ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel("Segemnt length (nm)")
    ax.set_ylabel("Domain radius (nm)")
    ax.set_zlabel("Count")

    print(nx_rad.max())

    pi_l /= np.sum(dl * pi_l)

    plt.figure()
    plt.plot(new_l, pi_l, label="")
    plt.xlabel("Segemnt length (nm)")
    plt.title("PDF (Segment length inside pore) - Pi(L)")

    plt.show()


if __name__ == '__main__':
    # main_part_struct()
    # main_pnm_psd_analizer()
    generate_distribution()
