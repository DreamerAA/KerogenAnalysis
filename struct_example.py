import sys
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
import scipy.stats as stats
from scipy.interpolate import UnivariateSpline
from scipy.stats import weibull_min, exponweib
from joblib import Parallel, delayed
from sklearn.metrics import pairwise_distances
from matplotlib.collections import PolyCollection
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from base.boundingbox import BoundingBox
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
    radiuses, throat_lengths = Reader.read_pnm_data(
        path_to_pnm, scale=scale, border=border
    )
    psd_params = exponweib.fit(radiuses)
    tld_params = exponweib.fit(throat_lengths)
    return psd_params, tld_params, radiuses, throat_lengths


def main_pnm_psd_analizer() -> None:
    psd_params, tld_params, radiuses, throat_lengths = extract_weibull_psd(
        "../data/tmp/1_pbc_atom/result", 1e9, 0.02
    )

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
    psd_params, tld_params, radiuses, throat_lengths = extract_weibull_psd(
        "../data/Kerogen/tmp/1_pbc_atom/result", 1e9, 0.02
    )

    max_rad = radiuses[-1]
    max_length = np.sqrt(3 * ((1.5 * max_rad) ** 2))

    cl = 100
    nx_len = np.linspace(0, max_length, cl)

    dl = nx_len[1] - nx_len[0]

    count_points = 10_000
    xyz = np.zeros(shape=(count_points, 3), dtype=np.float32)
    xyz[:, 0] = np.random.uniform(-1, 1, size=count_points)
    xyz[:, 1] = np.random.uniform(-1, 1, size=count_points)
    xyz[:, 2] = np.random.uniform(-1, 1, size=count_points)
    dist = np.sqrt(np.sum(xyz**2, axis=1))
    xyz = xyz[dist < 1, :]

    def upper_tri_masking(
        A: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        m = A.shape[0]
        r = np.arange(m)
        mask = r[:, None] < r
        return A[mask]

    def sim(num: int, m: int, radius: float) -> npt.NDArray[np.float32]:
        len_distr = np.zeros(shape=(cl - 1,), dtype=np.int32)
        xyz_scaled = xyz * radius
        distances = pairwise_distances(xyz_scaled, xyz_scaled, n_jobs=4)
        distances = upper_tri_masking(distances)

        for i in range(1, cl):
            ll, rl = nx_len[(i - 1) : (i + 1)]
            len_distr[i - 1] = np.sum(
                np.logical_and(distances > ll, distances <= rl)
            )
        print(f" --- Result of {num+1} from {m}")
        return len_distr / np.sum(len_distr * dl)

    def PiLD_3D():
        cr = 50
        nx_rad = np.linspace(0, max_rad, cr)

        pi_l_d = np.zeros(shape=(cr - 1, cl - 1))

        # for i, radius in enumerate(nx_rad[1:]):
        #     pi_l_d[i, :] = sim(i, len(nx_rad[1:]), radius)

        pres = Parallel(n_jobs=10)(
            delayed(sim)(i, len(nx_rad[1:]), rad)
            for i, rad in enumerate(nx_rad[1:])
        )
        for i, res in enumerate(pres):
            pi_l_d[i, :] = res

        new_l = nx_len[:-1] + (nx_len[1:] - nx_len[:-1]) * 0.5

        X = np.copy(new_l)
        Y = nx_rad[1:]
        X, Y = np.meshgrid(X, Y)
        Z = np.copy(pi_l_d)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # Plot the surface.
        surf = ax.plot_surface(
            X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False
        )
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.set_xlabel("Segemnt length (nm)")
        ax.set_ylabel("Domain radius (nm)")
        ax.set_zlabel("Count")

    # PiLD_3D()

    sample_rad = radiuses[::30]
    pi_l_d = np.zeros(shape=(sample_rad.shape[0], cl - 1))

    # for i, radius in enumerate(sample_rad):
    #     pi_l_d[i, :] = sim(i, len(sample_rad), radius)

    pres = Parallel(n_jobs=10)(
        delayed(sim)(i, len(sample_rad), rad)
        for i, rad in enumerate(sample_rad)
    )

    for i, res in enumerate(pres):
        pi_l_d[i, :] = res

    pi_l = np.mean(pi_l_d, axis=0)
    new_l = nx_len[:-1] + (nx_len[1:] - nx_len[:-1]) * 0.5

    p, bb = np.histogram(throat_lengths, bins=50)
    xdel = bb[1] - bb[0]
    x = bb[:-1] + xdel * 0.5
    pn = p / np.sum(p * xdel)

    pi_l_save = np.zeros(shape=(pi_l.shape[0], 2), dtype=np.float32)
    pi_l_save[:, 1] = pi_l
    pi_l_save[:, 0] = new_l
    np.save("../data/Kerogen/tmp/1_pbc_atom/pi_l.npy", pi_l_save)
    np.save("../data/Kerogen/tmp/1_pbc_atom/throat_lengths.npy", throat_lengths)

    plt.figure()
    plt.plot(new_l, pi_l, label="Pi(L)")
    plt.plot(x, pn, label='Throat lengths - histogram data')
    plt.xlabel("Segemnt length (nm)")
    plt.title("PDF (Segment length inside pore) - Pi(L)")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # main_part_struct()
    # main_pnm_psd_analizer()
    generate_distribution()
