import time
import math
from pathlib import Path
from os.path import realpath
import sys
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import networkx as nx
import numpy as np
import numpy.typing as npt
from joblib import Parallel, delayed

path = Path(realpath(__file__))
parent_dir = str(path.parent.parent.absolute())
sys.path.append(parent_dir)

from base.boundingbox import KerogenBox, Range
from base.kerogendata import AtomData, KerogenData


class Segmentator:
    def __init__(
        self,
        kerogen: KerogenData,
        img_size: Tuple[int, int, int],
        size_data: Callable[[int], float],
        radius_extention: Callable[[int], float],
        partitioning: int = 2,
        max_atom_size: float = 0.18,
    ):
        self.kerogen = kerogen
        self.img_size = img_size
        self.radius_extention = radius_extention
        vox_sizes = [
            ker_s / float(img_s)
            for ker_s, img_s in zip(kerogen.box.size(), img_size)
        ]

        steps = [ker_s / partitioning for ker_s in kerogen.box.size()]
        shifts = self.kerogen.box.min()
        self.bb: List[KerogenBox] = []
        for i in range(partitioning):
            for j in range(partitioning):
                for k in range(partitioning):
                    nums = [i, j, k]
                    aranges = [
                        Range(
                            num * step - 2 * vs - max_atom_size + s,
                            (num + 1) * step + 2 * vs + max_atom_size + s,
                        )
                        for num, step, vs, s in zip(
                            nums, steps, vox_sizes, shifts
                        )
                    ]
                    self.bb.append(KerogenBox(*aranges))
                    for ind, a in enumerate(kerogen.atoms):
                        if self.bb[-1].is_inside(a.pos):
                            self.bb[-1].add_atom(
                                ind,
                                size_data(a.type_id)
                                + self.radius_extention(a.type_id),
                                a.pos,  # type: ignore
                            )
        # print(f" --- Count Kerogen boxes: {len(self.bb)}")
        for i, bb in enumerate(self.bb):
            bb.commit()
            # print(f" --- Finish commit kerogen box: {i}")

    @staticmethod
    def cut_cell(
        size: Tuple[float, float, float], dev: float = 4.0
    ) -> KerogenBox:
        maxs = np.max(np.array(size))
        mins = np.min(np.array(size))
        ax_ns = min(maxs / dev, mins)
        new_cell_size = tuple(min(ax_ns, s) for s in size)
        minb = tuple((s - ns) / 2 for s, ns in zip(size, new_cell_size))
        maxb = tuple((s + ns) / 2 for s, ns in zip(size, new_cell_size))
        return KerogenBox(
            Range(minb[0], maxb[0]),
            Range(minb[1], maxb[1]),
            Range(minb[2], maxb[2]),
        )

    @staticmethod
    def full_cell(size: Tuple[float, float, float]) -> KerogenBox:
        return KerogenBox(
            Range(0.0, size[0]),
            Range(0.0, size[1]),
            Range(0.0, size[2]),
        )

    @staticmethod
    def calc_image_size(
        cell_size: Tuple[float, float, float],
        reference_size: int = 100,
        by_min: bool = True,
    ) -> Tuple[int, int, int]:
        l_cell_size = [s for s in cell_size]
        ref_cell_size = min(l_cell_size) if by_min else max(l_cell_size)
        return tuple(  # type: ignore
            int(math.ceil(reference_size * cs / ref_cell_size))
            for cs in l_cell_size
        )

    def binarize(self) -> npt.NDArray[np.int8]:
        vox_sizes = [
            ker_s / float(img_s)
            for ker_s, img_s in zip(self.kerogen.box.size(), self.img_size)
        ]

        def wrap(ix):
            s_img = np.ones(
                shape=(self.img_size[1], self.img_size[2]), dtype=np.int8
            )

            for iy in range(self.img_size[1]):
                for iz in range(self.img_size[2]):
                    pos = np.array(
                        [
                            mm + (float(i) + 0.5) * vs
                            for i, vs, mm in zip(
                                [ix, iy, iz], vox_sizes, self.kerogen.box.min()
                            )
                        ]
                    ).reshape(1, 3)
                    for bb in self.bb:
                        if bb.is_inside(pos) and bb.is_intersect_atom(pos):
                            s_img[iy, iz] = 0
            print(f" --- Slice {ix}-x finished!")
            return s_img

        num_proc = 15
        sim_results = Parallel(n_jobs=num_proc)(
            delayed(wrap)(ix) for ix in range(self.img_size[0])
        )
        # sim_results = [wrap(ix) for ix in range(self.img_size[0])]

        img = np.ones(shape=self.img_size, dtype=np.int8)
        for i, res in enumerate(sim_results):
            img[i, :, :] = res

        return img

    def dist_map(self) -> npt.NDArray[np.float32]:
        vox_sizes = [
            ker_s / float(img_s)
            for ker_s, img_s in zip(self.kerogen.box.size(), self.img_size)
        ]

        def dist_to_edge(a, b, p):
            # normalized tangent vector
            d = np.divide(b - a, np.linalg.norm(b - a))

            # signed parallel distance components
            s = np.dot(a - p, d)[0]
            t = np.dot(p - b, d)[0]

            # clamped parallel distance
            h = np.maximum.reduce([s, t, 0])

            # perpendicular distance component
            c = np.cross(p - a, d)

            return np.hypot(h, np.linalg.norm(c))

        def wrap(ix):
            s_img = 1e6 * np.ones(
                shape=(self.img_size[1], self.img_size[2]), dtype=np.float32
            )

            for iy in range(self.img_size[1]):
                for iz in range(self.img_size[2]):
                    pos = np.zeros(shape=(1, 3), dtype=np.float32)
                    pos[:] = [
                        mm + (float(i) + 0.5) * vs
                        for i, vs, mm in zip(
                            [ix, iy, iz], vox_sizes, self.kerogen.box.min()
                        )
                    ]

                    for bb in self.bb:
                        if not bb.is_inside(pos):
                            continue

                        d, id = bb.dist_nearest(pos)
                        de = -1
                        for n1, n2 in self.kerogen.graph.edges(id):
                            de = dist_to_edge(
                                self.kerogen.atoms[n1].pos,
                                self.kerogen.atoms[n2].pos,
                                pos,
                            )
                            s_img[iy, iz] = min(s_img[iy, iz], de)
                        if de == -1:
                            s_img[iy, iz] = min(s_img[iy, iz], d)
            print(f" --- Slice {ix}-x finished!")
            return s_img

        num_proc = 8
        sim_results = Parallel(n_jobs=num_proc)(
            delayed(wrap)(ix) for ix in range(self.img_size[0])
        )

        img = np.ones(shape=self.img_size, dtype=np.float32)
        for i, res in enumerate(sim_results):
            img[i, :, :] = res

        return img
