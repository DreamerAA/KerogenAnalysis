import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import networkx as nx
import numpy as np
import numpy.typing as npt
from joblib import Parallel, delayed

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
                    for a in kerogen.atoms:
                        if self.bb[-1].is_inside(a.pos):
                            self.bb[-1].add_atom(
                                a.type_id,
                                size_data(a.type_id)
                                + self.radius_extention(a.type_id),
                                a.pos,  # type: ignore
                            )
        # print(f" --- Count Kerogen boxes: {len(self.bb)}")
        for i, bb in enumerate(self.bb):
            bb.commit()
            # print(f" --- Finish commit kerogen box: {i}")

    @staticmethod
    def cut_cell(size: Tuple[float, float, float], dev: float = 4.0) -> KerogenBox:
        maxs = np.max(np.array(size))
        ax_ns = maxs / dev
        new_cell_size = tuple(min(ax_ns, s) for s in size)
        minb = tuple((s - ns) / 2 for s, ns in zip(size, new_cell_size))
        maxb = tuple((s + ns) / 2 for s, ns in zip(size, new_cell_size))
        return KerogenBox(
            Range(minb[0], maxb[0]),
            Range(minb[1], maxb[1]),
            Range(minb[2], maxb[2]),
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
            reference_size * cs // ref_cell_size for cs in l_cell_size
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
                    )
                    for bb in self.bb:
                        if bb.is_inside(pos) and bb.is_intersect_atom(pos):
                            s_img[iy, iz] = 0
            print(f" --- Slice {ix}-x finished!")
            return s_img

        num_proc = 16
        sim_results = Parallel(n_jobs=num_proc)(
            delayed(wrap)(ix) for ix in range(self.img_size[0])
        )
        img = np.ones(shape=self.img_size, dtype=np.int8)
        for i, res in enumerate(sim_results):
            img[i, :, :] = res

        return img
