import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from os.path import realpath
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import networkx as nx
import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from base.boundingbox import KerogenBox, Range
from base.kerogendata import AtomData, KerogenData
from utils.utils import kprint


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

        # Precompute positions and sizes for all atoms as numpy arrays once
        all_positions = np.array(
            [a.pos for a in kerogen.atoms], dtype=np.float32
        )
        sizes_arr = np.array(
            [
                size_data(a.type_id) + radius_extention(a.type_id)
                for a in kerogen.atoms
            ],
            dtype=np.float32,
        )

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
                    bb = KerogenBox(*aranges)
                    in_bb = (
                        (all_positions[:, 0] >= bb.xb_.min_)
                        & (all_positions[:, 0] <= bb.xb_.max_)
                        & (all_positions[:, 1] >= bb.yb_.min_)
                        & (all_positions[:, 1] <= bb.yb_.max_)
                        & (all_positions[:, 2] >= bb.zb_.min_)
                        & (all_positions[:, 2] <= bb.zb_.max_)
                    )
                    bb.atom_ids = np.where(in_bb)[0].astype(np.int32)
                    bb.atom_sizes = sizes_arr[in_bb]
                    bb.positions = all_positions[in_bb]
                    self.bb.append(bb)

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

    def binarize(
        self, num_workers: int = 15, atom_chunk: int = 1024
    ) -> npt.NDArray[np.int8]:
        """
        Rerurn 3D image where zero is atom
        Returns:
            npt.NDArray[np.int8]: binary image
        """
        vox_sizes = np.array(
            [
                ker_s / float(img_s)
                for ker_s, img_s in zip(self.kerogen.box.size(), self.img_size)
            ],
            dtype=np.float32,
        )
        mins = self.kerogen.box.min()

        Ny, Nz = self.img_size[1], self.img_size[2]
        gy = (mins[1] + (np.arange(Ny) + 0.5) * vox_sizes[1]).astype(np.float32)
        gz = (mins[2] + (np.arange(Nz) + 0.5) * vox_sizes[2]).astype(np.float32)
        yy, zz = np.meshgrid(gy, gz, indexing='ij')  # (Ny, Nz)
        yz_flat = np.stack([yy.ravel(), zz.ravel()], axis=1)  # (Ny*Nz, 2)

        img = np.ones(self.img_size, dtype=np.int8)

        def process_slice(ix):
            gx = float(mins[0] + (ix + 0.5) * vox_sizes[0])
            x_col = np.full((Ny * Nz, 1), gx, dtype=np.float32)
            pos = np.hstack([x_col, yz_flat])  # (Ny*Nz, 3)
            atom_mask = np.zeros(Ny * Nz, dtype=bool)

            for bb in self.bb:
                in_bb = (
                    (pos[:, 0] >= bb.xb_.min_)
                    & (pos[:, 0] <= bb.xb_.max_)
                    & (pos[:, 1] >= bb.yb_.min_)
                    & (pos[:, 1] <= bb.yb_.max_)
                    & (pos[:, 2] >= bb.zb_.min_)
                    & (pos[:, 2] <= bb.zb_.max_)
                )
                if not in_bb.any() or len(bb.positions) == 0:
                    continue
                pos_in = pos[in_bb]
                mask_in = np.zeros(int(in_bb.sum()), dtype=bool)
                for a_start in range(0, len(bb.positions), atom_chunk):
                    dists = cdist(
                        pos_in, bb.positions[a_start : a_start + atom_chunk]
                    )
                    mask_in |= (
                        dists < bb.atom_sizes[a_start : a_start + atom_chunk]
                    ).any(axis=1)
                atom_mask[in_bb] |= mask_in

            return (~atom_mask).reshape(Ny, Nz).astype(np.int8)

        if num_workers == 0:
            for ix in range(self.img_size[0]):
                img[ix] = process_slice(ix)
                print(f" --- Slice {ix}-x finished!")
        else:
            n = min(num_workers, self.img_size[0])
            with ThreadPoolExecutor(max_workers=n) as executor:
                for ix, result in enumerate(
                    executor.map(process_slice, range(self.img_size[0]))
                ):
                    img[ix] = result
                    print(f" --- Slice {ix}-x finished!")

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

        all_atom_positions = np.vstack([bb.positions for bb in self.bb])
        tree = cKDTree(all_atom_positions)

        def wrap(ix):
            ny, nz = self.img_size[1], self.img_size[2]
            box_min = self.kerogen.box.min()
            iy_idx = np.arange(ny, dtype=np.float32)
            iz_idx = np.arange(nz, dtype=np.float32)
            IY, IZ = np.meshgrid(iy_idx, iz_idx, indexing='ij')
            all_pos = np.column_stack(
                [
                    np.full(
                        ny * nz,
                        box_min[0] + (ix + 0.5) * vox_sizes[0],
                        dtype=np.float32,
                    ),
                    (box_min[1] + (IY + 0.5) * vox_sizes[1]).ravel(),
                    (box_min[2] + (IZ + 0.5) * vox_sizes[2]).ravel(),
                ]
            )  # (Ny*Nz, 3)
            dist, _ = tree.query(all_pos, k=1, workers=1)
            result = dist.astype(np.float32).reshape(ny, nz)
            return result

        img = np.ones(shape=self.img_size, dtype=np.float32)
        for ix in range(self.img_size[0]):
            img[ix] = wrap(ix)
            print(f" --- Slice {ix}-x finished!")

        return img
