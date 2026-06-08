import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from os.path import realpath
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from base.boundingbox import KerogenBox, Range
from base.kerogendata import AtomData, KerogenData
from utils.utils import kprint


class BinarizeAlgo(Enum):
    SEQUENTIAL    = "sequential"
    THREAD_SLICE  = "thread_slice"
    THREAD_CHUNK  = "thread_chunk"
    PROCESS_CHUNK = "process_chunk"


_proc_worker_state: Dict = {}


def _proc_worker_init(
    bb: list,
    vox_sizes: "np.ndarray",
    mins: "np.ndarray",
    img_size: tuple,
    atom_chunk: int,
    yz_flat: "np.ndarray",
) -> None:
    _proc_worker_state.update(
        bb=bb, vox_sizes=vox_sizes, mins=mins,
        img_size=img_size, atom_chunk=atom_chunk, yz_flat=yz_flat,
    )


def _proc_worker_process_chunk(chunk_indices: List[int]) -> List[tuple]:
    s = _proc_worker_state
    bb        = s["bb"]
    vox_sizes = s["vox_sizes"]
    mins      = s["mins"]
    img_size  = s["img_size"]
    atom_chunk = s["atom_chunk"]
    yz_flat   = s["yz_flat"]
    Ny, Nz = img_size[1], img_size[2]

    results = []
    for ix in chunk_indices:
        gx    = float(mins[0] + (ix + 0.5) * vox_sizes[0])
        x_col = np.full((Ny * Nz, 1), gx, dtype=np.float32)
        pos   = np.hstack([x_col, yz_flat])
        atom_mask = np.zeros(Ny * Nz, dtype=bool)
        for bb_entry in bb:
            in_bb = (
                (pos[:, 0] >= bb_entry.xb_.min_) & (pos[:, 0] <= bb_entry.xb_.max_)
                & (pos[:, 1] >= bb_entry.yb_.min_) & (pos[:, 1] <= bb_entry.yb_.max_)
                & (pos[:, 2] >= bb_entry.zb_.min_) & (pos[:, 2] <= bb_entry.zb_.max_)
            )
            if not in_bb.any() or len(bb_entry.positions) == 0:
                continue
            pos_in  = pos[in_bb]
            mask_in = np.zeros(int(in_bb.sum()), dtype=bool)
            for a_start in range(0, len(bb_entry.positions), atom_chunk):
                dists   = cdist(pos_in, bb_entry.positions[a_start : a_start + atom_chunk])
                mask_in |= (dists < bb_entry.atom_sizes[a_start : a_start + atom_chunk]).any(axis=1)
            atom_mask[in_bb] |= mask_in
        results.append((ix, (~atom_mask).reshape(Ny, Nz).astype(np.int8)))
    print(f" --- Chunk slices {chunk_indices[0]}..{chunk_indices[-1]}-x finished!")
    return results


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

    @staticmethod
    def _make_chunks(total: int, chunk_size: int) -> List[List[int]]:
        indices = list(range(total))
        return [indices[i : i + chunk_size] for i in range(0, total, chunk_size)]

    def binarize(
        self,
        num_workers: int = 15,
        atom_chunk: int = 1024,
        algo: Optional[BinarizeAlgo] = None,
        chunk_size: int = 8,
    ) -> npt.NDArray[np.int8]:
        """Return 3D image where 0 = atom, 1 = void."""
        if algo is None:
            algo = BinarizeAlgo.SEQUENTIAL if num_workers == 0 else BinarizeAlgo.THREAD_SLICE

        vox_sizes = np.array(
            [ker_s / float(img_s) for ker_s, img_s in zip(self.kerogen.box.size(), self.img_size)],
            dtype=np.float32,
        )
        mins = self.kerogen.box.min()
        Nx, Ny, Nz = self.img_size
        gy = (mins[1] + (np.arange(Ny) + 0.5) * vox_sizes[1]).astype(np.float32)
        gz = (mins[2] + (np.arange(Nz) + 0.5) * vox_sizes[2]).astype(np.float32)
        yy, zz = np.meshgrid(gy, gz, indexing='ij')
        yz_flat = np.stack([yy.ravel(), zz.ravel()], axis=1)  # (Ny*Nz, 2)

        img = np.ones(self.img_size, dtype=np.int8)

        def process_slice(ix: int) -> npt.NDArray[np.int8]:
            gx    = float(mins[0] + (ix + 0.5) * vox_sizes[0])
            x_col = np.full((Ny * Nz, 1), gx, dtype=np.float32)
            pos   = np.hstack([x_col, yz_flat])
            atom_mask = np.zeros(Ny * Nz, dtype=bool)
            for bb in self.bb:
                in_bb = (
                    (pos[:, 0] >= bb.xb_.min_) & (pos[:, 0] <= bb.xb_.max_)
                    & (pos[:, 1] >= bb.yb_.min_) & (pos[:, 1] <= bb.yb_.max_)
                    & (pos[:, 2] >= bb.zb_.min_) & (pos[:, 2] <= bb.zb_.max_)
                )
                if not in_bb.any() or len(bb.positions) == 0:
                    continue
                pos_in  = pos[in_bb]
                mask_in = np.zeros(int(in_bb.sum()), dtype=bool)
                for a_start in range(0, len(bb.positions), atom_chunk):
                    dists   = cdist(pos_in, bb.positions[a_start : a_start + atom_chunk])
                    mask_in |= (dists < bb.atom_sizes[a_start : a_start + atom_chunk]).any(axis=1)
                atom_mask[in_bb] |= mask_in
            return (~atom_mask).reshape(Ny, Nz).astype(np.int8)

        if algo == BinarizeAlgo.SEQUENTIAL:
            for ix in range(Nx):
                img[ix] = process_slice(ix)
                print(f" --- Slice {ix}-x finished!")

        elif algo == BinarizeAlgo.THREAD_SLICE:
            n = min(num_workers, Nx)
            with ThreadPoolExecutor(max_workers=n) as executor:
                for ix, result in enumerate(executor.map(process_slice, range(Nx))):
                    img[ix] = result
                    print(f" --- Slice {ix}-x finished!")

        elif algo == BinarizeAlgo.THREAD_CHUNK:
            n = min(num_workers, Nx)
            chunks = Segmentator._make_chunks(Nx, chunk_size)

            def process_chunk_thread(chunk_indices: List[int]) -> List[tuple]:
                results = [(ix, process_slice(ix)) for ix in chunk_indices]
                print(f" --- Chunk slices {chunk_indices[0]}..{chunk_indices[-1]}-x finished!")
                return results

            with ThreadPoolExecutor(max_workers=n) as executor:
                futures = [executor.submit(process_chunk_thread, ch) for ch in chunks]
                for fut in futures:
                    for ix, result in fut.result():
                        img[ix] = result

        elif algo == BinarizeAlgo.PROCESS_CHUNK:
            n = min(num_workers, Nx)
            chunks = Segmentator._make_chunks(Nx, chunk_size)
            with ProcessPoolExecutor(
                max_workers=n,
                initializer=_proc_worker_init,
                initargs=(self.bb, vox_sizes, mins, self.img_size, atom_chunk, yz_flat),
            ) as executor:
                futures = [executor.submit(_proc_worker_process_chunk, ch) for ch in chunks]
                for fut in futures:
                    for ix, result in fut.result():
                        img[ix] = result

        return img

    def benchmark_binarize(
        self,
        num_workers: int = 8,
        atom_chunk: int = 1024,
        chunk_size: int = 8,
        algos: Optional[List[BinarizeAlgo]] = None,
    ) -> Dict[str, float]:
        """Run binarize() with each algorithm variant and print a timing comparison."""
        if algos is None:
            algos = list(BinarizeAlgo)
        results: Dict[str, float] = {}
        for algo in algos:
            print(f" --- [benchmark] Starting {algo.value} ...")
            t0 = time.perf_counter()
            self.binarize(num_workers=num_workers, atom_chunk=atom_chunk, algo=algo, chunk_size=chunk_size)
            results[algo.value] = time.perf_counter() - t0
            print(f" --- [benchmark] {algo.value}: {results[algo.value]:.3f}s")

        baseline = results.get(BinarizeAlgo.SEQUENTIAL.value)
        print("\n --- Benchmark Results ---")
        for name, t in results.items():
            speedup = f"  ({baseline / t:.2f}x)" if baseline and t > 0 else ""
            print(f"     {name:<20s}: {t:8.3f}s{speedup}")
        print()
        return results

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
