from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import cdist


class Range(object):
    def __init__(
        self,
        mmin: float = None,
        mmax: float = None,
    ):
        if mmin is not None and mmax is not None:
            assert mmin <= mmax
        self.min_ = float(np.finfo(float).max) if mmin == None else mmin
        self.max_ = float(np.finfo(float).min) if mmax == None else mmax

    def update(self, v: float) -> None:
        self.min_ = min(v, self.min_)
        self.max_ = max(v, self.max_)

    def update_by_range(self, from_update: 'Range') -> None:
        self.min_ = min(self.min_, from_update.min_)
        self.max_ = max(self.max_, from_update.max_)

    def dilation(self, v: float) -> None:
        self.min_ -= v
        self.max_ += v

    def diff(self) -> float:
        return self.max_ - self.min_

    def is_inside(self, v: float) -> bool:
        return self.min_ <= v and v <= self.max_

    def center(self) -> float:
        return (self.max_ + self.min_) * 0.5


class BoundingBox(object):
    def __init__(
        self,
        xb: Optional[Range] = None,
        yb: Optional[Range] = None,
        zb: Optional[Range] = None,
    ):
        self.xb_ = Range() if xb is None else xb
        self.yb_ = Range() if yb is None else yb
        self.zb_ = Range() if zb is None else zb

    def is_inside(self, pos: npt.NDArray[np.float32]) -> bool:
        m = [
            b.is_inside(v)
            for b, v in zip([self.xb_, self.yb_, self.zb_], pos.flat)
        ]
        return np.all(m)

    def size(self) -> Tuple[float, float, float]:
        return (self.xb_.diff(), self.yb_.diff(), self.zb_.diff())

    def min(self) -> npt.NDArray[np.float32]:
        return np.array([self.xb_.min_, self.yb_.min_, self.zb_.min_])

    def max(self) -> npt.NDArray[np.float32]:
        return np.array([self.xb_.max_, self.yb_.max_, self.zb_.max_])

    def center(self) -> npt.NDArray[np.float32]:
        return np.array([r.center() for r in [self.xb_, self.yb_, self.zb_]])

    def update(self, p: npt.NDArray[np.float32]) -> None:
        for b, v in zip([self.xb_, self.yb_, self.zb_], p.flat):
            b.update(v)

    def update_by_box(self, bbox: 'BoundingBox') -> None:
        for nb, b in [
            (self.xb_, bbox.xb_),
            (self.yb_, bbox.yb_),
            (self.zb_, bbox.zb_),
        ]:
            nb.update_by_range(b)

    def aminmax(self):
        res = []
        for a in [self.xb_, self.yb_, self.zb_]:
            res += [a.min_, a.max_]
        return res


class KerogenBox(BoundingBox):
    def __init__(
        self,
        xb: Optional[Range] = None,
        yb: Optional[Range] = None,
        zb: Optional[Range] = None,
    ):
        self.xb_ = Range() if xb is None else xb
        self.yb_ = Range() if yb is None else yb
        self.zb_ = Range() if zb is None else zb
        self.tmp_atom_type_ids: List[int] = []
        self.tmp_atom_ids: List[int] = []
        self.tmp_atom_sizes: List[float] = []
        self.tmp_positions: List[Tuple[float, float, float]] = []

    def add_atom(
        self, id: int, size: float, pos: Tuple[float, float, float]
    ) -> None:
        self.tmp_atom_ids.append(id)
        self.tmp_atom_sizes.append(size)
        self.tmp_positions.append(pos)

    def commit(self) -> None:
        self.atom_ids = np.array(self.tmp_atom_ids, dtype=np.int32)
        self.atom_sizes = np.array(self.tmp_atom_sizes, dtype=np.float32)
        self.positions = np.zeros(
            shape=(len(self.tmp_positions), 3), dtype=np.float32
        )
        for i, p in enumerate(self.tmp_positions):
            self.positions[i] = p

        self.tmp_atom_type_ids.clear()
        self.tmp_positions.clear()
        self.tmp_atom_ids.clear()
        self.tmp_atom_sizes.clear()

    def is_intersect_atom(self, pos: npt.NDArray[np.float32]) -> bool:
        dist = cdist(pos, self.positions)
        return np.any(dist < self.atom_sizes)  # type: ignore

    def dist_nearest(self, pos: npt.NDArray[np.float32]) -> Tuple[float, int]:
        dist = cdist(pos, self.positions)
        index = dist.argmin()
        return dist.min(), self.atom_ids[index]
