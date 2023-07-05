from typing import Callable, Dict, Tuple

import networkx as nx
import numpy as np

from kerogendata import KerogenData


class Periodizer:
    @staticmethod
    def periodize(kerogen: KerogenData) -> None:
        def new_coord(old: float, s: float) -> float:
            if old < 0:
                return s + old
            elif old > s:
                return old - s
            return old

        sx, sy, sz = kerogen.size
        for a in kerogen.atoms:
            x, y, z = a.pos
            a.pos = (new_coord(x, sx), new_coord(y, sy), new_coord(z, sz))

        Periodizer.rm_long_edges(kerogen)

    @staticmethod
    def rm_long_edges(kerogen: KerogenData) -> None:
        sx_2, sy_2, sz_2 = np.array([s for s in kerogen.box.size()]) * 0.5

        def check_dist(n1: int, n2: int) -> bool:
            a1, a2 = kerogen.atoms[n1], kerogen.atoms[n2]
            if (
                abs(a1.pos[0] - a2.pos[0]) > sx_2
                or abs(a1.pos[1] - a2.pos[1]) > sy_2
                or abs(a1.pos[2] - a2.pos[2]) > sz_2
            ):
                return True
            return False

        list_edges_to_rm = []
        for n1, n2 in kerogen.graph.edges():
            if check_dist(n1, n2):
                list_edges_to_rm.append((n1, n2))
        kerogen.graph.remove_edges_from(list_edges_to_rm)
