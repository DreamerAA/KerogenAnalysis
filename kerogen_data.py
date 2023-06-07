from dataclasses import dataclass
from typing import Dict, List, Tuple
from boundingbox import KerogenBox
import networkx as nx
import numpy as np
import numpy.typing as npt


@dataclass
class AtomData:
    struct_number: int
    struct_type: str
    atom_id: str
    type_id: int
    pos: npt.NDArray[np.float32]

    def tuple_pos(self) -> Tuple[float, float, float]:
        return self.pos[0], self.pos[1], self.pos[2]


@dataclass
class KerogenData:
    graph: nx.Graph
    atoms: List[AtomData]
    box: KerogenBox

    def positionsAsDict(self) -> Dict[int, Tuple[float, float, float]]:
        return {i: a.tuple_pos() for i, a in enumerate(self.atoms)}
