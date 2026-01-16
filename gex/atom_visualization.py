import sys
import os
from pathlib import Path
from os.path import realpath
import random
import time
from typing import IO, List, Tuple, Any
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

path = Path(realpath(__file__))
parent_dir = str(path.parent.parent.absolute())
sys.path.append(parent_dir)

from base.boundingbox import BoundingBox, KerogenBox, Range
from base.kerogendata import AtomData, KerogenData
from base.periodizer import Periodizer
from base.reader import Reader
from base.trajectory import Trajectory
from processes.segmentaion import Segmentator
from visualizer.visualizer import Visualizer


atom_real_sizes = {
    i: s for i, s in enumerate([0.17, 0.152, 0.155, 0.109, 0.18])
}


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
        size_node=0.5,
        size_edge=0.00,
        colors_data=colors_data,
        scales_data=scales_data,
        scale='non',
    )


atoms = [AtomData(1, "KRG", 'c', 0, np.array([0., 0., 0.])), 
         AtomData(1, "KRG", 'o', 1, np.array([0., 0., 0.5])),
         AtomData(1, "KRG", 'n', 2, np.array([0., 0., 1.])),
         AtomData(1, "KRG", 'h', 3, np.array([0., 0., 1.5])),
         AtomData(1, "KRG", 's', 4, np.array([0., 0., 2.]))]


graph = nx.Graph()
graph.add_nodes_from(
    [
        (i, {"color_id": atom.type_id, "scale_id": atom.type_id})
        for i, atom in enumerate(atoms)
    ]
)
# graph.add_edges_from([(0, 1)])


bbox = BoundingBox(Range(0., 0.5), Range(0., 0.5), Range(0., 0.5))
kerogen_data = KerogenData(graph, atoms, bbox)
draw_kerogen_data(kerogen_data)
