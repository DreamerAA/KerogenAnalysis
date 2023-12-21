from typing import Tuple, Any
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.stats import exponweib
from joblib import Parallel, delayed
from sklearn.metrics import pairwise_distances
from matplotlib import cm
import sys
from pathlib import Path
from os.path import realpath
from processes.pil_distr_generator import PiLDistrGenerator

path = Path(realpath(__file__))
parent_dir = str(path.parent.parent.absolute())
sys.path.append(parent_dir)

from base.reader import Reader


def generate_distribution(path_to_pnm: str, path_to_save: str) -> None:
    radiuses, throat_lengths = Reader.read_pnm_data(
        path_to_pnm, scale=1e10, border=0.015
    )

    generator = PiLDistrGenerator()
    pi_l = generator.run(radiuses)

    np.save(path_to_save + "_pi_l.npy", pi_l)
    np.save(path_to_save + "_throat_lengths.npy", throat_lengths)


if __name__ == '__main__':
    path = "/home/andrey/PHD/Kerogen/data/Kerogen/time_trapping_results/ch4/"
    generate_distribution(
        path + "num=1597500000_500_500_500", path + "num=1597500000_500_500_500"
    )
