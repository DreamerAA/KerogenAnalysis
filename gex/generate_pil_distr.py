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
from os import listdir
from os.path import isfile, join, dirname, realpath


path = Path(realpath(__file__))
parent_dir = str(path.parent.parent.absolute())
sys.path.append(parent_dir)

from processes.pil_distr_generator import PiLDistrGenerator
from base.reader import Reader


def generate_distribution(path_to_pnms: str, path_to_save: str) -> None:
    if isfile(path_to_save + "pi_l.npy") and isfile(
        path_to_save + "throat_lengths.npy"
    ):
        return

    onlyfiles = [
        f for f in listdir(path_to_pnms) if isfile(join(path_to_pnms, f))
    ]
    onlyfiles = [
        join(path_to_pnms, file[:-10]) for file in onlyfiles if "_link1" in file
    ]
    radiuses, throat_lengths = [], []
    for f in onlyfiles:
        r, t = Reader.read_pnm_data(
            f, scale=1e10, border=0.03
        )  # scale=1e10 - to A
        radiuses = np.concatenate((radiuses, r))
        throat_lengths = np.concatenate((throat_lengths, t))

    generator = PiLDistrGenerator()
    pi_l = generator.run(radiuses)

    np.save(path_to_save + "pi_l.npy", pi_l)
    np.save(path_to_save + "throat_lengths.npy", throat_lengths)


if __name__ == '__main__':
    mpath = "/media/andrey/Samsung_T5/PHD/Kerogen/type2matrix/"
    
    # for tem in ["300K", "400K"]:
    for tem in ["300K"]:
        for el in ["h2", "ch4"]:
            path_to_save = mpath + f"{tem}/{el}/"
            path_to_pnm = path_to_save + "/pnm/"

            generate_distribution(path_to_pnm, path_to_save)
