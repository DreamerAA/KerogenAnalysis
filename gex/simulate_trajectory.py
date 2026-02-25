import argparse
import sys
from os.path import realpath, join, isfile
from pathlib import Path
import pickle

import numpy as np

path = Path(realpath(__file__))
parent_dir = str(path.parent.parent.absolute())
sys.path.append(parent_dir)

from base.reader import Reader
from utils.utils import create_empirical_cdf, ps_generate
from processes.kerogen_walk_simulator import KerogenWalkSimulator
from visualizer.visualizer import Visualizer, WrapMode
from base.bufferedsampler import BufferedSampler
from base.discretecdf import DiscreteCDF
from base.empiricalcdf import EmpiricalCDF

def run(path_to_main: str):
    ps_type = 'uniform'  # poisson uniform
    ps = ps_generate(ps_type, max_count_step=50)

    path_to_radiuses: str = join(path_to_main, "radiuses.npy")
    if isfile(path_to_radiuses):
        radiuses = np.load(path_to_radiuses)
    else:
        raise RuntimeError("radiuses not found")


    path_to_tl_wf: str = join(path_to_main, "throat_lengths_weibull_fitter.pkl")
    if isfile(path_to_tl_wf):
        with open(path_to_tl_wf, "rb") as f:
            throat_lengths_weibull_fitter = pickle.load(f)
    else:
        raise RuntimeError("throat_lengths not found")
    
    psd = create_empirical_cdf(radiuses)
    bs_ps = BufferedSampler(DiscreteCDF(ps), "ps", size=100_000)
    bs_psd = BufferedSampler(EmpiricalCDF(psd), "psd", size=100_000)
    bs_ptl = BufferedSampler(throat_lengths_weibull_fitter, "ptl", size=100_000)

    k = 0.
    p = 0.
    simulator = KerogenWalkSimulator(bs_psd, bs_ps, bs_ptl, k, p, with_history=False)
    traj = simulator.run(300)
    Visualizer.draw_trajectoryes(
        [traj],
        radius=0.1,
        periodic=False,
        wrap_mode=WrapMode.EMPTY,
        with_points=True,
        color_type='clusters'
    )
    Visualizer.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path_to_data',
        type=str,
        default="/media/andrey/Samsung_T5/PHD/Kerogen/type1matrix/300K/ch4",
    )
    args = parser.parse_args()

    run(args.path_to_data)
