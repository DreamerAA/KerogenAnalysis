import argparse
import sys
from os.path import realpath
from pathlib import Path

import numpy as np

path = Path(realpath(__file__))
parent_dir = str(path.parent.parent.absolute())
sys.path.append(parent_dir)

from base.reader import Reader
from examples.utils import create_cdf, ps_generate
from processes.kerogen_walk_simulator import KerogenWalkSimulator
from visualizer.visualizer import Visualizer, WrapMode


def run(path_to_pnm):
    radiuses, throat_lengths = Reader.read_pnm_data(
        path_to_pnm, scale=1e10, border=0.015
    )

    ps_type = 'uniform'  # poisson uniform
    ps = ps_generate(ps_type)

    ppl = create_cdf(radiuses)
    ptl = create_cdf(throat_lengths)

    simulator = KerogenWalkSimulator(ppl, ps, ptl, 1.0, 1.0)
    traj = simulator.run(1000)
    Visualizer.draw_trajectoryes(
        [traj],
        radius=0.2,
        periodic=False,
        wrap_mode=WrapMode.AXES,
        with_points=True,
    )
    Visualizer.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path_to_data',
        type=str,
        default="/media/andrey/Samsung_T5/PHD/Kerogen/type1matrix/300K/ch4/pnm/num=1640025000_500_500_500",
    )
    args = parser.parse_args()

    run(args.path_to_data)
