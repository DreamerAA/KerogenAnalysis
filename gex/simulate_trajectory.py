import argparse
from pathlib import Path
import sys
from os.path import realpath
import numpy as np

path = Path(realpath(__file__))
parent_dir = str(path.parent.parent.absolute())
sys.path.append(parent_dir)

from base.reader import Reader
from processes.kerogen_walk_simulator import KerogenWalkSimulator
from visualizer.visualizer import Visualizer
from examples.utils import create_cdf


def run(path_to_pnm):
    radiuses, throat_lengths = Reader.read_pnm_data(
        path_to_pnm, scale=1e10, border=0.015
    )

    steps = np.array([s for s in range(1, 101)], dtype=np.int32).reshape(100, 1)
    prob = ((steps.astype(np.float32)) * 0.01).reshape(100, 1)
    ps = np.hstack((steps, prob))

    ppl = create_cdf(radiuses)
    ptl = create_cdf(throat_lengths)

    simulator = KerogenWalkSimulator(ppl, ps, ptl, 0.5, 0.9)
    traj, _ = simulator.run(1000)
    Visualizer.draw_trajectoryes([traj], periodic=False, plot_box=False)
    Visualizer.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path_to_data',
        type=str,
        default="../data/Kerogen/time_trapping_results/ch4/num=1597500000_500_500_500",
    )
    args = parser.parse_args()

    run(args.path_to_data)
