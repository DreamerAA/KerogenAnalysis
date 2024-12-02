import argparse
from pathlib import Path
import sys
import os
from os import listdir
from os.path import isfile, join, dirname, realpath
import numpy as np
import matplotlib.pyplot as plt
import pickle
from typing import List
from joblib import Parallel, delayed
import seaborn as sns
import pandas as pd

path = Path(realpath(__file__))
parent_dir = str(path.parent.parent.absolute())
sys.path.append(parent_dir)

from examples.utils import get_params, visualize_trajectory
from base.trap_sequence import TrapSequence
from base.trajectory import Trajectory
from processes.trap_extractor import TrapExtractor
from processes.trajectory_extended_analizer import (
    TrajectoryExtendedAnalizer,
    ExtendedParams,
)
from processes.trajectory_extended_analizer import TrajectoryAnalizer


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError(
            "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
        )

    # s = np.r_[x[window_len - 1 : 0 : -1], x, x[-2 : -window_len - 1 : -1]]
    s = x
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='same')

    return y


def runClear(
    traj_path: str, save_path: str, prefix: str, step: int = 1
) -> None:
    if isfile(save_path):
        ddata = pd.read_csv(save_path)
    else:
        trajectories = Trajectory.read_trajectoryes(traj_path)
        trajectories = trajectories[::step]
        nums = []
        a_dr2 = []
        a_t = []
        for i, trj in enumerate(trajectories):
            dr2 = trj.msd()
            t = trj.times
            a_t = a_t + list(t)
            a_dr2 = a_dr2 + list(dr2)
            nums = nums + [i] * len(t)

        ddata = pd.DataFrame(
            list(zip(a_dr2, a_t, nums)),
            columns=['MSD', 'Time', 'Trajectory number'],
        )
        ddata.to_csv(save_path)
    sns.lineplot(data=ddata, x="Time", y="MSD", label=prefix)


def runSmooth(traj_path: str, prefix: str) -> None:
    trajectories = Trajectory.read_trajectoryes(traj_path)
    msd = np.zeros(shape=(trajectories[0].count_points,), dtype=np.float32)
    t = []
    wl = 20
    for trj in trajectories:
        dr2, t = trj.msd()
        msd += dr2

    print(f"Time window size", t[wl] - t[0])
    msd /= len(trajectories)

    msd = smooth(msd, 20, 'flat')

    plt.plot(t, msd, label=prefix)


def runAll(traj_path: str, element: str, temp: str, step: int = 1) -> None:
    trajectories = Trajectory.read_trajectoryes(traj_path)
    for i, trj in enumerate(trajectories[::step]):
        dr2 = trj.msd()
        t = trj.times
        plt.plot(t, dr2, label=element + '-' + temp + '-' + str(i))


def runTimeAvarage(
    traj_path: str,
    msd_path: str,
    prefix: str,
    step: int = 1,
    force_save: bool = False,
) -> None:
    if isfile(msd_path) and not force_save:
        ddata = pd.read_csv(msd_path)
    else:
        trajectories = Trajectory.read_trajectoryes(traj_path)

        nums = []
        a_msd = []
        a_t = []
        for i, trj in enumerate(trajectories[::step]):
            msd = trj.msd_average_time()
            t = trj.times
            a_t = a_t + list(t)
            a_msd = a_msd + list(msd)
            nums = nums + [i] * len(t)

        ddata = pd.DataFrame(
            list(zip(a_msd, a_t, nums)),
            columns=['MSD', 'Time', 'Trajectory number'],
        )
        ddata.to_csv(msd_path)
    sns.lineplot(data=ddata, x="Time", y="MSD", label=prefix)


if __name__ == '__main__':
    input_data = [
        (
            "/media/andrey/Samsung_T5/PHD/Kerogen/type1matrix/300K/ch4/",
            "type1-300K-CH4", 1
        ),
        (
            "/media/andrey/Samsung_T5/PHD/Kerogen/type1matrix/300K/h2/",
            "type1-300K-H2", 2
        ),
        (
            "/media/andrey/Samsung_T5/PHD/Kerogen/type1matrix/400K/ch4/",
            "type1-400K-CH4", 1
        ),
        (
            "/media/andrey/Samsung_T5/PHD/Kerogen/type1matrix/400K/h2/",
            "type1-400K-H2", 2
        ),
        (
            "/media/andrey/Samsung_T5/PHD/Kerogen/type2matrix/300K/ch4/",
            "type2-300K-CH4", 1
        ),
        (
            "/media/andrey/Samsung_T5/PHD/Kerogen/type2matrix/300K/h2/",
            "type2-300K-H2", 2
        ),
    ]
    
    for path, prefix, step in input_data:
        print("Run " + prefix)
        traj_path = path + "trj.gro"
            # runClear(
            # traj_path, prefix + f"{temp}/{el}/msd.csv", f"{el}-{temp}", step
            # )
        runTimeAvarage(
            traj_path,
            path + "msd_time_avarage.csv",
            prefix,
            step,
            False,
        )

    # runSmooth(traj_path, el)

    # temp = "400K"
    # el, step = "ch4", 1
    # traj_path = prefix + f"{temp}/{el}/trj.gro"
    # runAll(traj_path, temp, el, step)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Time (ps)")
    plt.ylabel("MSD (A^2)")
    plt.legend()
    plt.show()
