from pathlib import Path
import argparse
import time
from typing import List, Tuple
from skimage import measure
import matplotlib.pyplot as plt
import numpy as np

from base.trajectory import Trajectory
from processes.trajectory_analyzer import TrajectoryAnalizer, AnalizerParams
from visualizer.visualizer import Visualizer

# AnalizerParams(traj_type='Bm', nu=0.5, diag_percentile=0 , kernel_size=3, list_mu=np.array([0.5, 1. , 1.5, 2. , 2.5, 3.]), p_value=0.01),
# AnalizerParams(traj_type='Bm', nu=0.1, diag_percentile=0 , kernel_size=2, list_mu=np.array([0.5, 1. , 1.5, 2. , 2.5, 3.]), p_value=0.9),
# AnalizerParams(traj_type='Bm', nu=0.1, diag_percentile=0 , kernel_size=3, list_mu=np.array([0.5, 1. , 1.5, 2. , 2.5, 3.]), p_value=0.9),
# AnalizerParams(traj_type='Bm', nu=0.1, diag_percentile=50, kernel_size=3, list_mu=np.array([0.5, 1. , 1.5, 2. , 2.5, 3.]), p_value=0.01)


def visualize_dist_trajectory(traj: Trajectory) -> None:
    diff = traj.points[1:] - traj.points[:-1]
    sq_diff = diff * diff
    sq_dist = np.sum(sq_diff, axis=1)
    dist = np.sqrt(sq_dist)
    сdist = np.cumsum(dist)
    plt.subplot(2, 1, 1)
    plt.plot(traj.times[1:], dist)
    plt.subplot(2, 1, 2)
    plt.plot(traj.times[1:], сdist)


def visualize_trajectory(traj: Trajectory, color_type='dist') -> None:
    Visualizer.draw_trajectoryes([traj], color_type=color_type, plot_box=False)


def visualize_trajectories(trajs: List[Trajectory]) -> None:
    Visualizer.draw_trajectoryes(trajs)


def animate_trajectoryes(trajs: List[Trajectory]) -> None:
    Visualizer.animate_trajectoryes(trajs)


def find_best_params(traj, num, indexes, aparams):

    # for tt in ['fBm', 'Bm']:
    #         for nu in [0.1, 0.5, 1]:
    #             for dp in [0, 10, 50]:
    #                 for ks in [1, 2, 5]:
    #                     for pv in [0.01, 0.1, 0.5, 0.9]:
    #                         for lm in [[0.5], [2.], [3], [0.5, 1, 1.5], [1.5, 2, 2.5], [0.5, 1, 1.5, 2, 2.5, 3]]:

    lm = [0.5, 1, 1.5, 2, 2.5, 3]
    for i, par in zip(indexes, aparams):
        dp, pv, nu, tt, ks = par
        start = time.time()
        params = AnalizerParams(tt, nu, dp, ks, np.array(lm), pv)
        analizer = TrajectoryAnalizer(traj, params)
        end = time.time()
        print(f" --- Elapsed time analizer: {end-start}")

        print(f" --- Traj type: {tt}")
        print(f" --- NU: {nu}")
        print(f" --- Diag percentile: {dp}")
        print(f" --- Kernel size: {ks}")
        print(f" --- P value: {pv}")
        print(f" --- List mu: {lm}")
        clusters = measure.label(traj.traps, connectivity=1).astype(np.float32)
        print(f" --- Count clusters: {clusters.max()}")
        print("")
        np.save(f"./output/h2_traj/{num}/{i}.npy", traj.traps)


def all_params():
    params = {}
    i = 0
    for dp in [0, 10, 50]:  # 0 - bad
        for pv in [0.01, 0.1, 0.9]:  # 0.9 - bad
            for nu in [0.1, 0.5, 0.9]:
                for tt in ['fBm', 'Bm']:  # fBm - best
                    for ks in [0, 1, 2, 3]:
                        for lm in [[0.5, 1, 1.5, 2, 2.5, 3]]:
                            params[i] = (dp, pv, nu, tt, ks)
                            i = i + 1
    return params


def analizy_visualize(trj, params):
    analizer = TrajectoryAnalizer(trj, params)
    clusters = measure.label(trj.traps, connectivity=1).astype(np.float32)
    print(f" --- Count clusters: {clusters.max()}")
    visualize_trajectory(trj, 'clusters')


def run_and_plot_trap_time_distribution(trajectories: List[Trajectory], aparams: List[Tuple[int, AnalizerParams]]):
    fix, axs = plt.subplots(len(aparams))
    for i, ind_params in enumerate(aparams):
        for j, trj in enumerate(trajectories):
            Path(f"./output/h2_traj/{i}").mkdir(parents=True, exist_ok=True)
            path_to_result = f"./output/h2_traj/{i}/{ind_params[0]}.npy"
            my_file = Path(path_to_result)
            if my_file.is_file():
                trj.traps = np.load(my_file)
            else:
                analizer = TrajectoryAnalizer(trj, ind_params[1])

            np.save(f"./output/h2_traj/{i}/{ind_params[0]}.npy", trj.traps)
            print(f" --- Trajectory number: {j+1}, Params number: {i+1}")

        results = get_trap_time_distribution(trajectories)
        axs[i].hist(results)
    plt.show()


def get_trap_time_distribution(trajectories: List[Trajectory]):
    result = []
    for trj in trajectories:
        clusters = measure.label(trj.traps, connectivity=1).astype(np.int32)
        result = result + [np.sum(clusters == (c + 1)) for c in range(clusters.max())]
    return result


def analize_script(traj_num):
    #### analisys of parameters for h2
    params = all_params()
    indexes = [154, 162, 186]
    # indexes = [162]  # (50, 0.01, 0.9, 'fBm', 2)

    # find_best_params(trajectories[tnum], tnum, indexes, [params[i] for i in indexes])

    # dclusters = [measure.label(np.load(f"./output/h2_traj/{tnum}/{i}.npy"), connectivity=1).max() for i in indexes]
    # plt.hist(dclusters)
    # plt.show()

    # print([i for i in indexes if params[i][1] == 0.1])

    fix, axs = plt.subplots(2, 3)
    axs[0, 0].hist([params[i][0] for i in indexes])
    axs[0, 1].hist([params[i][1] for i in indexes])
    axs[0, 2].hist([params[i][2] for i in indexes])
    axs[1, 0].hist([params[i][3] for i in indexes])
    axs[1, 1].hist([params[i][4] for i in indexes])
    plt.show()

    print(f" -- count all params: {len(indexes)}")
    for i in indexes:
        trajectories[traj_num].traps = np.load(f"./output/h2_traj/{traj_num}/{i}.npy")
        clusters = measure.label(trajectories[traj_num].traps, connectivity=1)
        print(f" --- Index: {i}, Count clusters: {clusters.max()}")
        visualize_trajectory(trajectories[traj_num], color_type='clusters')

    # use this for methan_traj
    # AnalizerParams(traj_type='Bm', nu=0.5, diag_percentile=0 , kernel_size=3, list_mu=np.array([0.5, 1. , 1.5, 2. , 2.5, 3.]), p_value=0.01)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path',
        type=str,
        # default="../data/methan_traj/meth_1.7_micros.1.gro"
        default="../data/h2_micros/h2_micros.1.gro"
    )
    path_to_traj = parser.parse_args()

    trajectories = Trajectory.read_trajectoryes(path_to_traj.path)

    list_mu = np.array([0.5, 1. , 1.5, 2. , 2.5, 3.])
    atparams = all_params()
    # indexes = [154, 162, 186]
    indexes = [162]

    tparams = [atparams[i] for i in indexes]
    aparams = [AnalizerParams(tt, nu, dp, ks, list_mu, pv) for dp, pv, nu, tt, ks in tparams]
    params = aparams[0]

    # run_and_plot_trap_time_distribution(trajectories, list(zip(indexes, aparams)))
    # analize_script(4)

    # for i in [12, 14, 19]:  #
    #     visualize_trajectory(trajectories[i])

    # params = AnalizerParams(traj_type='fBm', nu=0.9, diag_percentile=50 , kernel_size=1, list_mu=np.array([0.5, 1. , 1.5, 2. , 2.5, 3.]), p_value=0.01)
    params.list_mu = np.array([1.5, 2])
    analizy_visualize(trajectories[5], params)

    # visualize_trajectories(trajectories)
    # animate_trajectoryes(trajectories)

    # plt.plot(trj.times, trj.clusters)
    # plt.xlabel('Time', size = 24)
    # plt.ylabel('Pore number', size = 24)
    # plt.xticks(size = 18)
    # plt.yticks(size = 18)
    # plt.show()

    # for t in trajectories:
    #     visualize_dist_trajectory(t)
    #     plt.show()
