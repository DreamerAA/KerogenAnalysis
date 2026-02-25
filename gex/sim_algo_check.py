import argparse
import os
import pickle
import sys
import time
from os.path import realpath, join, isfile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from base.discretecdf import DiscreteCDF

path = Path(realpath(__file__))
parent_dir = str(path.parent.parent.absolute())
sys.path.append(parent_dir)

from base.bufferedsampler import BufferedSampler

from base.empiricalcdf import EmpiricalCDF
from utils.utils import kprint, ps_generate, create_empirical_cdf
from processes.kerogen_walk_simulator import KerogenWalkSimulator

from processes.hybrid_trajectory_analizer import (
    HybridAnalizerParams,
    HybridTrajectoryAnalizer,
)
from processes.struct_trajectory_analyzer import (
    StructTrajectoryAnalizer,
    StructAnalizerParams,
)
from processes.probability_trajectory_analizer import (
    ProbabilityTrajectoryAnalizer,
    ProbabilityAnalizerParams,
)


np.random.default_rng(1111)


def _atomic_pickle_dump(obj: Any, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_npy_save(path: str, arr: np.ndarray) -> None:
    final_path = path if path.endswith(".npy") else path + ".npy"
    tmp_path = final_path + ".tmp"

    with open(tmp_path, "wb") as f:
        np.save(f, arr)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_path, final_path)


def _calc_total_jobs(count_trj: int, prob: np.ndarray) -> int:
    return count_trj * len(prob)


def run(
    path_to_main: str,
    count_trj=20,
    count_steps=2500,
):
    path_to_save: str = join(path_to_main, "errors")
    path_to_pil_gf: str = join(path_to_main, "pi_l_gamma_fitter.pkl")
    path_to_tl_wf: str = join(path_to_main, "throat_lengths_weibull_fitter.pkl")
    path_to_radiuses: str = join(path_to_main, "radiuses.npy")

    if not Path(path_to_save).exists():
        os.mkdir(path_to_save)

    params_struct_set = {
        0.1: StructAnalizerParams(
            traj_type='Bm',
            nu=0.9,
            diag_percentile=0,
            kernel_size=1,
            list_mu=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            p_value=0.9,
            num_jobs=1,
        ),
        0.5: StructAnalizerParams(
            traj_type='Bm',
            nu=0.1,
            diag_percentile=0,
            kernel_size=1,
            list_mu=[1.5],
            p_value=0.9,
            num_jobs=1,
        ),
        0.95: StructAnalizerParams(
            traj_type='fBm',
            nu=0.1,
            diag_percentile=0,
            kernel_size=0,
            list_mu=[1.0],
            p_value=0.01,
            num_jobs=1,
        ),
    }

    params_prob_set = {
        k: ProbabilityAnalizerParams(
            critical_probability=1e-3,
        )
        for k in [0.1, 0.5, 0.95]
    }
    params_hybrid_set = {
        k: HybridAnalizerParams(
            params_prob_set[k],
            params_struct_set[k],
            0.1,
        )
        for k in [0.1, 0.5, 0.95]
    }
    pset = {
        k: {
            "hybrid": params_hybrid_set[k],
            "struct": params_struct_set[k],
            "prob": params_prob_set[k],
        }
        for k in [0.1, 0.5, 0.95]
        # for k in [0.95]
    }

    if isfile(path_to_pil_gf):
        with open(path_to_pil_gf, "rb") as f:
            pil_gamma_fitter = pickle.load(f)
    else:
        raise RuntimeError("pi_l data not found")

    if isfile(path_to_radiuses):
        radiuses = np.load(path_to_radiuses)
    else:
        raise RuntimeError("radiuses not found")

    if isfile(path_to_tl_wf):
        with open(path_to_tl_wf, "rb") as f:
            throat_lengths_weibull_fitter = pickle.load(f)
    else:
        raise RuntimeError("throat_lengths not found")

    ps_type = 'uniform'  # poisson uniform
    ps = ps_generate(ps_type, max_count_step=10)
    psd = create_empirical_cdf(radiuses)
    bs_ps = BufferedSampler(DiscreteCDF(ps), "ps", size=100_000)
    bs_psd = BufferedSampler(EmpiricalCDF(psd), "psd", size=100_000)
    bs_ptl = BufferedSampler(throat_lengths_weibull_fitter, "ptl", size=100_000)

    header_trajs = [
        f"Trajectory_{i+1}" for i in range(count_trj)
    ]  # Названия траектори
    for k, params in pset.items():
        matrix_analyzer = StructTrajectoryAnalizer(params["struct"])
        prob_analizer = ProbabilityTrajectoryAnalizer(
            params["prob"], pil_gamma_fitter, throat_lengths_weibull_fitter
        )

        hybrid_analizer = HybridTrajectoryAnalizer(
            params["hybrid"], pil_gamma_fitter, throat_lengths_weibull_fitter
        )

        prob = np.arange(0.0, 1.05, 0.05)

        header = f"_k={k}_countSteps={count_steps}_countTrj={count_trj}.npy"
        matrix_er_fn = path_to_save + "/matrix" + header
        prob_er_fn = path_to_save + "/prob" + header
        hybrid_er_fn = path_to_save + "/hybrid" + header

        ckpt_fn = (
            path_to_save
            + f"/checkpoint_k={k}_countSteps={count_steps}_countTrj={count_trj}.pkl"
        )

        prob = np.arange(0.0, 1.1, 0.1)

        # Инициализация/загрузка прогресса
        next_ind = 0
        if Path(ckpt_fn).is_file():
            with open(ckpt_fn, "rb") as f:
                ckpt = pickle.load(f)

            # простая валидация, чтобы случайно не продолжить "чужой" прогресс
            if (
                ckpt.get("k") == k
                and ckpt.get("count_steps") == count_steps
                and ckpt.get("count_trj") == count_trj
                and np.allclose(ckpt.get("prob"), prob)
            ):
                ar_matrix_error = ckpt["ar_matrix_error"]
                ar_prob_error = ckpt["ar_prob_error"]
                ar_hybrid_error = ckpt["ar_hybrid_error"]
                next_ind = int(ckpt["next_ind"])
                kprint(f"[RESUME] k={k}: continue from ind={next_ind}")
            else:
                kprint(
                    f"[RESUME] k={k}: checkpoint params mismatch, starting from scratch"
                )
                ar_matrix_error = np.zeros(shape=(len(prob), count_trj))
                ar_prob_error = np.zeros(shape=(len(prob), count_trj))
                ar_hybrid_error = np.zeros(shape=(len(prob), count_trj))
        else:
            ar_matrix_error = np.zeros(shape=(len(prob), count_trj))
            ar_prob_error = np.zeros(shape=(len(prob), count_trj))
            ar_hybrid_error = np.zeros(shape=(len(prob), count_trj))

        total_jobs_k = _calc_total_jobs(count_trj, prob)

        # если по этому k уже всё посчитано, то можно сразу сохранить итоговые npy и идти дальше
        if next_ind >= total_jobs_k:
            kprint(
                f"[SKIP] k={k}: already finished ({next_ind}/{total_jobs_k})"
            )
        else:
            # основной цикл: плоский индекс -> (j, i)
            for flat in range(next_ind, total_jobs_k):
                j = flat // count_trj
                i = flat % count_trj
                p = float(prob[j])

                start_time = time.time()
                simulator = KerogenWalkSimulator(bs_psd, bs_ps, bs_ptl, k, p)

                traj = simulator.run(count_steps)
                real_traps = traj.traps.copy().astype(np.int32)

                matrix_traps_result = matrix_analyzer.run(traj).astype(np.int32)
                prob_traps_result = prob_analizer.run(traj).astype(np.int32)

                hybrid_analizer.set_trap_approx(matrix_traps_result)
                hybrid_traps_result = hybrid_analizer.run(traj).astype(np.int32)

                matrix_error = min(
                    np.sum(np.abs(real_traps - matrix_traps_result[:-1])),
                    np.sum(np.abs(real_traps - matrix_traps_result[1:])),
                )
                prob_error = np.sum(np.abs(real_traps - prob_traps_result))
                hybrid_error = np.sum(np.abs(real_traps - hybrid_traps_result))

                ar_prob_error[j, i] = prob_error
                ar_matrix_error[j, i] = matrix_error
                ar_hybrid_error[j, i] = hybrid_error

                # глобальный ind (как у тебя в логе) — сделаем совместимым по смыслу:
                # ind считает все k подряд, но у нас checkpoint на k.
                # Для печати оставим "локальный" ind внутри k (1..total_jobs_k).
                local_ind_1based = flat + 1

                kprint(
                    f"Ready {local_ind_1based} from {total_jobs_k}, trajectory num={i+1}, p={p}, k (non trap prob) = {k}, time = {time.time() - start_time}s "
                )

                # checkpoint после каждой траектории (можно реже — см. ниже)
                ckpt = {
                    "k": k,
                    "count_steps": count_steps,
                    "count_trj": count_trj,
                    "prob": prob,
                    "next_ind": flat + 1,
                    "ar_matrix_error": ar_matrix_error,
                    "ar_prob_error": ar_prob_error,
                    "ar_hybrid_error": ar_hybrid_error,
                }
                _atomic_pickle_dump(ckpt, ckpt_fn)

            # после завершения k — сохраняем итоговые npy атомарно
            _atomic_npy_save(matrix_er_fn, ar_matrix_error)
            _atomic_npy_save(prob_er_fn, ar_prob_error)
            _atomic_npy_save(hybrid_er_fn, ar_hybrid_error)

            # checkpoint можно удалить, если хочешь
            # Path(ckpt_fn).unlink(missing_ok=True)

        ar_matrix_error = np.load(matrix_er_fn)
        ar_prob_error = np.load(prob_er_fn)
        ar_hybrid_error = np.load(hybrid_er_fn)

        ar_matrix_error /= count_steps
        ar_prob_error /= count_steps
        ar_hybrid_error /= count_steps

        df_matrix = pd.DataFrame(ar_matrix_error, columns=header_trajs)
        df_prob = pd.DataFrame(ar_prob_error, columns=header_trajs)
        df_hybrid = pd.DataFrame(ar_hybrid_error, columns=header_trajs)

        df_matrix["Probability"] = prob
        df_prob["Probability"] = prob
        df_hybrid["Probability"] = prob

        df_matrix_long = df_matrix.melt(
            id_vars=["Probability"], var_name="Trajectory", value_name="Error"
        )
        df_prob_long = df_prob.melt(
            id_vars=["Probability"], var_name="Trajectory", value_name="Error"
        )
        df_hybrid_long = df_hybrid.melt(
            id_vars=["Probability"], var_name="Trajectory", value_name="Error"
        )

        # df_hybrid_long["Erorr"] = df_hybrid_long["Error"] / count_steps
        # df_prob_long["Erorr"] = df_prob_long["Error"] / count_steps
        # df_matrix_long["Erorr"] = df_matrix_long["Error"] / count_steps

        plt.figure()
        sns.lineplot(
            data=df_prob_long,
            x="Probability",
            y="Error",
            label="Probabilistic",
            legend=False,
        )
        sns.lineplot(
            data=df_hybrid_long,
            x="Probability",
            y="Error",
            label="Hybrid",
            legend=False,
        )
        sns.lineplot(
            data=df_matrix_long,
            x="Probability",
            y="Error",
            label="Structural",
            legend=False,
        )

        plt.xlabel("Probability move to next trap", fontsize=12)
        plt.ylabel("Avarage error as (Count Error)/(Count Steps)", fontsize=12)
        plt.title(
            f"Errors for k = {k}, count trajectories {count_trj},\n trajectory count steps {count_steps}",
            fontsize=14,
        )
        # plt.yticks([0.009, 0.013, 0.017],fontsize=24)
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        plt.legend(frameon=False, prop={'size': 12})

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path_to_main',
        type=str,
        default="/media/andrey/Samsung_T5/PHD/Kerogen/type1matrix/300K/ch4/",
    )
    args = parser.parse_args()

    run(args.path_to_main)
