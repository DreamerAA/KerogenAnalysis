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
from processes.trap_extractor import TrapExtractor

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


def save_error_corridor_png(
    out_dir: str,
    filename: str,
    prob_grid: np.ndarray,
    ar_prob_error: np.ndarray,  # shape: (P, T)
    ar_hybrid_error: np.ndarray,  # shape: (P, T)
    ar_struct_error: np.ndarray,  # shape: (P, T)
    title: str,
    q_low: float = 0.1,  # 10%
    q_high: float = 0.9,  # 90%
    center: str = "median",  # "mean" | "median"
) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    def band(arr: np.ndarray):
        if center == "mean":
            c = arr.mean(axis=1)
        else:
            c = np.quantile(arr, 0.5, axis=1)

        lo = np.quantile(arr, q_low, axis=1)
        hi = np.quantile(arr, q_high, axis=1)
        return lo, c, hi

    p_lo, p_c, p_hi = band(ar_prob_error)
    h_lo, h_c, h_hi = band(ar_hybrid_error)
    s_lo, s_c, s_hi = band(ar_struct_error)

    fig = plt.figure()

    # Probabilistic
    plt.fill_between(prob_grid, p_lo, p_hi, alpha=0.2)
    plt.plot(
        prob_grid,
        p_c,
        label=f"Prob ({center}, {int(q_low*100)}–{int(q_high*100)}%)",
    )

    # Hybrid
    plt.fill_between(prob_grid, h_lo, h_hi, alpha=0.2)
    plt.plot(
        prob_grid,
        h_c,
        label=f"Hybrid ({center}, {int(q_low*100)}–{int(q_high*100)}%)",
    )

    # Structural
    plt.fill_between(prob_grid, s_lo, s_hi, alpha=0.2)
    plt.plot(
        prob_grid,
        s_c,
        label=f"Structural ({center}, {int(q_low*100)}–{int(q_high*100)}%)",
    )

    # plt.grid(True)

    plt.xlabel(r"Probability move to new trap, $p$", fontsize=14)
    plt.ylabel(
        "Average error / Count steps", fontsize=14
    )  # или как у тебя подписано
    plt.title(title, fontsize=16)

    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.legend(frameon=False, prop={'size': 12})

    fig.savefig(join(out_dir, filename), dpi=200, bbox_inches="tight")
    plt.close(fig)


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
    count_trj=100,
    count_steps=3000,
):
    path_to_save: str = join(path_to_main, "errors")
    path_to_pil_gf: str = join(path_to_main, "pi_l_gamma_fitter.pkl")
    path_to_tl_wf: str = join(path_to_main, "throat_lengths_weibull_fitter.pkl")
    path_to_radiuses: str = join(path_to_main, "radiuses.npy")

    if not Path(path_to_save).exists():
        os.mkdir(path_to_save)

    params_struct_set = {
        0.1: StructAnalizerParams(
            traj_type='fBm',
            nu=0.1,
            diag_percentile=0,
            kernel_size=0,
            list_mu=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            p_value=0.9,
        ),
        0.5: StructAnalizerParams(
            traj_type='fBm',
            nu=0.1,
            diag_percentile=0,
            kernel_size=1,
            list_mu=[1.5, 2.0],
            p_value=0.9,
        ),
        0.9: StructAnalizerParams(
            traj_type='Bm',
            nu=0.5,
            diag_percentile=0,
            kernel_size=1,
            list_mu=[1.5],
            p_value=0.1,
        ),
    }

    params_prob_set = {
        k: ProbabilityAnalizerParams(
            critical_probability=1e-3,
        )
        for k in [0.1, 0.5, 0.9]
    }
    params_hybrid_set = {
        k: HybridAnalizerParams(
            params_prob_set[k],
            params_struct_set[k],
            0.3,
        )
        for k in [0.1, 0.5, 0.9]
    }
    pset = {
        k: {
            "hybrid": params_hybrid_set[k],
            "struct": params_struct_set[k],
            "prob": params_prob_set[k],
        }
        for k in [0.1, 0.5, 0.9]
        # for k in [0.5]
    }
    prob_grid = np.arange(0.0, 1.05, 0.05)

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

    psd = create_empirical_cdf(radiuses)
    bs_psd = BufferedSampler(EmpiricalCDF(psd), "psd", size=10_000)
    bs_ptl = BufferedSampler(throat_lengths_weibull_fitter, "ptl", size=10_000)

    # for ps_type in ["poisson", "uniform"]:
    for ps_type in ["uniform"]:
        for mean_count_steps in [100]:
            # for mean_count_steps in [20, 50]:

            ps = ps_generate(ps_type, mean_count=mean_count_steps)
            bs_ps = BufferedSampler(DiscreteCDF(ps), "ps", size=10_000)

            for k, params in pset.items():
                result_shape = (len(prob_grid), count_trj)

                matrix_analyzer = StructTrajectoryAnalizer(params["struct"])
                prob_analizer = ProbabilityTrajectoryAnalizer(
                    params["prob"],
                    pil_gamma_fitter,
                    throat_lengths_weibull_fitter,
                )

                hybrid_analizer = HybridTrajectoryAnalizer(
                    params["hybrid"],
                    pil_gamma_fitter,
                    throat_lengths_weibull_fitter,
                )

                exp_tag = f"ps={ps_type}_mean={mean_count_steps}_count_steps={count_steps}"
                add_tag = f"k={k}_count_trj={count_trj}"
                header = exp_tag + f"_{add_tag}.npy"
                matrix_er_fn = join(path_to_save, "matrix_" + header)
                prob_er_fn = join(path_to_save, "prob_" + header)
                hybrid_er_fn = join(path_to_save, "hybrid_" + header)

                ckpt_fn = path_to_save + f"/checkpoint_{exp_tag}_{add_tag}.pkl"

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
                        and np.allclose(ckpt.get("prob"), prob_grid)
                    ):
                        prob_est_k = ckpt["prob_est_k"]
                        matrix_est_k = ckpt["matrix_est_k"]
                        hybrid_est_k = ckpt["hybrid_est_k"]
                        ar_matrix_error = ckpt["ar_matrix_error"]
                        ar_prob_error = ckpt["ar_prob_error"]
                        ar_hybrid_error = ckpt["ar_hybrid_error"]
                        next_ind = int(ckpt["next_ind"])
                        kprint(f"[RESUME] k={k}: continue from ind={next_ind}")
                    else:
                        kprint(
                            f"[RESUME] k={k}: checkpoint params mismatch, starting from scratch"
                        )
                        ar_matrix_error = np.zeros(shape=result_shape)
                        ar_prob_error = np.zeros(shape=result_shape)
                        ar_hybrid_error = np.zeros(shape=result_shape)
                else:
                    prob_est_k = 0
                    matrix_est_k = 0
                    hybrid_est_k = 0
                    ar_matrix_error = np.zeros(shape=result_shape)
                    ar_prob_error = np.zeros(shape=result_shape)
                    ar_hybrid_error = np.zeros(shape=result_shape)

                total_jobs_k = _calc_total_jobs(count_trj, prob_grid)

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
                        p = float(prob_grid[j])

                        start_time = time.time()
                        simulator = KerogenWalkSimulator(
                            bs_psd, bs_ps, bs_ptl, k, p
                        )

                        traj = simulator.run(count_steps)
                        delta_time = traj.delta_time * 1e-12  # picoseconds
                        real_traps = traj.traps.copy().astype(np.int32)

                        matrix_traps_result = matrix_analyzer.run(traj).astype(
                            np.int32
                        )
                        seq = TrapExtractor.get_trap_seq(
                            matrix_traps_result[1:], delta_time
                        )
                        matrix_est_k_i = (
                            TrapExtractor.get_zero_trap_probability(seq)
                        )

                        prob_traps_result = prob_analizer.run(traj).astype(
                            np.int32
                        )
                        seq = TrapExtractor.get_trap_seq(
                            prob_traps_result, delta_time
                        )
                        prob_est_k_i = TrapExtractor.get_zero_trap_probability(
                            seq
                        )

                        hybrid_analizer.set_trap_approx(matrix_traps_result)
                        hybrid_traps_result = hybrid_analizer.run(traj).astype(
                            np.int32
                        )
                        seq = TrapExtractor.get_trap_seq(
                            hybrid_traps_result, delta_time
                        )
                        hybrid_est_k_i = (
                            TrapExtractor.get_zero_trap_probability(seq)
                        )

                        matrix_error = np.sum(
                            np.abs(real_traps - matrix_traps_result[1:])
                        )

                        prob_error = np.sum(
                            np.abs(real_traps - prob_traps_result)
                        )
                        hybrid_error = np.sum(
                            np.abs(real_traps - hybrid_traps_result)
                        )

                        ar_prob_error[j, i] = prob_error
                        ar_matrix_error[j, i] = matrix_error
                        ar_hybrid_error[j, i] = hybrid_error

                        prob_est_k += prob_est_k_i
                        matrix_est_k += matrix_est_k_i
                        hybrid_est_k += hybrid_est_k_i

                        # глобальный ind (как у тебя в логе) — сделаем совместимым по смыслу:
                        # ind считает все k подряд, но у нас checkpoint на k.
                        # Для печати оставим "локальный" ind внутри k (1..total_jobs_k).
                        local_ind_1based = flat + 1

                        kprint(
                            f"Ready {local_ind_1based} from {total_jobs_k}, trajectory num={i+1}, p={p}, k (non trap prob) = {k}, time = {(time.time() - start_time):.3f}s "
                        )

                        # checkpoint после каждой траектории (можно реже — см. ниже)
                        ckpt = {
                            "k": k,
                            "matrix_est_k": matrix_est_k,
                            "prob_est_k": prob_est_k,
                            "hybrid_est_k": hybrid_est_k,
                            "count_steps": count_steps,
                            "count_trj": count_trj,
                            "prob": prob_grid,
                            "next_ind": flat + 1,
                            "ar_matrix_error": ar_matrix_error,
                            "ar_prob_error": ar_prob_error,
                            "ar_hybrid_error": ar_hybrid_error,
                        }
                        _atomic_pickle_dump(ckpt, ckpt_fn)

                # после завершения k — сохраняем итоговые npy атомарно
                if not (
                    Path(matrix_er_fn).is_file()
                    and Path(prob_er_fn).is_file()
                    and Path(hybrid_er_fn).is_file()
                ):
                    _atomic_npy_save(matrix_er_fn, ar_matrix_error)
                    _atomic_npy_save(prob_er_fn, ar_prob_error)
                    _atomic_npy_save(hybrid_er_fn, ar_hybrid_error)

                    # checkpoint можно удалить, если хочешь
                    # Path(ckpt_fn).unlink(missing_ok=True)

                # нормализация
                ar_struct_n = ar_matrix_error / count_steps
                ar_prob_n = ar_prob_error / count_steps
                ar_hybrid_n = ar_hybrid_error / count_steps

                prob_est_k = prob_est_k / total_jobs_k
                matrix_est_k = matrix_est_k / total_jobs_k
                hybrid_est_k = hybrid_est_k / total_jobs_k

                # df_hybrid_long["Erorr"] = df_hybrid_long["Error"] / count_steps
                # df_prob_long["Erorr"] = df_prob_long["Error"] / count_steps
                # df_matrix_long["Erorr"] = df_matrix_long["Error"] / count_steps
                plots_dir = join(
                    path_to_save,
                    "plots",
                    f"ps={ps_type}",
                    f"mean={mean_count_steps}",
                )
                png_name = (
                    f"errors__k={k}__trj={count_trj}__steps={count_steps}.png"
                )
                # title = f"Errors | ps={ps_type} | mean={mean_count_steps} | k={k} | count trajectories={count_trj} | trajectory steps={count_steps}"
                title = str(r"Errors for $k$=") + str(f"{k}")

                save_error_corridor_png(
                    out_dir=plots_dir,
                    filename=png_name,
                    prob_grid=prob_grid,
                    ar_prob_error=ar_prob_n,
                    ar_hybrid_error=ar_hybrid_n,
                    ar_struct_error=ar_struct_n,
                    title=title,
                    q_low=0.2,
                    q_high=0.8,
                    center="median",  # или "mean"
                )
                kprint(
                    f"For k={k}: probability estimation k={prob_est_k:.3f} | hybrid estimation k={hybrid_est_k:.3f} | matrix estimation k={matrix_est_k:.3f}"
                )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path_to_main',
        type=str,
        default="/media/andrey/Samsung_T5/PHD/Kerogen/type1matrix/300K/ch4/",
    )
    args = parser.parse_args()

    run(args.path_to_main)
