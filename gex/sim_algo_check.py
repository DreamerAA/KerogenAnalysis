import argparse
import os
import pickle
import sys
from os.path import realpath, join, isfile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import Any
from base.discretecdf import DiscreteCDF
from processes.trap_extractor import TrapExtractor
from tqdm import tqdm

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
from processes.neumann_pearson_analizer import (
    NeumannPearsonTrajectoryAnalizer,
    NeumannPearsonAnalizerParams,
)
from processes.prob_np_analizer import (
    ProbabilityNPTrajectoryAnalizer,
    ProbabilityNPTrajectoryAnalizerParams,
)
from processes.neumann_pearson_struct_analizer import (
    NeumannPearsonStructTrajectoryAnalizer,
    NeumannPearsonStructAnalizerParams,
)


def save_error_corridor_png(
    out_dir: str,
    filename: str,
    prob_grid: np.ndarray,
    data: dict[str, tuple(np.ndarray, np.ndarray)],  # shape: (P, T)
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

    fig = plt.figure()

    def plot_error(lbl, error, k_est):
        lo, c, hi = band(error)
        # lbl = str(f" ({center}, {int(q_low*100)}–{int(q_high*100)}%)")

        lbl += str(r", $k_{est}=$") + str(f"{k_est:.3f}")
        plt.fill_between(prob_grid, lo, hi, alpha=0.2)
        plt.plot(prob_grid, c, label=lbl)

    for name, (error, k_est) in data.items():
        plot_error(name, error, k_est)

    plt.xlabel(r"Probability move to new trap, $p$", fontsize=14)
    plt.ylabel(
        "Average error / Count steps", fontsize=12
    )  # или как у тебя подписано
    plt.title(title, fontsize=12)

    plt.yticks(fontsize=10)
    plt.xticks(fontsize=10)
    plt.legend(frameon=False, prop={'size': 10})

    fig.savefig(join(out_dir, filename), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _atomic_pickle_dump(obj: Any, path: str) -> None:
    tmp = path
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def trajectories_simulation(
    path_to_save: str,
    count_trj: int,
    k: float,
    probs_grid: NPFArray,
    count_steps: int,
    bs_psd: BufferedSampler,
    bs_ps: BufferedSampler,
    bs_ptl: BufferedSampler,
):
    path_to_save_trj = join(path_to_save, "trajectories")
    Path(path_to_save_trj).mkdir(parents=True, exist_ok=True)

    trajectories = {}
    for p in probs_grid:
        filename_trajectories = join(path_to_save_trj, f"k={k}_p={p}.pkl")
        if not isfile(filename_trajectories):
            trjs = []
            for i in range(count_trj):
                simulator = KerogenWalkSimulator(bs_psd, bs_ps, bs_ptl, k, p)

                traj = simulator.run(count_steps + 1)
                trjs.append(traj)
            with open(filename_trajectories, "wb") as f:
                pickle.dump(trjs, f)

        with open(filename_trajectories, "rb") as f:
            trjs = pickle.load(f)
        trajectories[(k, p)] = trjs
    return trajectories


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
    params_np_set = {
        k: NeumannPearsonAnalizerParams(0.01) for k in [0.1, 0.5, 0.9]
    }
    params_pnp_set = {
        k: ProbabilityNPTrajectoryAnalizerParams(1e-3, 0.01)
        for k in [0.1, 0.5, 0.9]
    }
    params_np_struct_set = {
        k: NeumannPearsonStructAnalizerParams(0.01) for k in [0.1, 0.5, 0.9]
    }

    pset = {
        k: {
            HybridTrajectoryAnalizer.name(): params_hybrid_set[k],
            StructTrajectoryAnalizer.name(): params_struct_set[k],
            ProbabilityTrajectoryAnalizer.name(): params_prob_set[k],
            NeumannPearsonTrajectoryAnalizer.name(): params_np_set[k],
            ProbabilityNPTrajectoryAnalizer.name(): params_pnp_set[k],
            NeumannPearsonStructTrajectoryAnalizer.name(): params_np_struct_set[
                k
            ],
        }
        for k in [0.1, 0.5, 0.9]
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

    ps_type = 'uniform'
    mean_count_steps = 100

    ps = ps_generate(ps_type, mean_count=mean_count_steps)
    bs_ps = BufferedSampler(DiscreteCDF(ps), "ps", size=10_000)

    header = f"count_steps={count_steps}"
    for k, params in pset.items():
        result_shape = (len(prob_grid), count_trj)

        matrix_analyzer = StructTrajectoryAnalizer(
            params[StructTrajectoryAnalizer.name()]
        )

        neumann_pearson_analizer = NeumannPearsonTrajectoryAnalizer(
            params[NeumannPearsonTrajectoryAnalizer.name()],
            pil_gamma_fitter,
            throat_lengths_weibull_fitter,
        )

        prob_analizer = ProbabilityTrajectoryAnalizer(
            params[ProbabilityTrajectoryAnalizer.name()],
            pil_gamma_fitter,
            throat_lengths_weibull_fitter,
        )

        hybrid_analizer = HybridTrajectoryAnalizer(
            params[HybridTrajectoryAnalizer.name()],
            pil_gamma_fitter,
            throat_lengths_weibull_fitter,
        )
        prob_np_analizer = ProbabilityNPTrajectoryAnalizer(
            params[ProbabilityNPTrajectoryAnalizer.name()],
            pil_gamma_fitter,
            throat_lengths_weibull_fitter,
        )

        np_struct_analizer = NeumannPearsonStructTrajectoryAnalizer(
            params[NeumannPearsonStructTrajectoryAnalizer.name()],
            pil_gamma_fitter,
            throat_lengths_weibull_fitter,
        )

        # total_jobs_k = _calc_total_jobs(count_trj, prob_grid)

        trajectories = trajectories_simulation(
            path_to_save,
            count_trj,
            k,
            prob_grid,
            count_steps,
            bs_psd,
            bs_ps,
            bs_ptl,
        )

        currect_state = {}

        def set_approx_traps(analizer, pi, ti, name):
            results = currect_state[name].get("results")
            traps = results[pi, ti]
            analizer.set_trap_approx(traps)

        def set_approx_struct_traps(analizer, pi, ti):
            set_approx_traps(analizer, pi, ti, StructTrajectoryAnalizer.name())

        # def set_approx_np_traps(analizer, pi, ti):
        #     set_approx_traps(
        #         analizer, pi, ti, NeumannPearsonTrajectoryAnalizer.name()
        #     )

        def empty_func(analizer, pi, ti):
            pass

        def empty_init():
            k_est = np.zeros(shape=result_shape)
            errors = np.zeros(shape=result_shape)
            results = np.zeros(shape=(*result_shape, count_steps))
            return k_est, errors, results

        for analizer, approx_func in [
            (matrix_analyzer, empty_func),
            (neumann_pearson_analizer, empty_func),
            (prob_analizer, empty_func),
            (prob_np_analizer, empty_func),
            (hybrid_analizer, set_approx_struct_traps),
            (np_struct_analizer, set_approx_struct_traps),
        ]:
            exp_tag = (
                f"name={analizer.name()}_k={k}_count_trj={count_trj}_" + header
            )
            # fn = join(path_to_save, header)
            ckpt_fn = join(path_to_save, f"checkpoint_{exp_tag}.pkl")

            # Инициализация/загрузка прогресса
            flat_ind = 0
            if Path(ckpt_fn).is_file():
                with open(ckpt_fn, "rb") as f:
                    ckpt = pickle.load(f)

                # простая валидация, чтобы случайно не продолжить "чужой" прогресс
                if (
                    ckpt.get("k") == k
                    and ckpt.get("count_steps") == count_steps
                    and ckpt.get("count_trj") == count_trj
                    and np.allclose(ckpt.get("prob_grid"), prob_grid)
                ):
                    k_est = ckpt.get("k_est")
                    errors = ckpt.get("errors")
                    results = ckpt.get("results")
                    flat_ind = ckpt.get("flat_ind")

                    kprint(f"Restart for k={k} with flat_ind={flat_ind}")
                else:
                    k_est, errors, results = empty_init()
            else:
                k_est, errors, results = empty_init()

            for ind in tqdm(
                range(len(prob_grid) * count_trj),
                desc=f"Analyze {analizer.name()} for k={k}",
            ):
                if ind < flat_ind:
                    continue
                pi = ind // count_trj
                ti = ind % count_trj

                p = prob_grid[pi]
                trjs = trajectories[(k, p)]
                trj = trjs[ti]

                approx_func(analizer, pi, ti)
                result = analizer.run(trj).astype(np.int32)

                delta_time = trj.delta_time * 1e-12  # picoseconds
                seq = TrapExtractor.get_trap_seq(result, delta_time)

                real_traps = trj.traps.copy().astype(np.int32)
                errors[pi, ti] = np.sum(np.abs(real_traps - result))
                k_est[pi, ti] = seq.get_zero_trap_probability()
                results[pi, ti] = result

                if (ind + 1) % 10 == 0 and ind != 0:
                    ckpt = {
                        "k": k,
                        "prob_grid": prob_grid,
                        "count_steps": count_steps,
                        "count_trj": count_trj,
                        "flat_ind": ind,
                        "k_est": k_est,
                        "results": results,
                        "errors": errors,
                    }
                    _atomic_pickle_dump(ckpt, ckpt_fn)
            currect_state[analizer.name()] = {
                "k": k,
                "prob_grid": prob_grid,
                "count_steps": count_steps,
                "count_trj": count_trj,
                "k_est": k_est,
                "results": results,
                "errors": errors,
            }

        def get_norm_errors_k_est(name):
            errors = currect_state[name].get("errors")
            error = errors / count_steps
            k_est = currect_state[name].get("k_est")
            k_est = np.sum(k_est) / np.size(k_est)
            return error, k_est

        np_errors, np_k_est = get_norm_errors_k_est(
            NeumannPearsonTrajectoryAnalizer.name()
        )
        struct_errors, struct_k_est = get_norm_errors_k_est(
            StructTrajectoryAnalizer.name()
        )
        # prob_errors, prob_k_est = get_norm_errors_k_est(
        #     ProbabilityTrajectoryAnalizer.name()
        # )
        hybrid_errors, hybrid_k_est = get_norm_errors_k_est(
            HybridTrajectoryAnalizer.name()
        )
        prob_np_errors, prob_np_k_est = get_norm_errors_k_est(
            ProbabilityNPTrajectoryAnalizer.name()
        )
        # np_struct_errors, np_struct_k_est = get_norm_errors_k_est(
        #     NeumannPearsonStructTrajectoryAnalizer.name()
        # )

        plots_dir = join(
            path_to_save,
            "plots",
        )
        pdf_name = f"errors_k={k}_trj={count_trj}_steps={count_steps}.svg"
        title = str(r"Errors for $k$=") + str(f"{k}")

        save_error_corridor_png(
            out_dir=plots_dir,
            filename=pdf_name,
            prob_grid=prob_grid,
            data={
                "Structural": (struct_errors, struct_k_est),
                "Neumann-Pearson": (np_errors, np_k_est),
                "Neyman–Pearson + Bayes": (
                    prob_np_errors,
                    prob_np_k_est,
                ),  # prob+np
                # "Hybrid": (hybrid_errors, hybrid_k_est),
                # "Prob": (prob_errors, prob_k_est),
                # "Neumann-Pearson+Struct": (np_struct_errors, np_struct_k_est),
            },
            title=title,
            q_low=0.2,
            q_high=0.8,
            center="median",  # или "mean"
        )

        kprint(
            f"For k={k}: "
            # f"probability estimation k={prob_k_est:.3f} | "
            f"probability + neumann pearson estimation k={prob_np_k_est:.3f} | "
            f"hybrid estimation k={hybrid_k_est:.3f} | "
            f"struct estimation k={struct_k_est:.3f} | "
            # f"neumann pearson estimation k={np_k_est:.3f} | "
            # f"neumann pearson + struct estimation k={np_struct_k_est:.3f} | "
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
