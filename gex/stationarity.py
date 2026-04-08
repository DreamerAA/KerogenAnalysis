"""
KS-based stationarity check for time-indexed samples + vector plots.

Assumptions
-----------
You have samples for different time instants stored as 1D numpy arrays.
Example structure:
    samples_l = {t0: arr0, t1: arr1, ...}   # lengths
    samples_r = {t0: arr0, t1: arr1, ...}   # radii
or as a list in time order:
    samples_l = [arr_t0, arr_t1, ...]
    samples_r = [arr_t0, arr_t1, ...]

This script:
1) runs two-sample Kolmogorov–Smirnov tests:
   - baseline vs all times
   - adjacent times
   - full pairwise matrix (optional)
2) applies multiple-testing correction (Holm by default)
3) generates vector figures (PDF/SVG):
   - overlay hist (density) + optional KDE
   - overlay ECDF
   - D-stat and (corrected) p-values vs time
   - pairwise heatmaps for D and p (optional)
"""

from __future__ import annotations

from os import listdir
from os.path import dirname, isfile, join, realpath

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
from scipy.stats import kstwobign
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, gaussian_kde
import matplotlib.cm as cm
from base.reader import Reader


Array1D = np.ndarray
Samples = Union[Dict[float, Array1D], Dict[int, Array1D], Sequence[Array1D]]


AX_LABEL_FONTSIZE = 14
TICK_LABEL_FONTSIZE = 12
LEGEND_FONTSIZE = 11
TITLE_FONTSIZE = 14


def _select_indices(
    n: int, max_curves: int, highlight_pair: Optional[Tuple[int, int]]
) -> List[int]:
    idx = np.linspace(0, n - 1, min(max_curves, n)).round().astype(int)
    idx = np.unique(idx).tolist()

    if highlight_pair is not None:
        i, j = highlight_pair
        if 0 <= i < n and i not in idx:
            idx.append(i)
        if 0 <= j < n and j not in idx:
            idx.append(j)

    return sorted(set(idx))


def _make_color_map(idx: Sequence[int]):
    # стабильные цвета (tab10/tab20) — один и тот же idx -> один и тот же цвет
    cmap = cm.get_cmap("tab10" if len(idx) <= 10 else "tab20", len(idx))
    return {k: cmap(pos) for pos, k in enumerate(idx)}


def _style_axes(ax):
    # axis labels: bigger + bold
    ax.xaxis.label.set_size(AX_LABEL_FONTSIZE)
    ax.yaxis.label.set_size(AX_LABEL_FONTSIZE)

    # ticks: bigger, not bold
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONTSIZE)
    ax.tick_params(axis="both", which="minor", labelsize=TICK_LABEL_FONTSIZE)


def _style_title(ax):
    ax.title.set_size(TITLE_FONTSIZE)
    ax.title.set_weight("normal")


# -----------------------------
# Utilities
# -----------------------------
def _as_time_ordered(samples: Samples) -> Tuple[np.ndarray, List[Array1D]]:
    """
    Returns (times, arrays) in increasing time order.
    If samples is a list/tuple -> times = [0,1,2,...]
    """
    if isinstance(samples, dict):
        times = np.array(sorted(samples.keys()), dtype=int)
        arrays = [np.asarray(samples[t]).ravel() for t in times]
        return times, arrays
    else:
        arrays = [np.asarray(a).ravel() for a in samples]
        times = np.arange(len(arrays), dtype=int)
        return times, arrays


def _prepare_arrays(
    samples: Samples,
    *,
    drop_nan: bool = True,
    positive_only: bool = False,
    transform: Optional[str] = None,
) -> Tuple[np.ndarray, List[Array1D]]:
    times, arrays0 = _as_time_ordered(samples)
    arrays: List[Array1D] = []
    for a in arrays0:
        aa = _clean_sample(a, drop_nan=drop_nan, positive_only=positive_only)
        if transform == "log":
            aa = aa[aa > 0]
            aa = np.log(aa)
        elif transform == "log1p":
            aa = aa[aa >= 0]
            aa = np.log1p(aa)
        arrays.append(aa)
    return times, arrays


def plot_ks_null_tail(
    samples: Samples,
    outpath: Path,
    *,
    pair: Tuple[int, int],
    title: str,
    drop_nan: bool = True,
    positive_only: bool = False,
    transform: Optional[str] = None,
):
    times, arrays = _prepare_arrays(
        samples,
        drop_nan=drop_nan,
        positive_only=positive_only,
        transform=transform,
    )
    i, j = pair
    a, b = arrays[i], arrays[j]
    if a.size < 2 or b.size < 2:
        return

    # Compute observed KS D and corresponding lambda = sqrt(neff) * D
    res = ks_2samp(a, b, alternative="two-sided", method="auto")
    D_obs = float(res.statistic)
    n = a.size
    m = b.size
    neff = (n * m) / (n + m)
    lam_obs = np.sqrt(neff) * D_obs

    # Asymptotic null: lambda ~ kstwobign under H0 (large-sample)
    # We'll plot the survival function S(lambda) = P(Lambda >= lambda)
    x = np.linspace(0.0, max(2.5, lam_obs * 1.5), 512)
    sf = kstwobign.sf(x)  # tail probability

    p_asym = float(kstwobign.sf(lam_obs))  # tail area (asymptotic)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(x, sf)
    ax.axvline(lam_obs, linewidth=2)
    ax.annotate(
        f"lambda_obs = {lam_obs:.4g}\nD_obs = {D_obs:.4g}\np(asym) = {p_asym:.4g}",
        xy=(lam_obs, kstwobign.sf(lam_obs)),
        xytext=(8, 8),
        textcoords="offset points",
        ha="left",
        va="bottom",
    )

    # Shade the tail area visually (in SF-plot it’s the y-value at lambda)
    # If you prefer CDF/pdf with filled area, скажи — сделаю второй вариант.
    ax.set_title(title)
    ax.set_xlabel(r"$\lambda = \sqrt{n_{\mathrm{eff}}}\,D$")
    ax.set_ylabel(r"Survival function $P(\lambda \geq \lambda_{\mathrm{obs}})$")

    fig.tight_layout()
    fig.savefig(outpath, format=outpath.suffix.lstrip("."), bbox_inches="tight")
    plt.close(fig)


def _clean_sample(
    x: Array1D, *, drop_nan: bool = True, positive_only: bool = False
) -> Array1D:
    x = np.asarray(x).ravel()
    if drop_nan:
        x = x[np.isfinite(x)]
    if positive_only:
        x = x[x > 0]
    return x


def holm_correction(
    pvals: np.ndarray, alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Holm-Bonferroni multiple testing correction.

    Returns
    -------
    reject : boolean array (True if H0 rejected under Holm at level alpha)
    p_adj  : adjusted p-values (Holm)
    """
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    p_sorted = p[order]

    # Holm adjusted p-values (step-down)
    p_adj_sorted = np.empty_like(p_sorted)
    for i in range(m):
        p_adj_sorted[i] = (m - i) * p_sorted[i]
    # enforce monotonicity
    p_adj_sorted = np.maximum.accumulate(p_adj_sorted[::-1])[::-1]
    p_adj_sorted = np.clip(p_adj_sorted, 0.0, 1.0)

    p_adj = np.empty_like(p_adj_sorted)
    p_adj[order] = p_adj_sorted

    reject = p_adj <= alpha
    return reject, p_adj


def ecdf(x: Array1D) -> Tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(x).ravel())
    y = np.arange(1, x.size + 1) / x.size
    return x, y


@dataclass
class KSTestResult:
    pairs: List[Tuple[int, int]]  # indices in time-ordered list
    times: np.ndarray  # time stamps aligned with arrays
    D: np.ndarray  # KS statistic per pair
    p: np.ndarray  # raw p-values per pair
    p_adj: Optional[np.ndarray] = None  # adjusted p-values
    reject: Optional[np.ndarray] = None  # rejection after correction


# -----------------------------
# KS testing
# -----------------------------
def ks_pairs(
    samples: Samples,
    *,
    pair_mode: str = "baseline",  # "baseline" | "adjacent" | "all"
    baseline_index: int = 0,
    alpha: float = 0.05,
    correction: str = "holm",  # "holm" | "none"
    alternative: str = "two-sided",
    method: str = "auto",
    drop_nan: bool = True,
    positive_only: bool = False,
    transform: Optional[str] = None,  # None | "log" | "log1p"
) -> KSTestResult:
    """
    Run KS tests for chosen set of pairs.

    transform:
      - "log"   : log(x) on x>0
      - "log1p" : log(1+x) on x>=0 (recommended if zeros appear)
    """
    times, arrays0 = _as_time_ordered(samples)
    arrays = []
    for a in arrays0:
        aa = _clean_sample(a, drop_nan=drop_nan, positive_only=positive_only)
        if transform == "log":
            aa = aa[aa > 0]
            aa = np.log(aa)
        elif transform == "log1p":
            aa = aa[aa >= 0]
            aa = np.log1p(aa)
        arrays.append(aa)

    n = len(arrays)
    if n < 2:
        raise ValueError(
            "Need at least 2 time instants to compare distributions."
        )

    pairs: List[Tuple[int, int]] = []
    if pair_mode == "baseline":
        b = baseline_index
        if not (0 <= b < n):
            raise ValueError("baseline_index out of range.")
        pairs = [(b, j) for j in range(n) if j != b]
    elif pair_mode == "adjacent":
        pairs = [(i, i + 1) for i in range(n - 1)]
    elif pair_mode == "all":
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        raise ValueError("pair_mode must be one of: baseline|adjacent|all")

    D = np.zeros(len(pairs), dtype=float)
    p = np.zeros(len(pairs), dtype=float)

    for k, (i, j) in enumerate(pairs):
        a, b = arrays[i], arrays[j]
        if a.size < 2 or b.size < 2:
            D[k] = np.nan
            p[k] = np.nan
            continue
        res = ks_2samp(a, b, alternative=alternative, method=method)
        D[k] = float(res.statistic)
        p[k] = float(res.pvalue)

    out = KSTestResult(pairs=pairs, times=times, D=D, p=p)

    if correction == "holm":
        mask = np.isfinite(p)
        reject = np.full_like(p, False, dtype=bool)
        p_adj = np.full_like(p, np.nan, dtype=float)
        if mask.any():
            reject_m, p_adj_m = holm_correction(p[mask], alpha=alpha)
            reject[mask] = reject_m
            p_adj[mask] = p_adj_m
        out.reject = reject
        out.p_adj = p_adj
    elif correction == "none":
        out.reject = p <= alpha
        out.p_adj = p.copy()
    else:
        raise ValueError("correction must be 'holm' or 'none'")

    return out


def stationarity_summary(res: KSTestResult, *, alpha: float = 0.05) -> str:
    """
    Human-readable summary.
    """
    p_use = res.p_adj if res.p_adj is not None else res.p
    reject = res.reject if res.reject is not None else (p_use <= alpha)

    valid = np.isfinite(p_use)
    n_tests = int(valid.sum())
    n_rej = int(np.logical_and(valid, reject).sum())

    if n_tests == 0:
        return "No valid tests (insufficient data / all NaN)."

    msg = (
        f"KS tests: {n_tests} valid comparisons; "
        f"rejections at alpha={alpha}: {n_rej}.\n"
        f"Max KS D: {np.nanmax(res.D):.4g}, median D: {np.nanmedian(res.D):.4g}\n"
        f"Min p(adj): {np.nanmin(p_use):.4g}, median p(adj): {np.nanmedian(p_use):.4g}"
    )
    return msg


# -----------------------------
# Plotting (vector output)
# -----------------------------
def plot_hist_overlay(
    samples: Samples,
    outpath: Path,
    *,
    title: str,
    xname: str,
    bins: int = 50,
    density: bool = True,
    kde: bool = True,
    max_curves: int = 6,
    drop_nan: bool = True,
    positive_only: bool = False,
    transform: Optional[str] = None,
    idx: Optional[Sequence[int]] = None,
    color_map: Optional[dict] = None,
):
    times, arrays0 = _as_time_ordered(samples)
    arrays = []
    for a in arrays0:
        aa = _clean_sample(a, drop_nan=drop_nan, positive_only=positive_only)
        if transform == "log":
            aa = aa[aa > 0]
            aa = np.log(aa)
        elif transform == "log1p":
            aa = aa[aa >= 0]
            aa = np.log1p(aa)
        arrays.append(aa)

    # choose curves: evenly spaced in time
    n = len(arrays)
    if idx is None:
        idx = np.linspace(0, n - 1, min(max_curves, n)).round().astype(int)
        idx = np.unique(idx).tolist()
    else:
        idx = list(idx)

    if color_map is None:
        color_map = _make_color_map(idx)

    # common range
    allx = np.concatenate([arrays[i] for i in idx if arrays[i].size > 0])
    xmin, xmax = np.nanmin(allx), np.nanmax(allx)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in idx:
        x = arrays[i]
        if x.size < 2:
            continue
        c = color_map.get(i, None)
        t = times[i] / 1e6
        label = f"t = {t:.2f} us"
        ax.hist(
            x,
            bins=bins,
            range=(xmin, xmax),
            density=density,
            histtype="step",
            label=label,
            color=c,
        )
        if kde and x.size >= 10:
            grid = np.linspace(xmin, xmax, 512)
            pdf = gaussian_kde(x)(grid)
            ax.plot(grid, pdf)

    ax.set_title(title)
    ax.set_xlabel(xname)
    ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE)

    _style_title(ax)
    _style_axes(ax)

    fig.tight_layout()
    fig.savefig(outpath, format=outpath.suffix.lstrip("."), bbox_inches="tight")
    plt.close(fig)


def plot_ecdf_overlay(
    samples: Samples,
    outpath: Path,
    *,
    title: str,
    xname: str,
    max_curves: int = 6,
    highlight_pair: Optional[
        Tuple[int, int]
    ] = None,  # (i,j) in time-ordered arrays
    drop_nan: bool = True,
    positive_only: bool = False,
    transform: Optional[str] = None,
    idx: Optional[Sequence[int]] = None,
    color_map: Optional[dict] = None,
):
    times, arrays = _prepare_arrays(
        samples,
        drop_nan=drop_nan,
        positive_only=positive_only,
        transform=transform,
    )

    n = len(arrays)
    if n < 2:
        return

    if idx is None:
        idx = _select_indices(n, max_curves, highlight_pair)
    else:
        idx = sorted(set(list(idx)))

    if color_map is None:
        color_map = _make_color_map(idx)

    # Ensure highlight pair curves are always included
    if highlight_pair is not None:
        i, j = highlight_pair
        if 0 <= i < n and 0 <= j < n:
            if i not in idx:
                idx.append(i)
            if j not in idx:
                idx.append(j)

    idx = sorted(set(idx))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Plot ECDF curves
    for i in idx:
        x = arrays[i]
        if x.size < 2:
            continue
        xs, ys = ecdf(x)
        c = color_map.get(i, None)
        t = times[i] / 1e6
        label = f"t = {t:.2f} us"
        ax.step(
            xs,
            ys,
            where="post",
            label=label,
            color=c,
        )

    # If highlight_pair is given: draw the KS vertical segment on top
    if highlight_pair is not None:
        i, j = highlight_pair
        a = arrays[i]
        b = arrays[j]
        if a.size >= 2 and b.size >= 2:
            xa, ya = ecdf(a)
            xb, yb = ecdf(b)

            xgrid = np.unique(np.concatenate([xa, xb]))
            Fa = np.searchsorted(xa, xgrid, side="right") / xa.size
            Gb = np.searchsorted(xb, xgrid, side="right") / xb.size
            diff = np.abs(Fa - Gb)
            kmax = int(np.argmax(diff))
            x_star = float(xgrid[kmax])
            D_obs = float(diff[kmax])
            y1 = float(Fa[kmax])
            y2 = float(Gb[kmax])

            ax.vlines(x_star, min(y1, y2), max(y1, y2), linewidth=2)
            ax.annotate(
                f"D = {D_obs:.4g}",
                xy=(x_star, 0.5 * (y1 + y2)),
                xytext=(8, 8),
                textcoords="offset points",
                ha="left",
                va="top",
            )

    ax.set_title(title)
    ax.set_xlabel(xname)
    _style_axes(ax)

    ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE)
    fig.tight_layout()
    fig.savefig(outpath, format=outpath.suffix.lstrip("."), bbox_inches="tight")
    plt.close(fig)


def plot_ks_vs_time(
    res: KSTestResult,
    outpath: Path,
    *,
    title: str,
    use_adjusted_p: bool = True,
):
    p = res.p_adj if (use_adjusted_p and res.p_adj is not None) else res.p
    pairs = res.pairs
    times = res.times

    xs = []
    for i, j in pairs:
        xs.append(0.5 * (times[i] + times[j]))
    xs = np.asarray(xs)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    # D line
    l1 = ax1.plot(xs, res.D, marker="o", linestyle="-", label="KS statistic D")
    ax1.set_title(title)
    ax1.set_xlabel("Time, ps")
    ax1.set_ylabel("KS statistic D")

    # second axis for p-values
    # ax2 = ax1.twinx()
    # label_p = (
    #     "p-value (adj)"
    #     if (use_adjusted_p and res.p_adj is not None)
    #     else "p-value"
    # )
    # l2 = ax2.plot(xs, p, marker="x", linestyle="--", label=label_p)
    # ax2.set_ylabel(label_p)
    # ax2.set_ylim(0.8, 1.0)

    # Styling
    _style_title(ax1)
    _style_axes(ax1)
    # _style_axes(ax2)

    # Combined legend (handles from both axes)
    # handles = l1 + l2
    # labels = [h.get_label() for h in handles]

    # Reserve space on the right for legend
    fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])

    # ax1.legend(
    #     handles,
    #     labels,
    #     frameon=False,
    #     fontsize=LEGEND_FONTSIZE,
    #     loc="upper left",
    #     bbox_to_anchor=(1.02, 0.0),  # outside right
    #     borderaxespad=0.0,
    # )
    fig.tight_layout()
    fig.savefig(outpath, format=outpath.suffix.lstrip("."), bbox_inches="tight")
    plt.close(fig)


def plot_pairwise_heatmaps(
    samples: Samples,
    outdir: Path,
    *,
    name: str,
    alpha: float = 0.05,
    correction: str = "holm",
    transform: Optional[str] = None,
    drop_nan: bool = True,
    positive_only: bool = False,
):
    """
    Builds full pairwise matrix and plots heatmaps for D and p_adj.
    """
    res = ks_pairs(
        samples,
        pair_mode="all",
        alpha=alpha,
        correction=correction,
        transform=transform,
        drop_nan=drop_nan,
        positive_only=positive_only,
    )
    times, arrays0 = _as_time_ordered(samples)
    n = len(arrays0)

    Dm = np.full((n, n), np.nan, dtype=float)
    Pm = np.full((n, n), np.nan, dtype=float)
    p_use = res.p_adj if res.p_adj is not None else res.p

    for D, p, (i, j) in zip(res.D, p_use, res.pairs):
        Dm[i, j] = Dm[j, i] = D
        Pm[i, j] = Pm[j, i] = p

    # Heatmap D
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(Dm, aspect="auto")
    ax.set_title(f"{name}: KS D (pairwise)")
    ax.set_xlabel("time index")
    ax.set_ylabel("time index")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(outdir / f"{name}_ks_D_heatmap.svg", bbox_inches="tight")
    plt.close(fig)

    # Heatmap p (log scale in colors is useful; keep raw colormap for simplicity)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(Pm, aspect="auto")
    ax.set_title(
        f"{name}: KS p-value (pairwise, {'adj' if correction!='none' else 'raw'})"
    )
    ax.set_xlabel("time index")
    ax.set_ylabel("time index")
    fig.colorbar(im, ax=ax)

    _style_title(ax)
    _style_axes(ax)

    fig.tight_layout()
    fig.savefig(outdir / f"{name}_ks_p_heatmap.svg", bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Example "driver"
# -----------------------------
def run_stationarity_pipeline(
    samples: Samples,
    outdir: Union[str, Path],
    *,
    label: str,
    xname: str,
    alpha: float = 0.05,
    baseline_index: int = 0,
    transform: Optional[str] = None,  # None|"log"|"log1p"
    positive_only: bool = False,
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Tests
    res_base = ks_pairs(
        samples,
        pair_mode="baseline",
        baseline_index=baseline_index,
        alpha=alpha,
        correction="holm",
        transform=transform,
        positive_only=positive_only,
    )
    res_adj = ks_pairs(
        samples,
        pair_mode="adjacent",
        alpha=alpha,
        correction="holm",
        transform=transform,
        positive_only=positive_only,
    )

    print(
        f"[{label}] Baseline-vs-all summary:\n{stationarity_summary(res_base, alpha=alpha)}\n"
    )
    print(
        f"[{label}] Adjacent summary:\n{stationarity_summary(res_adj, alpha=alpha)}\n"
    )

    # Choose representative "worst-case" pair (baseline mode) for the ECDF+D illustration
    k_star = int(np.nanargmax(res_base.D))
    pair_star = res_base.pairs[k_star]
    # Shared curve selection + shared colors for hist/ecdf
    times, arrays = _prepare_arrays(
        samples, transform=transform, positive_only=positive_only
    )
    idx = _select_indices(len(arrays), max_curves=6, highlight_pair=pair_star)
    color_map = _make_color_map(idx)

    # 2) Plots (vector)
    plot_hist_overlay(
        samples,
        outpath=outdir / f"{label}_hist_overlay.svg",
        title=f"{label}: Density histogram overlay",
        xname=xname,
        kde=True,
        transform=transform,
        positive_only=positive_only,
        idx=idx,
        color_map=color_map,
    )

    plot_ecdf_overlay(
        samples,
        outpath=outdir / f"{label}_ecdf_overlay.svg",
        title=f"{label}: ECDF overlay (includes worst-case baseline pair)",
        transform=transform,
        xname=xname,
        positive_only=positive_only,
        highlight_pair=pair_star,
        idx=idx,
        color_map=color_map,
    )
    plot_ks_vs_time(
        res_base,
        outpath=outdir / f"{label}_ks_baseline_vs_time.svg",
        title=f"{label}: KS (baseline vs time)",
        use_adjusted_p=True,
    )
    # plot_ks_vs_time(
    #     res_adj,
    #     outpath=outdir / f"{label}_ks_adjacent_vs_time.svg",
    #     title=f"{label}: KS (adjacent times)",
    #     use_adjusted_p=True,
    # )

    k_star = int(np.nanargmax(res_base.D))
    pair_star = res_base.pairs[k_star]

    # plot_ks_null_tail(
    #     samples,
    #     outpath=outdir / f"{label}_ks_null_tail.svg",
    #     pair=pair_star,
    #     title=f"{label}: KS null tail illustration (asymptotic)",
    #     transform=transform,
    #     positive_only=positive_only,
    # )
    # Optional: full pairwise heatmaps (can be heavy if T is big)
    # plot_pairwise_heatmaps(samples, outdir, name=label, alpha=alpha, transform=transform, positive_only=positive_only)

    return res_base, res_adj


def analysis(main_path: str, pnm_dir: str, oputput_dir: str):
    path_to_pnms = join(main_path, pnm_dir)
    onlyfiles = [
        f for f in listdir(path_to_pnms) if isfile(join(path_to_pnms, f))
    ]
    onlyfiles = [
        file
        for file in onlyfiles
        if "_link1" in file and file.startswith("num")
    ]
    steps = [int((file.split("=")[1]).split("_")[0]) for file in onlyfiles]
    sorted_lfiles = list(zip(steps, onlyfiles))
    sorted_lfiles = sorted(sorted_lfiles, key=lambda x: x[0])

    samples_l = {}
    samples_r = {}

    for step, file in sorted_lfiles:
        if step == 25000:
            continue

        radiuses, throat_lengths = Reader.read_pnm_data(
            join(path_to_pnms, file[:-10]), scale=1e10, border=0.015
        )
        time = int(50 + (step - 25000.0) * 500.0 / 250000.0)

        samples_l[time] = throat_lengths
        samples_r[time] = radiuses
        print(f"Step: {step} Time: {time}")

    outdir = join(main_path, oputput_dir)

    # If distributions are heavy-tailed, consider transform="log1p" (especially if zeros possible)
    # For strictly positive values and no zeros, transform="log" is OK.
    if samples_l is not None:
        run_stationarity_pipeline(
            samples_l,
            outdir,
            label="P(l)",
            xname="Throat length ($\AA$)",
            alpha=0.05,
            transform=None,
            positive_only=False,
        )

    if samples_r is not None:
        run_stationarity_pipeline(
            samples_r,
            outdir,
            label="P(r)",
            xname="Pore radius ($\AA$)",
            alpha=0.05,
            transform=None,
            positive_only=False,
        )


if __name__ == "__main__":

    paths = ["/media/andrey/Samsung_T5/PHD/Kerogen/type1matrix/300K/ch4/"]
    for path in paths:
        analysis(path, "pnm", "ks_stationarity")
