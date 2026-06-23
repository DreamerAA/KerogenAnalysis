"""
Power-law analysis entry points for trap-duration distributions.

Two CLI modes (Clauset, Shalizi & Newman 2009):

  --mode xmin    Sweep x_min and show KS / p-value / alpha vs x_min.
  --mode sample  Sweep sample size and show p-value vs n.

Usage
-----
python gex/powerlaw_analysis.py <path> --prefix SIB --mode xmin \\
       [--n_synth 500] [--n_xmin 30] [--step 1] [--seed 42]

python gex/powerlaw_analysis.py <path> --prefix SIB --mode sample \\
       [--n_synth 500] [--xmin 1e-9] [--step 1] [--seed 42]
"""

import argparse
import glob
import os
import pickle
import re
import warnings
from dataclasses import dataclass, field
from os.path import join
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import numpy as np
import numpy.typing as npt

from base.trap_sequence import TrapSequence
from processes.powerlaw_fitter import (
    FitResult,
    MIN_TAIL_SIZE,
    exp_ccdf,
    exp_loglik,
    fit_exponential,
    fit_plc,
    fit_powerlaw,
    find_xmin_optimal,
    ks_statistic,
    mc_pvalue_pl,
    mc_pvalue_exp,
    mc_pvalue_plc,
    pl_ccdf,
    pl_loglik,
    plc_ccdf_grid,
    plc_loglik,
    vuong_test,
    _plc_normalization,
)
from utils.utils import kprint

_N_SAMPLE_SIZES: int = 20  # number of sample sizes for mode=sample


# ===================================================================== #
# Layer 1 – Data loading
# ===================================================================== #


def load_trap_times(
    path: str, prefix: str, step: int
) -> npt.NDArray[np.float64]:
    """
    Load all non-zero trap durations (seconds) for a given analyzer prefix.

    Scans ``{path}/traps/{prefix}/seq_*.pickle`` and applies ``step``:
    only files whose index is a multiple of step are loaded.
    """
    traps_dir = join(path, "traps", prefix)
    if not os.path.isdir(traps_dir):
        raise RuntimeError(f"Traps directory not found: {traps_dir}")

    pattern = join(traps_dir, "seq_*.pickle")
    files = glob.glob(pattern)
    if not files:
        raise RuntimeError(f"No seq_*.pickle files found in {traps_dir}")

    def _index(fp: str) -> int:
        m = re.search(r'seq_(\d+)\.pickle$', fp)
        return int(m.group(1)) if m else -1

    files_sorted = sorted(files, key=_index)
    # Apply step filter: keep files whose index % step == 0
    files_filtered = [f for f in files_sorted if _index(f) % step == 0]

    parts: List[npt.NDArray[np.float64]] = []
    for fp in files_filtered:
        with open(fp, 'rb') as fh:
            seq: TrapSequence = pickle.load(fh)
        t = np.asarray(seq.times, dtype=np.float64)
        parts.append(t[t > 0.0])

    if not parts:
        raise RuntimeError(
            f"All loaded sequences had zero non-zero times ({traps_dir})"
        )

    times = np.concatenate(parts)
    kprint(
        f"Loaded {len(times)} non-zero trap durations from '{prefix}' "
        f"({len(files_filtered)} trajectory files)"
    )
    return times


# ===================================================================== #
# Layer 2 – Per-xmin orchestrator
# ===================================================================== #


def _fit_all_at_xmin(
    x: npt.NDArray[np.float64],
    xmin: float,
    n_synth: int,
    rng: np.random.Generator,
    compute_mc: bool = True,
) -> Optional[FitResult]:
    """
    Fit PL, Exp, PLC at a single xmin and run all tests.
    Returns None if the tail is too small or critical fits fail.
    """
    x_tail = x[x >= xmin]
    if len(x_tail) < MIN_TAIL_SIZE:
        return None

    # --- Power law ---
    try:
        alpha, alpha_sigma, n_tail = fit_powerlaw(x, xmin)
    except ValueError:
        return None
    if alpha <= 1.0:
        return None

    D_pl = ks_statistic(x_tail, xmin, lambda t: pl_ccdf(t, xmin, alpha))

    p_mc = float('nan')
    if compute_mc:
        p_mc = mc_pvalue_pl(x, xmin, alpha, D_pl, n_synth, rng)

    # --- Exponential ---
    try:
        lam_exp, _ = fit_exponential(x, xmin)
    except ValueError:
        lam_exp = float('nan')
        D_exp = float('nan')
    else:
        D_exp = ks_statistic(x_tail, xmin, lambda t: exp_ccdf(t, xmin, lam_exp))

    p_mc_exp = float('nan')
    if compute_mc and np.isfinite(lam_exp) and lam_exp > 0.0:
        p_mc_exp = mc_pvalue_exp(x, xmin, lam_exp, D_exp, n_synth, rng)

    # --- Power law with cutoff ---
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            alpha_plc, lam_plc, _, _ = fit_plc(x, xmin)
    except (ValueError, Exception):
        alpha_plc = float('nan')
        lam_plc = float('nan')
        D_plc = float('nan')
    else:
        lam_plc = max(0.0, lam_plc)  # Nelder-Mead can return slightly negative
        D_plc = ks_statistic(
            x_tail, xmin, lambda t: plc_ccdf_grid(t, xmin, alpha_plc, lam_plc)
        )

    p_mc_plc = float('nan')
    if compute_mc and np.isfinite(alpha_plc) and np.isfinite(lam_plc):
        p_mc_plc = mc_pvalue_plc(
            x, xmin, alpha_plc, lam_plc, D_plc, n_synth, rng
        )

    # --- Vuong tests ---
    ll_pl = pl_loglik(x_tail, xmin, alpha)

    # PL vs Exp
    if np.isfinite(lam_exp):
        ll_exp = exp_loglik(x_tail, xmin, lam_exp)
        v_p_exp, v_s_exp = vuong_test(ll_pl, ll_exp)
    else:
        v_p_exp, v_s_exp = float('nan'), 0

    # PL vs PLC
    if np.isfinite(alpha_plc) and np.isfinite(lam_plc):
        C_plc = _plc_normalization(alpha_plc, lam_plc, xmin)
        ll_plc = plc_loglik(x_tail, xmin, alpha_plc, lam_plc, C_plc)
        v_p_plc, v_s_plc = vuong_test(ll_pl, ll_plc)
    else:
        v_p_plc, v_s_plc = float('nan'), 0

    return FitResult(
        xmin=xmin,
        n_tail=n_tail,
        alpha=alpha,
        alpha_sigma=alpha_sigma,
        D_pl=D_pl,
        p_mc_pl=p_mc,
        lam_exp=lam_exp,
        D_exp=D_exp,
        p_mc_exp=p_mc_exp,
        alpha_plc=alpha_plc,
        lam_plc=lam_plc,
        D_plc=D_plc,
        p_mc_plc=p_mc_plc,
        vuong_p_pl_vs_exp=v_p_exp,
        vuong_sign_pl_vs_exp=v_s_exp,
        vuong_p_pl_vs_plc=v_p_plc,
        vuong_sign_pl_vs_plc=v_s_plc,
    )


# ===================================================================== #
# Layer 3 – Mode orchestrators
# ===================================================================== #


def run_xmin_mode(
    x: npt.NDArray[np.float64],
    n_xmin: int,
    n_synth: int,
    rng: np.random.Generator,
) -> List[FitResult]:
    """
    Sweep x_min over a log-spaced grid and collect FitResults.
    Grid runs from data minimum to 99th percentile.
    """
    xmin_grid = np.logspace(
        np.log10(x.min()),
        np.log10(x.max() - 1e-10),
        n_xmin,
    )

    results: List[FitResult] = []
    r_pl = _fit_all_at_xmin(x, 9.000000000000001e-09, n_synth, rng, compute_mc=True)
    for i, xm in enumerate(xmin_grid):
        r = _fit_all_at_xmin(x, xm, n_synth, rng, compute_mc=True)
        if r is not None:
            n_str = f"n_tail={r.n_tail}"
            mc_str = (
                f"p_mc={r.p_mc_pl:.3f}"
                if np.isfinite(r.p_mc_pl)
                else "p_mc=nan"
            )
            kprint(
                f"  [{i+1:2d}/{n_xmin}] xmin={xm:.3e}  "
                f"alpha={r.alpha:.3f}  D_pl={r.D_pl:.4f}  {mc_str}  {n_str}"
            )
            results.append(r)
        else:
            kprint(
                f"  [{i+1:2d}/{n_xmin}] xmin={xm:.3e}  skipped (n_tail too small)"
            )

    return results


def run_sample_mode(
    x: npt.NDArray[np.float64],
    xmin: float,
    n_synth: int,
    rng: np.random.Generator,
) -> List[Tuple[int, FitResult]]:
    """
    Sweep sample size (sub-sampling without replacement) and collect FitResults.
    """
    n_total = len(x)
    raw_sizes = np.logspace(np.log10(50), np.log10(n_total), _N_SAMPLE_SIZES)
    sample_sizes = np.unique(np.round(raw_sizes).astype(int))

    results: List[Tuple[int, FitResult]] = []
    for i, n in enumerate(sample_sizes):
        n = int(n)
        x_sub = rng.choice(x, size=n, replace=False)
        r = _fit_all_at_xmin(x_sub, xmin, n_synth, rng, compute_mc=True)
        if r is not None:
            mc_str = (
                f"p_mc={r.p_mc_pl:.3f}"
                if np.isfinite(r.p_mc_pl)
                else "p_mc=nan"
            )
            kprint(
                f"  [{i+1:2d}/{len(sample_sizes)}] n={n:6d}  "
                f"alpha={r.alpha:.3f}  D_pl={r.D_pl:.4f}  {mc_str}"
            )
            results.append((n, r))
        else:
            kprint(
                f"  [{i+1:2d}/{len(sample_sizes)}] n={n:6d}  skipped (n_tail too small)"
            )

    return results


# ===================================================================== #
# Layer 4 – Plotting
# ===================================================================== #

_COLORS = {'pl': '#1f77b4', 'exp': '#ff7f0e', 'plc': '#2ca02c'}
_LABELS = {'pl': 'Power law', 'exp': 'Exponential', 'plc': 'PL + cutoff'}


def _log_axis(ax):
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(mticker.LogFormatterSciNotation())
    ax.xaxis.set_major_locator(mticker.LogLocator(numticks=6))


def plot_xmin_analysis(
    x: npt.NDArray[np.float64],
    results: List[FitResult],
    out_path: str,
    prefix: str,
) -> None:
    """
    Combined figure (GridSpec 4×2):
      Row 0  — KS vs x_min | MC p-values vs x_min
      Row 1  — PL fit (at x_min^pl) | Hill plot α
      Row 2  — Exp fit (at x_min^exp) | Hill plot λ
      Row 3  — PLC fit (at x_min^plc) | PLC parameter heatmap
    Each law uses its own optimal x_min (minimising its KS statistic).
    """
    if not results:
        kprint("No valid results to plot.")
        return

    # ---- Build arrays ----
    xmins_arr     = np.array([r.xmin        for r in results])
    d_pl_arr      = np.array([r.D_pl        for r in results])
    d_exp_arr     = np.array([r.D_exp       for r in results])
    d_plc_arr     = np.array([r.D_plc       for r in results])
    p_mc_pl_arr   = np.array([r.p_mc_pl     for r in results])
    p_mc_exp_arr  = np.array([r.p_mc_exp    for r in results])
    p_mc_plc_arr  = np.array([r.p_mc_plc    for r in results])
    alpha_arr     = np.array([r.alpha       for r in results])
    alpha_sig_arr = np.array([r.alpha_sigma for r in results])
    lam_arr       = np.array([r.lam_exp     for r in results])
    ntail_arr     = np.array([r.n_tail      for r in results], dtype=float)
    alpha_plc_arr = np.array([r.alpha_plc   for r in results])
    lam_plc_arr   = np.array([r.lam_plc     for r in results])

    # ---- Per-law optimal xmin ----
    idx_pl  = int(np.argmin(d_pl_arr))
    idx_exp = int(np.argmin(np.where(np.isfinite(d_exp_arr), d_exp_arr, np.inf)))
    idx_plc = int(np.argmin(np.where(np.isfinite(d_plc_arr), d_plc_arr, np.inf)))

    xmin_opt_pl  = xmins_arr[idx_pl]
    xmin_opt_exp = xmins_arr[idx_exp]
    xmin_opt_plc = xmins_arr[idx_plc]

    res_pl  = results[idx_pl]
    res_exp = results[idx_exp]
    res_plc = results[idx_plc]

    kprint(
        f"\n[per-law xmin]  pl={xmin_opt_pl:.3e}  "
        f"exp={xmin_opt_exp:.3e}  plc={xmin_opt_plc:.3e}"
    )

    # ---- Layout: 4 rows × 2 columns ----
    fig = plt.figure(figsize=(14, 20))
    gs = GridSpec(
        4, 2,
        figure=fig,
        hspace=0.50,
        wspace=0.38,
        height_ratios=[1.0, 1.2, 1.2, 1.2],
    )

    ax_ks  = fig.add_subplot(gs[0, 0])
    ax_pv  = fig.add_subplot(gs[0, 1])
    ax_pl  = fig.add_subplot(gs[1, 0])
    ax_hal = fig.add_subplot(gs[1, 1])
    ax_ex  = fig.add_subplot(gs[2, 0])
    ax_hlm = fig.add_subplot(gs[2, 1])
    ax_pc  = fig.add_subplot(gs[3, 0])
    ax_hm  = fig.add_subplot(gs[3, 1])

    # ---- Shared helper: per-law vertical markers ----
    _sup = {'pl': r'\mathrm{pl}', 'exp': r'\mathrm{exp}', 'plc': r'\mathrm{plc}'}
    _opt_xmins = [
        (xmin_opt_pl, 'pl'), (xmin_opt_exp, 'exp'), (xmin_opt_plc, 'plc')
    ]

    # ================================================================
    # [0,0] KS vs x_min
    # ================================================================
    ax_ks.plot(xmins_arr, d_pl_arr, color=_COLORS['pl'],
               label=_LABELS['pl'], linewidth=2)
    mask_e = np.isfinite(d_exp_arr)
    if mask_e.any():
        ax_ks.plot(xmins_arr[mask_e], d_exp_arr[mask_e],
                   color=_COLORS['exp'], label=_LABELS['exp'], linewidth=2)
    mask_p = np.isfinite(d_plc_arr)
    if mask_p.any():
        ax_ks.plot(xmins_arr[mask_p], d_plc_arr[mask_p],
                   color=_COLORS['plc'], label=_LABELS['plc'], linewidth=2)
    for xm, key in _opt_xmins:
        ax_ks.axvline(xm, color=_COLORS[key], linestyle='--', linewidth=1.2,
                      label=rf'$x_{{\min}}^{{{_sup[key]}}}={xm:.2e}$')
    _log_axis(ax_ks)
    ax_ks.set_xlabel(r'$x_{\min}\ (\mathrm{s})$', fontsize=12)
    ax_ks.set_ylabel('KS statistic $D$', fontsize=12)
    ax_ks.legend(frameon=False, fontsize=8)
    ax_ks.tick_params(labelsize=9)
    ax_ks.set_title(f'{prefix} — KS Goodness-of-Fit vs Threshold', fontsize=11)

    # ================================================================
    # [0,1] MC p-value vs x_min
    # ================================================================
    def _pv_line(ax, arr, color, label):
        m = np.isfinite(arr)
        if m.any():
            ax.plot(xmins_arr[m], arr[m], color=color, linewidth=2, label=label)

    _pv_line(ax_pv, p_mc_pl_arr,  _COLORS['pl'],  'MC p-val (PL)')
    _pv_line(ax_pv, p_mc_exp_arr, _COLORS['exp'], 'MC p-val (Exp)')
    _pv_line(ax_pv, p_mc_plc_arr, _COLORS['plc'], 'MC p-val (PLC)')

    ax_pv.axhline(0.1, color='red', linestyle='--', linewidth=1.2)
    ax_pv.text(xmins_arr[0], 0.12, '$p = 0.1$', color='red', fontsize=9)
    all_pv = np.concatenate([p_mc_pl_arr, p_mc_exp_arr, p_mc_plc_arr])
    ymax_pv = (
        max(float(np.nanmax(all_pv[np.isfinite(all_pv)])) * 1.1, 0.25)
        if np.any(np.isfinite(all_pv)) else 1.0
    )
    ax_pv.fill_between(xmins_arr, 0.1, ymax_pv, color='green', alpha=0.08,
                        label='plausible ($p>0.1$)')
    for xm, key in _opt_xmins:
        ax_pv.axvline(xm, color=_COLORS[key], linestyle='--', linewidth=1.2,
                      label=rf'$x_{{\min}}^{{{_sup[key]}}}$')
    _log_axis(ax_pv)
    ax_pv.set_ylim(0.0, min(ymax_pv, 1.05))
    ax_pv.set_xlabel(r'$x_{\min}\ (\mathrm{s})$', fontsize=12)
    ax_pv.set_ylabel('p-value', fontsize=12)
    ax_pv.legend(frameon=False, fontsize=8)
    ax_pv.tick_params(labelsize=9)
    ax_pv.set_title(f'{prefix} — Monte Carlo p-value vs Threshold', fontsize=11)

    # ================================================================
    # [1,1] Hill plot — α (power law)
    # ================================================================
    err_al = 1.96 * alpha_sig_arr
    ax_hal.plot(xmins_arr, alpha_arr, color=_COLORS['pl'], linewidth=2,
                label=r'$\hat{\alpha}$')
    ax_hal.fill_between(xmins_arr, alpha_arr - err_al, alpha_arr + err_al,
                        color=_COLORS['pl'], alpha=0.20, label=r'$95\%$ CI')
    ax_hal.axvline(xmin_opt_pl, color=_COLORS['pl'], linestyle='--', linewidth=1.2,
                   label=rf'$x_{{\min}}^{{\mathrm{{pl}}}}={xmin_opt_pl:.2e}$')
    _log_axis(ax_hal)
    ax_hal.set_xlabel(r'$x_{\min}\ (\mathrm{s})$', fontsize=12)
    ax_hal.set_ylabel(r'$\hat{\alpha}$', fontsize=12)
    ax_hal.legend(frameon=False, fontsize=9)
    ax_hal.tick_params(labelsize=9)
    ax_hal.set_title(r'Hill Plot — Power-Law Exponent $\alpha$', fontsize=11)

    # ================================================================
    # [2,1] Hill plot — λ (exponential)
    # ================================================================
    mask_lam = np.isfinite(lam_arr) & (lam_arr > 0) & (ntail_arr > 0)
    if mask_lam.any():
        lam_m   = lam_arr[mask_lam]
        xmins_m = xmins_arr[mask_lam]
        ntail_m = ntail_arr[mask_lam]
        err_lm  = 1.96 * lam_m / np.sqrt(ntail_m)
        ax_hlm.plot(xmins_m, lam_m, color=_COLORS['exp'], linewidth=2,
                    label=r'$\hat{\lambda}$')
        ax_hlm.fill_between(xmins_m, lam_m - err_lm, lam_m + err_lm,
                            color=_COLORS['exp'], alpha=0.20, label=r'$95\%$ CI')
    ax_hlm.axvline(xmin_opt_exp, color=_COLORS['exp'], linestyle='--', linewidth=1.2,
                   label=rf'$x_{{\min}}^{{\mathrm{{exp}}}}={xmin_opt_exp:.2e}$')
    _log_axis(ax_hlm)
    ax_hlm.set_xlabel(r'$x_{\min}\ (\mathrm{s})$', fontsize=12)
    ax_hlm.set_ylabel(r'$\hat{\lambda}\ (\mathrm{s}^{-1})$', fontsize=12)
    ax_hlm.legend(frameon=False, fontsize=9)
    ax_hlm.tick_params(labelsize=9)
    ax_hlm.set_title(r'Hill Plot — Exponential Rate $\lambda$', fontsize=11)

    # ================================================================
    # Distribution fit panels helper
    # ================================================================
    n_total  = len(x)
    n_bins   = 50
    bins_all = np.logspace(np.log10(x.min()), np.log10(x.max()), n_bins + 1)
    counts_all, edges_all = np.histogram(x, bins=bins_all, density=True)
    widths_all = np.diff(edges_all)
    pos_mask = counts_all > 0

    def _fit_panel(ax, r, method, color, title):
        xm     = r.xmin
        f_tail = np.sum(x >= xm) / n_total
        x_plot = np.logspace(np.log10(xm), np.log10(x.max()), 500)
        ax.bar(edges_all[:-1][pos_mask], counts_all[pos_mask],
               width=widths_all[pos_mask], align='edge',
               color='steelblue', alpha=0.4, label='all data', zorder=1)
        if method == 'pl' and np.isfinite(r.alpha) and r.alpha > 1.0:
            pdf = f_tail * (r.alpha - 1.0) / xm * (x_plot / xm) ** (-r.alpha)
            ax.plot(x_plot, pdf, color=color, linewidth=2,
                    label=rf'PL $\alpha={r.alpha:.2f}$', zorder=2)
        elif method == 'exp' and np.isfinite(r.lam_exp) and r.lam_exp > 0.0:
            pdf = f_tail * r.lam_exp * np.exp(-r.lam_exp * (x_plot - xm))
            ax.plot(x_plot, pdf, color=color, linewidth=2,
                    label=rf'Exp $\lambda={r.lam_exp:.2e}$', zorder=2)
        elif (method == 'plc' and np.isfinite(r.alpha_plc) and r.alpha_plc > 1.0
              and np.isfinite(r.lam_plc) and r.lam_plc >= 0.0):
            C = _plc_normalization(r.alpha_plc, r.lam_plc, xm)
            if C > 0.0:
                with np.errstate(over='ignore', under='ignore'):
                    pdf_raw = (x_plot ** (-r.alpha_plc)
                               * np.exp(-r.lam_plc * x_plot) / C)
                pdf_raw = np.where(np.isfinite(pdf_raw), pdf_raw, np.nan)
                ax.plot(x_plot, f_tail * pdf_raw, color=color, linewidth=2,
                        zorder=2,
                        label=(rf'PLC $\alpha={r.alpha_plc:.2f}$,'
                               rf' $\lambda={r.lam_plc:.2e}$'))
        ax.axvline(xm, color='grey', linestyle='--', linewidth=1.2, alpha=0.8)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$t\ (\mathrm{s})$', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(frameon=False, fontsize=8)
        ax.tick_params(labelsize=8)
        ax.set_title(title, fontsize=11)

    # [1,0] PL fit
    _fit_panel(ax_pl, res_pl,  'pl',  _COLORS['pl'],  'Power-Law Fit')
    # [2,0] Exp fit
    _fit_panel(ax_ex, res_exp, 'exp', _COLORS['exp'], 'Exponential Fit')
    # [3,0] PLC fit
    _fit_panel(ax_pc, res_plc, 'plc', _COLORS['plc'], 'Power Law + Cutoff Fit')

    # ================================================================
    # [3,1] PLC heatmap — (alpha_plc, lam_plc) coloured by D_plc
    # ================================================================
    mask_hm = (np.isfinite(alpha_plc_arr) & np.isfinite(lam_plc_arr)
               & np.isfinite(d_plc_arr))
    if mask_hm.any():
        sc = ax_hm.scatter(
            alpha_plc_arr[mask_hm], lam_plc_arr[mask_hm],
            c=d_plc_arr[mask_hm], cmap='viridis_r',
            s=25, alpha=0.85, zorder=2,
        )
        fig.colorbar(sc, ax=ax_hm, label='KS statistic $D$')
        if np.isfinite(res_plc.alpha_plc) and np.isfinite(res_plc.lam_plc):
            ax_hm.scatter(
                [res_plc.alpha_plc], [res_plc.lam_plc],
                marker='*', s=200, color='red', zorder=5,
                label=rf'$x_{{\min}}^{{\mathrm{{plc}}}}$ optimum',
            )
    ax_hm.set_xlabel(r'$\alpha_{\mathrm{plc}}$', fontsize=12)
    ax_hm.set_ylabel(r'$\lambda_{\mathrm{plc}}\ (\mathrm{s}^{-1})$', fontsize=12)
    ax_hm.legend(frameon=False, fontsize=9)
    ax_hm.tick_params(labelsize=9)
    ax_hm.set_title('PLC Parameter Space — KS Dispersion', fontsize=11)

    fig.suptitle(f'{prefix} — power-law analysis', fontsize=14, y=1.002)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    kprint(f"Saved: {out_path}")


def plot_sample_analysis(
    results: List[Tuple[int, FitResult]],
    out_path: str,
    prefix: str,
    xmin: float,
) -> None:
    """
    Single panel: p-value vs sample size for three models.
    Saved to ``out_path``.
    """
    if not results:
        kprint("No valid results to plot.")
        return

    ns = np.array([n for n, _ in results])
    p_mc = np.array([r.p_mc_pl for _, r in results])
    v_exp = np.array([r.vuong_p_pl_vs_exp for _, r in results])
    v_plc = np.array([r.vuong_p_pl_vs_plc for _, r in results])

    fig, ax = plt.subplots(figsize=(8, 5))

    def _plot_line(arr, color, label):
        mask = np.isfinite(arr)
        if mask.any():
            ax.plot(
                ns[mask],
                arr[mask],
                color=color,
                linewidth=2,
                marker='o',
                markersize=4,
                label=label,
            )

    _plot_line(p_mc, _COLORS['pl'], 'MC p-value — Power law')
    _plot_line(v_exp, _COLORS['exp'], 'Vuong p — PL vs Exp')
    _plot_line(v_plc, _COLORS['plc'], 'Vuong p — PL vs PLC')

    ax.axhline(0.1, color='red', linestyle='--', linewidth=1.2)
    ax.text(ns[0], 0.12, '$p = 0.1$', color='red', fontsize=10)

    ax.set_xscale('log')
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel('Sample size $n$', fontsize=13)
    ax.set_ylabel('p-value', fontsize=13)
    ax.set_title(
        f'{prefix} — p-value vs sample size\n'
        rf'($x_{{\min}} = {xmin:.2e}$ s)',
        fontsize=12,
    )
    ax.legend(frameon=False, fontsize=11)
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    kprint(f"Saved: {out_path}")


# ===================================================================== #
# CLI entry point
# ===================================================================== #

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Power-law analysis following Clauset et al. 2009"
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Data directory (must contain traps/<prefix>/seq_*.pickle)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="SIB",
        choices=["Distance-matrix", "SIB", "Hybrid"],
        help="Trap detection method prefix (default: SIB)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["xmin", "sample"],
        help="Analysis mode: 'xmin' sweeps x_min; 'sample' sweeps sample size",
    )
    parser.add_argument(
        "--n_synth",
        type=int,
        default=500,
        help="Monte Carlo synthetic datasets (default 500; 2500 for publication)",
    )
    parser.add_argument(
        "--n_xmin",
        type=int,
        default=30,
        help="Number of x_min grid points [mode=xmin only] (default 30)",
    )
    parser.add_argument(
        "--xmin",
        type=float,
        default=None,
        help="Fixed x_min (seconds) for mode=sample; auto-detected if omitted",
    )
    parser.add_argument(
        "--step", type=int, default=1, help="Trajectory file step (default 1)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default 42)"
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Load data
    x = load_trap_times(str(args.path), args.prefix, args.step)
    kprint(
        f"  min={x.min():.3e} s  max={x.max():.3e} s  "
        f"median={np.median(x):.3e} s"
    )

    figs_out = join(str(args.path), "figs")
    os.makedirs(figs_out, exist_ok=True)

    if args.mode == 'xmin':
        kprint(
            f"\n=== x_min sweep  (n_xmin={args.n_xmin}, n_synth={args.n_synth}) ==="
        )
        results = run_xmin_mode(x, args.n_xmin, args.n_synth, rng)

        if not results:
            kprint("Warning: no valid fit results obtained.")

        out_svg = join(figs_out, f"powerlaw_xmin_analysis_{args.prefix}.svg")
        plot_xmin_analysis(x, results, out_svg, args.prefix)

    else:  # sample
        # Determine x_min
        if args.xmin is not None:
            xmin_used = args.xmin
            kprint(f"\nUsing fixed x_min = {xmin_used:.4e} s")
        else:
            kprint("\nAuto-detecting optimal x_min via KS minimisation...")
            xmin_used, alpha_opt, D_opt = find_xmin_optimal(x)
            kprint(
                f"  x_min* = {xmin_used:.4e} s  alpha* = {alpha_opt:.3f}  D* = {D_opt:.4f}"
            )

        kprint(f"\n=== Sample size sweep  (n_synth={args.n_synth}) ===")
        sample_results = run_sample_mode(x, xmin_used, args.n_synth, rng)

        out_svg = join(figs_out, f"powerlaw_sample_analysis_{args.prefix}.svg")
        plot_sample_analysis(sample_results, out_svg, args.prefix, xmin_used)
