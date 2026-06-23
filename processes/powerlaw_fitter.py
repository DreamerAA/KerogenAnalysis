"""
Power-law distribution fitting following Clauset, Shalizi & Newman (2009)
"Power-Law Distributions in Empirical Data", SIAM Review 51(4):661-703.

Equation numbers in comments reference that paper.
"""

import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy.integrate import cumulative_trapezoid, quad
from scipy.optimize import minimize
from scipy.special import erfc

# --------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------- #
MIN_TAIL_SIZE: int = 10      # minimum observations above x_min for valid fit
_CCDF_GRID_PTS: int = 1000   # grid points for PLC CCDF numerical integration


# --------------------------------------------------------------------- #
# Result dataclass
# --------------------------------------------------------------------- #
@dataclass
class FitResult:
    xmin: float
    n_tail: int

    # Power law
    alpha: float          # MLE exponent (eq 3.1)
    alpha_sigma: float    # standard error (eq 3.2)
    D_pl: float           # KS statistic (eq 3.9)
    p_mc_pl: float        # Monte Carlo p-value (sec 4.1); NaN if not computed

    # Exponential
    lam_exp: float        # MLE rate parameter
    D_exp: float          # KS statistic

    # Power law with exponential cutoff
    alpha_plc: float      # MLE exponent
    lam_plc: float        # MLE cutoff rate
    D_plc: float          # KS statistic

    # Standalone MC goodness-of-fit p-values
    p_mc_exp: float      # MC p-value for exponential; NaN if not computed
    p_mc_plc: float      # MC p-value for PLC (fast bootstrap); NaN if not computed

    # Vuong likelihood ratio tests (Appendix C, eq C.6)
    vuong_p_pl_vs_exp: float    # two-sided p-value
    vuong_sign_pl_vs_exp: int   # +1 if PL favoured, -1 if Exp favoured
    vuong_p_pl_vs_plc: float
    vuong_sign_pl_vs_plc: int


# --------------------------------------------------------------------- #
# MLE estimators
# --------------------------------------------------------------------- #

def fit_powerlaw(
    x: npt.NDArray[np.float64],
    xmin: float,
) -> Tuple[float, float, int]:
    """
    Continuous power-law MLE (Clauset eq. 3.1–3.2).

    Returns
    -------
    alpha_hat : float
        Estimated scaling exponent.
    sigma : float
        Standard error on alpha_hat.
    n_tail : int
        Number of observations used (x >= xmin).

    Raises
    ------
    ValueError
        If n_tail < MIN_TAIL_SIZE or all values equal xmin.
    """
    x_tail = x[x >= xmin]
    n = len(x_tail)
    if n < MIN_TAIL_SIZE:
        raise ValueError(f"n_tail={n} < MIN_TAIL_SIZE={MIN_TAIL_SIZE}")

    sum_log = np.sum(np.log(x_tail / xmin))
    if sum_log <= 0.0:
        raise ValueError("sum_log <= 0: all values equal xmin or xmin too large")

    alpha_hat = 1.0 + n / sum_log
    sigma = (alpha_hat - 1.0) / np.sqrt(n)
    return float(alpha_hat), float(sigma), n


def fit_exponential(
    x: npt.NDArray[np.float64],
    xmin: float,
) -> Tuple[float, int]:
    """
    Shifted exponential MLE: p(t) = lam * exp(-lam*(t - xmin)) for t >= xmin.

    Returns (lam_hat, n_tail).
    """
    x_tail = x[x >= xmin]
    n = len(x_tail)
    if n < MIN_TAIL_SIZE:
        raise ValueError(f"n_tail={n} < MIN_TAIL_SIZE={MIN_TAIL_SIZE}")

    excess_mean = np.mean(x_tail) - xmin
    if excess_mean <= 0.0:
        lam_hat = np.finfo(float).max
    else:
        lam_hat = 1.0 / excess_mean
    return float(lam_hat), n


def _plc_normalization(alpha: float, lam: float, xmin: float) -> float:
    """
    Normalisation C = integral_{xmin}^{inf} t^{-alpha} exp(-lam*t) dt.
    Uses dimensionless y = t/xmin for numerical stability:
      C = xmin^(1-alpha) * integral_1^inf y^(-alpha) exp(-beta*y) dy,
    where beta = lam * xmin.
    """
    if alpha <= 1.0 or lam < 0.0 or xmin <= 0.0:
        return 0.0

    beta = lam * xmin

    def integrand(y: float) -> float:
        with np.errstate(over='ignore', under='ignore'):
            val = y ** (-alpha) * np.exp(-beta * y)
            return float(val) if np.isfinite(val) else 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        I, _ = quad(integrand, 1.0, np.inf, epsabs=1e-12, epsrel=1e-8,
                    limit=200)

    if not np.isfinite(I) or I <= 0.0:
        return 0.0

    C = xmin ** (1.0 - alpha) * I
    return float(C) if np.isfinite(C) and C > 0.0 else 0.0


def fit_plc(
    x: npt.NDArray[np.float64],
    xmin: float,
) -> Tuple[float, float, float, int]:
    """
    Power law with exponential cutoff MLE via numerical optimisation.

    Returns (alpha_plc, lam_plc, nll_min, n_tail).
    Falls back to L-BFGS-B if Nelder-Mead does not converge.
    """
    x_tail = x[x >= xmin]
    n = len(x_tail)
    if n < MIN_TAIL_SIZE:
        raise ValueError(f"n_tail={n} < MIN_TAIL_SIZE={MIN_TAIL_SIZE}")

    log_x_tail = np.log(x_tail)

    def nll(params: npt.NDArray) -> float:
        alpha, lam = params
        if alpha <= 1.0 + 1e-8 or lam < 0.0:
            return 1e15
        C = _plc_normalization(alpha, lam, xmin)
        if C <= 0.0:
            return 1e15
        log_C = np.log(C)
        val = n * log_C + alpha * np.sum(log_x_tail) + lam * np.sum(x_tail)
        return float(val) if np.isfinite(val) else 1e15

    # Initial guess: alpha close to PL estimate; lam from exponential excess mean
    try:
        alpha0, _, _ = fit_powerlaw(x, xmin)
    except ValueError:
        alpha0 = 2.0
    excess_mean = np.mean(x_tail) - xmin
    lam0 = 1.0 / excess_mean if excess_mean > 0.0 else 1.0 / xmin

    x0 = np.array([alpha0, lam0])
    res = minimize(nll, x0, method='Nelder-Mead',
                   options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 20000})

    if (not res.success) or (not np.isfinite(res.fun)) or (res.fun >= 1e14):
        # fallback: L-BFGS-B with explicit bounds
        res = minimize(nll, x0, method='L-BFGS-B',
                       bounds=[(1.001, 20.0), (0.0, 1e15)],
                       options={'ftol': 1e-12, 'gtol': 1e-8, 'maxiter': 5000})

    alpha_opt, lam_opt = float(res.x[0]), float(res.x[1])
    return alpha_opt, lam_opt, float(res.fun), n


# --------------------------------------------------------------------- #
# CCDF functions
# --------------------------------------------------------------------- #

def pl_ccdf(
    x: npt.NDArray[np.float64],
    xmin: float,
    alpha: float,
) -> npt.NDArray[np.float64]:
    """Theoretical CCDF for power law: P(X>=x) = (x/xmin)^{-(alpha-1)}."""
    return (x / xmin) ** (-(alpha - 1.0))


def exp_ccdf(
    x: npt.NDArray[np.float64],
    xmin: float,
    lam: float,
) -> npt.NDArray[np.float64]:
    """Theoretical CCDF for shifted exponential: P(X>=x) = exp(-lam*(x-xmin))."""
    return np.exp(-lam * (x - xmin))


def plc_ccdf_grid(
    x: npt.NDArray[np.float64],
    xmin: float,
    alpha: float,
    lam: float,
) -> npt.NDArray[np.float64]:
    """
    Theoretical CCDF for power law with cutoff via numerical integration on a
    log-spaced grid. Interpolated at query points x.
    """
    x_max = max(np.max(x) * 100.0, xmin * 1e6)
    grid = np.logspace(np.log10(xmin * (1.0 - 1e-10)),
                       np.log10(x_max),
                       _CCDF_GRID_PTS)

    with np.errstate(over='ignore', under='ignore'):
        exp_vals = np.exp(-lam * grid)
        exp_vals = np.where(np.isfinite(exp_vals), exp_vals, 0.0)
        pdf_unnorm = grid ** (-alpha) * exp_vals

    cdf_unnorm = cumulative_trapezoid(pdf_unnorm, grid, initial=0.0)
    total = cdf_unnorm[-1]
    if total <= 0.0:
        return np.zeros_like(x)
    cdf = cdf_unnorm / total
    ccdf = np.clip(1.0 - cdf, 0.0, 1.0)

    return np.clip(np.interp(x, grid, ccdf), 0.0, 1.0)


# --------------------------------------------------------------------- #
# KS statistic (Clauset eq. 3.9)
# --------------------------------------------------------------------- #

def ks_statistic(
    x: npt.NDArray[np.float64],
    xmin: float,
    ccdf_fn: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
) -> float:
    """
    Two-sided KS statistic between empirical CCDF and theoretical CCDF
    on the data above xmin.

    Uses both the right-continuous and left-continuous empirical CCDF
    to avoid tie-sensitivity bias.
    """
    x_tail = np.sort(x[x >= xmin])
    n = len(x_tail)
    if n < 2:
        return 1.0

    i = np.arange(n, dtype=float)
    s_right = (n - i - 1) / n   # right-continuous empirical CCDF
    s_left  = (n - i) / n       # left-continuous empirical CCDF

    p = ccdf_fn(x_tail)
    p = np.clip(p, 0.0, 1.0)

    d = max(np.max(np.abs(s_right - p)),
            np.max(np.abs(s_left  - p)))
    return float(d)


# --------------------------------------------------------------------- #
# Optimal x_min (Clauset sec. 3.3)
# --------------------------------------------------------------------- #

def find_xmin_optimal(
    x: npt.NDArray[np.float64],
) -> Tuple[float, float, float]:
    """
    Find x_min that minimises the KS distance between empirical data and
    best-fit power law (Clauset sec. 3.3).

    Returns
    -------
    xmin_opt  : float
    alpha_opt : float
    D_opt     : float
    """
    candidates = np.unique(x)
    # keep enough tail: at least MIN_TAIL_SIZE observations above xmin
    candidates = candidates[:-MIN_TAIL_SIZE] if len(candidates) > MIN_TAIL_SIZE else candidates[:1]

    best_D = np.inf
    best_xmin = candidates[0]
    best_alpha = 2.0

    for xm in candidates:
        try:
            alpha, _, _ = fit_powerlaw(x, xm)
        except ValueError:
            continue
        if alpha <= 1.0:
            continue
        D = ks_statistic(x, xm, lambda t: pl_ccdf(t, xm, alpha))
        if D < best_D:
            best_D = D
            best_xmin = xm
            best_alpha = alpha

    return float(best_xmin), float(best_alpha), float(best_D)


# --------------------------------------------------------------------- #
# Monte Carlo p-value (Clauset sec. 4.1)
# --------------------------------------------------------------------- #

def mc_pvalue_pl(
    x: npt.NDArray[np.float64],
    xmin: float,
    alpha_hat: float,
    D_emp: float,
    n_synth: int,
    rng: np.random.Generator,
) -> float:
    """
    Estimate goodness-of-fit p-value via Monte Carlo (Clauset sec. 4.1).

    For each synthetic dataset:
      - Tail drawn from fitted power law via inverse-CDF (eq D.4).
      - Below-xmin part resampled from empirical data with replacement.
      - Power law refit at same xmin; KS statistic computed.

    Returns p = fraction(KS_synth >= D_emp).
    """
    x_tail_emp = x[x >= xmin]
    x_below    = x[x < xmin]
    n_tail = len(x_tail_emp)
    n_below = len(x_below)
    n_total = len(x)

    if n_tail < MIN_TAIL_SIZE or alpha_hat <= 1.0:
        return float('nan')

    exp_inv = -1.0 / (alpha_hat - 1.0)

    count_gte = 0
    valid = 0

    for _ in range(n_synth):
        # --- generate synthetic tail (inverse-CDF, eq D.4) ---
        u = rng.uniform(0.0, 1.0, size=n_tail)
        synth_tail = xmin * (1.0 - u) ** exp_inv
        # guard against overflow
        synth_tail = np.clip(synth_tail, xmin, xmin * 1e15)

        # --- generate below-xmin part by resampling ---
        if n_below > 0:
            synth_below = rng.choice(x_below, size=n_below, replace=True)
            synth = np.concatenate([synth_below, synth_tail])
        else:
            synth = synth_tail

        # --- refit PL at same xmin ---
        try:
            alpha_s, _, _ = fit_powerlaw(synth, xmin)
        except ValueError:
            continue

        if alpha_s <= 1.0:
            continue

        D_s = ks_statistic(synth, xmin, lambda t: pl_ccdf(t, xmin, alpha_s))
        if D_s >= D_emp:
            count_gte += 1
        valid += 1

    if valid == 0:
        return float('nan')
    return count_gte / valid


# --------------------------------------------------------------------- #
# Per-observation log-likelihoods (for Vuong test)
# --------------------------------------------------------------------- #

def pl_loglik(
    x_tail: npt.NDArray[np.float64],
    xmin: float,
    alpha: float,
) -> npt.NDArray[np.float64]:
    """Log p(x_i) under power law (eq 2.2)."""
    return (np.log(alpha - 1.0) - np.log(xmin)
            - alpha * np.log(x_tail / xmin))


def exp_loglik(
    x_tail: npt.NDArray[np.float64],
    xmin: float,
    lam: float,
) -> npt.NDArray[np.float64]:
    """Log p(x_i) under shifted exponential."""
    return np.log(lam) - lam * (x_tail - xmin)


def plc_loglik(
    x_tail: npt.NDArray[np.float64],
    xmin: float,
    alpha: float,
    lam: float,
    C: float,
) -> npt.NDArray[np.float64]:
    """Log p(x_i) under power law with exponential cutoff."""
    if C <= 0.0:
        return np.full(len(x_tail), -np.inf)
    return -alpha * np.log(x_tail) - lam * x_tail - np.log(C)


# --------------------------------------------------------------------- #
# Vuong likelihood ratio test (Clauset Appendix C, eq C.6)
# --------------------------------------------------------------------- #

def vuong_test(
    ll1: npt.NDArray[np.float64],
    ll2: npt.NDArray[np.float64],
) -> Tuple[float, int]:
    """
    Vuong (1989) likelihood ratio test for non-nested models.

    Parameters
    ----------
    ll1, ll2 : per-observation log-likelihoods for models 1 and 2.

    Returns
    -------
    p_value : float
        Two-sided p-value; small p => results are significant.
        Positive R (sum(ll1-ll2) > 0) means model 1 is favoured.
    sign : int
        +1 if model 1 (ll1) is favoured, -1 if model 2.
    """
    lr = ll1 - ll2
    R = np.sum(lr)
    n = len(lr)
    var_lr = np.mean((lr - np.mean(lr)) ** 2)

    if var_lr < 1e-300:
        return 1.0, 0

    p_val = float(erfc(abs(R) / np.sqrt(2.0 * n * var_lr)))
    sign = int(np.sign(R)) if R != 0.0 else 0
    return p_val, sign


# --------------------------------------------------------------------- #
# MC p-values for Exponential and PLC (standalone goodness-of-fit)
# --------------------------------------------------------------------- #

def mc_pvalue_exp(
    x: npt.NDArray[np.float64],
    xmin: float,
    lam_hat: float,
    D_emp: float,
    n_synth: int,
    rng: np.random.Generator,
) -> float:
    """
    Monte Carlo goodness-of-fit p-value for the shifted exponential.

    Mirrors Clauset sec. 4.1 but for Exp: synthetic tails are drawn from
    Exp(lam_hat), refitted, and KS compared to D_emp.
    """
    x_tail_emp = x[x >= xmin]
    x_below = x[x < xmin]
    n_tail = len(x_tail_emp)
    n_below = len(x_below)

    if n_tail < MIN_TAIL_SIZE or not (lam_hat > 0.0):
        return float('nan')

    count_gte = 0
    valid = 0

    for _ in range(n_synth):
        synth_tail = xmin + rng.exponential(1.0 / lam_hat, size=n_tail)
        if n_below > 0:
            synth_below = rng.choice(x_below, size=n_below, replace=True)
            synth = np.concatenate([synth_below, synth_tail])
        else:
            synth = synth_tail

        try:
            lam_s, _ = fit_exponential(synth, xmin)
        except ValueError:
            continue
        if lam_s <= 0.0:
            continue

        D_s = ks_statistic(synth, xmin, lambda t, ls=lam_s: exp_ccdf(t, xmin, ls))
        if D_s >= D_emp:
            count_gte += 1
        valid += 1

    return count_gte / valid if valid > 0 else float('nan')


def mc_pvalue_plc(
    x: npt.NDArray[np.float64],
    xmin: float,
    alpha_hat: float,
    lam_hat: float,
    D_emp: float,
    n_synth: int,
    rng: np.random.Generator,
) -> float:
    """
    Monte Carlo goodness-of-fit p-value for PLC.

    Uses parametric bootstrap without refitting (fast approximation):
    synthetic tails are drawn via numerical inverse CDF and KS is compared
    against the precomputed theoretical CCDF.  Anti-conservative by ~5-10%
    relative to full Clauset procedure, but avoids expensive PLC refitting.
    """
    x_tail_emp = x[x >= xmin]
    x_below = x[x < xmin]
    n_tail = len(x_tail_emp)
    n_below = len(x_below)

    if n_tail < MIN_TAIL_SIZE:
        return float('nan')

    # Precompute CDF / CCDF grid once
    x_max = xmin * 1e6
    if lam_hat > 0.0:
        x_max = min(x_max, max(xmin * 100.0, 30.0 / lam_hat))
    grid = np.logspace(np.log10(xmin), np.log10(x_max), _CCDF_GRID_PTS)

    with np.errstate(over='ignore', under='ignore'):
        pdf_g = grid ** (-alpha_hat) * np.exp(-lam_hat * grid)
    pdf_g = np.where(np.isfinite(pdf_g), pdf_g, 0.0)

    cdf_g = cumulative_trapezoid(pdf_g, grid, initial=0.0)
    total_g = cdf_g[-1]
    if total_g <= 0.0:
        return float('nan')
    cdf_g /= total_g
    ccdf_g = np.clip(1.0 - cdf_g, 0.0, 1.0)

    def _plc_ccdf(t: npt.NDArray) -> npt.NDArray:
        return np.clip(np.interp(t, grid, ccdf_g), 0.0, 1.0)

    count_gte = 0
    for _ in range(n_synth):
        u = rng.uniform(0.0, 1.0 - 1e-10, size=n_tail)
        synth_tail = np.interp(u, cdf_g, grid)

        if n_below > 0:
            synth_below = rng.choice(x_below, size=n_below, replace=True)
            synth = np.concatenate([synth_below, synth_tail])
        else:
            synth = synth_tail

        D_s = ks_statistic(synth, xmin, _plc_ccdf)
        if D_s >= D_emp:
            count_gte += 1

    return count_gte / n_synth
