"""
dashboard_data.py
==================
Pure (non-Streamlit) data-handling functions for the interactive
rho-sensitivity dashboard. Kept separate from dashboard.py so the window
definitions and x/r construction can be unit-tested directly, without
needing to drive a Streamlit session to check them.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

# Same window definitions used in replicate_paper_tables.py's Tables 2-6,
# so the dashboard can reproduce any of the paper's reported regressions
# exactly, not just approximate them.
PRESET_WINDOWS = {
    "1946-2000 (Table 2)": ("1946-01-01", "2000-12-31"),
    "1946-1972 (Table 3a)": ("1946-01-01", "1972-12-31"),
    "1973-2000 (Table 3b)": ("1973-01-01", "2000-12-31"),
    "1946-1994 (Table 4a)": ("1946-01-01", "1994-12-31"),
    "1963-1994 (Table 5a/6a)": ("1963-06-01", "1994-12-31"),
    "1963-2000 (Table 5b/6b)": ("1963-06-01", "2000-12-31"),
}

PREDICTOR_LABELS = {
    "logDY": "Dividend yield (log)",
    "logB/M": "Book-to-market (log)",
    "logE/P": "Earnings-price ratio (log)",
}

# Raw (non-log) level column for each log predictor, for the dashboard
# shell's "current level" display (README Issue 4, bullet 1).
RAW_LEVEL_COL = {
    "logDY": "DY",
    "logB/M": "B/M",
    "logE/P": "E/P",
}

RETURN_LABELS = {
    "VWNY": "Value-weighted NYSE (nominal)",
    "EWNY": "Equal-weighted NYSE (nominal)",
    "ExcVWNY": "Value-weighted NYSE (excess)",
    "ExcEWNY": "Equal-weighted NYSE (excess)",
}


def load_panel(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def filter_window(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)


def build_x_r(df: pd.DataFrame, predictor_col: str, return_col: str):
    """
    Build (x, r) exactly matching estimators.run_all's convention: x is
    predictor levels x_0..x_T (length T+1), r is returns r_1..r_T (length T),
    aligning r_t with x_{t-1} via x_lag = x[:-1] in the estimator functions.

    Returns (x, r, sub) where `sub` is the row-aligned dataframe used to
    build them (useful for showing the user exactly which rows/dates were
    included, e.g. for a "data used" expander in the UI).
    """
    sub = df.dropna(subset=[predictor_col, return_col]).reset_index(drop=True)
    if len(sub) < 30:
        return None, None, sub
    x = sub[predictor_col].to_numpy()
    r = sub[return_col].to_numpy()[1:]
    return x, r, sub


def power_curve_grid(rho_min: float = 0.90, rho_max: float = 0.9999, n_points: int = 60) -> np.ndarray:
    """
    Grid of assumed-rho values for the power curve. Denser near 1 (where the
    conditional test's behavior changes fastest) than near rho_min, since
    the paper's whole point is that the test's power is highly sensitive to
    rho specifically in that region.
    """
    # Concentrate points near 1 using a log-spaced transform on (1 - rho).
    one_minus = np.geomspace(1 - rho_max, 1 - rho_min, n_points)
    grid = 1 - one_minus
    return np.sort(grid)


# --------------------------------------------------------------------------
# Bayesian ρ dial (README Issue 4's core feature)
# --------------------------------------------------------------------------
#
# The paper's own equivalence (Section 4.1): for a FIXED, known rho, the
# conditional test's one-sided p-value IS the Bayesian posterior probability
# that b<=0 (a standard result for a one-sided test with a flat prior on b).
# So:
#   - "point mass at rho=1"   -> just the conditional p-value at rho~1
#   - any other prior on rho  -> average that same p-value(rho) over the
#     prior/posterior density on rho, i.e.
#         Posterior(b<=0) = integral[ p(rho) * w(rho) drho ] / integral[ w(rho) drho ]
#     where w(rho) is the prior (or posterior) density over rho.
#
# This module supplies w(rho) for two of the paper's three named cases
# (flat prior on rho<=1, and a shifted-normal prior); the point-mass case
# doesn't need integration at all.

from scipy.stats import norm as _norm


def flat_prior_weights(rho_grid: np.ndarray, rho_hat: float, se_rho: float,
                        upper: float = 0.999999) -> np.ndarray:
    """
    "Flat prior on rho<=1" (paper's Section 4.1, second case): combined with
    a Gaussian likelihood for rho_hat, a flat prior on rho<=1 gives a
    posterior for rho that's Normal(rho_hat, se_rho) truncated to rho<=1.
    Returns unnormalized density weights at each rho_grid point (zero above
    `upper`); normalize via integrate_posterior below.
    """
    dens = _norm.pdf(rho_grid, loc=rho_hat, scale=se_rho)
    return np.where(rho_grid <= upper, dens, 0.0)


def shifted_normal_prior_weights(rho_grid: np.ndarray, rho_hat: float, se_rho: float,
                                  shift_in_se: float = 1.0, spread_multiplier: float = 1.0,
                                  upper: float = 0.999999) -> np.ndarray:
    """
    "Shifted-normal prior centered near rho_hat, user-adjustable spread"
    (paper's Section 4.1, third case): the paper's own example shifts the
    belief about rho upward by exactly one standard deviation (of rho_hat's
    own sampling distribution) and truncates at 1. Generalized here with
    `shift_in_se` (paper's example: 1.0) and `spread_multiplier` (paper's
    example: 1.0, i.e. the same spread as rho_hat's own SE) both adjustable,
    so the dashboard's dial can reproduce the paper's exact worked example
    as one setting among many, not the only option.
    """
    mean = rho_hat + shift_in_se * se_rho
    sd = max(spread_multiplier * se_rho, 1e-8)
    dens = _norm.pdf(rho_grid, loc=mean, scale=sd)
    return np.where(rho_grid <= upper, dens, 0.0)


def integrate_posterior(rho_grid: np.ndarray, p_values: np.ndarray, weights: np.ndarray) -> float:
    """
    Numerically integrate p(rho)*w(rho) / integral[w(rho)] over rho_grid via
    the trapezoidal rule (handles rho_grid's non-uniform spacing correctly,
    unlike a plain weighted average over grid points).
    """
    _trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy >=2.0 renamed trapz -> trapezoid
    denom = _trapz(weights, rho_grid)
    if denom <= 0:
        # Degenerate case (e.g. all weight above the truncation point) --
        # fall back to the point-mass-at-1 case rather than dividing by zero.
        return float(p_values[-1])
    numer = _trapz(p_values * weights, rho_grid)
    return float(numer / denom)


# --------------------------------------------------------------------------
# Power rule-of-thumb (README Issue 4, "does the conditional test have
# power right now?" indicator)
# --------------------------------------------------------------------------
#
# The paper states two calibration points directly (Section 2.4): with 25
# years of data, the conditional test needs monthly rho ~ 0.98 (annual
# ~ 0.85) to have power; with 50 years, ~0.99 (annual ~0.90). Linearly
# interpolated/extrapolated in years for anything in between or beyond --
# the paper doesn't give a closed-form rule, just these two calibration
# points, so linear interpolation is the simplest defensible choice, not a
# claimed exact formula.

_POWER_CALIBRATION_YEARS = np.array([25.0, 50.0])
_POWER_CALIBRATION_MONTHLY_RHO = np.array([0.98, 0.99])
_POWER_CALIBRATION_ANNUAL_RHO = np.array([0.85, 0.90])


def power_threshold(t_months: int, frequency: str = "monthly") -> float:
    """Interpolated/extrapolated rho threshold for the conditional test to plausibly have power."""
    years = t_months / 12.0
    table = _POWER_CALIBRATION_MONTHLY_RHO if frequency == "monthly" else _POWER_CALIBRATION_ANNUAL_RHO
    return float(np.interp(years, _POWER_CALIBRATION_YEARS, table))


def has_power(rho_hat: float, t_months: int) -> bool:
    return rho_hat >= power_threshold(t_months, "monthly")


# --------------------------------------------------------------------------
# Drop-last-N-years sensitivity (README Issue 4: generalizes Table 4's one
# fixed 1994-vs-2000 comparison to an adjustable N)
# --------------------------------------------------------------------------

def truncate_window_by_years(end: str, n_years_dropped: int) -> str:
    """Given a window's end date and a number of years to drop from the end,
    return the new (earlier) end date as an ISO string."""
    end_ts = pd.Timestamp(end)
    new_end = end_ts - pd.DateOffset(years=n_years_dropped)
    return new_end.strftime("%Y-%m-%d")
