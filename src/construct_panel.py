"""
construct_panel.py
===================
Issue 1 core logic -- turns the raw pulls (from wrds_pull.py or
public_fallback.py) into the one clean master monthly panel everything
downstream reads from.

Master panel schema (per README Issue 1):
    date, VWNY, EWNY, ExcVWNY, ExcEWNY, RealVWNY, RealEWNY,
    DY, logDY, B/M, logB/M, E/P, logE/P, bm_is_approx, ep_is_approx

Percent/log convention, matching the paper's Table 1 exactly (see its note:
"Observations are monthly and the variables are expressed in percent;
log(x) equals the natural log of x expressed in percent"):
    - VWNY, EWNY, ExcVWNY, ExcEWNY, RealVWNY, RealEWNY, DY, B/M, E/P are all
      in percent (e.g. a 3.8% dividend yield is stored as 3.8, not 0.038)
    - logDY = ln(DY), logB/M = ln(B/M), logE/P = ln(E/P) -- i.e. the log of
      the percent-valued number itself, not of a decimal fraction
"""

from __future__ import annotations
import numpy as np
import pandas as pd

MASTER_COLS = [
    "date", "VWNY", "EWNY", "ExcVWNY", "ExcEWNY", "RealVWNY", "RealEWNY",
    "DY", "logDY", "B/M", "logB/M", "E/P", "logE/P",
    "bm_is_approx", "ep_is_approx",
]


def _to_percent(series: pd.Series) -> pd.Series:
    """
    CRSP/Compustat pulls are typically decimal fractions (0.038); some public
    or synthetic sources may already be in percent (3.8). Detect which and
    normalize to percent, so everything downstream has one fixed convention.
    """
    return series * 100 if series.abs().max() < 1 else series


# --------------------------------------------------------------------------
# Step 1: dividend flow and dividend yield
# --------------------------------------------------------------------------

def construct_dividend_yield(crsp_index: pd.DataFrame) -> pd.DataFrame:
    """
    From monthly vwretd/vwretx/totval (returns as decimal fractions, as CRSP
    stores them), construct:
        div_flow_t   = (vwretd_t - vwretx_t) * totval_{t-1}   (paper, Section 3)
        div_trail12  = trailing 12-month sum of div_flow
        DY_t (%)     = 100 * div_trail12_t / totval_t
        logDY_t      = ln(DY_t)

    `crsp_index` must have columns vwretd, vwretx, totval (returns as decimal
    fractions), indexed by month. Returns the same frame with div_flow,
    div_trail12, DY, logDY appended (DY/logDY per the percent/log convention
    described in the module docstring).
    """
    df = crsp_index.copy().sort_index()
    totval_lag = df["totval"].shift(1)
    df["div_flow"] = (df["vwretd"] - df["vwretx"]) * totval_lag
    df["div_trail12"] = df["div_flow"].rolling(12, min_periods=12).sum()
    df["DY"] = 100 * df["div_trail12"] / df["totval"]
    df["logDY"] = np.log(df["DY"])
    return df


# --------------------------------------------------------------------------
# Step 2: excess and real returns
# --------------------------------------------------------------------------

def construct_excess_and_real_returns(panel: pd.DataFrame, rf: pd.DataFrame,
                                       cpi: pd.DataFrame) -> pd.DataFrame:
    """
    Adds ExcVWNY, ExcEWNY (net of the 1-month T-bill) and RealVWNY, RealEWNY
    (CPI-deflated), all in percent. `panel` must already have VWNY/EWNY in
    percent; `rf` must have column `rf` in percent; `cpi` must have column
    `cpi` (an index level, any base -- only its growth rate is used).
    """
    df = panel.join(rf, how="left").join(cpi, how="left")
    df["ExcVWNY"] = df["VWNY"] - df["rf"]
    df["ExcEWNY"] = df["EWNY"] - df["rf"]

    cpi_growth_pct = df["cpi"].pct_change() * 100
    df["RealVWNY"] = ((1 + df["VWNY"] / 100) / (1 + cpi_growth_pct / 100) - 1) * 100
    df["RealEWNY"] = ((1 + df["EWNY"] / 100) / (1 + cpi_growth_pct / 100) - 1) * 100
    return df


# --------------------------------------------------------------------------
# Step 3: B/M and E/P, with the required >=4-month reporting lag
# --------------------------------------------------------------------------

def assign_lagged_fiscal_year(month_dates: pd.DatetimeIndex,
                               fye_dates: pd.DatetimeIndex,
                               min_lag_months: int = 4) -> pd.Series:
    """
    For each month in `month_dates`, find the most recent fiscal-year-end in
    `fye_dates` that is at least `min_lag_months` old -- i.e. the fiscal year
    whose accounting data would actually have been public knowledge by that
    month (the paper's "do not update accounting numbers until four months
    after the fiscal year" rule). Returns a Series indexed by month_dates
    giving the matched FYE (NaT where no FYE is old enough yet).
    """
    fye_sorted = pd.Series(sorted(fye_dates))
    out = pd.Series(index=month_dates, dtype="datetime64[ns]")
    for m in month_dates:
        cutoff = m - pd.DateOffset(months=min_lag_months)
        eligible = fye_sorted[fye_sorted <= cutoff]
        out[m] = eligible.iloc[-1] if len(eligible) else pd.NaT
    return out


def construct_bm_and_ep(panel: pd.DataFrame, compustat_annual: pd.DataFrame,
                         is_approximation: bool = False) -> pd.DataFrame:
    """
    Adds B/M, logB/M, E/P, logE/P to `panel` (in percent / log-of-percent, per
    the module convention), using `compustat_annual` (indexed by fiscal-
    year-end date, columns book_equity_sum, oibdp_sum -- see
    wrds_pull.pull_compustat_be_and_earnings) merged in with a >=4-month
    reporting lag, and dividing by `panel["totval"]` (aggregate NYSE market
    equity, contemporaneous, from the CRSP index pull).

    If `is_approximation` is True (i.e. compustat_annual actually came from
    a non-Compustat fallback such as the French 25-Size-BM-portfolio proxy,
    not real Compustat book equity/earnings), the bm_is_approx / ep_is_approx
    flag columns are set accordingly. This flag must never be silently
    dropped downstream -- Issue 1 explicitly requires flagging any
    approximation as such, not presenting it as a true aggregate NYSE ratio.

    Also attaches diagnostic-only columns (matched_fye, book_equity_sum_used,
    oibdp_sum_used) so an outlier B/M or E/P month can be traced back to
    exactly which fiscal year's data produced it, without re-deriving
    anything -- see diagnose_bm.py.
    """
    df = panel.copy()
    matched_fye = assign_lagged_fiscal_year(df.index, compustat_annual.index)
    be = matched_fye.map(compustat_annual["book_equity_sum"])
    earn = matched_fye.map(compustat_annual["oibdp_sum"])
    be.index = df.index
    earn.index = df.index

    df["B/M"] = 100 * be / df["totval"]
    df["logB/M"] = np.log(df["B/M"])
    df["E/P"] = 100 * earn / df["totval"]
    df["logE/P"] = np.log(df["E/P"])
    df["bm_is_approx"] = is_approximation
    df["ep_is_approx"] = is_approximation

    # Diagnostic-only, not part of MASTER_COLS -- dropped by assemble_master_panel
    # unless explicitly requested via keep_diagnostics=True.
    df["matched_fye"] = matched_fye.values
    df["book_equity_sum_used"] = be.values
    df["oibdp_sum_used"] = earn.values
    return df


# --------------------------------------------------------------------------
# Step 4: assemble the master panel
# --------------------------------------------------------------------------

def assemble_master_panel(crsp_index: pd.DataFrame, rf: pd.DataFrame, cpi: pd.DataFrame,
                           compustat_annual: pd.DataFrame | None = None,
                           bm_ep_is_approximation: bool = False,
                           keep_diagnostics: bool = False) -> pd.DataFrame:
    """
    Full Issue 1 pipeline: dividend yield -> excess/real returns -> B/M & E/P
    (if compustat_annual is supplied) -> final column selection matching
    MASTER_COLS.

    Parameters
    ----------
    crsp_index : DataFrame with vwretd, vwretx, ewretd, ewretx, totval
                 (returns as decimal fractions), indexed by month.
                 See wrds_pull.pull_crsp_nyse_index / public_fallback.synthetic_crsp_index.
    rf : DataFrame with column `rf` in percent, indexed by month.
    cpi : DataFrame with column `cpi` (index level), indexed by month.
    compustat_annual : DataFrame with book_equity_sum, oibdp_sum, indexed by
                 fiscal-year-end date, or None if no Compustat/approximation
                 data is available at all (B/M, E/P columns will be all-NaN).
    bm_ep_is_approximation : set True if `compustat_annual` was itself built
                 from a non-Compustat approximation (e.g. French 25 Size-BM
                 portfolios) rather than real Compustat data.
    keep_diagnostics : if True, also return totval, matched_fye,
                 book_equity_sum_used, oibdp_sum_used in the output (not part
                 of the official MASTER_COLS schema) -- use this to trace an
                 outlier B/M/E/P month back to its source data. See
                 diagnose_bm.py for a ready-made analysis of these columns.
    """
    df = construct_dividend_yield(crsp_index)
    df["VWNY"] = _to_percent(df["vwretd"])
    df["EWNY"] = _to_percent(df["ewretd"])

    df = construct_excess_and_real_returns(df, rf, cpi)

    if compustat_annual is not None:
        df = construct_bm_and_ep(df, compustat_annual, is_approximation=bm_ep_is_approximation)
    else:
        df["B/M"] = np.nan
        df["logB/M"] = np.nan
        df["E/P"] = np.nan
        df["logE/P"] = np.nan
        df["bm_is_approx"] = True
        df["ep_is_approx"] = True
        df["matched_fye"] = pd.NaT
        df["book_equity_sum_used"] = np.nan
        df["oibdp_sum_used"] = np.nan

    df = df.reset_index().rename(columns={"index": "date"})
    if keep_diagnostics:
        diag_cols = ["totval", "matched_fye", "book_equity_sum_used", "oibdp_sum_used"]
        return df[MASTER_COLS + [c for c in diag_cols if c in df.columns]]
    return df[MASTER_COLS]
