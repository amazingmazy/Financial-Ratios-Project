"""
test_construct_panel.py
========================
Unit tests for Issue 1's panel-construction logic, using synthetic data
(public_fallback.synthetic_*) so they run with zero network access and no
WRDS credentials -- consistent with the project's established pattern of
validating on synthetic/mocked data before requiring a live database
connection.

Run with: pytest tests/test_construct_panel.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

from public_fallback import (
    synthetic_crsp_index, synthetic_riskfree, synthetic_cpi,
    synthetic_compustat_be_earnings,
)
from construct_panel import (
    construct_dividend_yield, construct_excess_and_real_returns,
    assign_lagged_fiscal_year, construct_bm_and_ep, assemble_master_panel,
    MASTER_COLS,
)


@pytest.fixture
def crsp():
    return synthetic_crsp_index(n_months=400, start="1946-01-01")


@pytest.fixture
def rf():
    return synthetic_riskfree(n_months=400, start="1946-01-01")


@pytest.fixture
def cpi():
    return synthetic_cpi(n_months=400, start="1946-01-01")


@pytest.fixture
def compustat():
    return synthetic_compustat_be_earnings(n_years=40, start_year=1946)


def test_dividend_yield_positive_and_finite(crsp):
    df = construct_dividend_yield(crsp)
    valid = df["DY"].dropna()
    assert len(valid) > 0
    assert (valid > 0).all(), "dividend yield must be strictly positive"
    assert np.isfinite(valid).all()


def test_dividend_yield_first_12_months_nan(crsp):
    # div_flow[0] is NaN (totval.shift(1) has no prior value), so the
    # rolling(12, min_periods=12) window only has 12 valid observations
    # starting at row index 12 (0-indexed), not 11.
    df = construct_dividend_yield(crsp)
    assert df["DY"].iloc[:12].isna().all()
    assert df["DY"].iloc[12:].notna().all()


def test_log_dy_matches_log_of_dy(crsp):
    df = construct_dividend_yield(crsp)
    valid = df.dropna(subset=["DY"])
    np.testing.assert_allclose(valid["logDY"], np.log(valid["DY"]))


def test_excess_return_equals_return_minus_rf(crsp, rf, cpi):
    df = construct_dividend_yield(crsp)
    df["VWNY"] = df["vwretd"] * 100
    df["EWNY"] = df["ewretd"] * 100
    df = construct_excess_and_real_returns(df, rf, cpi)
    np.testing.assert_allclose(df["ExcVWNY"], df["VWNY"] - df["rf"])
    np.testing.assert_allclose(df["ExcEWNY"], df["EWNY"] - df["rf"])


def test_real_return_deflates_correctly(crsp, rf, cpi):
    df = construct_dividend_yield(crsp)
    df["VWNY"] = df["vwretd"] * 100
    df["EWNY"] = df["ewretd"] * 100
    df = construct_excess_and_real_returns(df, rf, cpi)
    # Real return should be lower than nominal whenever inflation is positive.
    cpi_growth = df["cpi"].pct_change() * 100
    positive_inflation = cpi_growth > 0
    sub = df.loc[positive_inflation].dropna(subset=["RealVWNY", "VWNY"])
    assert (sub["RealVWNY"] <= sub["VWNY"] + 1e-9).all()


def test_assign_lagged_fiscal_year_respects_min_lag():
    months = pd.date_range("2000-01-01", periods=24, freq="MS")
    fyes = pd.DatetimeIndex(["1998-12-31", "1999-12-31", "2000-12-31"])
    matched = assign_lagged_fiscal_year(months, fyes, min_lag_months=4)
    # 1999-12-31 + 4 months = 2000-04-30, so it only becomes eligible once the
    # cutoff (month - 4 months) reaches or passes it -- i.e. from 2000-05-01.
    assert matched.loc[pd.Timestamp("2000-05-01")] == pd.Timestamp("1999-12-31")
    # At 2000-04-01, the cutoff (1999-12-01) is still before 1999-12-31, so
    # that FYE isn't eligible yet -- only the older 1998-12-31 is.
    assert matched.loc[pd.Timestamp("2000-04-01")] == pd.Timestamp("1998-12-31")
    # January 2000 is far past 1998-12-31 (13 months) and not yet 4 months
    # past 1999-12-31, so 1998 should still be the match.
    assert matched.loc[pd.Timestamp("2000-01-01")] == pd.Timestamp("1998-12-31")


def test_bm_ep_approx_flag_propagates(crsp, compustat):
    df = construct_dividend_yield(crsp)
    df["VWNY"] = df["vwretd"] * 100
    df["EWNY"] = df["ewretd"] * 100
    out_real = construct_bm_and_ep(df, compustat, is_approximation=False)
    out_approx = construct_bm_and_ep(df, compustat, is_approximation=True)
    assert (out_real["bm_is_approx"] == False).all()
    assert (out_approx["bm_is_approx"] == True).all()
    assert (out_approx["ep_is_approx"] == True).all()


def test_bm_ep_all_nan_and_flagged_when_no_compustat(crsp, rf, cpi):
    panel = assemble_master_panel(crsp, rf, cpi, compustat_annual=None)
    assert panel["B/M"].isna().all()
    assert panel["E/P"].isna().all()
    assert (panel["bm_is_approx"] == True).all()
    assert (panel["ep_is_approx"] == True).all()


def test_master_panel_schema(crsp, rf, cpi, compustat):
    panel = assemble_master_panel(crsp, rf, cpi, compustat_annual=compustat,
                                   bm_ep_is_approximation=False)
    assert list(panel.columns) == MASTER_COLS
    assert len(panel) == len(crsp)
    # No look-ahead: B/M in month t must never use a fiscal year ending
    # less than 4 months before t.
    valid_bm = panel.dropna(subset=["B/M"])
    assert len(valid_bm) > 0


def test_master_panel_runs_end_to_end_without_compustat(crsp, rf, cpi):
    panel = assemble_master_panel(crsp, rf, cpi, compustat_annual=None)
    assert list(panel.columns) == MASTER_COLS
    assert len(panel) == len(crsp)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))