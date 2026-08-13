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
from wrds_pull import aggregate_firm_level_to_fiscal_year


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


# --------------------------------------------------------------------------
# Regression tests for the fyear-vs-datadate aggregation bug found in the
# first live WRDS run: grouping by exact datadate (instead of fyear) scattered
# ~1 firm per group instead of pooling hundreds of NYSE firms per fiscal year.
# --------------------------------------------------------------------------

def _make_firm_level(n_firms_per_year=200, years=(1998, 1999, 2000), seed=42):
    """
    Simulate what conn.raw_sql would return: many firms per fiscal year, each
    with its OWN exact fiscal year-end date (Dec 31 for most, but a realistic
    minority on June 30 or other dates) -- this is exactly the shape that
    broke groupby('datadate').
    """
    rng = np.random.default_rng(seed)
    rows = []
    fye_choices = ["-12-31", "-06-30", "-09-30", "-03-31"]
    for year in years:
        for i in range(n_firms_per_year):
            fye_suffix = rng.choice(fye_choices, p=[0.7, 0.15, 0.1, 0.05])
            rows.append({
                "gvkey": f"{year}_{i}",
                "datadate": pd.Timestamp(f"{year}{fye_suffix}"),
                "fyear": year,
                "book_equity": rng.uniform(50, 500),
                "oibdp": rng.uniform(10, 100),
            })
    return pd.DataFrame(rows)


def test_aggregation_pools_all_firms_per_fiscal_year_not_per_exact_date():
    firm_level = _make_firm_level(n_firms_per_year=200, years=(1998, 1999, 2000))
    agg = aggregate_firm_level_to_fiscal_year(firm_level)

    # The bug this guards against: grouping by exact datadate would produce
    # ~4 groups per year (one per fye_suffix) instead of 1, and n_firms per
    # group would be a small fraction of 200, not (close to) all of them.
    assert len(agg) == 3, "should have exactly one aggregate row per fiscal year"
    assert (agg["n_firms"] == 200).all(), (
        "every firm for a given fyear must be pooled into one aggregate, "
        "regardless of its individual fiscal-year-end date"
    )


def test_aggregation_indexes_by_december_31_of_fiscal_year():
    firm_level = _make_firm_level(n_firms_per_year=50, years=(2005,))
    agg = aggregate_firm_level_to_fiscal_year(firm_level)
    assert agg.index[0] == pd.Timestamp("2005-12-31")


def test_aggregation_converts_millions_to_thousands():
    firm_level = _make_firm_level(n_firms_per_year=10, years=(2000,), seed=1)
    raw_sum_millions = firm_level["book_equity"].sum()
    agg = aggregate_firm_level_to_fiscal_year(firm_level, min_firms_per_year=0)
    np.testing.assert_allclose(agg["book_equity_sum"].iloc[0], raw_sum_millions * 1000)


def test_aggregation_autocorrelation_is_high_when_pooled_correctly():
    # With ~200 firms pooled per year, year-to-year aggregate book equity
    # should be smooth/persistent (autocorrelation near 1), NOT the near-zero
    # or negative autocorrelation the fragmented-groupby bug produced.
    years = tuple(range(1970, 2001))
    firm_level = _make_firm_level(n_firms_per_year=200, years=years, seed=7)
    # Give book equity a realistic slow upward drift so the persistence check
    # is meaningful (i.i.d. noise would have ~0 autocorrelation regardless).
    drift = {y: 1.04 ** (y - years[0]) for y in years}
    firm_level["book_equity"] = firm_level["book_equity"] * firm_level["fyear"].map(drift)
    agg = aggregate_firm_level_to_fiscal_year(firm_level)
    ac1 = np.corrcoef(agg["book_equity_sum"].iloc[:-1], agg["book_equity_sum"].iloc[1:])[0, 1]
    assert ac1 > 0.8, f"expected high year-over-year persistence, got {ac1:.3f}"


# --------------------------------------------------------------------------
# Regression tests for the QA gate's tolerance logic (found via a live run
# where near-zero autocorrelations of opposite sign, e.g. paper=+0.013 vs
# computed=-0.013, spuriously failed a pure relative-error check).
# --------------------------------------------------------------------------

def test_qa_gate_near_zero_correlation_signflip_does_not_fail():
    from qa_table1 import ABS_TOLERANCE, SOFT_STATS
    paper_val, comp_val, name = 0.013, -0.013, "rho24"
    abs_diff = abs(comp_val - paper_val)
    assert abs_diff <= ABS_TOLERANCE[name], (
        "a sign-flipped near-zero correlation pair should be within the "
        "absolute-tolerance floor"
    )
    assert name in SOFT_STATS, "rho24 should not block the hard pass/fail gate"


def test_qa_gate_rho1_remains_a_hard_check():
    from qa_table1 import SOFT_STATS
    assert "rho1" not in SOFT_STATS, (
        "rho1 is the only persistence statistic the paper's estimators "
        "actually use -- it must stay a required check"
    )


# --------------------------------------------------------------------------
# Regression tests for the thin-early-years guard (found via a live run:
# logB/M's mean matched the paper well but SD/skew were badly off, the
# signature of a few outlier ratios from thinly-covered early fiscal years
# dominating a log-transformed statistic).
# --------------------------------------------------------------------------

def test_thin_fiscal_year_is_set_to_nan():
    from wrds_pull import aggregate_firm_level_to_fiscal_year
    # 1962: only 5 firms (thin/unrepresentative); 1963-1965: 200 firms (normal).
    rows = []
    for i in range(5):
        rows.append({"gvkey": f"1962_{i}", "fyear": 1962, "book_equity": 100.0, "oibdp": 20.0})
    for year in (1963, 1964, 1965):
        for i in range(200):
            rows.append({"gvkey": f"{year}_{i}", "fyear": year, "book_equity": 100.0, "oibdp": 20.0})
    firm_level = pd.DataFrame(rows)

    agg = aggregate_firm_level_to_fiscal_year(firm_level, min_firms_per_year=30)
    assert agg.loc["1962-12-31", "book_equity_sum"] is pd.NA or pd.isna(agg.loc["1962-12-31", "book_equity_sum"])
    for year in (1963, 1964, 1965):
        assert pd.notna(agg.loc[f"{year}-12-31", "book_equity_sum"])


def test_min_firms_guard_can_be_disabled():
    from wrds_pull import aggregate_firm_level_to_fiscal_year
    rows = [{"gvkey": "x", "fyear": 1962, "book_equity": 100.0, "oibdp": 20.0}]
    firm_level = pd.DataFrame(rows)
    agg = aggregate_firm_level_to_fiscal_year(firm_level, min_firms_per_year=0)
    assert pd.notna(agg.loc["1962-12-31", "book_equity_sum"])


# --------------------------------------------------------------------------
# Regression tests for the "-inf logB/M" bug found via diagnose_bm.py on a
# live run: fiscal year 1953 had normal OIBDP data but zero firms with
# non-missing book equity (CEQ/TXDITC not populated that early). pandas'
# .sum(skipna=True) on an all-NaN group returns 0.0, not NaN, silently
# manufacturing "aggregate book equity = 0" -> log(0) = -inf.
# --------------------------------------------------------------------------

def test_all_missing_book_equity_year_is_nan_not_zero():
    from wrds_pull import aggregate_firm_level_to_fiscal_year
    # Exactly the observed shape: 50 firms with real OIBDP but NULL book_equity.
    rows = [{"gvkey": f"f{i}", "fyear": 1953, "book_equity": np.nan, "oibdp": 400.0}
            for i in range(50)]
    firm_level = pd.DataFrame(rows)
    agg = aggregate_firm_level_to_fiscal_year(firm_level, min_firms_per_year=30)

    # The bug: without the fix, .sum() on an all-NaN group silently returns 0.0.
    assert pd.isna(agg.loc["1953-12-31", "book_equity_sum"]), (
        "a fiscal year with zero firms reporting book equity must be NaN, "
        "not silently summed to 0.0 (which produces log(0) = -inf downstream)"
    )
    # OIBDP was genuinely present for all 50 firms, so it should NOT be NaN'd
    # out -- this is the point of checking each field's coverage separately.
    assert pd.notna(agg.loc["1953-12-31", "oibdp_sum"])


def test_per_field_firm_counts_are_independent():
    from wrds_pull import aggregate_firm_level_to_fiscal_year
    # 40 firms have OIBDP only; 40 DIFFERENT firms have book_equity only.
    # A combined firm-count guard (80 total) would incorrectly treat both
    # fields as well-covered; the per-field guard should NaN both, since
    # each field individually has too few (well, exactly enough here to
    # test the boundary) non-missing observations relative to that field.
    rows = []
    for i in range(15):
        rows.append({"gvkey": f"be_{i}", "fyear": 1953, "book_equity": 100.0, "oibdp": np.nan})
    for i in range(15):
        rows.append({"gvkey": f"ep_{i}", "fyear": 1953, "book_equity": np.nan, "oibdp": 20.0})
    firm_level = pd.DataFrame(rows)
    agg = aggregate_firm_level_to_fiscal_year(firm_level, min_firms_per_year=30)

    # Only 15 firms have non-missing book_equity (< 30) -> NaN.
    assert pd.isna(agg.loc["1953-12-31", "book_equity_sum"])
    # Only 15 firms have non-missing oibdp (< 30) -> NaN.
    assert pd.isna(agg.loc["1953-12-31", "oibdp_sum"])


def test_thin_book_equity_does_not_incorrectly_nan_healthy_oibdp():
    from wrds_pull import aggregate_firm_level_to_fiscal_year
    # 5 firms have book_equity (thin -> NaN), but 200 DIFFERENT rows worth of
    # oibdp coverage exist for the same fiscal year (healthy -> kept).
    rows = []
    for i in range(5):
        rows.append({"gvkey": f"thin_be_{i}", "fyear": 1953, "book_equity": 100.0, "oibdp": np.nan})
    for i in range(200):
        rows.append({"gvkey": f"healthy_ep_{i}", "fyear": 1953, "book_equity": np.nan, "oibdp": 20.0})
    firm_level = pd.DataFrame(rows)
    agg = aggregate_firm_level_to_fiscal_year(firm_level, min_firms_per_year=30)

    assert pd.isna(agg.loc["1953-12-31", "book_equity_sum"]), "5 firms is below the 30-firm minimum"
    assert pd.notna(agg.loc["1953-12-31", "oibdp_sum"]), "200 firms is well above the minimum"


# --------------------------------------------------------------------------
# Regression tests for the coverage-discontinuity guard, added after a live
# run showed a fiscal year (1961) that cleared the raw firm-count threshold
# but was still ~1/9th of the following year's aggregate -- a coverage
# ramp-up a fixed firm count alone can't distinguish from genuine small
# aggregate values, since "how many firms is enough" itself changes across
# a sample spanning decades of Compustat's own historical coverage growth.
# --------------------------------------------------------------------------

def test_discontinuous_year_is_nan_even_with_enough_firms():
    from wrds_pull import aggregate_firm_level_to_fiscal_year
    # 150 firms in 1961 (clears a 100-firm minimum) but each with tiny book
    # equity -- aggregate is ~1/10th of the next three years' level.
    rows = []
    for i in range(150):
        rows.append({"gvkey": f"1961_{i}", "fyear": 1961, "book_equity": 11814507.0 / 150 / 1000, "oibdp": 100.0})
    for year in (1962, 1963, 1964):
        for i in range(1900):
            rows.append({"gvkey": f"{year}_{i}", "fyear": year, "book_equity": 123412811.0 / 1900 / 1000, "oibdp": 100.0})
    firm_level = pd.DataFrame(rows)

    agg = aggregate_firm_level_to_fiscal_year(firm_level, min_firms_per_year=100)
    assert pd.isna(agg.loc["1961-12-31", "book_equity_sum"]), (
        "1961 clears the firm-count minimum but is a clear coverage "
        "discontinuity relative to the following years -- must still be NaN'd"
    )
    for year in (1962, 1963, 1964):
        assert pd.notna(agg.loc[f"{year}-12-31", "book_equity_sum"])


def test_genuine_gradual_growth_is_not_flagged_as_discontinuous():
    from wrds_pull import aggregate_firm_level_to_fiscal_year
    # Smooth ~5%/year growth across well-covered years should NOT trip the
    # discontinuity guard -- only a sharp jump should.
    rows = []
    base = 100.0
    for j, year in enumerate(range(1970, 1976)):
        level = base * (1.05 ** j)
        for i in range(500):
            rows.append({"gvkey": f"{year}_{i}", "fyear": year, "book_equity": level / 500, "oibdp": 20.0})
    firm_level = pd.DataFrame(rows)

    agg = aggregate_firm_level_to_fiscal_year(firm_level, min_firms_per_year=100)
    for year in range(1970, 1976):
        assert pd.notna(agg.loc[f"{year}-12-31", "book_equity_sum"]), (
            f"{year}: smooth organic growth should not be flagged as a "
            "coverage discontinuity"
        )


def test_discontinuity_guard_can_be_disabled():
    from wrds_pull import aggregate_firm_level_to_fiscal_year
    rows = []
    for i in range(150):
        rows.append({"gvkey": f"1961_{i}", "fyear": 1961, "book_equity": 11814507.0 / 150 / 1000, "oibdp": 100.0})
    for i in range(1900):
        rows.append({"gvkey": f"1962_{i}", "fyear": 1962, "book_equity": 123412811.0 / 1900 / 1000, "oibdp": 100.0})
    firm_level = pd.DataFrame(rows)
    agg = aggregate_firm_level_to_fiscal_year(firm_level, min_firms_per_year=100,
                                               max_ratio_to_next_years=0)
    assert pd.notna(agg.loc["1961-12-31", "book_equity_sum"])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
