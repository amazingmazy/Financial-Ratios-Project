"""
test_totval_construction.py
============================
Regression tests for a real bug found via a live run: totval was being
computed from LAGGED market cap (the correct weighting scheme for
value-weighted returns) instead of CURRENT-month market cap (what
construct_dividend_yield actually needs it to be). This silently lagged the
whole dividend-yield series by one month, severing the contemporaneous
return-DY link the paper's entire methodology depends on -- confirmed by
running the (correct) estimator code on real data and finding corr(e,m)
near zero instead of the expected ~-0.9.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

from wrds_pull import aggregate_security_level_to_monthly_index


def _make_security_panel(T=120, n_firms=30, seed=0):
    """
    Simulate security-level monthly data with a genuine common market factor
    driving both returns and prices, so totval SHOULD track vwretd almost
    exactly if constructed correctly.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("1970-01-01", periods=T, freq="MS")
    mkt_ret = rng.normal(0.008, 0.04, T)
    prc0 = rng.uniform(10, 100, n_firms)
    shrout = rng.uniform(1e6, 5e7, n_firms)
    prc = np.zeros((T, n_firms))
    prc[0] = prc0
    for t in range(1, T):
        idio = rng.normal(0, 0.01, n_firms)
        prc[t] = prc[t - 1] * (1 + mkt_ret[t] + idio)

    rows = []
    for t, d in enumerate(dates):
        for i in range(n_firms):
            rows.append({
                "permno": i, "date": d,
                "ret": mkt_ret[t] + rng.normal(0, 0.001),
                "retx": mkt_ret[t] + rng.normal(0, 0.001),
                "prc": prc[t, i], "shrout": shrout[i],
            })
    return pd.DataFrame(rows)


def _old_buggy_aggregate(sec: pd.DataFrame) -> pd.DataFrame:
    """Reproduction of the pre-fix logic, for direct before/after comparison."""
    sec = sec.copy()
    sec["mktcap"] = sec["prc"].abs() * sec["shrout"]
    sec = sec.dropna(subset=["ret", "mktcap"])
    sec["mktcap_lag"] = sec.groupby("permno")["mktcap"].shift(1)
    sec = sec.dropna(subset=["mktcap_lag"])

    def _agg(g):
        w = g["mktcap_lag"]
        return pd.Series({
            "vwretd": (g["ret"] * w).sum() / w.sum(),
            "totval": w.sum(),  # the bug
        })

    out = sec.groupby(sec["date"].values.astype("datetime64[M]")).apply(_agg)
    out.index.name = "date"
    return out.sort_index()


def test_totval_tracks_contemporaneous_returns():
    sec = _make_security_panel()
    out = aggregate_security_level_to_monthly_index(sec)
    implied_ret = out["totval"].pct_change()
    corr = np.corrcoef(implied_ret.dropna(), out["vwretd"].iloc[1:])[0, 1]
    assert corr > 0.95, (
        f"totval's month-over-month growth should closely track vwretd "
        f"(a return this month should show up in totval this month), got "
        f"corr={corr:.3f}"
    )


def test_old_buggy_totval_does_not_track_contemporaneous_returns():
    """
    Confirms the bug this guards against was real: the pre-fix logic
    (totval = sum of LAGGED market cap) produces a much weaker correlation
    with the same month's return, on identical data.
    """
    sec = _make_security_panel()
    out = _old_buggy_aggregate(sec)
    implied_ret = out["totval"].pct_change()
    corr = np.corrcoef(implied_ret.dropna(), out["vwretd"].iloc[1:])[0, 1]
    assert corr < 0.3, (
        f"expected the old buggy logic to show a weak contemporaneous "
        f"correlation (reproducing the real-run symptom of corr(e,m) near "
        f"zero), got corr={corr:.3f} -- if this is high, the 'bug' isn't "
        f"actually being reproduced by this test"
    )


def test_totval_uses_current_not_lagged_cap_directly():
    """More direct check: totval in month t should equal the sum of month t's
    market cap (current prc * shrout), not month t-1's."""
    sec = _make_security_panel(T=6, n_firms=5)
    out = aggregate_security_level_to_monthly_index(sec)

    sec = sec.copy()
    sec["mktcap"] = sec["prc"].abs() * sec["shrout"]
    expected_totval = sec.groupby(sec["date"].values.astype("datetime64[M]"))["mktcap"].sum()

    # First eligible month is dropped (no lagged cap yet for weighting), so
    # compare from the second available month onward.
    common_dates = out.index.intersection(expected_totval.index)
    np.testing.assert_allclose(
        out.loc[common_dates, "totval"].to_numpy(),
        expected_totval.loc[common_dates].to_numpy(),
    )


def test_vwretd_still_uses_lagged_weights():
    """Confirm the fix didn't accidentally also change the (correct) return
    weighting scheme -- only totval's own definition should have changed."""
    sec = _make_security_panel(T=6, n_firms=5)
    out = aggregate_security_level_to_monthly_index(sec)

    sec = sec.copy()
    sec["mktcap"] = sec["prc"].abs() * sec["shrout"]
    sec["mktcap_lag"] = sec.groupby("permno")["mktcap"].shift(1)
    sec_valid = sec.dropna(subset=["mktcap_lag"])
    expected_vwretd = (sec_valid.groupby(sec_valid["date"].values.astype("datetime64[M]"))
                        .apply(lambda g: (g["ret"] * g["mktcap_lag"]).sum() / g["mktcap_lag"].sum()))

    common_dates = out.index.intersection(expected_vwretd.index)
    np.testing.assert_allclose(
        out.loc[common_dates, "vwretd"].to_numpy(),
        expected_vwretd.loc[common_dates].to_numpy(),
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
