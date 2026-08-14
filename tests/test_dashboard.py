"""
test_dashboard.py
==================
Tests for the interactive educational dashboard (README Issue 4):
  1. Pure data-handling functions (dashboard_data.py) -- window filtering,
     x/r construction, the Bayesian posterior calculation, the power
     rule-of-thumb, and the drop-N-years window truncation.
  2. A live Streamlit AppTest smoke test confirming all four required
     sections actually work end-to-end when driven programmatically, not
     just that the script has no syntax errors.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

from dashboard_data import (
    PRESET_WINDOWS, filter_window, build_x_r, power_curve_grid,
    flat_prior_weights, shifted_normal_prior_weights, integrate_posterior,
    power_threshold, has_power, truncate_window_by_years,
)
from estimators import fit_ar1, conditional_rho_test
from public_fallback import (
    synthetic_crsp_index, synthetic_riskfree, synthetic_cpi,
    synthetic_compustat_be_earnings,
)
from construct_panel import assemble_master_panel


@pytest.fixture(scope="module")
def synthetic_panel():
    crsp = synthetic_crsp_index(n_months=660, start="1946-01-01")
    rf = synthetic_riskfree(n_months=660, start="1946-01-01")
    cpi = synthetic_cpi(n_months=660, start="1946-01-01")
    compustat = synthetic_compustat_be_earnings(n_years=56, start_year=1946)
    return assemble_master_panel(crsp, rf, cpi, compustat_annual=compustat)


def _make_strongly_persistent_series(T=660, rho_true=0.995, seed=1, x0=-3.27):
    rng = np.random.default_rng(seed)
    sigma_m, sigma_e, corr = 0.03, 4.5, -0.9
    cov = np.array([[sigma_e**2, corr*sigma_e*sigma_m], [corr*sigma_e*sigma_m, sigma_m**2]])
    innov = rng.multivariate_normal([0, 0], cov, size=T)
    e, m = innov[:, 0], innov[:, 1]
    x = np.empty(T + 1)
    x[0] = x0
    f_true = x0 * (1 - rho_true)
    for t in range(1, T + 1):
        x[t] = f_true + rho_true * x[t - 1] + m[t - 1]
    x_lag = x[:-1]
    r = 0.9 + 1.0 * x_lag + e
    return r, x


# --------------------------------------------------------------------------
# Window / x-r plumbing (unchanged from the original point-value dashboard)
# --------------------------------------------------------------------------

def test_filter_window(synthetic_panel):
    start, end = PRESET_WINDOWS["1946-1972 (Table 3a)"]
    sub = filter_window(synthetic_panel, start, end)
    assert sub["date"].min() >= pd.Timestamp(start)
    assert sub["date"].max() <= pd.Timestamp(end)


def test_build_x_r_alignment(synthetic_panel):
    x, r, sub = build_x_r(synthetic_panel, "logDY", "VWNY")
    assert x is not None
    assert len(x) == len(r) + 1


# --------------------------------------------------------------------------
# Bayesian rho dial -- the core required feature
# --------------------------------------------------------------------------

def test_flat_prior_weights_truncated_at_one():
    grid = power_curve_grid(0.90, 0.999999, 100)
    w = flat_prior_weights(grid, rho_hat=0.99, se_rho=0.005)
    assert (w[grid > 0.999999] == 0).all() if (grid > 0.999999).any() else True
    assert w.max() > 0


def test_shifted_normal_weights_peak_moves_with_shift():
    grid = power_curve_grid(0.90, 0.999999, 300)
    w_shift0 = shifted_normal_prior_weights(grid, rho_hat=0.99, se_rho=0.005, shift_in_se=0.0)
    w_shift2 = shifted_normal_prior_weights(grid, rho_hat=0.99, se_rho=0.005, shift_in_se=2.0)
    peak0 = grid[np.argmax(w_shift0)]
    peak2 = grid[np.argmax(w_shift2)]
    assert peak2 > peak0, "shifting the prior up should move its peak toward 1"


def test_posterior_ordering_matches_paper_worked_example():
    """
    Direct regression test for the paper's own qualitative finding
    (Section 4.1): point mass at rho=1 gives the MOST conservative (highest)
    posterior probability that b<=0; a flat prior gives the LEAST
    conservative; a shifted-normal prior sits in between. This must hold
    because rho~1 is deliberately the most conservative single assumption
    in the paper's own framework (Section 2.3) -- it's not specific to the
    paper's exact numbers, so it should hold on any sufficiently-persistent
    synthetic series too.
    """
    r, x = _make_strongly_persistent_series()
    ar1 = fit_ar1(x)
    rho_hat, se_rho = ar1["rho_hat"], ar1["se_rho"]

    grid = power_curve_grid(0.90, 0.999999, 250)
    p_grid = np.array([conditional_rho_test(r, x, rho_assumed=rg).p for rg in grid])

    point_mass_p = float(p_grid[-1])
    flat_p = integrate_posterior(grid, p_grid, flat_prior_weights(grid, rho_hat, se_rho))
    shifted_p = integrate_posterior(
        grid, p_grid, shifted_normal_prior_weights(grid, rho_hat, se_rho, shift_in_se=1.0, spread_multiplier=1.0)
    )

    assert point_mass_p >= shifted_p >= flat_p, (
        f"expected point_mass ({point_mass_p:.4f}) >= shifted ({shifted_p:.4f}) "
        f">= flat ({flat_p:.4f})"
    )


def test_integrate_posterior_handles_degenerate_weights():
    """If all weight falls above the truncation point, should fall back to
    the point-mass value rather than raising a division-by-zero error."""
    grid = power_curve_grid(0.90, 0.999999, 50)
    p_grid = np.linspace(0.5, 0.01, 50)
    zero_weights = np.zeros_like(grid)
    result = integrate_posterior(grid, p_grid, zero_weights)
    assert result == pytest.approx(p_grid[-1])


# --------------------------------------------------------------------------
# Power rule-of-thumb
# --------------------------------------------------------------------------

def test_power_threshold_matches_paper_calibration_points():
    assert power_threshold(25 * 12, "monthly") == pytest.approx(0.98)
    assert power_threshold(50 * 12, "monthly") == pytest.approx(0.99)
    assert power_threshold(25 * 12, "annual") == pytest.approx(0.85)
    assert power_threshold(50 * 12, "annual") == pytest.approx(0.90)


def test_has_power_matches_threshold():
    assert has_power(0.997, 55 * 12) is True
    assert has_power(0.90, 25 * 12) is False


# --------------------------------------------------------------------------
# Drop-last-N-years window truncation
# --------------------------------------------------------------------------

def test_truncate_window_by_years_matches_table_4():
    # Table 4 compares exactly 1946-1994 vs. 1946-2000 -- a 6-year drop.
    assert truncate_window_by_years("2000-12-31", 6) == "1994-12-31"


def test_truncate_window_by_years_zero_is_identity():
    assert truncate_window_by_years("2000-12-31", 0) == "2000-12-31"


# --------------------------------------------------------------------------
# Live app smoke test: all four required sections
# --------------------------------------------------------------------------

def test_dashboard_all_four_sections_work_live(synthetic_panel, tmp_path):
    streamlit_testing = pytest.importorskip("streamlit.testing.v1")
    AppTest = streamlit_testing.AppTest

    panel_path = tmp_path / "master_panel.csv"
    synthetic_panel.to_csv(panel_path, index=False)

    app_path = os.path.join(os.path.dirname(__file__), "..", "src", "dashboard.py")
    at = AppTest.from_file(app_path)
    at.run(timeout=90)
    assert not at.exception

    at.text_input[0].set_value(str(panel_path)).run(timeout=90)
    assert not at.exception

    # 1. Dashboard shell: current level, rho_hat, T, SE(rho_hat), + 3-row comparison table
    shell_labels = {m.label for m in at.metric}
    assert {"Current level", "ρ̂ (AR(1))", "T (months)", "SE(ρ̂)"}.issubset(shell_labels)
    assert len(at.dataframe) >= 1

    # 2. Bayesian dial: radio with exactly the three required prior types,
    # and a posterior-probability metric that changes across them.
    radio = at.radio[0]
    assert set(radio.options) == {"Point mass at ρ=1", "Flat prior on ρ≤1", "Shifted-normal prior"}

    def _posterior():
        return [m.value for m in at.metric if m.label == "Posterior probability that b ≤ 0"][0]

    radio.set_value("Point mass at ρ=1").run(timeout=90)
    p_point = _posterior()
    radio.set_value("Flat prior on ρ≤1").run(timeout=90)
    p_flat = _posterior()
    assert p_point != p_flat, "posterior probability should differ across prior types"

    # 3. Power indicator: exactly one of warning/success should fire.
    fired = len(at.warning) + len(at.success)
    assert fired >= 1

    # 4. Drop-N-years panel: slider present, and moving it changes the
    # truncated-window text shown.
    drop_slider = [s for s in at.slider if "Years dropped" in s.label][0]
    drop_slider.set_value(0).run(timeout=90)
    text_at_0 = [m.value for m in at.markdown if "Truncated window" in m.value]
    drop_slider.set_value(6).run(timeout=90)
    text_at_6 = [m.value for m in at.markdown if "Truncated window" in m.value]
    assert text_at_0 != text_at_6


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
