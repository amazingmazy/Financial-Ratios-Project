"""
test_estimators.py
===================
Unit tests for the shared estimator library (src/estimators.py). Focus is on
the regression test at the bottom for a real bug found on live data: the
Stambaugh Monte Carlo simulation omitted the AR(1) intercept, which
systematically deflated its simulated standard error below OLS's -- backwards
from theory (Stambaugh should reflect *more* parameter uncertainty than OLS,
never less) and from the paper's own reported numbers.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from estimators import fit_ar1, stambaugh_correction, conditional_rho_test, run_all


def _simulate_ar1_predictive_pair(T=660, rho_true=0.995, b_true=1.0,
                                   sigma_m=0.03, sigma_e=4.5, corr=-0.9,
                                   x0=-3.27, seed=1):
    """
    Build a synthetic (r, x) pair with the AR(1) intercept correctly included
    -- x0 deliberately set far from 0 (like a real log dividend yield, which
    sits around -3 to -3.5, not near 0) so a missing-intercept bug in the
    *simulation* step (not this generator) would show up as a detectably
    wrong Stambaugh SE.
    """
    rng = np.random.default_rng(seed)
    cov = np.array([[sigma_e ** 2, corr * sigma_e * sigma_m],
                    [corr * sigma_e * sigma_m, sigma_m ** 2]])
    innov = rng.multivariate_normal([0, 0], cov, size=T)
    e, m = innov[:, 0], innov[:, 1]
    x = np.empty(T + 1)
    x[0] = x0
    f_true = x0 * (1 - rho_true)  # intercept consistent with stationary mean x0
    for t in range(1, T + 1):
        x[t] = f_true + rho_true * x[t - 1] + m[t - 1]
    x_lag = x[:-1]
    r = 0.9 + b_true * x_lag + e
    return r, x


def test_fit_ar1_returns_intercept():
    _, x = _simulate_ar1_predictive_pair()
    ar1 = fit_ar1(x)
    assert "f_hat" in ar1
    # For a stationary process anchored near x0=-3.27, the intercept should
    # be small in magnitude (f = mean*(1-rho), and (1-rho) is tiny) but
    # clearly nonzero and of the sign implied by a negative stationary mean.
    assert ar1["f_hat"] < 0


def test_fit_ar1_recovers_true_rho_and_intercept():
    rho_true = 0.995
    x0 = -3.27
    _, x = _simulate_ar1_predictive_pair(rho_true=rho_true, x0=x0, T=2000)
    ar1 = fit_ar1(x)
    assert abs(ar1["rho_hat"] - rho_true) < 0.01
    f_true = x0 * (1 - rho_true)
    assert abs(ar1["f_hat"] - f_true) < 0.05


# --------------------------------------------------------------------------
# Regression test for the missing-intercept bug found on live data: every
# Table 2/3/5 result showed Stambaugh's SE smaller than OLS's SE, backwards
# from theory. Traced to stambaugh_correction's AR(1) simulation loop
# omitting f_hat, which -- when x0 starts far from 0 and rho is close to 1
# (slow mean reversion) -- caused the simulated series to spend the sample
# decaying toward the wrong target (0 instead of the true historical mean),
# inflating the simulated variance of x_lag and deflating the simulated
# slope's standard error.
# --------------------------------------------------------------------------

def test_stambaugh_se_is_not_smaller_than_ols_se():
    """
    Stambaugh's correction integrates over additional parameter uncertainty
    (the AR(1) persistence) on top of what OLS already reflects -- its
    standard error should never come out smaller than OLS's own SE for a
    single-predictor regression like this. If it does, something is
    inflating the simulated slope's precision artificially (exactly what the
    missing-intercept bug did).
    """
    r, x = _simulate_ar1_predictive_pair()
    x_lag = x[:-1]
    from estimators import fit_predictive_ols
    ols = fit_predictive_ols(r, x_lag)
    stam = stambaugh_correction(r, x, n_sims=3000, random_state=0)
    assert stam.se >= ols["se_b"] * 0.9, (
        f"Stambaugh SE ({stam.se:.4f}) should not be meaningfully smaller "
        f"than OLS SE ({ols['se_b']:.4f}) -- this was the exact symptom of "
        f"the missing-intercept bug (buggy SE came out ~4x too small)"
    )


def test_stambaugh_simulation_uses_the_fitted_intercept():
    """
    Directly checks the bug's mechanism: reproducing the old (buggy) inline
    simulation without f_hat should give a materially smaller SE than the
    current implementation, on identical data and random seed.
    """
    r, x = _simulate_ar1_predictive_pair()
    x_lag = x[:-1]
    T = len(r)

    ar1 = fit_ar1(x)
    from estimators import fit_predictive_ols
    pred = fit_predictive_ols(r, x_lag)
    rho_hat = ar1["rho_hat"]
    m_hat = ar1["resid"]
    e_hat = pred["resid"]
    sigma_m = m_hat.std(ddof=1)
    sigma_e = e_hat.std(ddof=1)
    corr_em = np.corrcoef(e_hat, m_hat)[0, 1]
    cov = np.array([[sigma_e ** 2, corr_em * sigma_e * sigma_m],
                    [corr_em * sigma_e * sigma_m, sigma_m ** 2]])

    rng = np.random.default_rng(0)
    n_sims = 3000
    sims_buggy = np.empty(n_sims)
    x0 = x_lag[0]
    for s in range(n_sims):
        innov = rng.multivariate_normal([0, 0], cov, size=T)
        e_sim, m_sim = innov[:, 0], innov[:, 1]
        x_sim = np.empty(T + 1)
        x_sim[0] = x0
        for t in range(1, T + 1):
            x_sim[t] = rho_hat * x_sim[t - 1] + m_sim[t - 1]  # the old bug: no f_hat
        x_lag_sim = x_sim[:-1]
        Xd = np.column_stack([np.ones(T), x_lag_sim])
        beta_sim = np.linalg.lstsq(Xd, e_sim, rcond=None)[0]
        sims_buggy[s] = beta_sim[1]
    buggy_se = sims_buggy.std(ddof=0)

    fixed = stambaugh_correction(r, x, n_sims=n_sims, random_state=0)
    assert fixed.se > buggy_se * 2, (
        f"fixed SE ({fixed.se:.4f}) should be substantially larger than the "
        f"reproduced buggy SE ({buggy_se:.4f}) on identical data/seed"
    )


def test_run_all_end_to_end_ordering_matches_theory():
    """
    Loose sanity check on the full pipeline: for data simulated with strong
    persistence and a strong e/m correlation (the regime the paper's method
    is designed for), rho~1's SE should be smaller than OLS's SE (the
    variance-reduction the conditional test is supposed to deliver).
    """
    r, x = _simulate_ar1_predictive_pair(rho_true=0.997, corr=-0.95)
    res = run_all(r, x, n_sims=3000)
    assert res["rho1_se"] < res["ols_se"], (
        "the conditional (rho~1) test should show a variance reduction "
        "relative to OLS when persistence and corr(e,m) are both strong, "
        "matching the paper's own Table 2 pattern"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
