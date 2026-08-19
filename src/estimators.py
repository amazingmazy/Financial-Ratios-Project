"""
estimators.py
=============
Implements the three predictive-regression estimators compared in
Lewellen (2004), "Predicting returns with financial ratios," JFE 74.

Model (paper's Eqs. 3a-3b):
    r_t   = a + b * x_{t-1} + e_t
    x_t   = f + rho * x_{t-1} + m_t

Three estimators of b:
  1. OLS            -- ignores small-sample bias entirely (benchmark).
  2. Stambaugh (1999) -- bias-adjusts using the *marginal* (unconditional)
                         distribution of b_hat, obtained by Monte Carlo
                         simulation calibrated to the estimated (rho, Sigma).
                         This mirrors exactly what Lewellen describes doing
                         in the paper (Section 4.1): "The distribution
                         depends on the unknown parameters rho and Sigma,
                         for which I substitute the OLS estimates."
  3. rho ~ 1 (Lewellen's conditional test) -- bias-adjusts using the
                         *conditional* distribution of b_hat given rho_hat,
                         under the conservative assumption that rho is as
                         close to 1 as stationarity allows. Closed-form,
                         Eqs. (7)-(10) and the Appendix (A.1)-(A.7).

Also implements the modified-Bonferroni joint test of Section 2.4.

None of this requires proprietary data -- it operates on any (return,
lagged predictor) pair you hand it, in monthly or any other frequency.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from scipy import stats


# --------------------------------------------------------------------------
# Basic building blocks
# --------------------------------------------------------------------------

def ols_beta_se(y: np.ndarray, X: np.ndarray):
    """Plain OLS. X must include an intercept column. Returns (beta, se, resid, XtX_inv)."""
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    n, k = X.shape
    sigma2 = (resid @ resid) / (n - k)
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    return beta, se, resid, XtX_inv


def fit_ar1(x: np.ndarray):
    """
    Fit x_t = f + rho * x_{t-1} + m_t by OLS.
    Returns dict with f_hat (intercept), rho_hat, se(rho), residuals m_hat
    (length T-1), the Kendall/Marriott bias approx -(1+3*rho)/T, and sigma_m.
    """
    x_t = x[1:]
    x_lag = x[:-1]
    T = len(x_t)
    X = np.column_stack([np.ones(T), x_lag])
    beta, se, resid, XtX_inv = ols_beta_se(x_t, X)
    f_hat = beta[0]
    rho_hat = beta[1]
    kendall_bias = -(1 + 3 * rho_hat) / T
    return {
        "f_hat": f_hat,
        "rho_hat": rho_hat,
        "se_rho": se[1],
        "resid": resid,       # m_hat_t, length T
        "T": T,
        "kendall_bias": kendall_bias,
        "sigma_m": resid.std(ddof=2),
        "X": X,
        "XtX_inv": XtX_inv,
    }


def fit_predictive_ols(r: np.ndarray, x_lag: np.ndarray):
    """
    Fit r_t = a + b * x_{t-1} + e_t by OLS.
    Returns dict with b_hat, se(b), t-stat, one-sided p-value, residuals e_hat.
    """
    T = len(r)
    X = np.column_stack([np.ones(T), x_lag])
    beta, se, resid, XtX_inv = ols_beta_se(r, X)
    b_hat, se_b = beta[1], se[1]
    tstat = b_hat / se_b
    pval_one_sided = 1 - stats.t.cdf(tstat, df=T - 2)
    return {
        "b_hat": b_hat,
        "se_b": se_b,
        "t": tstat,
        "p": pval_one_sided,
        "resid": resid,
        "T": T,
    }


# --------------------------------------------------------------------------
# Estimator 2: Stambaugh (1999) bias correction via Monte Carlo
# --------------------------------------------------------------------------

@dataclass
class StambaughResult:
    b_hat_adj: float
    se: float
    p: float
    n_sims: int


def stambaugh_correction(r: np.ndarray, x: np.ndarray, n_sims: int = 20000,
                          random_state: int = 0) -> StambaughResult:
    """
    Bias-adjust the predictive slope using the *marginal* small-sample
    distribution of b_hat, following Stambaugh (1999) / Mankiw-Shapiro (1986)
    / Nelson-Kim (1993). Implemented via Monte Carlo simulation under the
    null b=0, calibrated to the data's estimated rho, sigma_e, sigma_m, and
    corr(e,m) -- exactly the substitution Lewellen describes using in the
    paper (p. 220): "The distribution depends on the unknown parameters rho
    and Sigma, for which I substitute the OLS estimates."

    r    : return series, length T (r_1 ... r_T), aligned with x_lag = x_0..x_{T-1}
    x    : predictor series, length T+1 (x_0 ... x_T), i.e. one longer than r
    """
    rng = np.random.default_rng(random_state)

    x_lag = x[:-1]
    x_cur = x[1:]
    T = len(r)
    assert len(x_lag) == T

    ar1 = fit_ar1(x)  # uses x_0..x_T -> T obs of m_t
    pred = fit_predictive_ols(r, x_lag)

    f_hat = ar1["f_hat"]
    rho_hat = ar1["rho_hat"]
    m_hat = ar1["resid"]
    e_hat = pred["resid"]
    sigma_m = m_hat.std(ddof=1)
    sigma_e = e_hat.std(ddof=1)
    corr_em = np.corrcoef(e_hat, m_hat)[0, 1]

    b_obs = pred["b_hat"]

    # Simulate under the null b = 0, with the estimated rho, intercept, and
    # residual covariance structure. Draw correlated (e_t, m_t) innovations,
    # build x recursively from x_0 = x[0], then run the same two regressions.
    #
    # BUG FIX: earlier versions omitted f_hat here, using x_sim[t] =
    # rho_hat*x_sim[t-1] + m_sim[t-1] with no drift term. For a stationary
    # AR(1), the intercept doesn't change the process's variance once it
    # reaches its stationary distribution -- but it does change what level
    # the process reverts *toward*. Without it, the simulation reverts
    # toward 0 instead of toward the true historical mean of x. When rho_hat
    # is close to 1 (slow mean reversion, as it always is for these
    # predictors) and x_0 starts far from 0 (e.g. logDY around -3, not 0),
    # the simulated series spends the whole sample decaying from x_0 toward
    # the wrong target, inflating the empirical variance of x_lag_sim within
    # the simulated window beyond what the true stationary process would
    # show. Since the simulated slope's variance is inversely proportional
    # to var(x_lag_sim), this systematically *deflates* Stambaugh's
    # simulated standard error -- exactly the symptom found on real data:
    # Stambaugh's SE coming out smaller than OLS's, backwards from theory
    # (Stambaugh should reflect *more* uncertainty than OLS, not less).
    cov = np.array([[sigma_e ** 2, corr_em * sigma_e * sigma_m],
                    [corr_em * sigma_e * sigma_m, sigma_m ** 2]])
    sims = np.empty(n_sims)
    x0 = x_lag[0]
    for s in range(n_sims):
        innov = rng.multivariate_normal([0, 0], cov, size=T)
        e_sim, m_sim = innov[:, 0], innov[:, 1]
        x_sim = np.empty(T + 1)
        x_sim[0] = x0
        for t in range(1, T + 1):
            x_sim[t] = f_hat + rho_hat * x_sim[t - 1] + m_sim[t - 1]
        x_lag_sim = x_sim[:-1]
        r_sim = x_lag_sim * 0.0 + e_sim  # b = 0 under the null
        Xd = np.column_stack([np.ones(T), x_lag_sim])
        beta_sim = np.linalg.lstsq(Xd, r_sim, rcond=None)[0]
        sims[s] = beta_sim[1]

    bias = sims.mean()          # E[b_hat] - 0 under the null, at (rho_hat, Sigma_hat)
    b_adj = b_obs - bias
    se_sim = sims.std(ddof=0)
    p_val = np.mean(sims >= b_obs)  # one-sided p-value under the null distribution

    return StambaughResult(b_hat_adj=b_adj, se=se_sim, p=p_val, n_sims=n_sims)


# --------------------------------------------------------------------------
# Estimator 3: Lewellen's conditional (rho ~ 1) test  -- closed form
# --------------------------------------------------------------------------

@dataclass
class ConditionalResult:
    gamma_hat: float
    b_hat_adj: float
    se: float
    t: float
    p: float
    rho_assumed: float


def conditional_rho_test(r: np.ndarray, x: np.ndarray, rho_assumed: float = 0.9999) -> ConditionalResult:
    """
    Implements Eqs. (8)-(10) and Appendix A.1 (Eq. A.4):
        r_t = a + b*x_{t-1} + gamma*(x_t - rho*x_{t-1}) + n_t

    Estimating this regression by OLS for a given rho gives b_hat directly
    equal to Lewellen's bias-adjusted estimator b_hat_adj = b_hat - gamma_hat*(rho_hat-rho),
    with correct standard errors and a genuine Student-t null distribution
    (T-3 degrees of freedom), as proved in Appendix A.1.
    """
    x_lag = x[:-1]
    x_cur = x[1:]
    T = len(r)
    assert len(x_lag) == T

    m_rho = x_cur - rho_assumed * x_lag  # m_t(rho), observable given rho
    X = np.column_stack([np.ones(T), x_lag, m_rho])
    beta, se, resid, XtX_inv = ols_beta_se(r, X)
    b_adj, se_b = beta[1], se[1]
    gamma_hat = beta[2]

    tstat = b_adj / se_b
    df = T - 3
    p_val = 1 - stats.t.cdf(tstat, df=df)

    return ConditionalResult(
        gamma_hat=gamma_hat, b_hat_adj=b_adj, se=se_b, t=tstat, p=p_val,
        rho_assumed=rho_assumed,
    )


# --------------------------------------------------------------------------
# Unit-root-ish test for rho=1 (used in the modified-Bonferroni joint test)
# --------------------------------------------------------------------------

def unit_root_pvalue(x: np.ndarray, n_sims: int = 20000, random_state: int = 1) -> float:
    """
    D = P(rho_hat_sim <= rho_hat_obs | rho = 1), by simulation. Used in the
    modified-Bonferroni joint p-value of Section 2.4: min(2P, P + D).
    """
    rng = np.random.default_rng(random_state)
    ar1 = fit_ar1(x)
    rho_obs = ar1["rho_hat"]
    sigma_m = ar1["sigma_m"]
    T = ar1["T"]
    x0 = x[0]

    sims = np.empty(n_sims)
    for s in range(n_sims):
        m_sim = rng.normal(0, sigma_m, size=T)
        x_sim = np.empty(T + 1)
        x_sim[0] = x0
        for t in range(1, T + 1):
            x_sim[t] = 1.0 * x_sim[t - 1] + m_sim[t - 1]  # rho = 1
        x_lag_sim = x_sim[:-1]
        x_cur_sim = x_sim[1:]
        Xd = np.column_stack([np.ones(T), x_lag_sim])
        beta_sim = np.linalg.lstsq(Xd, x_cur_sim, rcond=None)[0]
        sims[s] = beta_sim[1]

    D = np.mean(sims <= rho_obs)
    return D


def modified_bonferroni(p_conditional: float, p_stambaugh: float, D: float) -> float:
    """Section 2.4: overall p-value = min(2P, P + D), P = min of the two stand-alone p-values."""
    P = min(p_conditional, p_stambaugh)
    return min(1.0, 2 * P, P + D)


# --------------------------------------------------------------------------
# Convenience wrapper: run all three estimators + joint test for one series
# --------------------------------------------------------------------------

def run_all(r: np.ndarray, x: np.ndarray, n_sims: int = 20000, rho_assumed: float = 0.9999,
            random_state: int = 0):
    """
    r : length T (returns r_1..r_T)
    x : length T+1 (predictor levels x_0..x_T), so x_lag = x[:-1] aligns with r.
    Returns a dict with OLS, Stambaugh, rho~1, and joint-test results, plus
    the AR(1) diagnostics needed to reproduce the "Table 2/3/5/6" style rows.
    """
    x_lag = x[:-1]
    ar1 = fit_ar1(x)
    ols = fit_predictive_ols(r, x_lag)
    # Thread the seed through explicitly. Both Monte Carlo steps previously
    # took their own hard-coded defaults, so a caller that seeded its own run
    # still got different numbers here -- which showed up as the driver and an
    # ad-hoc script disagreeing on the Stambaugh slope by ~0.03 on identical
    # data and n_sims. The offset keeps the two simulations independent rather
    # than sharing a stream.
    stam = stambaugh_correction(r, x, n_sims=n_sims, random_state=random_state)
    cond = conditional_rho_test(r, x, rho_assumed=rho_assumed)
    D = unit_root_pvalue(x, n_sims=n_sims, random_state=random_state + 1)
    joint_p = modified_bonferroni(cond.p, stam.p, D)

    # corr(e, m): the correlation between return shocks and predictor-innovation
    # shocks. This single number is the engine behind both the Stambaugh bias
    # correction's magnitude and the rho~1 test's standard-error reduction --
    # a weak corr(e,m) mutes both effects even when everything else (including
    # rho_hat itself) looks like the paper's regime. Worth checking directly
    # rather than inferring from how strong/weak the downstream effects look,
    # since e_hat and m_hat are already computed as part of the AR(1) and OLS
    # fits above -- see docs/ISSUE2.md for how the paper's own reported
    # corr(e,m) (-0.955 for VWNY, Table 2) compares.
    m_hat = ar1["resid"]
    e_hat = ols["resid"]
    corr_em = float(np.corrcoef(e_hat, m_hat)[0, 1])

    return {
        "T": ols["T"],
        "rho_hat": ar1["rho_hat"],
        "se_rho": ar1["se_rho"],
        "kendall_bias": ar1["kendall_bias"],
        "corr_em": corr_em,
        "gamma_hat": cond.gamma_hat,
        "ols_b": ols["b_hat"], "ols_se": ols["se_b"], "ols_p": ols["p"],
        "stambaugh_b": stam.b_hat_adj, "stambaugh_se": stam.se, "stambaugh_p": stam.p,
        "rho1_b": cond.b_hat_adj, "rho1_se": cond.se, "rho1_t": cond.t, "rho1_p": cond.p,
        "unit_root_D": D,
        "joint_p": joint_p,
    }
