# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Walkthrough: Replicating Lewellen (2004)
#
# **Predicting returns with financial ratios**, *Journal of Financial Economics*
# 74:209–235.
#
# This notebook is a guided tour of the cleaned data and the analysis code in
# this repository. It is meant to be read top to bottom by someone who has not
# seen the project before: what the panel contains, where the numbers come from,
# and how the three estimators differ.
#
# It is **not** the deliverable that produces the report — that is `dodo.py`,
# which runs the pipeline end to end. Everything here reads artefacts the
# pipeline has already built, so nothing in this notebook can silently disagree
# with the report.
#
# ## The problem in one paragraph
#
# Regress next month's return on this month's dividend yield and you get a
# positive slope. But the yield's own innovations are almost perfectly
# *negatively* correlated with return innovations — a price rise raises the
# return and mechanically lowers the yield, in the same month. Stambaugh (1986,
# 1999) showed this makes the slope badly biased upward in small samples, and
# the standard correction wipes out most of the apparent predictability.
#
# Lewellen's contribution is that the standard correction throws away something
# we know for free: a dividend yield cannot explode, so its autocorrelation
# $\rho$ must be below 1. Conditioning on that bound instead of integrating over
# every possible $\rho$ gives a tighter estimator — and flips the conclusion
# back to "returns are predictable."
#
# ## What you will see
#
# 1. The cleaned panel and how each column was built
# 2. The correlation that causes the bias, measured directly
# 3. The three estimators, side by side, on one regression
# 4. Our replication against the paper's published Table 2
# 5. Where the method stops working — and why that is the paper's own prediction

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path.cwd() if (Path.cwd() / "settings.py").exists() else Path.cwd() / "src"
sys.path.insert(0, str(SRC))

from settings import config
import paper_values as pv
from estimators import fit_ar1, fit_predictive_ols, conditional_rho_test, stambaugh_correction
from qa_table1 import compute_stats

DATA_DIR = config("DATA_DIR")
pd.set_option("display.float_format", lambda v: f"{v:0.4f}")
print(f"reading panels from: {DATA_DIR}")

# %% [markdown]
# ## 1. The cleaned panel
#
# `run_issue1.py` pulls from CRSP and Compustat and writes a single tidy monthly
# panel. Cleaning lives entirely in `src/construct_panel.py`; nothing in that
# file does analysis, and nothing in `src/estimators.py` touches raw data.
#
# Two panels exist, and the difference matters. CRSP's legacy **SIZ** tables
# stopped being updated after 2024-12-31; the current **CIZ** schema runs a year
# further but compounds returns differently. We replicate on SIZ (closer to the
# convention Lewellen's own data used) and extend on CIZ.

# %%
panel_siz = pd.read_csv(DATA_DIR / "master_panel.csv", parse_dates=["date"])
panel_ciz = pd.read_csv(DATA_DIR / "master_panel_ciz.csv", parse_dates=["date"])

for name, p in [("SIZ", panel_siz), ("CIZ", panel_ciz)]:
    print(f"{name}: {len(p):>5} months   {p.date.min():%Y-%m} .. {p.date.max():%Y-%m}")

panel_siz.head()

# %% [markdown]
# ### What each column is
#
# | column | construction |
# |---|---|
# | `VWNY`, `EWNY` | NYSE value- and equal-weighted monthly returns, in percent |
# | `ExcVWNY`, `ExcEWNY` | the same, net of the one-month T-bill rate |
# | `DY` | trailing 12-month dividends ÷ current index level |
# | `B/M`, `E/P` | aggregate Compustat book equity / operating income ÷ market equity |
# | `logDY`, `logB/M`, `logE/P` | natural logs — what every regression actually uses |
#
# The logs are not cosmetic. Raw `DY` is positively skewed and its volatility
# scales with its level, so the AR(1) model the estimators assume fits the log
# series far better. The paper makes the same choice (Section 3).
#
# One construction detail worth surfacing, because it caused a real bug: the
# dividend flow is inferred as
# $\text{Div}_t \approx (\text{vwretd}_t - \text{vwretx}_t) \times \text{TotVal}_{t-1}$,
# the difference between returns with and without dividends. The *denominator*
# of `DY`, though, is the **current** month's index level. Using the lagged one
# instead shifted the whole series by a month and quietly destroyed the
# correlation the next section is about.

# %%
cols = ["VWNY", "EWNY", "DY", "logDY", "B/M", "logB/M", "E/P", "logE/P"]
panel_siz[["date"] + cols].describe().T[["count", "mean", "std", "min", "max"]]

# %% [markdown]
# ## 2. The correlation that causes the bias
#
# Everything in the paper follows from one number. Fit the two equations
#
# $$r_t = a + b\,x_{t-1} + e_t \qquad x_t = f + \rho\,x_{t-1} + m_t$$
#
# and correlate their residuals. Because a price rise pushes returns up and the
# yield down *in the same month*, $\text{corr}(e, m)$ is near $-1$ by
# construction — not by any forecasting relationship.

# %%
start, end, _ = pv.WINDOWS["1946-2000"]
win = panel_siz[(panel_siz.date >= start) & (panel_siz.date <= end)].dropna(
    subset=["logDY", "VWNY"]
).reset_index(drop=True)

x = win["logDY"].to_numpy()          # x_0 .. x_T
r = win["VWNY"].to_numpy()[1:]       # r_1 .. r_T

ar1 = fit_ar1(x)
ols = fit_predictive_ols(r, x[:-1])
corr_em = float(np.corrcoef(ols["resid"], ar1["resid"])[0, 1])

print(f"T                 = {len(r)}")
print(f"rho_hat           = {ar1['rho_hat']:.4f}   (paper: {pv.AR1['logDY']['1946-2000'].rho})")
print(f"corr(e, m)        = {corr_em:.4f}   (paper: {pv.PREDICTIVE['logDY']['1946-2000']['VWNY'].corr_em})")

# %% [markdown]
# That $-0.95$ is the engine of the whole paper. It is also the project's best
# diagnostic: if a data-construction change breaks the alignment between returns
# and the ratio, this number collapses toward zero while every univariate
# statistic still looks fine. `tests/test_replication_vs_paper.py` asserts it for
# exactly that reason.
#
# ## 3. The three estimators
#
# All three answer "what is $b$?" and differ only in how they handle the bias.

# %%
stam = stambaugh_correction(r, x, n_sims=2000, random_state=0)
cond = conditional_rho_test(r, x)
blk = pv.PREDICTIVE["logDY"]["1946-2000"]["VWNY"]

pd.DataFrame(
    {
        "estimator": ["OLS", "Stambaugh (1999)", "rho ~ 1 (Lewellen)"],
        "b_hat":     [ols["b_hat"], stam.b_hat_adj, cond.b_hat_adj],
        "std_err":   [ols["se_b"], stam.se, cond.se],
        "p_value":   [ols["p"], stam.p, cond.p],
        "paper_b":   [blk.ols.b, blk.stambaugh.b, blk.rho1.b],
        "paper_p":   [blk.ols.p, blk.stambaugh.p, blk.rho1.p],
    }
).set_index("estimator")

# %% [markdown]
# Read the `p_value` column downward. OLS says predictable; Stambaugh's
# correction says nothing is there; the $\rho \approx 1$ test says predictable
# again, decisively. Same data, same model — the entire disagreement is about
# how to handle a known bias.
#
# The mechanism is a single line, `estimators.conditional_rho_test`:
#
# $$\hat b_{adj} = \hat b - \hat\gamma(\hat\rho - 1), \qquad
#   \hat\gamma = \frac{\text{cov}(e,m)}{\text{var}(m)}$$
#
# With $\hat\gamma \approx -90$, the correction is violently sensitive to
# $\hat\rho$. That is why the test suite pins $\hat\rho$ to ±0.002 rather than a
# looser tolerance that would *look* tight next to a value of 0.997.

# %%
gamma = float(np.cov(ols["resid"], ar1["resid"], ddof=1)[0, 1] / np.var(ar1["resid"], ddof=1))
print(f"gamma_hat = {gamma:.1f}")
for rho_assumed in [0.9999, 0.997, 0.99, 0.98]:
    adj = ols["b_hat"] - gamma * (ar1["rho_hat"] - rho_assumed)
    print(f"  if rho were {rho_assumed:<7} -> b_adj = {adj:+.3f}")

# %% [markdown]
# A move of 0.02 in the assumed $\rho$ swings the answer by more than the
# estimate itself. The conservative choice — assume $\rho \approx 1$ — gives the
# *smallest* $b$ consistent with stationarity, which is why rejecting under it
# is strong evidence rather than an assumption doing the work.
#
# ## 4. How close is the replication?
#
# `src/paper_values.py` holds Lewellen's published tables, transcribed by hand
# and checked against relationships the paper states in prose. The test suite
# asserts our numbers against them within per-statistic tolerances chosen
# *before* the numbers were seen.

# %%
rows = []
for w in ["1946-2000", "1946-1972", "1973-2000"]:
    s, e, _ = pv.WINDOWS[w]
    sub = panel_siz[(panel_siz.date >= s) & (panel_siz.date <= e)]
    for series in ["VWNY", "EWNY", "DY", "logDY"]:
        ours = dict(zip(["mean", "sd", "skew", "rho1", "rho12", "rho24"],
                        compute_stats(sub[series])))
        paper = pv.TABLE_1[w][series]
        rows.append({"window": w, "series": series,
                     "paper_mean": paper.mean, "our_mean": ours["mean"],
                     "paper_rho1": paper.rho1, "our_rho1": ours["rho1"]})
pd.DataFrame(rows).set_index(["window", "series"])

# %% [markdown]
# The dividend-yield rows reproduce cleanly. `B/M` and `E/P` do not, and the
# reason is documented rather than hidden: the paper never states its
# book-equity formula, and including deferred taxes (`TXDITC`) inflates our
# aggregate `B/M` by about 10%. Dropping the term closes most of the gap —
# `pull_compustat_be_and_earnings(include_deferred_taxes=...)` exposes the
# choice rather than burying it.
#
# One finding from building this notebook's Table 1 comparison is worth
# repeating, because it reversed an earlier conclusion. Lewellen reports
# $\rho_{24} = 1.062$ for log DY — impossible for a correlation. Rather than an
# erratum, that is the clue identifying his estimator: only an *unbounded*
# statistic can exceed one, so his autocorrelations are lag-$k$ OLS slopes, not
# Pearson correlations. Recomputing that way reproduced every lag-12 and lag-24
# cell we had previously written off.

# %%
def lag_k_pearson(v, k):
    v = np.asarray(v, float)
    return float(np.corrcoef(v[:-k], v[k:])[0, 1])

sub = panel_siz[(panel_siz.date >= "1973-01-01") & (panel_siz.date <= "2000-12-31")]
series = sub["logDY"].dropna().to_numpy()
print(f"logDY, 1973-2000, lag 24")
print(f"  paper                     = {pv.TABLE_1['1973-2000']['logDY'].rho24}")
print(f"  Pearson correlation       = {lag_k_pearson(series, 24):.4f}   <- bounded by 1, cannot match")
print(f"  lag-k OLS slope (correct) = {compute_stats(sub['logDY'])[5]:.4f}")

# %% [markdown]
# ## 5. Where the method stops working
#
# The conditional test only has power when $\hat\rho$ is already close to one.
# Lewellen states the rule of thumb himself (Section 2.4): roughly 0.98 for 25
# years of monthly data, 0.99 for 50. Since his sample ends in 2000, he could
# not check whether that stayed true.
#
# It did not.

# %%
for label, s, e in [("1946-2000  (the paper)", "1946-01-01", "2000-12-31"),
                    ("1946-2025  (extended)", "1946-01-01", "2025-12-31"),
                    ("2001-2025  (out of sample)", "2001-01-01", "2025-12-31")]:
    sub = panel_ciz[(panel_ciz.date >= s) & (panel_ciz.date <= e)].dropna(
        subset=["logDY", "VWNY"]).reset_index(drop=True)
    xx = sub["logDY"].to_numpy()
    rr = sub["VWNY"].to_numpy()[1:]
    a, o, c = fit_ar1(xx), fit_predictive_ols(rr, xx[:-1]), conditional_rho_test(rr, xx)
    verdict = "applicable" if a["rho_hat"] >= 0.98 else "NO POWER (rho below 0.98)"
    print(f"{label:<28} rho={a['rho_hat']:.4f}  OLS b={o['b_hat']:+.3f} (p={o['p']:.3f})"
          f"  rho~1 b={c.b_hat_adj:+.3f} (p={c.p:.3f})   {verdict}")

# %% [markdown]
# Post-2000, log DY's autocorrelation falls to about 0.95 — below the paper's
# own threshold — and the conditional test duly returns a negative estimate with
# a p-value near 1. **This is not a refutation.** Lewellen's Table A.1
# tabulates exactly this: the test's power falls toward zero as $\rho$ drops
# away from one, and its estimate is biased downward once $\rho$ is genuinely
# below the assumed value.
#
# Notice too that OLS reports a *larger* slope on the shorter window with an
# unchanged p-value. Read alone that looks like stronger predictability; it is
# the small-sample bias the paper is about, made more visible by a shorter and
# less persistent sample.
#
# ## Where the code lives
#
# | file | responsibility |
# |---|---|
# | `src/wrds_pull.py` | CRSP and Compustat queries, both SIZ and CIZ schemas |
# | `src/construct_panel.py` | cleaning only — builds the tidy panel |
# | `src/qa_table1.py` | QA gate against the paper's Table 1 |
# | `src/estimators.py` | OLS, Stambaugh, $\rho\approx1$, modified Bonferroni |
# | `src/replicate_paper_tables.py` | Tables 2–6 and the extension windows |
# | `src/paper_values.py` | published values + tolerance rationale |
# | `tests/test_replication_vs_paper.py` | asserts our numbers against the paper |
# | `dodo.py` | the whole thing, end to end |
#
# To rebuild everything from raw pulls:
#
# ```bash
# conda env create -f environment.yml && conda activate financial-ratios
# cp .env.example .env        # then set WRDS_USERNAME
# doit
# ```
