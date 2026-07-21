# Lewellen (2004) Replication — "Predicting Returns with Financial Ratios"

Replication of Lewellen, J. (2004), *Journal of Financial Economics* 74(2): 209–235.
Companion project to a Stambaugh (1999) replication — same underlying data, opposing
conclusions about return predictability.

This README is organized as a set of issues, in the order they should be worked.
Each issue lists scope, dependencies, and a rough "done" bar.

---

## Issue 1 — Data wrangling & cleaning

**Goal:** produce one clean monthly panel that everything downstream reads from.

- [ ] Pull monthly NYSE value-weighted (VW) and equal-weighted (EW) returns, both
      *including* and *excluding* dividends, from the Ken French Data Library
      (or CRSP directly if WRDS access is available)
- [ ] Pull the one-month T-bill rate (French Library `RF`, or CRSP risk-free file)
- [ ] Pull CPI from FRED (`CPIAUCSL`) for real returns
- [ ] Construct the dividend flow: `Div_t ≈ (VWretd_t − VWretx_t) × Index_{t-1}`;
      roll to trailing 12 months; divide by current index level → `DY_t`; take logs
- [ ] Construct excess and real returns for VW and EW
- [ ] **[Compustat-dependent]** Construct aggregate B/M and E/P for NYSE firms:
  - sum book equity / operating income (before depreciation) across Compustat-covered
    NYSE firms with ≥3 years of history
  - lag accounting data 4 months
  - divide by contemporaneous NYSE VW market equity; take logs
  - if no WRDS/Compustat access: build an approximate version from Ken French's
    25 Size–BE/ME portfolios and **flag it explicitly as an approximation**, not a
    true aggregate NYSE ratio
- [ ] Assemble master panel: `date, VWNY, EWNY, ExcVWNY, ExcEWNY, DY, logDY, B/M,
      logB/M, E/P, logE/P`, monthly, 1946–present (pull from 1926 if useful for
      pre-sample checks)
- [ ] **QA gate:** reproduce Table 1 (mean, SD, skew, ρ₁, ρ₁₂, ρ₂₄) for full sample
      and both halves; do not proceed to Issue 2 until these match the paper within
      rounding

**Depends on:** nothing (this is the foundation)
**Blocks:** everything else

---

## Issue 2 — Core replication

**Goal:** implement the three estimators and reproduce the paper's main tables.
Broken into sub-issues per table so data and model dependencies are explicit and
each table can be picked up/tested independently once Issue 1 is done.

**Depends on:** Issue 1
**Blocks:** Issues 3–5

---

### Issue 2.0 — Shared estimator library

The three estimators are reused across every table below; build once, test against
the paper's full-sample VWNY numbers, then treat as a dependency for 2.1–2.5.

- [ ] **OLS** — standard predictive regression `r_t = a + b·x_{t-1} + e_t` — benchmark
      only, not bias-adjusted
- [ ] **Stambaugh (1999) bias-adjusted estimator** — AR(1) for `x_t`, analytic
      small-sample distribution (preferred) or Monte Carlo calibrated to `(ρ̂, Σ̂)`
      as fallback
- [ ] **ρ≈1 conditional estimator** — `b̂_adj = b̂ - ĝ(ρ̂-1)` where
      `ĝ = cov(ê,m̂)/var(m̂)` from the auxiliary regression `e_t = g·m_t + n_t`
      (paper's Appendix A.1); test statistic is exact Student-t(T-3) under the null
- [ ] **Modified Bonferroni joint p-value**: `min(2P, P+D)`, `D` = p-value for
      testing `ρ=1`
- [ ] Unit test: full-sample nominal VWNY, 1946–2000 →
      OLS b=0.92 (SE 0.48), Stambaugh b=0.20/p=0.308, ρ≈1 b=0.66/p=0.000 (Table 2)

**Data used:** none directly — this is pure estimator code, tested against
Issue 1's DY panel restricted to the VWNY/1946–2000 slice.

---

### Issue 2.1 — Table 1: summary statistics

- [ ] Compute mean, SD, skewness, and autocorrelations (ρ₁, ρ₁₂, ρ₂₄) for VWNY,
      EWNY, DY, log(DY) — full sample 1946–2000 and both halves (1946–72, 1973–2000)
- [ ] Same for B/M, log(B/M), E/P, log(E/P) — Compustat era, 1963–2000

**Data used:** VW/EW NYSE returns, DY (from Issue 1's CRSP/French-Library
construction); B/M and E/P (from Issue 1's Compustat aggregate construction, or
flagged approximation)
**Model used:** none — descriptive statistics only (this is the QA gate before
Issue 2.2 onward)

---

### Issue 2.2 — Table 2: dividend yield, full sample 1946–2000

- [ ] AR(1) regression: `log(DY_t) = f + ρ·log(DY_{t-1}) + m_t`
- [ ] Predictive regressions of nominal VWNY, nominal EWNY, excess VWNY, excess
      EWNY on lagged log(DY), each under OLS / Stambaugh / ρ≈1

**Data used:** VW/EW NYSE returns (nominal and excess, i.e. net of the one-month
T-bill), log(DY) — all from Issue 1, sample restricted to Jan 1946–Dec 2000 (660
months)
**Model used:** shared estimator library (Issue 2.0), applied to the single-regressor
log(DY) specification

---

### Issue 2.3 — Table 3: dividend yield, subsamples

- [ ] Repeat the Table 2 regressions (AR(1) + OLS/Stambaugh/ρ≈1) separately for
      Jan 1946–Dec 1972 (324 months) and Jan 1973–Dec 2000 (336 months)

**Data used:** same series as Issue 2.2, split at Dec 1972
**Model used:** shared estimator library (Issue 2.0) — no new estimator work, purely
a sample-filtering exercise once Issue 2.2 is done

---

### Issue 2.4 — Table 4: sensitivity to 1995–2000

- [ ] Repeat the Table 2 regressions for Jan 1946–Dec 1994 (588 months) and compare
      directly to the Jan 1946–Dec 2000 results from Issue 2.2
- [ ] Report the change in `ρ̂` and the implied change in `ĝ(ρ̂-1)` (the "realized
      bias") between the two samples — this is the number that explains why the
      conditional test is stable while OLS/Stambaugh are not

**Data used:** same VWNY/EWNY/log(DY) series as Issue 2.2, truncated at Dec 1994
**Model used:** shared estimator library (Issue 2.0); no new estimator work

---

### Issue 2.5 — Table 5 or 6: book-to-market or earnings-price ratio

- [ ] Pick B/M (Table 5) or E/P (Table 6) — recommend B/M first since it needs only
      book equity, not the operating-earnings line
- [ ] AR(1) regression for log(B/M) or log(E/P)
- [ ] Predictive regressions of nominal/excess VWNY/EWNY on the lagged log ratio,
      OLS / Stambaugh / ρ≈1, for June 1963–Dec 1994 (379 months) and June
      1963–Dec 2000 (451 months)

**Data used:** aggregate B/M or E/P from Issue 1's Compustat construction (or
flagged French-Library-portfolio approximation), plus VW/EW NYSE returns
**Model used:** shared estimator library (Issue 2.0), applied to the log(B/M) or
log(E/P) specification — identical mechanics to Table 2, different regressor

---

### Issue 2.6 — *(Optional)* Appendix: power simulation (Table A.1 / Fig. A.1)

- [ ] Monte Carlo simulation calibrated to the 1946–1972 VWNY/log(DY) OLS estimates,
      varying `b` ∈ {0, 0.4, 0.8, 1.2, 1.6} and `ρ` ∈ {0.999, …, 0.975}
- [ ] Rejection rates at 5% for Stambaugh, ρ≈1, and the joint Bonferroni test

**Data used:** none — pure simulation, parameters taken from Table 3's reported OLS
estimates for VWNY 1946–1972
**Model used:** shared estimator library (Issue 2.0), run on simulated rather than
real data — primarily a correctness check on the estimator code, not a replication
deliverable per se

---

## Issue 3 — Extension: extend the sample to the present

**Goal:** rerun Tables 1 and 2 (and ideally 3) through the most recent available month.

- [ ] Extend the master panel from 2000 through present (~2026) using the same
      construction as Issue 1
- [ ] Re-run Table 1 summary stats for the extended sample and a natural third
      subperiod (e.g., 2001–present)
- [ ] Re-run Table 2 (and Table 3, time permitting) on the extended sample
- [ ] Report how `ρ̂` (autocorrelation of log DY) has evolved since 2000 — this is
      the single number the paper's whole argument hinges on, so it's worth
      tracking explicitly
- [ ] Note regime-relevant events for interpretation: 2008 financial crisis,
      near-zero rate era (2009–2021), 2020 COVID crash/recovery, 2022 rate-hiking
      cycle, and the general rise of buybacks over dividends
- [ ] Write up: does dividend yield still predict returns out to today, and does
      the conditional test still dominate the unconditional one?

**Depends on:** Issues 1–2

---

## Issue 4 — Extension: interactive educational dashboard

**Goal:** a teaching tool that makes the paper's core argument tangible, centered on
a **Bayesian dial for ρ**.

- [ ] Dashboard shell showing, for a selected ratio (DY / B/M / E/P) and sample
      window: current level, `ρ̂`, and OLS / Stambaugh / ρ≈1 estimates + p-values
      side by side
- [ ] **Bayesian ρ dial** (core feature): a slider/control over the assumed prior
      for ρ, spanning at least:
  - point mass at ρ=1 (recreates the paper's conditional test exactly)
  - flat prior on ρ≤1
  - shifted-normal prior centered near ρ̂ with user-adjustable spread
  - live-updating posterior probability that b≤0, recreating the paper's own
    worked example (Section 4.1: 0.147 → 0.017 → 0.032 for nominal EWNY, 1946–72)
- [ ] Live "does the conditional test have power right now?" indicator — flags
      whether current `ρ̂` clears the paper's own rule-of-thumb threshold
      (≈0.98 monthly / 0.85 annual with 25 years of data; ≈0.99/0.90 with 50 years)
- [ ] Panel showing the 1995–2000-style sensitivity check: how much do OLS,
      Stambaugh, and ρ≈1 estimates move if you drop the last N years of data?
- [ ] *(Stretch)* small embedded version of the Table A.1 / Fig. A.1 power
      simulation, letting the user vary `b`, `ρ`, `T` and see rejection rates update

**Depends on:** Issues 1–3 (needs the full historical + extended panel and all
three estimators already working)

---

## Issue 5 — Extension: ML / alternative predictive model *(optional)*

**Goal:** stress-test whether a more flexible model changes the predictability
conclusion — lower priority, pursue only if time allows after Issues 1–4.

- [ ] Define an out-of-sample evaluation protocol (expanding or rolling window,
      historical-mean benchmark à la Goyal-Welch) — the paper is entirely in-sample,
      so this is the main gap an ML extension would fill
- [ ] Baseline: out-of-sample R² for OLS and ρ≈1-adjusted forecasts vs. the
      historical mean
- [ ] Candidate alternative model(s) — pick one or two, not all:
  - regularized linear (ridge/lasso) combining DY, B/M, E/P jointly instead of
    one-at-a-time
  - simple nonlinear model (e.g., random forest / gradient boosting) on the same
    predictors, primarily to check for nonlinearity/interaction effects rather than
    to chase raw predictive accuracy
- [ ] Compare in-sample significance (paper's framework) against out-of-sample
      performance (ML framework) — the interesting result either way is the
      *disagreement*, not just accuracy numbers
- [ ] Write up honestly: this is exploratory and secondary to the main replication;
      frame conclusions accordingly given known low signal-to-noise in return
      prediction

**Depends on:** Issues 1–3

---

## Issue 6 — Extension: cross-sectional application to other markets *(optional)*

**Goal:** test whether the paper's central claim — that the ρ≈1 conditional test
dominates Stambaugh's unconditional test whenever the predictor's autocorrelation is
close to one — generalizes outside NYSE/CRSP data, or is a US-specific artifact.

- [ ] Select a small set of non-US developed markets with long, clean dividend-yield
      histories (e.g., UK, Japan, Germany — MSCI or Datastream country indices;
      Global Financial Data is a paid alternative with longer history)
- [ ] Construct market-level DY (and B/M/E/P if data allows) for each market using
      the same trailing-12-month construction as Issue 1
- [ ] Report `ρ̂` per market first — this determines up front whether the conditional
      test can even be informative there (per the paper's own rule of thumb), before
      running any regressions
- [ ] Run the same three estimators (Issue 2.0 library — no new estimator code
      needed) market-by-market
- [ ] Compare: does the conditional test's power advantage over Stambaugh hold
      cross-market, or is it concentrated in markets/periods with unusually
      persistent dividend yields?
- [ ] *(Stretch)* pool markets into a panel version of the test, flagging that
      cross-sectional correlation of returns (not addressed in the original paper)
      would need its own treatment if pursued rigorously

**Data used:** MSCI/Datastream (or equivalent) country-level total- and price-return
indices for market-level dividend yield construction; local short-term rates for
excess returns where available
**Model used:** shared estimator library (Issue 2.0) — no new estimator work, applied
market-by-market; this is a data-acquisition and interpretation exercise more than a
modeling one
**Depends on:** Issue 2.0 (estimator library); independent of Issues 3–5

---

## Suggested order of work

1. Issue 1 (data) → QA gate on Table 1
2. Issue 2 (replication) → QA gate on Table 2 headline numbers
3. Issue 3 (extend sample) — cheap once 1–2 are solid
4. Issue 4 (dashboard) — the main deliverable beyond replication
5. Issue 5 (ML) — only if time remains
6. Issue 6 (cross-market) — only if time remains; can run in parallel with Issue 5
   since both depend only on Issue 2.0, not on each other
