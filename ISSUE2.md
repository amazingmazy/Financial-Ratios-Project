# Issue 2 — Core Replication

Implementation of the README's Issue 2: runs the shared estimator library
(Issue 2.0, already built and validated) against the real, QA-gated master
panel from Issue 1, reproducing Tables 2, 3, 4, and 5.

## Files

```
src/estimators.py               Issue 2.0 -- OLS, Stambaugh (1999) Monte Carlo
                                 bias correction, Lewellen's rho~1 conditional
                                 test, modified-Bonferroni joint test
src/replicate_paper_tables.py   Issue 2.2-2.5 -- Tables 2, 3, 4, 5
tests/test_replicate_paper_tables.py   Plumbing tests (window filtering,
                                 output shape) on synthetic data
```

## Running it

```bash
python src/replicate_paper_tables.py data/master_panel.csv
```

This produces, in order:

- **Table 2** — dividend yield predicts VWNY/EWNY/ExcVWNY/ExcEWNY, full
  sample 1946-2000
- **Table 3a/3b** — same regressions, split at 1946-1972 / 1973-2000
- **Table 4a/4b** — sensitivity check: 1946-1994 vs. 1946-2000 (nominal
  returns only, per the paper), plus a printed summary of how the AR(1) ρ
  and OLS slope move between the two windows — this is the number the
  paper's whole argument about the 1995-2000 run-up hinges on
- **Table 5a/5b** — book-to-market predicts returns, June 1963-Dec 1994 and
  June 1963-Dec 2000
- **Table 6a/6b** — earnings-price ratio predicts returns, same two windows,
  identical machinery to Table 5 with `logE/P` swapped in for `logB/M`

Each table prints a `corr(e,m)` diagnostic alongside AR(1) ρ — the
correlation between return shocks and predictor-innovation shocks, which
drives both the Stambaugh correction's magnitude and the ρ≈1 test's
standard-error reduction. For DY this should land close to the paper's
reported -0.955; a value near zero was the symptom of a real bug (see
`docs/ISSUE1.md`'s eighth-round changelog) rather than a property of the
data. For B/M and E/P, expect it to run somewhat weaker than DY's, since
their "shocks" combine monthly price moves with much less frequent annual
accounting updates.

Each table prints in a layout matching the paper's own (AR(1) ρ, corr(e,m),
then one row per estimator — OLS, Stambaugh, ρ≈1 — each showing the
coefficient, standard error, and p-value), and all results are also written
to `output/issue2_tables_2_3_4_5_6.csv` in tidy long format for further
analysis or plotting.

**No year exclusion is applied to the B/M or E/P samples.** Per your
instruction, Tables 5 and 6 use the full 1963-2000 window as constructed in
Issue 1, including the 1963-1966 period whose data characteristics
(gradual Compustat book-equity coverage backfill) are documented in
`docs/ISSUE1.md` but not excluded here.

## Runtime

The Stambaugh correction is Monte Carlo (default 8,000 simulations per
regression); with 24 regressions total across all four tables, expect this
to take a few minutes. Use `--n-sims` to trade off speed vs. precision —
e.g. `--n-sims 2000` for a quick look, `--n-sims 20000` (the default used
elsewhere in this project) for final numbers.

## What's not yet built

- **Issue 2.1 (Table 1)** is effectively already covered by
  `src/qa_table1.py`, which computes the same summary statistics as part of
  the Issue 1 QA gate rather than as a separate deliverable here.
- **Issue 2.6 (Appendix power simulation, Table A.1 / Fig. A.1)** is
  optional per the README and not built.

## Tolerance tests against the published values

`src/paper_values.py` holds Lewellen's Tables 1, 2, 3, 5 and 6 as published,
plus the per-statistic tolerances; `tests/test_replication_vs_paper.py`
asserts our numbers against them. Every other test file in the project
checks the code against itself on synthetic data, which cannot say whether
the replication actually worked -- this one can.

Tolerances follow *propagation*, not statistic type. Only three quantities
reach the estimates: rho_hat, the dispersion of the log ratio, and
corr(e,m), since `b_adj = b_ols - gamma*(rho_hat - 1)` with
`gamma = cov(e,m)/var(m)`. Those are gated tightly; everything else in
Table 1 is descriptive and gated loosely. rho_hat gets abs 0.002 because
gamma runs near -90 for DY: a 0.01 error in rho_hat moves the bias-adjusted
slope by ~0.9 and flips the sign of the paper's headline result, so a
tolerance that looks tight next to a value of 0.997 is not tight at all.

## Changelog: first run against real CRSP data

Ran 1926-2024 via `run_issue1.py --source wrds` (SIZ schema). Table 1 QA
gate passes. 266 of the 320 published-value assertions pass.

**The dividend-yield replication is clean -- Tables 1, 2 and 3 have zero
failures.** Table 2 VWNY 1946-2000:

| | paper | ours |
|---|---|---|
| OLS b (s.e.) | 0.917 (0.476) | 0.978 (0.477) |
| rho~1 b (s.e.) | 0.663 (0.142) | 0.686 (0.150) |
| rho~1 t | 4.67 | 4.56 |
| corr(e,m) | -0.955 | -0.949 |

The OLS standard error agreeing to 0.001 is the strongest single signal
that the index and dividend-flow construction is right -- it depends on
var(logDY) and on the sample being the same 660 months.

## Changelog: autocorrelation estimator was the wrong one

`compute_stats` used `np.corrcoef`, which is a Pearson correlation.
Lewellen's are lag-k OLS slopes. Both share the numerator cov(x_t, x_{t-k});
the slope divides by var(x_{t-k}) alone where Pearson divides by
sd(x_t)*sd(x_{t-k}), and they agree only when the two subsamples have equal
variance -- which a near-unit-root series over a long window does not. On
logDY 1946-2000 this was 0.9915 (Pearson) against 0.997 published, while
`fit_ar1` in `estimators.py`, an actual OLS slope, already gave 0.9966.

Found by taking the rho24 of 1.062 in Table 1 seriously instead of
treating it as a typo. A correlation cannot exceed one, so the statistic
had to be something unbounded. That identifies the estimator rather than
discrediting the cell.

This **reverses the earlier decision to demote rho12 and rho24 to
`SOFT_STATS`**. The premise there was right -- 1.062 is impossible for a
correlation -- but the conclusion was wrong: an out-of-range value means
the estimator was misidentified, not that the number is unreproducible.
Recomputed as slopes, every published lag-12 and lag-24 cell lands within
0.02, the 1.062 included (we get 1.044). Both are gated now at abs 0.03,
just above the largest observed miss of 0.018. `skew` remains the only
informational statistic. This recovered about two dozen of the paper's
numbers that had been written off as uncheckable.

## Known gaps: Tables 5 and 6 run high

All 54 remaining failures are B/M or E/P. Over 1963-2000:

| | paper | ours |
|---|---|---|
| B/M mean | 53.13 | 58.69 |
| E/P mean | 20.02 | 21.03 |
| corr(e,m), B/M | -0.890 | -0.783 |

**Not a timing or alignment problem.** Both ratios move in 99.9% of months
and corr(e,m) stays strongly negative -- the near-zero signature that
exposed the `totval` lag bug (ISSUE1.md, eighth round) is absent. The
levels are simply high by ~10% (B/M) and ~5% (E/P), with persistence
slightly low.

Not the early-Compustat coverage ramp either, despite that being the
obvious suspect from ISSUE1.md's seventh round: trimming the start year
moves B/M's mean *further* from the paper (58.69 at 1963-06, 65.54 at
1975), because our 1960s decade mean of 33.14 is pulling the average down,
not up. The decade shape (1960s 33, 1970s 69, 1980s 77, 1990s 49) is
economically sensible for aggregate NYSE book-to-market.

**Decision: document rather than chase.** Per the instructor, we are to use
the whole Compustat dataset, so a divergence from whatever firm screen
Lewellen applied is expected rather than a defect -- the paper specifies
neither the exact universe nor the book-equity formula, and states only
that a firm needs three years of accounting data. A ~10% gap on an
aggregate ratio built from an unspecified screen is a defensible
replication difference, and the qualitative conclusions are unaffected.
The assignment requires Table 5 *or* 6, so Table 5 (B/M) is the designated
deliverable and Table 6 is carried as a robustness check.

If it is worth revisiting later, the candidates in order are: the
three-years-of-history screen in `pull_compustat_be_and_earnings`, the
preferred-stock preference order (pstkrv -> pstkl -> pstk), and whether the
numerator's firm set matches the `totval` denominator's.

## Known gaps: legacy CRSP tables are frozen at 2024

`wrds_pull.py` queries the SIZ schema (`crsp.msi`, `crsp.msf`,
`crsp.msenames`, `crsp.mcti`). Checked directly:

| table | schema | max date |
|---|---|---|
| `crsp.msi`, `crsp.msf`, `crsp.mcti` | SIZ | 2024-12-31 |
| `crsp.msf_v2` | CIZ | 2025-12-31 |
| `comp.funda` | Compustat | 2026-07-31 |

CIZ (Flat File 2.0) replaced SIZ in January 2025 and the legacy tables stop
a year short. This does not affect the replication -- 1946-2000 is well
inside the SIZ range -- but **Issue 3 cannot reach the present without
porting the pull**, and it would fail silently rather than error.

Mapping for when we do: `crsp.indmthseriesdata` carries `mthtotret`,
`mthprcret` and `mthtotval`, which map directly onto the
`vwretd`/`vwretx`/`totval` triple `construct_dividend_yield` expects;
`crsp.msf_v2` supplies `mthcap` and `mthprevcap` so market cap need not be
rebuilt from `abs(prc)*shrout`; `crsp.stksecurityinfohist` replaces
`crsp.msenames` (join on `secinfostartdt <= mthcaldt <= secinfoenddt`), with
`exchcd = 1` becoming `primaryexch = 'N'` and the shrcd filter becoming
`securitytype='EQTY' AND securitysubtype='COM' AND sharetype='NS' AND
usincflg='Y'`. Delisting returns are already inside `mthret`.

Note one consequence for tolerances: CIZ `mthret` compounds daily returns
with dividends reinvested on the ex-date, where legacy `ret` was a
month-to-month holding-period return reinvested at month end. Since the
dividend flow is reconstructed as `(vwretd - vwretx) * totval_{t-1}`, that
convention change propagates into DY, and the replication and the extension
will not sit on identical footing.

## Changelog: CIZ pull added, and which schema to use where

`pull_crsp_nyse_index_ciz` builds the same strict-NYSE index from
`crsp.msf_v2` and hands off to the same
`aggregate_security_level_to_monthly_index`, so the `totval` fix is shared
rather than duplicated. `crsp.msf_v2` carries the security descriptors
inline, so no `msenames`-style date-range join is needed. Reached via
`--index-source ciz`.

The risk-free rate had to move too, since `crsp.mcti` is frozen at the same
date. CRSP has no drop-in replacement: `crsp.tfz_mth_rf` carries the right
series (kytreasnox 2000001, 1-Month Nominal, 1925-2025) but publishes yields
to maturity, correlating only 0.977 with `t30ret`; `crsp.tfz_mth_bp` holds
returns but they are Fama bond portfolios and start in 1952. `ciz` therefore
takes RF from the Ken French library, validated against `t30ret` across the
948 overlapping months: correlation 0.9958, mean absolute difference
0.012pp, essentially all of it French printing two decimals (0.39 against
0.3907). Only excess returns depend on this.

**Both schemas run over 1946-2000, and SIZ reproduces the paper better:**

| | paper | SIZ | CIZ |
|---|---|---|---|
| assertions passing | -- | 216 | 204 |
| VWNY mean | 1.040 | 1.042 | 1.038 |
| logDY s.d. | 0.330 | 0.333 | 0.349 |
| Table 2 OLS b | 0.917 | 0.978 | 0.827 |
| Table 2 rho~1 t | 4.67 | 4.56 | 3.79 |
| corr(e,m) | -0.955 | -0.949 | -0.928 |

This is the return-compounding difference showing up where it was predicted
to, not a defect in either pull. **Decision: SIZ for the replication, CIZ
for the extension.** The replication should sit on the convention closest to
Lewellen's own, and the extension has no choice. The two are reported as
separate exercises rather than spliced into one series, and the convention
change is the reason -- worth a sentence in the write-up, since a reader will
otherwise wonder why the 1946-2000 numbers move between the two tables.

## Environment note

`requirements.txt` pins neither `pandas` nor a working `wrds` floor. A
fresh install today resolves `pandas` 3.0, which drops the raw DBAPI2
support `wrds` relies on, so every `raw_sql` call fails with
`'Connection' object has no attribute 'cursor'`; and a bare `wrds`
resolves to 3.1.6 rather than the `>=3.2.0` the file asks for. Working
combination is `pandas<3` with `wrds>=3.5`. `requests`, `streamlit` and
`plotly` are imported by the code but missing from the file entirely.
