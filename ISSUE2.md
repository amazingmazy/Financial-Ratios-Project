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
python src/replicate_paper_tables.py _data/master_panel.csv
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
`ISSUE1.md`'s eighth-round changelog) rather than a property of the
data. For B/M and E/P, expect it to run somewhat weaker than DY's, since
their "shocks" combine monthly price moves with much less frequent annual
accounting updates.

Each table prints in a layout matching the paper's own (AR(1) ρ, corr(e,m),
then one row per estimator — OLS, Stambaugh, ρ≈1 — each showing the
coefficient, standard error, and p-value), and all results are also written
to `_output/issue2_tables_2_3_4_5_6.csv` in tidy long format for further
analysis or plotting.

**No year exclusion is applied to the B/M or E/P samples.** Per your
instruction, Tables 5 and 6 use the full 1963-2000 window as constructed in
Issue 1, including the 1963-1966 period whose data characteristics
(gradual Compustat book-equity coverage backfill) are documented in
`ISSUE1.md` but not excluded here.

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

## Changelog: Tables 5 and 6 -- it was the book-equity formula

Tables 5 and 6 initially ran high: B/M averaged 58.69 over 1963-2000 against
the paper's 53.13 (+10.5%) and E/P 21.03 against 20.02 (+5.0%). Tracked down
to deferred taxes.

We had been computing book equity the Fama-French way, `CEQ + TXDITC -
preferred stock`. TXDITC -- deferred taxes and investment tax credit -- turns
out to be 12.1% of our aggregate book equity, and dropping it closes almost
the whole gap:

| aggregate B/M, 1963-2000 mean | value | vs paper |
|---|---|---|
| paper | 53.13 | -- |
| `CEQ + TXDITC - preferred` | 58.69 | +10.5% |
| `CEQ - preferred` | 52.13 | -1.9% |

Three things say this is the cause rather than a coincidence:

1. **It cannot touch E/P.** That numerator is OIBDP, an income-statement flow
   with no deferred-tax component -- which is exactly why E/P diverged only
   about half as much as B/M. A cause acting on the firm set would have hit
   both roughly equally; one acting on the book-equity formula hits only B/M.
   The asymmetry was the clue.
2. **The decade pattern fits.** The gap is widest in the 1970s-80s (decade
   means 69 and 77, against 63 and 66 once TXDITC is removed), precisely when
   accelerated depreciation and high inflation made deferred taxes largest.
3. **The convention is genuinely unsettled.** The paper says only "the ratio
   of book equity to market equity" and never gives a formula, and Ken
   French's own site documents that the deferred-tax treatment changed after
   FASB 109 -- so it is not stable even within the Fama-French lineage.

Exposed as `pull_compustat_be_and_earnings(include_deferred_taxes=...)` and
`run_issue1.py --include-deferred-taxes`, defaulting to **False**, the variant
that reproduces the paper. Neither setting is "right": this is a definitional
choice the source paper leaves open, so it is a flag rather than a hard-coded
constant, and the choice travels with whichever panel was built.

### It fixes the level, not the persistence

Worth being precise about what the adjustment does and does not buy, since the
headline improvement in the mean oversells it:

| B/M, 1963-2000 | paper | with TXDITC | without |
|---|---|---|---|
| mean | 53.13 | 58.63 | **52.08** |
| s.d. | 18.28 | 21.00 | **17.77** |
| skew | 0.39 | 0.31 | **0.38** |
| rho1 | 0.990 | 0.986 | 0.984 |
| log B/M rho1 | 0.995 | 0.989 | 0.986 |
| log B/M rho24 | 0.923 | 0.817 | 0.746 |

Level, dispersion and shape all improve markedly. Persistence gets slightly
*worse*, which makes mechanical sense: deferred taxes accumulate gradually, so
removing them strips a smooth component out of book equity and leaves the
series noisier year to year.

The gap in persistence is present under **both** definitions (rho1 of 0.989
against 0.986, versus the paper's 0.995), so it is not caused by this choice --
it belongs to the residual below. It deserves flagging anyway, because rho is
the load-bearing statistic for this paper: `b_adj = b_ols - gamma*(rho_hat-1)`,
so an error in rho propagates into every Table 5 estimate in a way an error in
the level does not. Across the published-value assertions the net effect of the
adjustment was +6 cells matching (53 xfail / 51 xpass before, 47 / 57 after);
B/M now matches on 42% of its cells against E/P's 67%, and the difference is
concentrated in the autocorrelation rows.

### What was ruled out on the way

- **CRSP/Compustat link duplication.** The leading suspect, and wrong: the
  join returns 58,392 firm-year rows and 58,392 distinct `(gvkey, fyear)`
  pairs. No double-counting from dual share classes.
- **Timing or alignment.** Both ratios move in 99.9% of months and corr(e,m)
  stays strongly negative -- the near-zero signature that exposed the `totval`
  lag bug (ISSUE1.md, eighth round) is absent.
- **The early-Compustat coverage ramp**, despite being the obvious suspect
  from ISSUE1.md's seventh round. Trimming the start year moves B/M *further*
  from the paper (58.69 at 1963-06, 65.54 at 1975), because our 1960s decade
  mean is pulling the average down rather than up.

### The residual

About 5% remains after the adjustment, and it shows up in E/P too, so it is
common to both ratios rather than specific to book equity. That is the
firm-screen bundle: the "three years of accounting data" requirement the paper
states but never operationalises, whether the numerator's Compustat-matched
firm set should match the `totval` denominator's full CRSP NYSE universe
(it does not, which biases our ratio *down* relative to a true aggregate),
aggregating-then-lagging versus lagging-then-aggregating for the ~1/3 of firms
with non-December fiscal year ends, and 20+ years of Compustat restatement
since the paper's extract. Not chased -- per the instructor we use the whole
Compustat dataset, so some divergence from Lewellen's screen is expected
rather than a defect.

The assignment requires Table 5 *or* 6, so Table 5 (B/M) is the designated
deliverable and Table 6 is carried as a robustness check.

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

## Issue 3 — extending the sample

`replicate_paper_tables.py --extended` adds two windows, with the end date
read from the panel rather than hard-coded, since the two schemas reach
different dates and a hard-coded end would silently truncate one of them.
`--tag` suffixes the output CSV so both schemas can be run side by side.

```bash
python src/replicate_paper_tables.py _data/master_panel.csv     --extended --tag siz
python src/replicate_paper_tables.py _data/master_panel_ciz.csv --extended --tag ciz
# matched window, for the schema comparison below
python src/replicate_paper_tables.py _data/master_panel_ciz.csv --extended \
    --end 2024-12-31 --tag ciz2024
```

All numbers below are from these commands at `--n-sims 8000`. The Stambaugh
row is Monte Carlo, so it moves with the seed: at 2,000 draws two seeds gave
0.2508 and 0.2851 on the same data, a spread of 0.034. `run_all` now takes a
`random_state` and threads it into both simulations, which it previously did
not -- the driver and an ad-hoc script disagreed by ~0.03 on identical inputs
because each silently used its own default seed. Quote Stambaugh values to
two decimals at most, and re-run at higher `--n-sims` for anything reported.

**The paper's conclusion survives on the full extended sample.** For
1946-2025 (CIZ), VWNY: OLS b = 0.862 (p = 0.005), Stambaugh b = 0.411
(p = 0.186), rho~1 b = 0.413 (t = 3.50, p = 0.000). The pattern Lewellen
built the paper around -- Stambaugh failing to reject where the conditional
test rejects decisively -- holds twenty-five years past his cutoff.

**The post-2000 subsample is where it gets interesting, and it is not a
counterexample.** For 2001-2025, DY's autocorrelation falls to 0.9515,
*below the paper's own stated threshold* of roughly 0.99 for 25 years of
monthly data (Section 2.4). The conditional test duly collapses: rho~1
b = -0.590, t = -1.28, p = 0.899, while OLS reports b = 3.862 (p = 0.006).
That is exactly the behaviour Table A.1 tabulates -- the conditional test's
power drops toward zero as rho falls away from one, and the estimate is
biased downward when rho is truly below one. So the extension demonstrates
the paper's stated limitation rather than contradicting its result, and the
rho column is the number to read first. The driver prints the rule-of-thumb
check alongside the table for that reason.

Worth noting the OLS slope more than quadruples post-2000 (0.862 -> 3.862)
while its p-value stays around 0.006. Taken alone that reads as *stronger*
predictability; it is the small-sample bias the paper is about, unmasked by
a shorter sample and a less persistent regressor.

## Robustness: the schema choice does not move any conclusion

Both panels truncated at 2024-12-31 so the windows match exactly (identical
T), making this apples-to-apples rather than a comparison contaminated by
CIZ's extra year:

| window | series | | SIZ | CIZ |
|---|---|---|---|---|
| 1946-2024 | VWNY | rho | 0.9930 | 0.9948 |
| | | corr(e,m) | -0.955 | -0.937 |
| | | OLS b (p) | 1.079 (0.002) | 0.884 (0.005) |
| | | Stambaugh b (p) | 0.634 (0.120) | 0.425 (0.183) |
| | | rho~1 b (t) | 0.460 (4.11) | 0.428 (3.58) |
| | EWNY | OLS b (p) | 1.218 (0.003) | 1.006 (0.010) |
| | | rho~1 b (t) | 0.534 (2.59) | 0.469 (2.16) |
| 2001-2024 | VWNY | rho | 0.9485 | 0.9502 |
| | | corr(e,m) | -0.968 | -0.957 |
| | | OLS b (p) | 3.685 (0.010) | 4.023 (0.006) |
| | | Stambaugh b (p) | 2.704 (0.084) | 3.018 (0.070) |
| | | rho~1 b (t) | -1.008 (-2.54) | -0.563 (-1.20) |
| | EWNY | OLS b (p) | 4.159 (0.014) | 5.222 (0.010) |
| | | rho~1 b (t) | -1.252 (-1.69) | -0.847 (-0.84) |

Slope magnitudes differ by up to ~25% on the short window -- the compounding
convention again, amplified where the sample is small -- but rho agrees to
within 0.002 everywhere and **every inference is identical**: OLS
significant, Stambaugh not, and the conditional test decisively significant
over the full sample and negative and insignificant after 2000.

Two things this buys us. First, **the post-2000 collapse is not an artifact
of the port**: SIZ shows it more starkly than CIZ (rho~1 b of -1.008 against
-0.563 for VWNY), so the legacy schema, if anything, strengthens the
finding. Second, **2025 is not driving anything** -- extending CIZ from
2001-2024 to 2001-2025 moves rho from 0.9502 to 0.9515 and the conditional
slope from -0.563 to -0.590. The result is about the 2000s, not about the
most recent year.

The honest caveat: the schemas agree on every *sign and significance* call,
but not closely on magnitudes, and neither is "right." Report the
replication on SIZ and the extension on CIZ, state that both were run, and
do not present a slope from one as comparable to a slope from the other at
two decimal places.

## Environment

`conda env create -f environment.yml && conda activate financial-ratios`, or
`pip install -r requirements.txt` directly. `environment.yml` installs
`requirements.txt` rather than restating it, so conda and pip cannot resolve
different versions of anything.

**Correction to an earlier note here.** This section previously claimed that
`requirements.txt` was broken -- that a fresh install resolved pandas 3.0 and
every `raw_sql` call then failed with `'Connection' object has no attribute
'cursor'`. That was wrong, and the error was ours rather than the file's. A
clean `pip install -r requirements.txt` has always resolved pandas 2.2.3 with
wrds 3.5.0, and the WRDS client works from it; verified by building an empty
environment from the unmodified file, querying `crsp.msi`, and running the
suite (216 passed). What actually happened is that packages were installed
ad-hoc without the file, a bare `wrds` resolved to the 2023-era 3.1.6 -- which
passes a raw DBAPI2 connection to `pandas.read_sql_query`, unsupported since
pandas 3 -- and the resulting breakage was attributed to `requirements.txt`.

The `wrds>=3.5` floor now in the file is still worth keeping, for exactly that
reason: it is what stops a resolver from satisfying a newer pandas by reaching
back to an ancient wrds. wrds 3.5 constrains pandas to `>=2.2,<2.3` itself, so
pandas deliberately carries no separate pin -- pinning both invites a conflict
without adding a guarantee.

The file did have two real problems, both fixed: `streamlit` was absent, so the
dashboard could not run and `test_dashboard.py` silently skipped; and roughly
ten unused packages were carried along, the FastAPI stack among them, from the
same unrelated template that left a `Makefile` and `Dockerfile` pointing at an
`app/` directory this project has never had (both since deleted, along with
`main.py`, a uv hello-world stub). After the rewrite the suite runs 217 passed,
0 skipped from a clean environment.
