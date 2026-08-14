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
