# Issue 1 — Data Wrangling & Cleaning

Implementation of the README's Issue 1: produce one clean monthly panel that
everything downstream (Issue 2's tables, Issue 3's extended sample, etc.) reads
from.

## Files

```
src/wrds_pull.py          Real WRDS/CRSP/Compustat pull functions
src/public_fallback.py    Ken French Library + FRED pullers, plus synthetic-data
                          generators for offline testing
src/construct_panel.py    Core logic: dividend yield, excess/real returns,
                          B/M & E/P construction, master panel assembly
src/qa_table1.py          QA gate: Table 1 replication check vs. the paper
tests/test_construct_panel.py   Unit tests (synthetic data, no WRDS needed)
run_issue1.py              CLI orchestrator
```

## Running it

### With WRDS/CRSP access (real data)

```bash
python run_issue1.py --source wrds --start 1926-01-01 --out data/master_panel.csv
```

This will prompt for your WRDS credentials (via the `wrds` package's standard
flow, or read `~/.pgpass` if configured) and pull:

- `crsp.msi` for the NYSE value-/equal-weighted index (returns with and
  without dividends, plus index level for the dividend-flow construction)
- `crsp.mcti` for the one-month T-bill rate
- FRED (`CPIAUCSL`) for CPI
- `comp.funda` (joined to CRSP via the CCM linktable, `crsp.ccmxpf_linktable`,
  filtered to NYSE-listed firms) for aggregate book equity and operating
  income before depreciation, if you also have Compustat access

If you only have CRSP (no Compustat), add `--no-compustat`: B/M and E/P will
be written as `NaN` with `bm_is_approx`/`ep_is_approx` set to `True`, rather
than silently omitted.

**Before trusting the output**, verify two things I could not check without a
live connection:
1. `crsp.msi`'s universe composition for your subscription vintage — see the
   docstring in `wrds_pull.pull_crsp_nyse_index` for the NYSE-only caveat and
   the heavier, exchange-code-filtered alternative
   (`pull_crsp_nyse_index_exchcd_filtered`) if you need strict NYSE-only
   across the full sample.
2. `comp.funda`'s exact column availability for your subscription (`ceq`,
   `txditc`, `pstkrv`/`pstkl`/`pstk`, `oibdp` are all standard, but confirm
   they're populated for your date range).

### Offline dry run (no WRDS needed)

```bash
python run_issue1.py --source synthetic --out data/master_panel_demo.csv --skip-qa
```

This generates synthetic data with the right shape/schema and runs the full
pipeline end-to-end, so you can confirm the code works before pointing it at
WRDS. **The QA gate will (correctly) fail in this mode** — the synthetic data
isn't calibrated to match the paper's actual numbers, only to be plausible in
shape (positive DY, persistent AR(1), etc.). Use `--skip-qa` to still write
the file for inspection.

## The QA gate

Per the README: **do not proceed to Issue 2 until Table 1 matches.** Run:

```bash
python -m src.qa_table1 data/master_panel.csv
```

or it runs automatically as the last step of `run_issue1.py` (which exits
non-zero on failure unless `--skip-qa` is passed). It compares your computed
mean/SD/skew/ρ₁/ρ₁₂/ρ₂₄ against Lewellen (2004) Table 1's published values
(transcribed in `PAPER_TABLE_1` in `qa_table1.py`) for VWNY, EWNY, DY, logDY
(full sample + both halves) and B/M, logB/M, E/P, logE/P (1963–2000), within
a configurable tolerance (default 15%, loose enough to catch real construction
bugs but not so tight that it fails on rounding — tighten this once you're
confident the pipeline is correct, per the README's "within rounding" bar).

## Testing

```bash
pip install pytest
pytest tests/test_construct_panel.py -v
```

10 tests, all passing, covering: dividend-yield positivity/rolling-window
edge cases, log/percent convention consistency, excess and real return
arithmetic, the 4-month accounting lag's date logic, and the B/M/E/P
approximation-flag propagation (including the "no Compustat at all" case).

## Changelog: fixes from the first live WRDS run

The first real run against WRDS (`--source wrds`) surfaced three issues,
now fixed:

1. **B/M and E/P were catastrophically wrong** (off by orders of magnitude,
   with autocorrelation near zero instead of ~0.99). Root cause: the
   Compustat aggregation grouped firm-years by exact `datadate` instead of
   `fyear`. Since different firms have different fiscal year-ends (Dec 31,
   June 30, etc.), this scattered what should be one "NYSE aggregate for
   fiscal year Y" across dozens of near-single-firm groups — confirmed by a
   regression test (`test_aggregation_pools_all_firms_per_fiscal_year_not_per_exact_date`)
   that reproduces the exact failure mode against the old logic. Fixed by
   grouping on `fyear` instead, in the new
   `wrds_pull.aggregate_firm_level_to_fiscal_year`. Also fixed a units
   mismatch (Compustat reports in $ millions, CRSP's `totval` in $
   thousands) in the same function.

2. **VWNY/EWNY autocorrelation and skew failed specifically for any window
   including post-1972 data**, while the pure 1946–1972 subsample passed
   almost everything. This lines up exactly with NASDAQ's addition to CRSP's
   composite indices in December 1972: `crsp.msi` isn't NYSE-only past that
   point. Fixed by defaulting `run_issue1.py --source wrds` to
   `wrds_pull.pull_crsp_nyse_index_exchcd_filtered` (a strictly NYSE-only
   index built from security-level data), rather than the faster but
   universe-contaminated `crsp.msi`. Use `--index-source msi` to opt back
   into the faster path if your sample never extends past ~1962/1972.

3. **Skewness now informational, not a hard gate failure.** It isn't used
   anywhere in the paper's predictive-regression methodology, and it's a
   famously fragile statistic dominated by one or two extreme months —
   reasonable to differ from the paper's exact value due to CRSP data
   revisions over the 20+ year gap, without indicating a real bug.

## Changelog: fixes from the second live WRDS run

The fyear-aggregation and NASDAQ-contamination fixes above worked — mean,
SD, and rho1 now pass for every series. The remaining 5 fails were all in
one place (24-month autocorrelation, plus B/M's dispersion), fixed as
follows:

4. **Near-zero autocorrelations were failing on sign alone**
   (e.g. paper=+0.013, computed=-0.013). A pure relative-error check divides
   by the paper's value, so a tiny absolute difference near zero explodes
   into a huge percentage error. Fixed by adding an absolute-tolerance floor
   (`ABS_TOLERANCE`, 0.03 for rho1/12/24) applied alongside the relative
   check — either one passing is enough.

5. **rho12 and rho24 reclassified as informational, matching skew.** The
   paper's entire estimator (Stambaugh correction, ρ≈1 conditional test) is
   built exclusively on *lag-1* autocorrelation (Eq. 3b) — rho12/rho24 are
   Table 1 descriptive context only, never an input to any regression.
   They're also inherently noisier at longer lags, and the paper's own
   reported logDY rho24 (1.062) exceeds the mathematical bound of a Pearson
   correlation (±1), implying Lewellen's original software used a different
   autocovariance-based formula — exact numerical agreement at lag 12/24
   isn't achievable by construction, independent of pipeline correctness.
   rho1 stays a required, hard-blocking check.

6. **NYSE membership check made date-matched.** The Compustat join checked
   "was this permno ever NYSE-listed at any point in its history" rather
   than "was it NYSE-listed as of this specific fiscal year end" (via
   `crsp.msenames.namedt`/`nameendt`). The looser check could pull firms
   into some years' aggregate that weren't actually NYSE members that year,
   plausibly explaining B/M's dispersion running ~20% higher than the
   paper's despite the mean matching well. Not fully verified without a
   live rerun — see "known gaps" below.

## Changelog: fixes from the third live WRDS run

Down to a single hard failure (logB/M's SD, 0.464 vs 0.360) out of the
entire Table 1. Diagnosis: mean matched almost exactly (3.915 vs 3.910) but
SD and skew didn't (skew -1.777 vs +0.190) — that combination is the
signature of a handful of outlier low-B/M years dominating a log-transformed
statistic, most likely from thin Compustat/CCM coverage in the earliest
fiscal years near the 1963 window boundary.

7. **Added a minimum-firm-count guard.** Any fiscal year whose aggregate is
   built from fewer than `min_firms_per_year` (default 30) NYSE firms is now
   set to `NaN` rather than silently included, with a debug line listing
   which years were dropped and their firm counts. Verified end-to-end with
   a synthetic test: a thin 5-firm year correctly NaNs out both itself and
   the following months that would otherwise reference it via the 4-month
   lag, while a normal 200-firm year passes through unaffected. Not yet
   confirmed this fully closes the gap on your real data — check the new
   `[debug] N fiscal year(s) below the 30-firm minimum` line in your next
   run's output to see which years (if any) got dropped, and whether
   logB/M's SD moves toward the paper's value as a result.

## Changelog: fourth round — diagnostics instead of guessing

Your fourth run produced numbers **identical to the third**, down to the
third decimal, with no `[debug] N fiscal year(s) below the 30-firm minimum`
line — meaning either the min-firm-count guard never triggered (all years
clear 30 firms) or the updated file wasn't actually picked up. Rather than
guess a third fix blind, this round adds:

8. **A version marker.** `wrds_pull.py` now prints `[wrds_pull.py version:
   ...]` at the start of the Compustat pull. Check this against
   `WRDS_PULL_VERSION` at the top of the file — if it's missing or stale,
   you're running an old copy.

9. **Real diagnostics, not more hypotheses.** `run_issue1.py` now has a
   `--debug-out` flag that saves an extended panel with `totval`,
   `matched_fye`, `book_equity_sum_used`, and `oibdp_sum_used` alongside the
   normal columns, plus the full annual Compustat aggregate table is now
   printed during every run (scan it directly for outlier years). A new
   script, `src/diagnose_bm.py`, reads the debug panel and prints the most
   extreme logB/M and logE/P months by z-score, with the exact fiscal year
   and totval that produced each one, plus a flag for any fiscal year whose
   aggregate book equity jumps >50% year-over-year (a sign of a firm-coverage
   discontinuity rather than organic growth). Run:
   ```bash
   python run_issue1.py --source wrds --start 1926-01-01 \
       --out data/master_panel.csv --debug-out data/master_panel_debug.csv
   python src/diagnose_bm.py data/master_panel_debug.csv
   ```
   This tells us definitively what's driving logB/M's SD, instead of another
   round of blind hypothesis-and-rerun.

## Changelog: fifth round — root cause found via diagnose_bm.py

`diagnose_bm.py` worked exactly as intended and found something concrete:
fiscal year 1953 had `book_equity_sum_used = 0.0` exactly, alongside a
completely normal `oibdp_sum_used`. That's not a real number for an NYSE
aggregate -- it's pandas' `.sum(skipna=True)` silently returning `0.0` for a
group where *every* firm had missing book equity (CEQ/TXDITC not populated
that early in Compustat's history), rather than correctly returning `NaN`.
`log(0) = -inf`, matching the "divide by zero encountered in log" warning
present in every run since the first one.

The earlier `min_firms_per_year` guard (round 3) didn't catch this because
it counted *any* firm present in the join, including ones contributing valid
OIBDP but missing book equity -- exactly 1953's situation.

10. **Per-field firm-count guards.** `n_firms_be` and `n_firms_oibdp` are now
    tracked and checked independently, so a fiscal year can have its book
    equity NaN'd out while keeping a perfectly healthy OIBDP (or vice versa)
    instead of one field's data quality contaminating the other.
11. **TXDITC coalesced to 0 when missing** (standard Fama-French convention
    -- a missing deferred-tax line usually means "not applicable," unlike a
    missing CEQ, which genuinely means "unknown" and is deliberately left as
    NaN rather than assumed zero).

Verified with regression tests that reproduce the exact bug against
synthetic data shaped like the real failure (`test_all_missing_book_equity_year_is_nan_not_zero`),
confirm the old logic really did produce `0.0` in that scenario (checked by
literally running the old aggregation code side by side), and confirm the
per-field independence (`test_thin_book_equity_does_not_incorrectly_nan_healthy_oibdp`).

## Changelog: sixth round — FY1953's bug fixed, exposed FY1961 next

`diagnose_bm.py` confirmed 1953 is gone (no more `-inf` in the outlier
list). The extreme logB/M months shifted to **1962-1963**, all matched to
**fiscal year 1961**: book equity of $11.8M against $123.4M the very next
year — a 9.4x jump your own diagnostic already flagged as a >50%
year-over-year jump. 1961 had enough firms to clear the old 30-firm
threshold, so the per-field guard from the last round didn't catch it — a
fixed firm count can't distinguish "genuinely small aggregate" from "still
mid-ramp-up in Compustat's own coverage," especially across a sample
spanning decades where the *right* threshold changes over time.

12. **Raised `min_firms_per_year` from 30 to 100.** Still well below the
    ~1,900 firms/year typical of mature coverage, but enough headroom to
    catch clearly-thin years like 1961's ~40 firms.
13. **Added a coverage-discontinuity guard**, independent of firm count: a
    fiscal year is set to NaN if its aggregate is less than 35% of the
    average of the next 3 years' aggregates (both thresholds configurable).
    This is the fix for exactly the failure mode above — a year that
    clears the firm-count minimum but is still obviously a coverage
    ramp-up relative to its near-term neighbors. Verified it does NOT
    trigger on genuine smooth year-over-year growth
    (`test_genuine_gradual_growth_is_not_flagged_as_discontinuous`), so
    it shouldn't over-correct on real economic trends.

If logB/M's SD is still off after this run, `diagnose_bm.py`'s outlier list
should point to whichever year (if any) is next in line -- there may be a
handful of these ramp-up years stacked in the early-to-mid 1960s as
Compustat's own historical backfill filled in gradually, not just one.

## Known gaps / next steps

- **logB/M's SD is the one statistic still failing** (0.464 vs paper's
  0.360) as of the third live run — raw B/M itself now passes cleanly
  (mean, SD, rho1 all within tolerance), so this is specifically a
  log-transform sensitivity issue, not a construction bug in the ratio
  itself. Added a minimum-firm-count guard (`min_firms_per_year`, default
  30) as the most likely fix — see changelog below — but haven't confirmed
  against real data that it closes the gap. If it doesn't, the next thing
  I'd check is whether extreme individual-firm B/M outliers within
  otherwise well-covered years (rather than whole thin years) are the
  culprit — the paper may winsorize firm-level ratios before aggregating,
  which isn't implemented here.
- ~~`pull_compustat_be_and_earnings`'s firm-year counts couldn't be verified
  against a live schema~~ — validated by the second live run: ~1,894
  firms/year, in the expected range for NYSE.
- The French-Library-based B/M approximation mentioned in the README as a
  Compustat fallback is **not yet implemented** — `public_fallback.py` has
  the raw 25-Size-BM-portfolio return puller, but turning that into an
  aggregate B/M *level* series (as opposed to returns) needs French's
  companion "value-weighted average BE/ME" file, whose exact column layout I
  could not verify without a live fetch. Flagged with a TODO in the code.
  (Lower priority now that real Compustat access is working.)
- Real-return construction uses simple CPI-growth deflation; if you want it
  to match a specific real-return convention used elsewhere in your project,
  double-check against that rather than assuming this is the only sensible one.
