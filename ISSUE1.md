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

## Known gaps / next steps

- `pull_compustat_be_and_earnings` implements the standard Fama-French book
  equity construction and the standard CCM-link join, but I could not test it
  against a live schema — check firm-year counts against a known benchmark
  (e.g. compare `n_firms` to published NYSE listing counts) before trusting
  the aggregate B/M/E/P series.
- The French-Library-based B/M approximation mentioned in the README as a
  Compustat fallback is **not yet implemented** — `public_fallback.py` has
  the raw 25-Size-BM-portfolio return puller, but turning that into an
  aggregate B/M *level* series (as opposed to returns) needs French's
  companion "value-weighted average BE/ME" file, whose exact column layout I
  could not verify without a live fetch. Flagged with a TODO in the code.
- Real-return construction uses simple CPI-growth deflation; if you want it
  to match a specific real-return convention used elsewhere in your project,
  double-check against that rather than assuming this is the only sensible one.
