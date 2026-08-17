"""
run_issue1.py
=============
Orchestrator for Issue 1 (data wrangling & cleaning). Pulls data from either
a live WRDS connection or the synthetic/public fallback, assembles the master
panel, writes it to disk, and runs the Table 1 QA gate -- refusing to
silently "succeed" if the gate fails, per the README's explicit instruction
not to proceed to Issue 2 until Table 1 matches.

Usage:
    # Real run, with WRDS/CRSP (and optionally Compustat) access:
    python run_issue1.py --source wrds --out _data/master_panel.csv

    # Offline dry run (synthetic data; validates the pipeline, not real numbers):
    python run_issue1.py --source synthetic --out _data/master_panel_demo.csv
"""

from __future__ import annotations
import argparse
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from construct_panel import assemble_master_panel
from qa_table1 import qa_gate
from settings import config

# Defaults come from settings.py (which reads .env, then falls back to its own
# defaults); CLI arguments still override them. One place to change a path or a
# date, and it is the same place the rest of the project reads from.
DATA_DIR = config("DATA_DIR")
START_DATE = config("START_DATE").strftime("%Y-%m-%d")


def run_wrds(start: str, end: str | None, compustat: bool, index_source: str = "exchcd_filtered",
             keep_diagnostics: bool = False, include_deferred_taxes: bool = False) -> "pd.DataFrame":
    import wrds
    import wrds_pull

    # Read the username from the environment when it is available so the pull
    # can run unattended. A bare wrds.Connection() falls back to an interactive
    # input() prompt, which raises EOFError under any non-interactive runner --
    # cron, CI, or a doit task -- and makes end-to-end automation impossible.
    # The password is never handled here: wrds resolves it from ~/.pgpass (on
    # Windows, %APPDATA%/postgresql/pgpass.conf), which lives outside the repo.
    wrds_username = os.environ.get("WRDS_USERNAME")
    if wrds_username:
        print(f"Connecting to WRDS as {wrds_username}...")
        db = wrds.Connection(wrds_username=wrds_username)
    else:
        print("Connecting to WRDS (set WRDS_USERNAME to avoid the prompt)...")
        db = wrds.Connection()
    try:
        if index_source == "ciz":
            print("Pulling strictly NYSE-only index (crsp.msf_v2, CIZ schema)... "
                  "required for any sample past 2024-12-31, where the legacy SIZ "
                  "tables stop -- see wrds_pull.pull_crsp_nyse_index_ciz.")
            crsp_index = wrds_pull.pull_crsp_nyse_index_ciz(db, start=start, end=end)
        elif index_source == "exchcd_filtered":
            print("Pulling strictly NYSE-only index (crsp.msf, exchcd==1)... "
                  "this is slower than crsp.msi but avoids AMEX/NASDAQ contamination "
                  "post-1962/1972 -- see wrds_pull.py docstring.")
            crsp_index = wrds_pull.pull_crsp_nyse_index_exchcd_filtered(db, start=start, end=end)
        else:
            print("Pulling CRSP NYSE index (crsp.msi)... "
                  "NOTE: crsp.msi's universe drifts to include AMEX (~1962) and "
                  "NASDAQ (~1972) -- use --index-source exchcd_filtered if your "
                  "sample extends past those dates and you need strict NYSE-only.")
            crsp_index = wrds_pull.pull_crsp_nyse_index(db, start=start, end=end)
        print(f"  -> {len(crsp_index)} months "
              f"({crsp_index.index.min().date()} .. {crsp_index.index.max().date()})")

        # crsp.mcti is frozen at 2024-12-31, so a CIZ-era pull would silently
        # lose the risk-free rate for exactly the months the extension is about.
        if index_source == "ciz":
            print("Pulling risk-free rate from the Ken French Data Library "
                  "(crsp.mcti is frozen at 2024-12-31)...")
            rf = wrds_pull.pull_riskfree_french(start=start, end=end)
        else:
            print("Pulling CRSP risk-free rate (crsp.mcti)...")
            rf = wrds_pull.pull_crsp_riskfree(db, start=start, end=end)
        print(f"  -> {len(rf)} months "
              f"({rf.index.min().date()} .. {rf.index.max().date()})")

        print("Pulling CPI from FRED...")
        from public_fallback import pull_fred_cpi
        cpi = pull_fred_cpi(start=start)
        print(f"  -> {len(cpi)} months")

        compustat_annual, is_approx = None, False
        if compustat:
            print("Pulling Compustat book equity & operating earnings (comp.funda, via CCM link)...")
            compustat_annual = wrds_pull.pull_compustat_be_and_earnings(
                db, start=start, end=end,
                include_deferred_taxes=include_deferred_taxes)
            print(f"  -> {compustat_annual['n_firms'].sum()} firm-years aggregated to "
                  f"{len(compustat_annual)} fiscal year-ends")
            print("\n  Annual aggregate table (scan for outlier years -- e.g. book_equity_sum\n"
                  "  or n_firms wildly different from neighboring years):")
            with pd.option_context("display.max_rows", None, "display.width", 120):
                print(compustat_annual.to_string())
            print()
        else:
            print("Skipping Compustat (--no-compustat) -- B/M and E/P will be all-NaN, flagged accordingly.")

        panel = assemble_master_panel(crsp_index, rf, cpi,
                                       compustat_annual=compustat_annual,
                                       bm_ep_is_approximation=is_approx,
                                       keep_diagnostics=keep_diagnostics)
        return panel
    finally:
        db.close()


def run_synthetic(n_months: int, start: str, keep_diagnostics: bool = False) -> "pd.DataFrame":
    from public_fallback import (
        synthetic_crsp_index, synthetic_riskfree, synthetic_cpi,
        synthetic_compustat_be_earnings,
    )
    print("Generating synthetic data (offline dry run -- NOT real replication numbers)...")
    crsp_index = synthetic_crsp_index(n_months=n_months, start=start)
    rf = synthetic_riskfree(n_months=n_months, start=start)
    cpi = synthetic_cpi(n_months=n_months, start=start)
    compustat_annual = synthetic_compustat_be_earnings(
        n_years=n_months // 12 + 1, start_year=int(start[:4]))
    panel = assemble_master_panel(crsp_index, rf, cpi,
                                   compustat_annual=compustat_annual,
                                   bm_ep_is_approximation=False,
                                   keep_diagnostics=keep_diagnostics)
    return panel


def main():
    p = argparse.ArgumentParser(description="Issue 1: build the master monthly panel")
    p.add_argument("--source", choices=["wrds", "synthetic"], default="synthetic",
                    help="'wrds' requires a live WRDS connection; 'synthetic' is an offline dry run")
    p.add_argument("--index-source", choices=["exchcd_filtered", "msi", "ciz"], default="exchcd_filtered",
                    help="'exchcd_filtered' (default) builds a strict NYSE-only index from security-level "
                         "data -- slower but avoids crsp.msi's AMEX(~1962)/NASDAQ(~1972) universe drift. "
                         "'msi' is faster but only reliable for samples entirely before ~1962. "
                         "'ciz' is the same strict-NYSE construction against CRSP's current CIZ schema "
                         "(crsp.msf_v2) and is REQUIRED for any sample past 2024-12-31, where the legacy "
                         "SIZ tables stop -- they return a short panel rather than an error. 'ciz' also "
                         "sources the risk-free rate from the Ken French library, since crsp.mcti is "
                         "frozen at the same date.")
    p.add_argument("--start", default=START_DATE)
    p.add_argument("--end", default=None)
    p.add_argument("--n-months", type=int, default=660, help="only used with --source synthetic")
    p.add_argument("--include-deferred-taxes", action="store_true",
                    help="add TXDITC (deferred taxes and investment tax credit) to book "
                         "equity, the Fama-French convention. Default is to omit it, which "
                         "is what reproduces the paper: aggregate B/M averages 52.13 over "
                         "1963-2000 without it against the paper's 53.13, and 58.69 with it. "
                         "The paper never states its formula -- see "
                         "wrds_pull.pull_compustat_be_and_earnings.")
    p.add_argument("--no-compustat", action="store_true",
                    help="skip Compustat pull even in --source wrds mode (e.g. if you only have CRSP access)")
    p.add_argument("--out", default=str(DATA_DIR / "master_panel.csv"))
    p.add_argument("--debug-out", default=None,
                    help="if set, also write a version of the panel with diagnostic columns "
                         "(totval, matched_fye, book_equity_sum_used, oibdp_sum_used) to this "
                         "path -- use with diagnose_bm.py to trace an outlier B/M or E/P month "
                         "back to its source data")
    p.add_argument("--qa-tolerance", type=float, default=0.15)
    p.add_argument("--skip-qa", action="store_true",
                    help="write the panel even if the QA gate fails (not recommended -- Issue 2 depends on this passing)")
    args = p.parse_args()

    keep_diagnostics = args.debug_out is not None

    if args.source == "wrds":
        panel = run_wrds(args.start, args.end, compustat=not args.no_compustat,
                         include_deferred_taxes=args.include_deferred_taxes,
                          index_source=args.index_source, keep_diagnostics=keep_diagnostics)
    else:
        panel = run_synthetic(args.n_months, args.start, keep_diagnostics=keep_diagnostics)

    if args.debug_out:
        os.makedirs(os.path.dirname(args.debug_out) or ".", exist_ok=True)
        panel.to_csv(args.debug_out, index=False)
        print(f"Wrote diagnostic panel: {args.debug_out} ({len(panel)} rows, "
              f"{len(panel.columns)} columns including diagnostics)")
        from construct_panel import MASTER_COLS
        panel_for_qa = panel[MASTER_COLS]
    else:
        panel_for_qa = panel

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    panel_for_qa.to_csv(args.out, index=False)
    print(f"\nWrote master panel: {args.out} ({len(panel_for_qa)} rows)")

    print("\n" + "=" * 70)
    print("QA GATE: Table 1 replication check")
    print("=" * 70)
    passed = qa_gate(panel_for_qa, tolerance=args.qa_tolerance)

    if not passed and not args.skip_qa:
        print("\nQA gate failed -- per Issue 1, do NOT proceed to Issue 2 until this "
              "passes. (Use --skip-qa to write the panel anyway for debugging.)")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
