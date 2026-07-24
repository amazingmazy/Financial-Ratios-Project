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
    python run_issue1.py --source wrds --start 1926-01-01 --out data/master_panel.csv

    # Offline dry run (synthetic data; validates the pipeline, not real numbers):
    python run_issue1.py --source synthetic --out data/master_panel_demo.csv
"""

from __future__ import annotations
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from construct_panel import assemble_master_panel
from qa_table1 import qa_gate


def run_wrds(start: str, end: str | None, compustat: bool) -> "pd.DataFrame":
    import wrds
    import wrds_pull

    print("Connecting to WRDS...")
    db = wrds.Connection()
    try:
        print("Pulling CRSP NYSE index (crsp.msi)...")
        crsp_index = wrds_pull.pull_crsp_nyse_index(db, start=start, end=end)
        print(f"  -> {len(crsp_index)} months")

        print("Pulling CRSP risk-free rate (crsp.mcti)...")
        rf = wrds_pull.pull_crsp_riskfree(db, start=start, end=end)
        print(f"  -> {len(rf)} months")

        print("Pulling CPI from FRED...")
        from public_fallback import pull_fred_cpi
        cpi = pull_fred_cpi(start=start)
        print(f"  -> {len(cpi)} months")

        compustat_annual, is_approx = None, False
        if compustat:
            print("Pulling Compustat book equity & operating earnings (comp.funda, via CCM link)...")
            compustat_annual = wrds_pull.pull_compustat_be_and_earnings(db, start=start, end=end)
            print(f"  -> {len(compustat_annual)} firm-years aggregated to {compustat_annual.index.nunique()} fiscal year-ends")
        else:
            print("Skipping Compustat (--no-compustat) -- B/M and E/P will be all-NaN, flagged accordingly.")

        panel = assemble_master_panel(crsp_index, rf, cpi,
                                       compustat_annual=compustat_annual,
                                       bm_ep_is_approximation=is_approx)
        return panel
    finally:
        db.close()


def run_synthetic(n_months: int, start: str) -> "pd.DataFrame":
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
                                   bm_ep_is_approximation=False)
    return panel


def main():
    p = argparse.ArgumentParser(description="Issue 1: build the master monthly panel")
    p.add_argument("--source", choices=["wrds", "synthetic"], default="synthetic",
                    help="'wrds' requires a live WRDS connection; 'synthetic' is an offline dry run")
    p.add_argument("--start", default="1946-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--n-months", type=int, default=660, help="only used with --source synthetic")
    p.add_argument("--no-compustat", action="store_true",
                    help="skip Compustat pull even in --source wrds mode (e.g. if you only have CRSP access)")
    p.add_argument("--out", default="data/master_panel.csv")
    p.add_argument("--qa-tolerance", type=float, default=0.15)
    p.add_argument("--skip-qa", action="store_true",
                    help="write the panel even if the QA gate fails (not recommended -- Issue 2 depends on this passing)")
    args = p.parse_args()

    if args.source == "wrds":
        panel = run_wrds(args.start, args.end, compustat=not args.no_compustat)
    else:
        panel = run_synthetic(args.n_months, args.start)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    panel.to_csv(args.out, index=False)
    print(f"\nWrote master panel: {args.out} ({len(panel)} rows)")

    print("\n" + "=" * 70)
    print("QA GATE: Table 1 replication check")
    print("=" * 70)
    passed = qa_gate(panel, tolerance=args.qa_tolerance)

    if not passed and not args.skip_qa:
        print("\nQA gate failed -- per Issue 1, do NOT proceed to Issue 2 until this "
              "passes. (Use --skip-qa to write the panel anyway for debugging.)")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()