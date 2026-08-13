"""
diagnose_bm.py
==============
Standalone diagnostic for tracing an outlier B/M or E/P statistic back to its
source data, instead of guessing at fixes blind. Reads the diagnostic panel
produced by `run_issue1.py --debug-out ...` (which includes totval,
matched_fye, book_equity_sum_used, oibdp_sum_used alongside the normal
columns) and prints:

  1. The most extreme logB/M and logE/P months (by absolute deviation from
     the series mean), with the exact totval and fiscal-year data that
     produced them -- this is the direct answer to "which month(s) are
     driving the inflated SD/skew, and why."
  2. A z-score style outlier flag so you don't have to eyeball a few hundred
     rows.

Usage:
    python run_issue1.py --source wrds --start 1926-01-01 \
        --out data/master_panel.csv --debug-out data/master_panel_debug.csv
    python src/diagnose_bm.py data/master_panel_debug.csv
"""

from __future__ import annotations
import sys
import pandas as pd
import numpy as np


def find_outlier_months(df: pd.DataFrame, col: str, n: int = 8) -> pd.DataFrame:
    """Top-n months by |z-score| for the given column, with full diagnostic context."""
    sub = df.dropna(subset=[col]).copy()
    mean, sd = sub[col].mean(), sub[col].std()
    sub["z"] = (sub[col] - mean) / sd
    sub["abs_z"] = sub["z"].abs()
    cols_to_show = ["date", col, "z", "totval", "matched_fye",
                     "book_equity_sum_used", "oibdp_sum_used"]
    cols_to_show = [c for c in cols_to_show if c in sub.columns]
    return sub.sort_values("abs_z", ascending=False).head(n)[cols_to_show]


def print_annual_series(df: pd.DataFrame, start_year: int | None = None, end_year: int | None = None):
    """
    Print the full year-by-year book_equity_sum_used / oibdp_sum_used series
    with year-over-year % change, regardless of whether any single year trips
    the >50%-jump flag elsewhere in this script. A >50% flag only catches a
    single sharp jump; it won't tell you whether a milder outlier (e.g. a
    z-score around -3, well short of the -5.9 a genuine ramp-up year showed)
    is still-thin coverage continuing to fill in gradually, or just the low
    point of an otherwise smooth, real economic trend -- for that judgment
    call, you need to see the whole shape, not one flagged number.
    """
    if "book_equity_sum_used" not in df.columns:
        return
    by_fye = (df.dropna(subset=["matched_fye"])
                .groupby("matched_fye")
                .agg(book_equity_sum_used=("book_equity_sum_used", "first"),
                     oibdp_sum_used=("oibdp_sum_used", "first"))
                .sort_index())
    if start_year is not None:
        by_fye = by_fye[by_fye.index.year >= start_year]
    if end_year is not None:
        by_fye = by_fye[by_fye.index.year <= end_year]
    if by_fye.empty:
        print(f"(no fiscal years found in range {start_year}-{end_year})")
        return

    be_pct = by_fye["book_equity_sum_used"].pct_change() * 100
    ep_pct = by_fye["oibdp_sum_used"].pct_change() * 100

    print(f"{'fiscal year':12s} {'book_equity_sum':>18s} {'YoY %':>9s} "
          f"{'oibdp_sum':>15s} {'YoY %':>9s}")
    for fye in by_fye.index:
        be = by_fye.loc[fye, "book_equity_sum_used"]
        ep = by_fye.loc[fye, "oibdp_sum_used"]
        be_str = f"{be:,.0f}" if pd.notna(be) else "NaN"
        ep_str = f"{ep:,.0f}" if pd.notna(ep) else "NaN"
        be_pct_str = f"{be_pct.loc[fye]:+.0f}%" if pd.notna(be_pct.loc[fye]) else "--"
        ep_pct_str = f"{ep_pct.loc[fye]:+.0f}%" if pd.notna(ep_pct.loc[fye]) else "--"
        print(f"{str(fye.date()):12s} {be_str:>18s} {be_pct_str:>9s} "
              f"{ep_str:>15s} {ep_pct_str:>9s}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python diagnose_bm.py <master_panel_debug.csv> [start_year] [end_year]")
        sys.exit(2)
    path = sys.argv[1]
    year_range_start = int(sys.argv[2]) if len(sys.argv) > 2 else None
    year_range_end = int(sys.argv[3]) if len(sys.argv) > 3 else None
    df = pd.read_csv(path, parse_dates=["date", "matched_fye"])

    required = {"totval", "matched_fye", "book_equity_sum_used", "oibdp_sum_used"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: {path} is missing diagnostic columns {missing}.")
        print("Rerun with: python run_issue1.py ... --debug-out <this path>")
        sys.exit(1)

    for col in ["logB/M", "logE/P"]:
        if col not in df.columns or df[col].isna().all():
            continue
        print("=" * 78)
        print(f"Most extreme {col} months (by |z-score| within the full sample)")
        print("=" * 78)
        outliers = find_outlier_months(df, col, n=8)
        with pd.option_context("display.max_columns", None, "display.width", 140):
            print(outliers.to_string(index=False))
        print()

    # Also flag any fiscal year that appears in matched_fye with unusually
    # low book_equity_sum_used relative to its neighbors -- a level shift
    # that a single-month z-score check might miss if it persists for many
    # months (the ratio is constant within a fiscal year, changing only
    # because totval moves month to month).
    if "book_equity_sum_used" in df.columns:
        by_fye = (df.dropna(subset=["matched_fye"])
                    .groupby("matched_fye")["book_equity_sum_used"]
                    .first()
                    .sort_index())
        pct_change = by_fye.pct_change().abs()
        big_jumps = pct_change[pct_change > 0.5]  # >50% jump year over year
        if len(big_jumps):
            print("=" * 78)
            print("Fiscal years with a >50% jump in aggregate book equity vs. the prior")
            print("year (possible sign of a firm-coverage discontinuity, not organic growth)")
            print("=" * 78)
            for fye, pct in big_jumps.items():
                print(f"  {fye.date()}: book_equity_sum_used={by_fye[fye]:,.0f} "
                      f"({pct*100:+.0f}% vs. prior fiscal year)")
            print()

    # Full annual series, always printed -- a >50% single-jump flag or a
    # z-score outlier flag each only tell part of the story; seeing the
    # whole shape is what actually distinguishes "still-thin coverage
    # ramping up" from "real economic trend whose low point looks extreme
    # in isolation." Defaults to a window around whatever outlier months
    # were found above, if no explicit range was given on the command line.
    print("=" * 78)
    print("Full annual book_equity_sum / oibdp_sum series"
          + (f" ({year_range_start}-{year_range_end})" if year_range_start else ""))
    print("Look at the shape, not just one flagged year: is the outlier a sharp")
    print("one-year spike/dip against otherwise-smooth neighbors (coverage bug),")
    print("or part of a gradual, monotonic trend (real economic history)?")
    print("=" * 78)
    if year_range_start is None and "matched_fye" in df.columns:
        outlier_years = set()
        for col in ["logB/M", "logE/P"]:
            if col in df.columns and df[col].notna().any():
                top = find_outlier_months(df, col, n=8)
                outlier_years |= set(pd.to_datetime(top["matched_fye"]).dt.year)
        if outlier_years:
            year_range_start = min(outlier_years) - 5
            year_range_end = max(outlier_years) + 5
    print_annual_series(df, year_range_start, year_range_end)


if __name__ == "__main__":
    main()
