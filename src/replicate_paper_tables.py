"""
replicate_paper_tables.py
==========================
Issue 2 (core replication). Runs the shared estimator library (Issue 2.0,
src/estimators.py -- OLS, Stambaugh (1999) Monte Carlo bias correction,
Lewellen's rho~1 conditional test, and the modified-Bonferroni joint test)
against the real, QA-gated master panel from Issue 1, reproducing:

  - Table 2: dividend yield, full sample 1946-2000
  - Table 3: dividend yield, subsamples 1946-1972 and 1973-2000
  - Table 4: sensitivity to 1995-2000 (1946-1994 vs. 1946-2000)
  - Table 5: book-to-market, June 1963-Dec 1994 and June 1963-Dec 2000
  - Table 6: earnings-price ratio, June 1963-Dec 1994 and June 1963-Dec 2000

Per instruction: no year exclusion is applied to the B/M or E/P samples --
the full 1963-2000 window is used as-is, including the 1963-1966 period
whose data characteristics are documented in docs/ISSUE1.md (not excluded,
just noted).

Usage:
    python src/replicate_paper_tables.py data/master_panel.csv
    python src/replicate_paper_tables.py data/master_panel.csv --n-sims 8000
"""

from __future__ import annotations
import argparse
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from estimators import run_all

pd.set_option("display.float_format", lambda x: f"{x:0.4f}")


# --------------------------------------------------------------------------
# Core: one predictive regression (one return series x one ratio) -> run_all
# --------------------------------------------------------------------------

def run_one_series(df: pd.DataFrame, ratio_col: str, return_col: str, n_sims: int) -> dict | None:
    """
    df must be pre-filtered to the desired sample window and sorted by date.
    Builds x (predictor levels, length T+1) and r (returns, length T) per
    estimators.run_all's convention (x_lag = x[:-1] aligns with r), i.e. the
    paper's r_t = a + b*x_{t-1} + e_t.
    """
    sub = df.dropna(subset=[ratio_col, return_col]).reset_index(drop=True)
    if len(sub) < 30:
        return None
    x = sub[ratio_col].to_numpy()      # x_0 .. x_T
    r = sub[return_col].to_numpy()[1:]  # r_1 .. r_T
    return run_all(r, x, n_sims=n_sims)


def run_table(df: pd.DataFrame, ratio_col: str, return_cols: list[str],
              window_label: str, start: str, end: str, n_sims: int = 8000) -> pd.DataFrame:
    """
    One "Table N"-style block: predictive regressions of each return_col on
    lagged ratio_col, over [start, end], using all three estimators.
    Returns a tidy long-format DataFrame, one row per return series.
    """
    d = df[(df["date"] >= start) & (df["date"] <= end)].sort_values("date").reset_index(drop=True)
    rows = []
    for col in return_cols:
        res = run_one_series(d, ratio_col, col, n_sims)
        if res is None:
            print(f"  [skip] {window_label} {col} ~ {ratio_col}: fewer than 30 valid observations")
            continue
        rows.append({
            "window": window_label, "predictor": ratio_col, "series": col, "T": res["T"],
            "rho_hat": res["rho_hat"], "kendall_bias": res["kendall_bias"],
            "corr_em": res["corr_em"], "gamma_hat": res["gamma_hat"],
            "OLS_b": res["ols_b"], "OLS_se": res["ols_se"], "OLS_p": res["ols_p"],
            "Stambaugh_b": res["stambaugh_b"], "Stambaugh_se": res["stambaugh_se"], "Stambaugh_p": res["stambaugh_p"],
            "rho1_b": res["rho1_b"], "rho1_se": res["rho1_se"], "rho1_t": res["rho1_t"], "rho1_p": res["rho1_p"],
            "joint_p": res["joint_p"],
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Paper-style console formatting (rows = estimator, columns = series)
# --------------------------------------------------------------------------

def print_paper_style(table: pd.DataFrame, title: str):
    print("=" * 100)
    print(title)
    print("=" * 100)
    if table.empty:
        print("(no series had enough data in this window)")
        print()
        return
    series_list = table["series"].tolist()
    print(f"{'':12s}" + "".join(f"{s:>18s}" for s in series_list))
    print(f"{'T':12s}" + "".join(f"{int(table.loc[table.series==s,'T'].iloc[0]):>18d}" for s in series_list))
    print(f"{'AR(1) rho':12s}" + "".join(f"{table.loc[table.series==s,'rho_hat'].iloc[0]:>18.4f}" for s in series_list))
    print(f"{'corr(e,m)':12s}" + "".join(f"{table.loc[table.series==s,'corr_em'].iloc[0]:>18.4f}" for s in series_list))
    print("-" * (12 + 18 * len(series_list)))
    for est, bcol, secol, pcol in [("OLS", "OLS_b", "OLS_se", "OLS_p"),
                                    ("Stambaugh", "Stambaugh_b", "Stambaugh_se", "Stambaugh_p"),
                                    ("rho~1", "rho1_b", "rho1_se", "rho1_p")]:
        b_row = f"{est:12s}" + "".join(f"{table.loc[table.series==s,bcol].iloc[0]:>18.4f}" for s in series_list)
        se_row = f"{'':12s}" + "".join(f"{'(' + format(table.loc[table.series==s,secol].iloc[0], '.4f') + ')':>18s}" for s in series_list)
        p_row = f"{'':12s}" + "".join(f"{'p=' + format(table.loc[table.series==s,pcol].iloc[0], '.3f'):>18s}" for s in series_list)
        print(b_row)
        print(se_row)
        print(p_row)
    print(f"{'joint p':12s}" + "".join(f"{table.loc[table.series==s,'joint_p'].iloc[0]:>18.4f}" for s in series_list))
    print()


# --------------------------------------------------------------------------
# Main: Tables 2, 3, 4, 5
# --------------------------------------------------------------------------

def run_extended(df: pd.DataFrame, n_sims: int) -> list[pd.DataFrame]:
    """
    Issue 3: the same regressions carried past the paper's 2000 cutoff.

    The end date is taken from the panel rather than hard-coded, because the
    two CRSP schemas reach different dates -- the legacy SIZ tables stop at
    2024-12-31 and CIZ runs a year further (see ISSUE2.md). Hard-coding an end
    would silently truncate one of them, which is the same class of failure the
    frozen SIZ tables already caused once.

    Two windows, answering different questions:

      - 1946 to present: does the paper's conclusion survive the extra 25
        years? This is the headline extension result.
      - 2001 to present: does it hold in data Lewellen never saw? This is the
        stricter test, and it is where the interesting answer is -- DY's
        autocorrelation falls to ~0.95 here, below the paper's own stated
        threshold of roughly 0.99 for 25 years of monthly data (Section 2.4),
        so the conditional test is expected to lose power. Its Table A.1
        tabulates precisely that collapse. Reporting the window is therefore a
        demonstration of the paper's stated limitation rather than a
        counterexample to it, and the rho column is the number to read first.
    """
    end = df["date"].max()
    end_str = end.strftime("%Y-%m-%d")
    end_lbl = end.strftime("%Y")
    tables = []

    for start, start_lbl, title in [
        ("1946-01-01", "1946", "full sample extended to the present"),
        ("2001-01-01", "2001", "out of sample -- data the paper never saw"),
    ]:
        t = run_table(df, "logDY", ["VWNY", "EWNY", "ExcVWNY", "ExcEWNY"],
                      f"{start_lbl}-{end_lbl}", start, end_str, n_sims=n_sims)
        print_paper_style(t, f"EXTENDED — Dividend yield predicts NYSE returns, "
                             f"{start_lbl}-{end_lbl} ({title})")
        tables.append(t)

    if tables and not tables[-1].empty:
        rho = tables[-1]["rho_hat"].iloc[0]
        print(f"Post-2000 AR(1) rho for logDY: {rho:.4f}")
        if rho < 0.98:
            print("  Below the paper's ~0.98 rule of thumb (Section 2.4): the rho~1\n"
                  "  conditional test has little power on this window by the paper's\n"
                  "  own analysis, so a null result here is expected rather than\n"
                  "  contradictory. Read the OLS and Stambaugh rows instead.\n")
        else:
            print("  Clears the paper's ~0.98 rule of thumb; the conditional test\n"
                  "  remains applicable on this window.\n")
    return tables


def main():
    p = argparse.ArgumentParser(description="Issue 2: core replication (Tables 2, 3, 4, 5)")
    p.add_argument("panel_csv", help="path to the QA-gated master panel from Issue 1 (e.g. data/master_panel.csv)")
    p.add_argument("--n-sims", type=int, default=8000, help="Monte Carlo draws for the Stambaugh correction")
    p.add_argument("--out-dir", default="output")
    p.add_argument("--extended", action="store_true",
                    help="also run the Issue 3 extension windows (1946-present and "
                         "2001-present), with the end date read from the panel")
    p.add_argument("--tag", default=None,
                    help="suffix for the output CSV, e.g. --tag siz / --tag ciz, so the "
                         "two CRSP schemas can be run side by side without overwriting")
    p.add_argument("--end", default=None,
                    help="truncate the panel here (YYYY-MM-DD) before running anything. "
                         "Needed to compare the two CRSP schemas on a matched window: the "
                         "legacy SIZ panel ends at 2024-12-31 and CIZ a year later, so "
                         "without this the extension windows have different lengths and "
                         "the comparison is not like for like.")
    args = p.parse_args()

    df = pd.read_csv(args.panel_csv, parse_dates=["date"])
    if args.end:
        df = df[df["date"] <= pd.to_datetime(args.end)]
        print(f"Panel truncated at {args.end}: {len(df)} months "
              f"({df['date'].min().date()} .. {df['date'].max().date()})\n")
    os.makedirs(args.out_dir, exist_ok=True)
    all_tables = []

    # ---- Table 2: DY, full sample 1946-2000 ----
    t2 = run_table(df, "logDY", ["VWNY", "EWNY", "ExcVWNY", "ExcEWNY"],
                   "1946-2000", "1946-01-01", "2000-12-31", n_sims=args.n_sims)
    print_paper_style(t2, "TABLE 2 — Dividend yield predicts NYSE returns, 1946-2000")
    all_tables.append(t2)

    # ---- Table 3: DY, subsamples ----
    t3a = run_table(df, "logDY", ["VWNY", "EWNY", "ExcVWNY", "ExcEWNY"],
                     "1946-1972", "1946-01-01", "1972-12-31", n_sims=args.n_sims)
    print_paper_style(t3a, "TABLE 3a — Dividend yield predicts NYSE returns, 1946-1972")
    t3b = run_table(df, "logDY", ["VWNY", "EWNY", "ExcVWNY", "ExcEWNY"],
                     "1973-2000", "1973-01-01", "2000-12-31", n_sims=args.n_sims)
    print_paper_style(t3b, "TABLE 3b — Dividend yield predicts NYSE returns, 1973-2000")
    all_tables += [t3a, t3b]

    # ---- Table 4: sensitivity to 1995-2000 (nominal returns only, per the paper) ----
    t4a = run_table(df, "logDY", ["VWNY", "EWNY"],
                     "1946-1994", "1946-01-01", "1994-12-31", n_sims=args.n_sims)
    print_paper_style(t4a, "TABLE 4a — Dividend yield predicts NYSE returns, 1946-1994")
    t4b = run_table(df, "logDY", ["VWNY", "EWNY"],
                     "1946-2000 (Table 4 comparison)", "1946-01-01", "2000-12-31", n_sims=args.n_sims)
    print_paper_style(t4b, "TABLE 4b — Dividend yield predicts NYSE returns, 1946-2000 (repeated for comparison)")
    if not t4a.empty and not t4b.empty:
        print("Change in AR(1) rho and OLS slope, 1946-1994 -> 1946-2000:")
        for s in t4a["series"]:
            rho_a = t4a.loc[t4a.series == s, "rho_hat"].iloc[0]
            rho_b = t4b.loc[t4b.series == s, "rho_hat"].iloc[0]
            b_a = t4a.loc[t4a.series == s, "OLS_b"].iloc[0]
            b_b = t4b.loc[t4b.series == s, "OLS_b"].iloc[0]
            print(f"  {s}: rho {rho_a:.3f} -> {rho_b:.3f}   OLS b {b_a:.3f} -> {b_b:.3f}")
        print()
    all_tables += [t4a, t4b]

    # ---- Table 5: B/M, June 1963-Dec 1994 and June 1963-Dec 2000 ----
    # No year exclusion applied -- full sample used as-is, per instruction.
    t5a = run_table(df, "logB/M", ["VWNY", "EWNY", "ExcVWNY", "ExcEWNY"],
                     "1963-1994", "1963-06-01", "1994-12-31", n_sims=args.n_sims)
    print_paper_style(t5a, "TABLE 5a — Book-to-market predicts NYSE returns, 1963-1994")
    t5b = run_table(df, "logB/M", ["VWNY", "EWNY", "ExcVWNY", "ExcEWNY"],
                     "1963-2000", "1963-06-01", "2000-12-31", n_sims=args.n_sims)
    print_paper_style(t5b, "TABLE 5b — Book-to-market predicts NYSE returns, 1963-2000")
    all_tables += [t5a, t5b]

    # ---- Table 6: E/P, June 1963-Dec 1994 and June 1963-Dec 2000 ----
    # Identical machinery to Table 5, swapping logB/M for logE/P. Per Issue 1's
    # QA gate, E/P's underlying data (mean, SD, rho1) passed every check
    # cleanly -- unlike B/M, there's no known coverage-ramp-up caveat here.
    t6a = run_table(df, "logE/P", ["VWNY", "EWNY", "ExcVWNY", "ExcEWNY"],
                     "1963-1994", "1963-06-01", "1994-12-31", n_sims=args.n_sims)
    print_paper_style(t6a, "TABLE 6a — Earnings-price ratio predicts NYSE returns, 1963-1994")
    t6b = run_table(df, "logE/P", ["VWNY", "EWNY", "ExcVWNY", "ExcEWNY"],
                     "1963-2000", "1963-06-01", "2000-12-31", n_sims=args.n_sims)
    print_paper_style(t6b, "TABLE 6b — Earnings-price ratio predicts NYSE returns, 1963-2000")
    all_tables += [t6a, t6b]

    if args.extended:
        all_tables += run_extended(df, n_sims=args.n_sims)

    combined = pd.concat(all_tables, ignore_index=True)
    suffix = f"_{args.tag}" if args.tag else ""
    name = ("issue2_tables_2_3_4_5_6_extended" if args.extended
            else "issue2_tables_2_3_4_5_6")
    out_path = os.path.join(args.out_dir, f"{name}{suffix}.csv")
    combined.to_csv(out_path, index=False)
    print(f"Wrote combined results: {out_path} ({len(combined)} rows)")


if __name__ == "__main__":
    main()
