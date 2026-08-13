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

Per instruction: no year exclusion is applied to the B/M sample -- the full
1963-2000 window is used as-is, including the 1963-1966 period whose data
characteristics are documented in docs/ISSUE1.md (not excluded, just noted).

Table 6 (E/P) is not run by default since the README recommends B/M first,
but uses identical machinery -- see run_table(df, "logE/P", ...) to add it;
E/P's underlying data passed every Table 1 QA check cleanly, including SD,
so it's arguably in even better shape than B/M for a next pass.

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

def main():
    p = argparse.ArgumentParser(description="Issue 2: core replication (Tables 2, 3, 4, 5)")
    p.add_argument("panel_csv", help="path to the QA-gated master panel from Issue 1 (e.g. data/master_panel.csv)")
    p.add_argument("--n-sims", type=int, default=8000, help="Monte Carlo draws for the Stambaugh correction")
    p.add_argument("--out-dir", default="output")
    args = p.parse_args()

    df = pd.read_csv(args.panel_csv, parse_dates=["date"])
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

    combined = pd.concat(all_tables, ignore_index=True)
    out_path = os.path.join(args.out_dir, "issue2_tables_2_3_4_5.csv")
    combined.to_csv(out_path, index=False)
    print(f"Wrote combined results: {out_path} ({len(combined)} rows)")


if __name__ == "__main__":
    main()
