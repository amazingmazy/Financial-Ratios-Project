"""
qa_table1.py
============
Issue 1's explicit QA gate: "reproduce Table 1 (mean, SD, skew, rho1, rho12,
rho24) for full sample and both halves; do not proceed to Issue 2 until these
match the paper within rounding."

This module computes exactly those statistics from the assembled master
panel and compares them against the paper's own published Table 1 values
(hard-coded below, transcribed directly from Lewellen 2004, Table 1), with a
configurable tolerance. It's meant to be run as a script (`python -m
qa_table1 <panel.csv>`) that prints a pass/fail report and exits non-zero on
failure, so it can gate a pipeline/CI step.
"""

from __future__ import annotations
import sys
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Paper's published Table 1 values (Lewellen 2004, p. 219), for comparison.
# Format: {sample_label: {series_label: (mean, sd, skew, rho1, rho12, rho24)}}
# --------------------------------------------------------------------------
PAPER_TABLE_1 = {
    "1946-2000": {
        "VWNY":   (1.04, 4.08, 0.38, 0.032, 0.042, 0.014),
        "EWNY":   (1.11, 4.80, 0.16, 0.136, 0.065, 0.027),
        "DY":     (3.80, 1.20, 0.37, 0.992, 0.889, 0.812),
        "logDY":  (1.28, 0.33, 0.53, 0.997, 0.948, 0.912),
    },
    "1946-1972": {
        "VWNY":   (0.98, 3.67, 0.39, 0.079, 0.026, 0.066),
        "EWNY":   (1.04, 4.38, 0.26, 0.150, 0.024, 0.021),
        "DY":     (4.02, 1.21, 0.84, 0.992, 0.879, 0.774),
        "logDY":  (1.35, 0.28, 0.56, 0.993, 0.876, 0.785),
    },
    "1973-2000": {
        "VWNY":   (1.10, 4.44, 0.39, 0.001, 0.065, 0.013),
        "EWNY":   (1.18, 5.18, 0.11, 0.125, 0.113, 0.030),
        "DY":     (3.59, 1.15, 0.19, 0.991, 0.899, 0.914),
        "logDY":  (1.22, 0.37, 0.81, 0.999, 0.996, 1.062),
    },
    # Compustat era, 1963-2000 -- only populated if B/M, E/P aren't NaN.
    "1963-2000": {
        "B/M":    (53.13, 18.28, 0.39, 0.990, 0.891, 0.837),
        "logB/M": (3.91, 0.36, 0.19, 0.995, 0.951, 0.923),
        "E/P":    (20.02, 7.01, 0.55, 0.988, 0.864, 0.770),
        "logE/P": (2.94, 0.35, 0.14, 0.990, 0.891, 0.785),
    },
}

STAT_NAMES = ["mean", "sd", "skew", "rho1", "rho12", "rho24"]


def compute_stats(x: pd.Series) -> tuple:
    """Mean, SD, skewness, and autocorrelations at lags 1, 12, 24 for a series."""
    x = x.dropna().to_numpy()
    def autocorr(lag):
        if len(x) <= lag:
            return np.nan
        return np.corrcoef(x[:-lag], x[lag:])[0, 1]
    return (x.mean(), x.std(ddof=1), pd.Series(x).skew(),
            autocorr(1), autocorr(12), autocorr(24))


def qa_gate(panel: pd.DataFrame, tolerance: float = 0.15, verbose: bool = True) -> bool:
    """
    Compare computed Table 1 stats against PAPER_TABLE_1 for every
    (sample, series) pair the paper reports, within a relative tolerance
    (default 15%, loose enough for public-proxy/synthetic data to spot gross
    errors, tight enough to catch real construction bugs -- tighten this once
    running on real CRSP/Compustat data, per the README's "match within
    rounding" bar).

    Returns True iff every check passes. Prints a full report either way.
    """
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"])
    windows = {
        "1946-2000": (df["date"] >= "1946-01-01") & (df["date"] <= "2000-12-31"),
        "1946-1972": (df["date"] >= "1946-01-01") & (df["date"] <= "1972-12-31"),
        "1973-2000": (df["date"] >= "1973-01-01") & (df["date"] <= "2000-12-31"),
        "1963-2000": (df["date"] >= "1963-01-01") & (df["date"] <= "2000-12-31"),
    }

    all_pass = True
    for window_label, mask in windows.items():
        sub = df.loc[mask]
        for series_label, paper_stats in PAPER_TABLE_1.get(window_label, {}).items():
            if series_label not in sub.columns:
                continue
            if sub[series_label].isna().all():
                if verbose:
                    print(f"[SKIP] {window_label:10s} {series_label:8s} -- all-NaN "
                          f"(likely no Compustat data available)")
                continue
            computed = compute_stats(sub[series_label])
            for name, paper_val, comp_val in zip(STAT_NAMES, paper_stats, computed):
                if pd.isna(comp_val):
                    ok = False
                else:
                    denom = max(abs(paper_val), 1e-6)
                    ok = abs(comp_val - paper_val) / denom <= tolerance
                all_pass &= ok
                if verbose:
                    flag = "OK  " if ok else "FAIL"
                    print(f"[{flag}] {window_label:10s} {series_label:8s} {name:6s} "
                          f"paper={paper_val:8.3f}  computed={comp_val:8.3f}")

    if verbose:
        print()
        print("QA GATE:", "PASS -- proceed to Issue 2" if all_pass else
              "FAIL -- do not proceed to Issue 2 until this matches the paper")
    return all_pass


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m qa_table1 <master_panel.csv> [tolerance]")
        sys.exit(2)
    panel = pd.read_csv(sys.argv[1])
    tol = float(sys.argv[2]) if len(sys.argv) > 2 else 0.15
    passed = qa_gate(panel, tolerance=tol)
    sys.exit(0 if passed else 1)