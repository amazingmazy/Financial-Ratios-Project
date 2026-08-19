"""
analysis.py
================
Specific exhibit, beyond what the paper asks to replicate (rubric item 5).

Lewellen's whole argument rests on log(DY) staying close enough to a unit
root that the rho~1 conditional test has power (Section 2.4: he states the
rule of thumb explicitly -- roughly 0.98 monthly autocorrelation for 25
years of data, 0.99 for 50). This script asks a question the paper never
does, because its sample ends in 2000: has that stayed true since?

It produces two things from the extended CIZ panel:

1. A decade-by-decade summary table of log(DY), log(B/M), and log(E/P) --
   mean, s.d., and lag-1 autocorrelation (rho_hat) per decade -- written as
   a LaTeX fragment.
2. A rolling-window rho_hat series for log(DY), plotted against the paper's
   own stated power thresholds (dashboard_data.power_threshold), showing
   when and whether the series crosses below them.

Run as:
    python src/analysis.py data/master_panel_ciz.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this runs inside `doit`, no display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from settings import config
from estimators import fit_ar1
from dashboard_data import power_threshold

OUTPUT_DIR = Path(config("OUTPUT_DIR"))

# Series to summarize: (log column in the panel, display label)
SERIES = [
    ("logDY", "Log(DY)"),
    ("logB/M", "Log(B/M)"),
    ("logE/P", "Log(E/P)"),
]

# Rolling-window length for the rho figure. 25 years matches the shorter of
# the paper's own two calibration points (Section 2.4), so the reference
# line the figure plots against is the same one the window length implies.
ROLLING_YEARS = 25
ROLLING_MONTHS = ROLLING_YEARS * 12
STEP_MONTHS = 12  # re-estimate annually rather than every month -- plenty
                   # of resolution for a 1946-2025 series and far cheaper


# --------------------------------------------------------------------------
# 1. Decade table
# --------------------------------------------------------------------------

def decade_label(year: int) -> str:
    start = (year // 10) * 10
    return f"{start}s"


def decade_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (decade, series): mean, s.d., rho_hat of the log series.

    rho_hat is computed within each decade's own data (not carried over from
    a longer window), so it reflects that decade's persistence specifically.
    Decades with fewer than 24 months of data for a series are dropped --
    too short for fit_ar1's OLS to be meaningful.
    """
    df = df.copy()
    df["decade"] = df["date"].dt.year.map(decade_label)

    rows = []
    for decade, block in df.groupby("decade", sort=True):
        for col, label in SERIES:
            x = block[col].dropna().to_numpy()
            if len(x) < 24:
                continue
            ar1 = fit_ar1(x)
            rows.append(
                {
                    "Decade": decade,
                    "Series": label,
                    "Mean": x.mean(),
                    "S.D.": x.std(ddof=1),
                    "rho_hat": ar1["rho_hat"],
                    "N months": len(x),
                }
            )
    out = pd.DataFrame(rows)
    # Chronological, not alphabetical -- "1990s" < "2000s" would otherwise
    # sort as a string and put the 2000s before the 1990s.
    out["_sort"] = out["Decade"].str[:-1].astype(int)
    out = out.sort_values(["_sort", "Series"]).drop(columns="_sort")
    return out.reset_index(drop=True)


def write_summary_table_tex(table: pd.DataFrame, path: Path) -> None:
    """Write the decade table as a LaTeX fragment: 
    a bare tabular, meant to be \\input{} inside a
    \\table environment in the report."""
    pivoted = table.pivot(index="Decade", columns="Series", values=["Mean", "S.D.", "rho_hat"])
    # Flatten and reorder so each series' three stats sit together, rather
    # than grouped by statistic -- easier to read as "one block per series".
    pivoted = pivoted.reorder_levels([1, 0], axis=1)
    pivoted = pivoted[[(label, stat) for _, label in SERIES for stat in ["Mean", "S.D.", "rho_hat"]]]
    stat_display = {"Mean": "Mean", "S.D.": "S.D.", "rho_hat": r"$\hat\rho$"}
    pivoted.columns = [f"{label} {stat_display[stat]}" for label, stat in pivoted.columns]

    float_format = lambda x: f"{x:.3f}"
    latex = pivoted.to_latex(float_format=float_format, na_rep="--", escape=False)
    path.write_text(latex)
    print(f"wrote {path}")


# --------------------------------------------------------------------------
# 2. Rolling rho_hat vs. the paper's power threshold
# --------------------------------------------------------------------------

def rolling_rho(df: pd.DataFrame, col: str = "logDY") -> pd.DataFrame:
    """rho_hat of `col`, estimated on trailing ROLLING_MONTHS windows,
    stepped every STEP_MONTHS. Each point's window length in months is
    carried alongside it, since power_threshold's reference line depends on
    window length, not just on the series."""
    x = df[col].to_numpy()
    dates = df["date"].to_numpy()
    rows = []
    for end_idx in range(ROLLING_MONTHS, len(x) + 1, STEP_MONTHS):
        start_idx = max(0, end_idx - ROLLING_MONTHS)
        window = x[start_idx:end_idx]
        if np.isnan(window).any() or len(window) < 24:
            continue
        ar1 = fit_ar1(window)
        rows.append(
            {
                "date": dates[end_idx - 1],
                "rho_hat": ar1["rho_hat"],
                "window_months": len(window),
            }
        )
    return pd.DataFrame(rows)


def plot_rolling_rho(rolling: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(rolling["date"], rolling["rho_hat"], color="black", lw=1.5,
            label=r"Rolling $\hat\rho$, log(DY) (25-yr trailing window)")

    # Reference line: power_threshold expects the window length in months,
    # which is constant here (ROLLING_MONTHS) since every plotted point uses
    # the same trailing-window length by construction.
    threshold = power_threshold(ROLLING_MONTHS, "monthly")
    ax.axhline(threshold, color="firebrick", ls="--", lw=1.2,
               label=rf"Paper's power threshold, {ROLLING_YEARS}-yr window ($\hat\rho \geq {threshold:.3f}$)")

    ax.axvline(pd.Timestamp("2000-12-31"), color="grey", ls=":", lw=1,
               label="Paper's sample end (2000)")

    ax.set_ylabel(r"$\hat\rho$")
    ax.set_xlabel("Window end date")
    ax.set_title(r"Rolling autocorrelation of log(DY) vs. the conditional test's power threshold")
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main(panel_path: str) -> None:
    df = pd.read_csv(panel_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    table = decade_summary_table(df)
    write_summary_table_tex(table, OUTPUT_DIR / "own_summary_table.tex")

    rolling = rolling_rho(df, "logDY")
    plot_rolling_rho(rolling, OUTPUT_DIR / "own_rho_evolution.png")

    # Printed for the console/log, not for the report -- a quick sanity
    # check that the crossing the figure shows is real and not a plotting
    # artefact.
    below = rolling[rolling["rho_hat"] < power_threshold(ROLLING_MONTHS, "monthly")]
    if len(below):
        first_cross = below.iloc[0]["date"]
        print(f"rho_hat first drops below the {ROLLING_YEARS}-yr power threshold "
              f"around window ending {pd.Timestamp(first_cross).date()}")
    else:
        print(f"rho_hat never drops below the {ROLLING_YEARS}-yr power threshold in this sample")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python src/analysis.py <path/to/master_panel.csv>")
        sys.exit(1)
    main(sys.argv[1])