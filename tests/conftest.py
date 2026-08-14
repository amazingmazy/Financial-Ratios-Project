"""
conftest.py
===========
Shared pytest configuration and fixtures.

The replication tests need a built master panel, which requires WRDS
entitlements. Rather than skipping them invisibly, this module provides one
session-scoped ``panel`` fixture that skips with an actionable message naming
the exact command needed to produce the missing file.

Point the tests at a panel elsewhere with the ``MASTER_PANEL`` environment
variable, e.g.::

    MASTER_PANEL=_data/master_panel.parquet pytest tests/ -q
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_PANEL = REPO_ROOT / "data" / "master_panel.csv"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "wrds: needs a built master panel (run the Issue 1 pipeline first)",
    )
    config.addinivalue_line(
        "markers", "slow: Monte Carlo heavy; excluded by `-m 'not slow'`"
    )


@pytest.fixture(scope="session")
def panel() -> pd.DataFrame:
    """The built master panel, or a skip with instructions for producing it."""
    path = Path(os.environ.get("MASTER_PANEL", DEFAULT_PANEL))
    if not path.exists():
        pytest.skip(
            f"no master panel at {path}. Build one with:\n"
            f"    python run_issue1.py --source wrds --out {path}\n"
            f"or point at an existing one with MASTER_PANEL=<path>."
        )
    df = (
        pd.read_parquet(path)
        if path.suffix == ".parquet"
        else pd.read_csv(path, parse_dates=["date"])
    )
    return df.sort_values("date").reset_index(drop=True)


def slice_window(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Rows with ``start <= date <= end``, sorted, index reset."""
    mask = (df["date"] >= start) & (df["date"] <= end)
    return df.loc[mask].sort_values("date").reset_index(drop=True)


def build_x_r(
    df: pd.DataFrame, ratio_col: str, return_col: str
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``(x, r)`` for the estimators from an already-windowed frame.

    Mirrors ``replicate_paper_tables.run_one_series`` exactly, so the tests
    exercise the same alignment the production tables use: ``x`` holds the
    predictor levels x_0..x_T (length N) and ``r`` holds returns r_1..r_T
    (length N-1), giving the paper's ``r_t = a + b*x_{t-1} + e_t``.

    Unlike ``run_one_series`` this asserts the surviving rows are contiguous
    months. That check is the point: dropping NaNs and then reindexing makes
    non-adjacent months look adjacent, and an AR(1) fitted across such a seam
    silently corrupts rho_hat -- the one statistic the whole paper turns on.
    See the note in ``test_replication_vs_paper.py``.
    """
    sub = df.dropna(subset=[ratio_col, return_col]).reset_index(drop=True)
    gaps = sub["date"].diff().dt.days.dropna()
    if len(gaps) and (gaps.max() > 45 or gaps.min() < 20):
        bad = sub.loc[gaps.idxmax(), "date"] if gaps.max() > 45 else None
        raise AssertionError(
            f"{ratio_col}/{return_col}: rows are not contiguous months after "
            f"dropna (largest gap {gaps.max():.0f} days near {bad}). Fitting "
            f"an AR(1) across this seam would corrupt rho_hat."
        )
    return sub[ratio_col].to_numpy(), sub[return_col].to_numpy()[1:]
