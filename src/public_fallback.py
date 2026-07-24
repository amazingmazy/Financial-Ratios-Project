"""
public_fallback.py
===================
Issue 1 -- public-data fallback path, used for:
  (a) developing/testing the panel-construction logic without a live WRDS
      connection, and
  (b) a non-WRDS demo mode for anyone without CRSP/Compustat access.

Sources:
  - Ken French Data Library ("F-F_Research_Data_Factors" gives Mkt-RF and RF;
    "Portfolios_Formed_on_ME" gives a genuine CRSP-based equal-weighted proxy)
  - FRED (CPIAUCSL) for CPI
  - "25_Portfolios_ME_BEME" (25 Size x B/M portfolios) as the explicitly-flagged
    approximation for aggregate B/M when Compustat isn't available (per the
    README's Issue 1 fallback instruction)

NOTE: these are all real network fetches (zip/csv downloads) and require
unrestricted internet access -- they will not run inside a sandboxed
environment with an allowlisted proxy. For sandboxed testing, use the
`synthetic_*` functions at the bottom of this file instead, which generate
data with the right shape/schema so `construct_panel.py` can be unit tested
end-to-end without any network access at all.
"""

from __future__ import annotations
import io
import zipfile
import requests
import pandas as pd

FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def _download_french_zip(dataset_csv_zip: str) -> str:
    """Download a Ken French dataset zip and return the inner CSV text."""
    url = f"{FRENCH_BASE}/{dataset_csv_zip}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    inner_name = zf.namelist()[0]
    return zf.read(inner_name).decode("latin-1")


def pull_french_factors_and_rf(start: str = "1926-01") -> pd.DataFrame:
    """
    Pull the market factor (Mkt-RF) and risk-free rate (RF) from French's
    'F-F_Research_Data_Factors' monthly file. Mkt-RF + RF reconstructs the
    CRSP value-weighted market return -- a very close public proxy for VWNY
    (NYSE+AMEX+NASDAQ combined rather than NYSE-only, but the standard
    substitute used across the literature when CRSP access isn't available).

    Returns a DataFrame indexed by month (date, first-of-month) with columns:
        mkt_rf, rf, vwretd_proxy (= mkt_rf + rf)
    """
    text = _download_french_zip("F-F_Research_Data_Factors_CSV.zip")
    # The monthly block is the first table in the file, terminated by a blank
    # line before the annual block begins.
    lines = text.splitlines()
    start_idx = next(i for i, l in enumerate(lines) if l.strip().startswith(("19", "20")))
    end_idx = next(i for i, l in enumerate(lines[start_idx:], start_idx) if l.strip() == "") 
    data_lines = lines[start_idx:end_idx]
    rows = [l.split(",") for l in data_lines]
    df = pd.DataFrame(rows, columns=["date", "mkt_rf", "smb", "hml", "rf"])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m")
    for c in ["mkt_rf", "smb", "hml", "rf"]:
        df[c] = pd.to_numeric(df[c])
    df["vwretd_proxy"] = df["mkt_rf"] + df["rf"]
    df = df.set_index("date").sort_index()
    return df.loc[df.index >= pd.Timestamp(start)]


def pull_french_25_size_bm() -> pd.DataFrame:
    """
    Pull the 25 Size x B/M portfolios (monthly returns), for use as the
    Issue-1-specified approximation of aggregate B/M when Compustat isn't
    available. The intended use in construct_panel.py: convert the return
    spread pattern across the 5 B/M quintiles into an implied aggregate B/M
    proxy, or (more simply/robustly) use the published portfolio-level
    average B/M ratios that accompany this dataset on French's site
    ("Value Weight Average of BE/ME") if pulled alongside.

    Returns raw monthly returns for the 25 portfolios, indexed by date.
    """
    text = _download_french_zip("25_Portfolios_5x5_CSV.zip")
    lines = text.splitlines()
    start_idx = next(i for i, l in enumerate(lines) if l.strip().startswith(("19", "20")))
    end_idx = next(i for i, l in enumerate(lines[start_idx:], start_idx) if l.strip() == "")
    header = [h.strip() for h in lines[start_idx - 1].split(",") if h.strip()]
    data_lines = lines[start_idx:end_idx]
    rows = [l.split(",") for l in data_lines]
    df = pd.DataFrame(rows, columns=["date"] + header[:len(rows[0]) - 1])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m")
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.set_index("date").sort_index()


def pull_fred_cpi(start: str = "1926-01-01") -> pd.DataFrame:
    """Pull CPI (CPIAUCSL) from FRED. Returns a DataFrame indexed by date with column `cpi`."""
    resp = requests.get(FRED_BASE, params={"id": "CPIAUCSL", "cosd": start}, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = ["date", "cpi"]
    df["date"] = pd.to_datetime(df["date"])
    df["cpi"] = pd.to_numeric(df["cpi"], errors="coerce")
    return df.set_index("date").sort_index()


# --------------------------------------------------------------------------
# Synthetic data generators -- for offline unit testing of construct_panel.py
# --------------------------------------------------------------------------

def synthetic_crsp_index(n_months: int = 660, start: str = "1946-01-01", seed: int = 0) -> pd.DataFrame:
    """
    Generate a synthetic monthly NYSE index panel with the right schema and
    roughly the right statistical properties (persistent AR(1)-like dividend
    yield, plausible return volatility) to exercise construct_panel.py without
    any network access. NOT for producing real replication numbers.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_months, freq="MS")

    # Simulate a price index with a slow-moving trailing dividend yield.
    rho = 0.995
    sigma_m = 0.03
    stationary_mean = np.log(0.04)  # anchor the AR(1) at a plausible ~4% DY
    intercept = stationary_mean * (1 - rho)
    log_dy = np.empty(n_months)
    log_dy[0] = stationary_mean
    for t in range(1, n_months):
        log_dy[t] = intercept + rho * log_dy[t - 1] + rng.normal(0, sigma_m)
    dy = np.exp(log_dy)

    ret_incl = rng.normal(0.008, 0.04, n_months) - 0.3 * np.diff(np.concatenate([[log_dy[0]], log_dy]))
    index_level = 100 * np.cumprod(1 + ret_incl)
    div_flow = dy * index_level  # implied trailing-12mo dividend flow, annualized
    ret_excl = ret_incl - (div_flow / 12) / index_level  # crude split of price vs. dividend return

    ew_ret_incl = ret_incl + rng.normal(0, 0.01, n_months)  # small-cap-like extra noise
    ew_ret_excl = ret_excl + rng.normal(0, 0.01, n_months)

    df = pd.DataFrame({
        "vwretd": ret_incl, "vwretx": ret_excl,
        "ewretd": ew_ret_incl, "ewretx": ew_ret_excl,
        "totval": index_level, "totcnt": 1000,
    }, index=dates)
    df.index.name = "date"
    return df


def synthetic_riskfree(n_months: int = 660, start: str = "1946-01-01", seed: int = 1) -> pd.DataFrame:
    import numpy as np
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_months, freq="MS")
    rf = np.clip(rng.normal(0.3, 0.15, n_months), 0, None)
    return pd.DataFrame({"rf": rf}, index=dates).rename_axis("date")


def synthetic_compustat_be_earnings(n_years: int = 60, start_year: int = 1946, seed: int = 2) -> pd.DataFrame:
    import numpy as np
    rng = np.random.default_rng(seed)
    years = pd.date_range(f"{start_year}-12-31", periods=n_years, freq="YE")
    book_equity = 1000 * np.cumprod(1 + rng.normal(0.04, 0.05, n_years))
    oibdp = 150 * np.cumprod(1 + rng.normal(0.04, 0.06, n_years))
    return pd.DataFrame({
        "book_equity_sum": book_equity, "oibdp_sum": oibdp, "n_firms": 400,
    }, index=years).rename_axis("datadate")


def synthetic_cpi(n_months: int = 660, start: str = "1946-01-01", seed: int = 3) -> pd.DataFrame:
    import numpy as np
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_months, freq="MS")
    monthly_inflation = rng.normal(0.003, 0.003, n_months)
    cpi = 20 * np.cumprod(1 + monthly_inflation)
    return pd.DataFrame({"cpi": cpi}, index=dates).rename_axis("date")