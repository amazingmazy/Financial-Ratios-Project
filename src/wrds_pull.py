"""
wrds_pull.py
============
Issue 1 (data wrangling & cleaning) — WRDS/CRSP/Compustat pull functions.

These functions assume you have a live WRDS connection (`wrds.Connection()`),
which requires a WRDS account with the appropriate CRSP/Compustat subscriptions.
They are written against the standard, well-documented WRDS table/column names
used throughout the academic literature (Fama-French, Stambaugh, Lewellen, and
countless replications) -- nothing here is proprietary knowledge, only the
underlying data itself is.

Usage:
    import wrds
    db = wrds.Connection()   # prompts for WRDS username/password, or reads ~/.pgpass
    vw_ew = pull_crsp_nyse_index(db, start="1926-01-01")
    rf = pull_crsp_riskfree(db, start="1926-01-01")
    be_earn = pull_compustat_be_and_earnings(db, start="1962-01-01")
    db.close()

Every function also accepts a `conn=None` and will raise a clear error telling
you to either pass a live `wrds.Connection()` or use `public_fallback.py`
instead -- so the rest of the pipeline (construct_panel.py) can be developed,
tested, and unit-tested without ever touching WRDS.
"""

from __future__ import annotations
import pandas as pd


class WRDSNotAvailable(RuntimeError):
    """Raised when a WRDS pull is attempted without a live connection."""
    pass


def _require_conn(conn):
    if conn is None:
        raise WRDSNotAvailable(
            "No WRDS connection supplied. Pass a live `wrds.Connection()` object, "
            "or use the functions in public_fallback.py for a non-WRDS demo/test run."
        )


# --------------------------------------------------------------------------
# CRSP: NYSE value-/equal-weighted index returns (with and without dividends)
# --------------------------------------------------------------------------

def pull_crsp_nyse_index(conn, start: str = "1926-01-01", end: str | None = None) -> pd.DataFrame:
    """
    Pull monthly NYSE value-weighted and equal-weighted returns, both including
    dividends (`vwretd`/`ewretd`) and excluding them (`vwretx`/`ewretx`), plus the
    index level (`totval`, used to construct the dividend flow) and the count of
    securities, from CRSP's monthly stock-index file (`crsp.msi`).

    NOTE ON UNIVERSE: `crsp.msi` is CRSP's headline "market index" file, which is
    NYSE-only prior to AMEX's addition in the 1960s and NASDAQ's addition in the
    1970s (exactly the composition-drift issue Lewellen's paper flags in Section 3
    -- "to avoid changes in the market's composition as AMEX and NASDAQ firms enter
    the database"). For an exchange-code-filtered, guaranteed NYSE-only series
    across the whole 1926-present window (recommended for the full replication),
    use `pull_crsp_nyse_index_exchcd_filtered` instead, which builds the index
    directly from `crsp.msf`/`crsp.msenames` restricted to `exchcd == 1`.

    Returns a DataFrame indexed by month-end date with columns:
        vwretd, vwretx, ewretd, ewretx, totval, totcnt
    """
    _require_conn(conn)
    query = """
        SELECT date, vwretd, vwretx, ewretd, ewretx, totval, totcnt
        FROM crsp.msi
        WHERE date >= %(start)s
        {end_clause}
        ORDER BY date
    """.format(end_clause="AND date <= %(end)s" if end else "")
    params = {"start": start}
    if end:
        params["end"] = end
    df = conn.raw_sql(query, params=params, date_cols=["date"])
    df["date"] = pd.to_datetime(df["date"]).values.astype("datetime64[M]")
    return df.set_index("date").sort_index()


def pull_crsp_nyse_index_exchcd_filtered(conn, start: str = "1926-01-01",
                                          end: str | None = None) -> pd.DataFrame:
    """
    Build a strictly NYSE-only (exchcd == 1) value-/equal-weighted monthly return
    index directly from security-level CRSP data, for users who want to avoid
    `crsp.msi`'s universe drift entirely. Value-weights by lagged market cap
    (`prc * shrout`); equal-weights with a simple mean. Both computed with and
    without dividends using `ret`/`retx`.

    This is slower and heavier than `pull_crsp_nyse_index` (it operates at the
    security-month level, not the pre-aggregated index level) but is the more
    faithful reproduction of "NYSE only" for the post-1962 period.
    """
    _require_conn(conn)
    query = """
        SELECT a.permno, a.date, a.ret, a.retx, a.prc, a.shrout,
               b.exchcd
        FROM crsp.msf a
        LEFT JOIN crsp.msenames b
            ON a.permno = b.permno
            AND a.date BETWEEN b.namedt AND b.nameendt
        WHERE b.exchcd = 1
            AND a.date >= %(start)s
            {end_clause}
    """.format(end_clause="AND a.date <= %(end)s" if end else "")
    params = {"start": start}
    if end:
        params["end"] = end
    sec = conn.raw_sql(query, params=params, date_cols=["date"])
    sec["date"] = pd.to_datetime(sec["date"])
    sec["mktcap"] = sec["prc"].abs() * sec["shrout"]
    sec = sec.dropna(subset=["ret", "mktcap"])
    sec["mktcap_lag"] = sec.groupby("permno")["mktcap"].shift(1)
    sec = sec.dropna(subset=["mktcap_lag"])

    def _agg(g):
        w = g["mktcap_lag"]
        return pd.Series({
            "vwretd": (g["ret"] * w).sum() / w.sum(),
            "vwretx": (g["retx"] * w).sum() / w.sum(),
            "ewretd": g["ret"].mean(),
            "ewretx": g["retx"].mean(),
            "totval": w.sum(),
            "totcnt": g["permno"].nunique(),
        })

    out = sec.groupby(sec["date"].values.astype("datetime64[M]")).apply(_agg)
    out.index.name = "date"
    return out.sort_index()


# --------------------------------------------------------------------------
# Risk-free rate
# --------------------------------------------------------------------------

def pull_crsp_riskfree(conn, start: str = "1926-01-01", end: str | None = None) -> pd.DataFrame:
    """
    Pull the one-month T-bill return from CRSP's risk-free-rate file
    (`crsp.mcti` -- "Monthly Treasury/Risk-Free File", `t30ret` = 1-month bill
    return). Returns a DataFrame indexed by month-end date with column `rf`
    (in the same percent units as CRSP/French-Library returns).
    """
    _require_conn(conn)
    query = """
        SELECT caldt AS date, t30ret AS rf
        FROM crsp.mcti
        WHERE caldt >= %(start)s
        {end_clause}
        ORDER BY caldt
    """.format(end_clause="AND caldt <= %(end)s" if end else "")
    params = {"start": start}
    if end:
        params["end"] = end
    df = conn.raw_sql(query, params=params, date_cols=["date"])
    df["date"] = pd.to_datetime(df["date"]).values.astype("datetime64[M]")
    df["rf"] = df["rf"] * 100  # CRSP stores as a decimal fraction; match percent convention
    return df.set_index("date").sort_index()


# --------------------------------------------------------------------------
# Compustat: aggregate NYSE book equity and operating earnings, via CCM link
# --------------------------------------------------------------------------

def pull_compustat_be_and_earnings(conn, start: str = "1962-01-01",
                                    end: str | None = None,
                                    min_years_history: int = 3) -> pd.DataFrame:
    """
    Construct aggregate NYSE book equity and operating income before depreciation,
    following the paper's construction (Section 3): book equity =
    CEQ + TXDITC - preferred stock; operating earnings = OIBDP; both summed across
    NYSE-listed firms (via the CRSP-Compustat merged linktable) with at least
    `min_years_history` years of Compustat history, as of each fiscal year-end.

    Book equity per Fama-French (1993): CEQ + TXDITC - preferred stock, where
    preferred stock is PSTKRV (redemption value), falling back to PSTKL
    (liquidating value), falling back to PSTK (par value), in that order --
    this fallback chain is the standard convention in the empirical literature.

    Returns a DataFrame indexed by fiscal-year-end date with columns:
        book_equity_sum, oibdp_sum, n_firms
    NOTE: this is deliberately at annual/fiscal-year-end granularity. The
    4-month reporting lag and the monthly panel merge (assigning each fiscal
    year's aggregate to the correct set of *months*) happen in
    `construct_panel.py`, not here -- this function's job is just the
    Compustat-side aggregation.
    """
    _require_conn(conn)
    query = """
        WITH be AS (
            SELECT
                f.gvkey, f.datadate, f.fyear,
                f.ceq, f.txditc,
                COALESCE(f.pstkrv, f.pstkl, f.pstk, 0) AS pstk_any,
                f.oibdp
            FROM comp.funda f
            WHERE f.indfmt = 'INDL'
              AND f.datafmt = 'STD'
              AND f.popsrc = 'D'
              AND f.consol = 'C'
              AND f.datadate >= %(start)s
              {end_clause}
        ),
        first_year AS (
            SELECT gvkey, MIN(fyear) AS first_fyear
            FROM comp.funda
            WHERE indfmt = 'INDL' AND datafmt = 'STD' AND popsrc = 'D' AND consol = 'C'
            GROUP BY gvkey
        ),
        link AS (
            SELECT gvkey, lpermno AS permno, linkdt, linkenddt
            FROM crsp.ccmxpf_linktable
            WHERE linktype IN ('LU','LC') AND linkprim IN ('P','C')
        ),
        nyse_names AS (
            SELECT DISTINCT permno
            FROM crsp.msenames
            WHERE exchcd = 1
        )
        SELECT be.gvkey, be.datadate, be.fyear,
               (be.ceq + be.txditc - be.pstk_any) AS book_equity,
               be.oibdp
        FROM be
        JOIN first_year fy ON be.gvkey = fy.gvkey
        JOIN link l ON be.gvkey = l.gvkey
            AND be.datadate BETWEEN l.linkdt AND COALESCE(l.linkenddt, CURRENT_DATE)
        JOIN nyse_names n ON l.permno = n.permno
        WHERE (be.fyear - fy.first_fyear) >= %(min_years)s
    """.format(end_clause="AND f.datadate <= %(end)s" if end else "")
    params = {"start": start, "min_years": min_years_history}
    if end:
        params["end"] = end
    firm_level = conn.raw_sql(query, params=params, date_cols=["datadate"])
    firm_level["datadate"] = pd.to_datetime(firm_level["datadate"])

    agg = (firm_level
           .groupby("datadate")
           .agg(book_equity_sum=("book_equity", "sum"),
                oibdp_sum=("oibdp", "sum"),
                n_firms=("gvkey", "nunique"))
           .sort_index())
    return agg