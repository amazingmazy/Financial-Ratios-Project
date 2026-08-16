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
import numpy as np
import pandas as pd

# Bump this whenever wrds_pull.py changes, and check it's printed at the top
# of your run's console output -- if it's missing or shows an old value,
# you're running a stale copy of this file, not the one you think you are.
WRDS_PULL_VERSION = "2024-issue2-v7-totval-current-month-fix"


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
    print(f"  [wrds_pull.py version: {WRDS_PULL_VERSION}]")
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
    return aggregate_security_level_to_monthly_index(sec)


def pull_crsp_nyse_index_ciz(conn, start: str = "1926-01-01",
                              end: str | None = None) -> pd.DataFrame:
    """
    CIZ-schema equivalent of `pull_crsp_nyse_index_exchcd_filtered`.

    CRSP's Flat File Format 2.0 ("CIZ") replaced the legacy "SIZ" schema in
    January 2025, and the legacy tables stopped being updated: as of this
    writing `crsp.msf` and `crsp.msi` end at 2024-12-31 while `crsp.msf_v2`
    runs to 2025-12-31. The replication window (1946-2000) sits entirely
    inside the legacy range, so this function exists for Issue 3 -- extending
    the sample to the present -- rather than to change the replication. Note
    that the legacy path fails *silently*: it returns a short panel rather
    than an error, so an extension built on it would quietly stop a year early.

    Two simplifications relative to the SIZ path:

      - No join. `crsp.msf_v2` carries the security descriptors (exchange,
        share type, issuer type) inline, so `crsp.msenames` and its
        namedt/nameendt date-range join are not needed.
      - Delisting returns are already incorporated into `mthret`, so there is
        no separate delisting file to merge.

    Filter mapping from the SIZ version:

        exchcd = 1                ->  primaryexch = 'N'
        shrcd in (10, 11)         ->  securitytype = 'EQTY'
                                      AND securitysubtype = 'COM'
                                      AND sharetype = 'NS'
                                      AND usincflg = 'Y'
                                      AND issuertype in ('ACOR', 'CORP')
        (n/a)                     ->  conditionaltype = 'RW'      (regular way)
        (n/a)                     ->  tradingstatusflg = 'A'      (active)

    The last two have no SIZ analogue and exclude securities that are halted,
    suspended, or settling on non-standard terms.

    Returns the same column set as the SIZ path (permno/date/ret/retx/prc/
    shrout, renamed from the CIZ names) and hands off to the same
    `aggregate_security_level_to_monthly_index`, so the aggregation logic --
    and the `totval` fix that logic encodes -- is shared rather than
    duplicated. `crsp.msf_v2` also publishes `mthcap` directly, which would
    avoid recomputing abs(prc)*shrout; we deliberately do not use it, so that
    both schemas produce market cap by exactly the same arithmetic and any
    difference between the two pulls is attributable to the source data
    rather than to two different cap definitions.

    One consequence worth carrying into the write-up: CIZ `mthret` compounds
    daily returns with dividends reinvested on the ex-date, whereas legacy
    `ret` was a month-to-month holding-period return reinvested at month end.
    Because the dividend flow is reconstructed as
    (vwretd - vwretx) * totval_{t-1}, that convention change propagates into
    DY, so the replication and the extension do not sit on identical footing.
    """
    _require_conn(conn)
    print(f"  [wrds_pull.py version: {WRDS_PULL_VERSION}]")
    query = """
        SELECT permno,
               mthcaldt AS date,
               mthret   AS ret,
               mthretx  AS retx,
               mthprc   AS prc,
               shrout
        FROM crsp.msf_v2
        WHERE primaryexch = 'N'
            AND securitytype = 'EQTY'
            AND securitysubtype = 'COM'
            AND sharetype = 'NS'
            AND usincflg = 'Y'
            AND issuertype IN ('ACOR', 'CORP')
            AND conditionaltype = 'RW'
            AND tradingstatusflg = 'A'
            AND mthcaldt >= %(start)s
            {end_clause}
    """.format(end_clause="AND mthcaldt <= %(end)s" if end else "")
    params = {"start": start}
    if end:
        params["end"] = end
    sec = conn.raw_sql(query, params=params, date_cols=["date"])
    sec["date"] = pd.to_datetime(sec["date"])
    return aggregate_security_level_to_monthly_index(sec)


def aggregate_security_level_to_monthly_index(sec: pd.DataFrame) -> pd.DataFrame:
    """
    Pure aggregation step, factored out of pull_crsp_nyse_index_exchcd_filtered
    so it can be unit-tested without a live WRDS connection.

    `sec` must have columns permno, date, ret, retx, prc, shrout (one row per
    security-month). Value-weights returns by LAGGED market cap (standard
    practice -- weighting by this month's realized value would create a
    mechanical look-ahead correlation between weights and the return being
    weighted), but reports `totval` using CURRENT-month market cap.

    This distinction matters concretely: `totval` is later used by
    construct_dividend_yield as DY_t's denominator, matching the paper's own
    definition of "dividends divided by the CURRENT level of the index." An
    earlier version of this function reused the lagged weights for `totval`
    too, which silently lagged the entire dividend-yield series by one
    month -- severing the contemporaneous link between a month's return and
    that same month's DY. Verified on a live run: this collapsed corr(e,m)
    from an expected ~-0.9 (mechanically, price up -> DY down, same month)
    to ~-0.03 to -0.15, which in turn muted both the Stambaugh correction's
    bias adjustment and the rho~1 test's standard-error reduction -- neither
    of which is a bug in the estimators themselves (verified separately by
    running the same estimator code on controlled synthetic data, where it
    correctly recovers corr(e,m) ~ -0.996).
    """
    sec = sec.copy()
    sec["mktcap"] = sec["prc"].abs() * sec["shrout"]
    sec = sec.dropna(subset=["ret", "mktcap"])
    sec["mktcap_lag"] = sec.groupby("permno")["mktcap"].shift(1)
    sec = sec.dropna(subset=["mktcap_lag"])

    def _agg(g):
        w = g["mktcap_lag"]  # weighting scheme for returns only
        return pd.Series({
            "vwretd": (g["ret"] * w).sum() / w.sum(),
            "vwretx": (g["retx"] * w).sum() / w.sum(),
            "ewretd": g["ret"].mean(),
            "ewretx": g["retx"].mean(),
            "totval": g["mktcap"].sum(),  # CURRENT-month cap, not lagged
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


def pull_riskfree_french(start: str = "1926-01-01",
                          end: str | None = None) -> pd.DataFrame:
    """
    One-month T-bill return from the Ken French Data Library, as a current
    substitute for `crsp.mcti` (which is frozen at 2024-12-31 -- see
    `pull_crsp_nyse_index_ciz`). Same output contract as
    `pull_crsp_riskfree`: indexed by month, single column `rf`, in percent.
    Takes no `conn`; French's library is a public download.

    Why French rather than a CIZ treasury table. CRSP's current treasury files
    do not expose a drop-in replacement for `mcti.t30ret`. `crsp.tfz_mth_rf`
    carries the right series (kytreasnox 2000001, "CRSP Risk Free Rates -
    1-Month (Nominal)", 1925-2025) but publishes *yields to maturity*, not
    realised monthly returns; checked against `t30ret` over 1946-2024 it
    correlates only 0.977, which is a yield-versus-return difference rather
    than noise. `crsp.tfz_mth_bp` does hold returns but they are the Fama bond
    maturity portfolios, and it starts in 1952 -- too late for a 1946 sample.

    French's RF is the one-month T-bill rate and is the standard definition of
    "excess return" in this literature, including the papers Lewellen builds
    on. Validated against `crsp.mcti.t30ret` over the 948 overlapping months
    from 1946-01 to 2024-12: correlation 0.9958, mean absolute difference
    0.012pp, maximum 0.175pp. Essentially all of the residual is French
    publishing to two decimals (0.39 against CRSP's 0.3907), so these are the
    same series at different printed precision rather than two measurements.

    Only excess returns depend on this; nominal VWNY/EWNY are untouched.
    """
    from public_fallback import pull_french_factors_and_rf

    fr = pull_french_factors_and_rf(start)
    df = fr[["rf"]].copy()
    df.index = pd.to_datetime(df.index).values.astype("datetime64[M]")
    df.index.name = "date"
    if end:
        df = df.loc[df.index <= pd.to_datetime(end).to_numpy().astype("datetime64[M]")]
    return df.sort_index()


# --------------------------------------------------------------------------
# Compustat: aggregate NYSE book equity and operating earnings, via CCM link
# --------------------------------------------------------------------------

def aggregate_firm_level_to_fiscal_year(firm_level: pd.DataFrame,
                                         units_millions_to_thousands: int = 1000,
                                         min_firms_per_year: int = 100,
                                         max_ratio_to_next_years: float = 0.35,
                                         discontinuity_lookahead: int = 3) -> pd.DataFrame:
    """
    Pure aggregation step, factored out of pull_compustat_be_and_earnings so it
    can be unit-tested without a live WRDS connection.

    `firm_level` must have columns gvkey, fyear, book_equity, oibdp (one row
    per firm-year, already NYSE-filtered and history-filtered). Groups by
    `fyear` (not exact datadate -- see module-level note on why), sums book
    equity and OIBDP across firms, converts from Compustat's $ millions to
    CRSP totval's $ thousands, and indexes the result by December 31 of each
    fiscal year.

    Two independent guards against unrepresentative early-coverage years,
    checked per-field (book equity and OIBDP separately -- see below):

    1. `min_firms_per_year` (default 100, was 30): any fiscal year built from
       fewer than this many firms with non-missing data for that field is set
       to NaN. Raised from the original default after a live run showed 30
       was too permissive -- mature years in this dataset average ~1,900
       NYSE firms, so 30 let through years that were still deep in Compustat's
       coverage ramp-up (see next point).

    2. `max_ratio_to_next_years` / `discontinuity_lookahead`: a year is ALSO
       set to NaN if its aggregate is less than `max_ratio_to_next_years`
       (default 0.35) times the average of the next `discontinuity_lookahead`
       years' aggregates. This catches exactly the failure mode a fixed firm
       count can miss: a live run found fiscal year 1961 had enough firms to
       clear a 30-firm threshold, but its aggregate book equity ($11.8M) was
       still only ~1/9th of 1962's ($123M) -- a coverage discontinuity, not
       real one-year growth, that a raw firm count alone doesn't distinguish
       from a genuinely small but representative year. A fixed threshold
       can't be tuned to catch this in general, since "how many firms is
       enough" itself changes across a sample spanning decades of Compustat's
       own historical coverage growth; comparing each year to its near-term
       neighbors adapts automatically.

    Both guards are applied independently to book_equity_sum and oibdp_sum
    (n_firms_be for the former, n_firms_oibdp for the latter) -- a live run
    showed one field can be well-covered in a year where the other isn't
    (e.g. 1953 had healthy OIBDP but zero non-missing book equity), so a
    single combined check would silently let a bad field's aggregate through
    riding on the other field's coverage.

    Set either guard's threshold to 0 / None to disable it.
    """
    agg = (firm_level
           .groupby("fyear")
           .agg(book_equity_sum=("book_equity", "sum"),
                oibdp_sum=("oibdp", "sum"),
                n_firms=("gvkey", "nunique"),
                n_firms_be=("book_equity", lambda s: s.notna().sum()),
                n_firms_oibdp=("oibdp", lambda s: s.notna().sum()))
           .sort_index())
    agg["book_equity_sum"] *= units_millions_to_thousands
    agg["oibdp_sum"] *= units_millions_to_thousands

    def apply_guards(sum_col, count_col, label):
        thin = agg[count_col] < min_firms_per_year
        if thin.any():
            print(f"  [debug] {thin.sum()} fiscal year(s) below the "
                  f"{min_firms_per_year}-firm minimum for {label}, set to NaN: "
                  f"{agg.index[thin].tolist()} ({count_col}: {agg.loc[thin, count_col].tolist()})")
            agg.loc[thin, sum_col] = np.nan

        if max_ratio_to_next_years:
            # Compare each year to the mean of the next N years, using
            # values BEFORE this function's own NaN-ing above so a bad year
            # doesn't get compared against neighbors already blanked out.
            raw = agg[sum_col].where(~thin)  # exclude already-thin years from the reference too
            forward_mean = pd.Series(
                [raw.iloc[i + 1:i + 1 + discontinuity_lookahead].mean() for i in range(len(raw))],
                index=raw.index,
            )
            ratio = agg[sum_col] / forward_mean
            discontinuous = (ratio < max_ratio_to_next_years) & forward_mean.notna() & ~thin
            if discontinuous.any():
                print(f"  [debug] {discontinuous.sum()} fiscal year(s) for {label} look like a "
                      f"coverage ramp-up (< {max_ratio_to_next_years:.0%} of the next "
                      f"{discontinuity_lookahead} years' average), set to NaN: "
                      f"{agg.index[discontinuous].tolist()} "
                      f"(ratio to forward average: {ratio[discontinuous].round(3).tolist()})")
                agg.loc[discontinuous, sum_col] = np.nan

    apply_guards("book_equity_sum", "n_firms_be", "BOOK EQUITY")
    apply_guards("oibdp_sum", "n_firms_oibdp", "OIBDP")

    agg.index = pd.to_datetime(agg.index.astype(int).astype(str) + "-12-31")
    agg.index.name = "datadate"
    return agg


def pull_compustat_be_and_earnings(conn, start: str = "1962-01-01",
                                    end: str | None = None,
                                    min_years_history: int = 3,
                                    include_deferred_taxes: bool = False) -> pd.DataFrame:
    """
    Construct aggregate NYSE book equity and operating income before depreciation,
    following the paper's construction (Section 3): book equity =
    CEQ [+ TXDITC] - preferred stock; operating earnings = OIBDP; both summed
    across NYSE-listed firms (via the CRSP-Compustat merged linktable) with at
    least `min_years_history` years of Compustat history.

    DEFERRED TAXES (`include_deferred_taxes`, default False)
    -------------------------------------------------------
    Whether to add TXDITC -- deferred taxes and investment tax credit -- to book
    equity. The Fama-French convention adds it; Lewellen appears not to. The
    paper says only "the ratio of book equity to market equity" and never gives
    a formula, so the text does not settle it. The data does:

        aggregate B/M, 1963-2000 mean      value    vs paper
        paper                              53.13      --
        CEQ + TXDITC - preferred           58.69    +10.5%
        CEQ - preferred                    52.13     -1.9%

    TXDITC is 12.1% of our aggregate book equity, and dropping it closes almost
    the entire gap. Three things corroborate that this is the cause rather than
    a coincidence:

      1. It cannot affect E/P, whose numerator is OIBDP -- an income-statement
         flow with no deferred-tax component. That is exactly why E/P diverges
         only about half as much as B/M (+5.0% against +10.5%).
      2. The gap is widest in the 1970s-80s (decade means 69 and 77, against 63
         and 66 once TXDITC is removed), precisely when accelerated depreciation
         and high inflation made deferred taxes largest.
      3. Ken French's own site documents that the deferred-tax treatment changed
         after FASB 109, so the convention is not stable even within the
         Fama-French lineage.

    Default is False, i.e. the variant that reproduces the paper, since
    replicating it is the point of Tables 5 and 6. Pass True for the
    Fama-French convention. Neither is "right" -- this is a definitional
    choice the source paper leaves open, so it is exposed rather than
    hard-coded, and the flag is recorded in the output so a panel can be
    traced back to the definition that produced it.

    A residual of roughly 5% remains after this adjustment and shows up in E/P
    too, so it is common to both ratios rather than specific to book equity.
    That is the firm-screen bundle -- the "three years of accounting data"
    requirement, whether the numerator's firm set should match the market-equity
    denominator's, aggregating-then-lagging versus lagging-then-aggregating for
    non-December fiscal year ends, and 20+ years of Compustat restatement. Not
    chased; see ISSUE2.md.

    NULL HANDLING: TXDITC is coalesced to 0 if missing -- standard Fama-French
    convention, since a missing value here usually means "not applicable" rather
    than "unknown" (immaterial when `include_deferred_taxes` is False, since the
    term is then dropped entirely). CEQ (common equity) is deliberately NOT
    coalesced -- if it's genuinely missing, the firm's book equity is unknown,
    not zero, and it should drop out of that year's sum rather than be
    misrepresented. See aggregate_firm_level_to_fiscal_year for how a fiscal
    year with too few non-missing CEQ values is handled.

    UNITS: Compustat dollar fields (ceq, txditc, pstk*, oibdp) are reported in
    $ millions; CRSP's crsp.msi.totval (used as the market-equity denominator
    downstream in construct_panel.py) is in $ thousands. The aggregation step
    (aggregate_firm_level_to_fiscal_year) multiplies book equity and OIBDP by
    1000 before returning, so both sides of the B/M and E/P ratios are in the
    same ($ thousands) units. If you change the totval source, revisit this.

    AGGREGATION GRANULARITY: firms are grouped by Compustat's `fyear` (the
    fiscal-year label), not by the exact `datadate` -- different firms have
    different fiscal year-end dates (Dec 31, June 30, etc.), so grouping by
    the literal date fragments what should be one "NYSE aggregate for fiscal
    year Y" into dozens of near-empty single-firm groups. Each fiscal year's
    aggregate is indexed by December 31 of that year, used as the assumed
    fiscal-year-end for the >=4-month reporting lag in construct_panel.py.

    NYSE MEMBERSHIP: checked as-of each firm's fiscal-year-end date against
    crsp.msenames' namedt/nameendt validity range, not "was this permno ever
    NYSE-listed at any point in its history." The latter, looser check would
    let a firm's book equity/earnings count toward the NYSE aggregate even in
    years it was actually listed elsewhere (or not yet listed at all),
    adding spurious composition noise across years -- a plausible contributor
    if you see B/M or E/P's standard deviation running higher than the
    paper's despite the mean matching well.

    Returns a DataFrame indexed by (Dec-31-of-fyear) date with columns:
        book_equity_sum, oibdp_sum, n_firms
    """
    _require_conn(conn)
    print(f"  [wrds_pull.py version: {WRDS_PULL_VERSION}]")
    query = """
        WITH be AS (
            SELECT
                f.gvkey, f.datadate, f.fyear,
                f.ceq, COALESCE(f.txditc, 0) AS txditc,
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
        )
        SELECT be.gvkey, be.datadate, be.fyear,
               (be.ceq {txditc_term} - be.pstk_any) AS book_equity,
               be.oibdp
        FROM be
        JOIN first_year fy ON be.gvkey = fy.gvkey
        JOIN link l ON be.gvkey = l.gvkey
            AND be.datadate::date BETWEEN l.linkdt::date AND COALESCE(l.linkenddt::date, CURRENT_DATE)
        JOIN crsp.msenames n ON l.permno = n.permno
            AND n.exchcd = 1
            AND be.datadate::date BETWEEN n.namedt::date AND n.nameendt::date
        WHERE (be.fyear - fy.first_fyear) >= %(min_years)s
    """.format(
        end_clause="AND f.datadate <= %(end)s" if end else "",
        txditc_term="+ be.txditc" if include_deferred_taxes else "",
    )
    print(f"  [info] book equity = CEQ "
          f"{'+ TXDITC ' if include_deferred_taxes else ''}- preferred stock "
          f"({'Fama-French convention' if include_deferred_taxes else 'paper-matching; see docstring'})")
    params = {"start": start, "min_years": min_years_history}
    if end:
        params["end"] = end
    firm_level = conn.raw_sql(query, params=params, date_cols=["datadate"])
    firm_level["datadate"] = pd.to_datetime(firm_level["datadate"])

    print(f"  [debug] {len(firm_level)} firm-year rows after all joins, "
          f"{firm_level['gvkey'].nunique()} unique firms, "
          f"{firm_level['fyear'].nunique()} unique fiscal years "
          f"({firm_level.groupby('fyear')['gvkey'].nunique().mean():.0f} firms/year on average -- "
          f"this should be in the hundreds for NYSE, not close to 1)")

    return aggregate_firm_level_to_fiscal_year(firm_level)
