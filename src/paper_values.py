"""
paper_values.py
===============
Lewellen (2004), "Predicting returns with financial ratios," JFE 74:209-235 --
the paper's *published* numbers, transcribed by hand, plus the tolerance
specification used to decide whether our replication matches them.

This module contains DATA ONLY. It has no dependency on our pipeline, so it
can be read as a standalone reference and diffed against the PDF. Two
consumers:

  1. ``tests/test_replication_vs_paper.py`` -- asserts our computed numbers
     land within tolerance of these.
  2. ``src/tables_to_latex.py`` (Phase 5) -- builds the paper-vs-ours-vs-delta
     comparison table for the write-up.

--------------------------------------------------------------------------
TRANSCRIPTION HAZARD -- read before editing
--------------------------------------------------------------------------
The published PDF's *text layer* silently drops the minus sign on negative
numbers (an artifact of how the 2004 Elsevier typesetting encodes the
en-dash used for minus). Copying from selected text therefore yields
`0.38` where the rendered page shows `-0.38`. The pre-existing
``PAPER_TABLE_1`` in ``qa_table1.py`` was transcribed that way and has the
sign dropped on every negative skewness and on VWNY's rho24 in the second
half; it did not surface as a failure only because skew/rho12/rho24 are
informational there.

Everything below was transcribed from the *rendered* table, and every
negative value is spelled out explicitly. When editing, verify against the
rendered page, not the text layer. The named-field dataclasses (rather than
bare positional tuples) exist so that a misplaced value is visible at the
call site.
"""

from __future__ import annotations

from dataclasses import dataclass

# ==========================================================================
# Record types
# ==========================================================================


@dataclass(frozen=True)
class SummaryStats:
    """One row of the paper's Table 1 (p. 219).

    Variables are in percent; ``log(x)`` is the natural log of ``x``
    *expressed in percent* (so log(DY) is around 1.28, not around -3.3).
    """

    mean: float
    sd: float
    skew: float
    rho1: float
    rho12: float
    rho24: float


@dataclass(frozen=True)
class AR1Row:
    """The AR(1) block heading each of Tables 2, 3, 5, 6.

    Model: ``x_t = f + rho * x_{t-1} + m_t`` where x is the log ratio.
    ``bias`` is the paper's own "Bias" column (its estimate of E[rho_hat - rho]);
    ``kendall`` is its "-(1+3rho)/T" column, the Kendall-Marriott approximation.
    """

    rho: float
    se_rho: float
    bias: float
    kendall: float
    adj_r2: float
    sd_m: float


@dataclass(frozen=True)
class EstimatorRow:
    """One estimator's (slope, standard error, one-sided p-value) triple.

    The paper reports no standard error for its Stambaugh row in some
    printings; where it does, it is the simulated marginal SD of b_hat.
    """

    b: float
    se: float
    p: float


@dataclass(frozen=True)
class PredictiveBlock:
    """One return series' full block within a table.

    ``corr_em`` is the correlation between the predictive-regression residual
    e_t and the AR(1) residual m_t. It is the single most useful diagnostic in
    the whole exercise: it is mechanically close to -1 by construction (a
    positive return pushes the price into the ratio's denominator the same
    month), it drives both the size of the Stambaugh bias and the size of the
    rho~1 standard-error reduction, and a near-zero value means the panel is
    misaligned even when every univariate moment looks correct. That is
    exactly how the `totval` lag bug was found -- see ISSUE1.md, eighth round.
    """

    ols: EstimatorRow
    stambaugh: EstimatorRow
    rho1: EstimatorRow
    adj_r2: float
    sd_e: float
    corr_em: float


# ==========================================================================
# Sample windows and their published lengths
# ==========================================================================

# (start, end, T) exactly as the paper states them. T is asserted exactly --
# it is free to check, it cannot drift for a legitimate reason, and it is the
# fastest way to catch an off-by-one window or a stray dropna(). The paper
# states each of these counts explicitly in its table captions.
WINDOWS: dict[str, tuple[str, str, int]] = {
    # Tables 1-4: dividend yield. "January 1946-December 2000 (660 months)".
    "1946-2000": ("1946-01-01", "2000-12-31", 660),
    "1946-1972": ("1946-01-01", "1972-12-31", 324),
    "1973-2000": ("1973-01-01", "2000-12-31", 336),
    "1946-1994": ("1946-01-01", "1994-12-31", 588),
    # Tables 5-6: B/M and E/P. Note these start in *June* 1963, not January --
    # "June 1963-December 1994 (379 months)" / "June 1963-December 2000 (451
    # months)". Starting in January would give 384/456 and silently shift
    # every estimate.
    "1963-1994": ("1963-06-01", "1994-12-31", 379),
    "1963-2000": ("1963-06-01", "2000-12-31", 451),
}


# ==========================================================================
# Table 1 -- Summary statistics (p. 219)
# ==========================================================================

TABLE_1: dict[str, dict[str, SummaryStats]] = {
    "1946-2000": {
        "VWNY":  SummaryStats(1.04, 4.08, -0.38, 0.032, 0.042, 0.014),
        "EWNY":  SummaryStats(1.11, 4.80, -0.16, 0.136, 0.065, 0.027),
        "DY":    SummaryStats(3.80, 1.20, 0.37, 0.992, 0.889, 0.812),
        "logDY": SummaryStats(1.28, 0.33, -0.53, 0.997, 0.948, 0.912),
    },
    "1946-1972": {
        "VWNY":  SummaryStats(0.98, 3.67, -0.39, 0.079, 0.026, 0.066),
        "EWNY":  SummaryStats(1.04, 4.38, -0.26, 0.150, 0.024, 0.021),
        "DY":    SummaryStats(4.02, 1.21, 0.84, 0.992, 0.879, 0.774),
        # Positive skew here is correct and is the paper's point: logging makes
        # DY *more* symmetric than the raw ratio (0.84 -> 0.56) in the first
        # half, while the second half turns negative because of 1995-2000.
        "logDY": SummaryStats(1.35, 0.28, 0.56, 0.993, 0.876, 0.785),
    },
    "1973-2000": {
        "VWNY":  SummaryStats(1.10, 4.44, -0.39, 0.001, 0.065, -0.013),
        "EWNY":  SummaryStats(1.18, 5.18, -0.11, 0.125, 0.113, 0.030),
        "DY":    SummaryStats(3.59, 1.15, -0.19, 0.991, 0.899, 0.914),
        # rho24 = 1.062 is impossible for a Pearson correlation, and it is
        # printed as-is in the paper. Rather than an erratum, it is the single
        # most useful clue in Table 1: it identifies the estimator. Only an
        # unbounded statistic can exceed one, so Lewellen's autocorrelations
        # must be lag-k OLS slopes (cov / var(x_{t-k})), not Pearson
        # correlations (cov / sd*sd). Computing them that way reproduces this
        # cell at 1.044 and every other lag-12/lag-24 cell to within 0.02.
        # See qa_table1.lag_k_slope.
        "logDY": SummaryStats(1.22, 0.37, -0.81, 0.999, 0.996, 1.062),
    },
    "1963-2000": {
        "B/M":    SummaryStats(53.13, 18.28, 0.39, 0.990, 0.891, 0.837),
        "logB/M": SummaryStats(3.91, 0.36, -0.19, 0.995, 0.951, 0.923),
        "E/P":    SummaryStats(20.02, 7.01, 0.55, 0.988, 0.864, 0.770),
        "logE/P": SummaryStats(2.94, 0.35, 0.14, 0.990, 0.891, 0.785),
    },
}


# ==========================================================================
# Tables 2, 3, 5, 6 -- AR(1) and predictive regressions
# ==========================================================================
# Keyed as AR1[predictor][window] and PREDICTIVE[predictor][window][series].

AR1: dict[str, dict[str, AR1Row]] = {
    "logDY": {
        "1946-2000": AR1Row(0.997, 0.005, -0.008, -0.006, 0.984, 0.043),
        "1946-1972": AR1Row(0.993, 0.008, -0.016, -0.012, 0.981, 0.039),
        "1973-2000": AR1Row(0.999, 0.007, -0.016, -0.012, 0.984, 0.046),
        "1946-1994": AR1Row(0.986, 0.007, -0.008, -0.007, 0.971, 0.043),
    },
    "logB/M": {
        "1963-1994": AR1Row(0.987, 0.009, -0.013, -0.010, 0.972, 0.047),
        "1963-2000": AR1Row(0.995, 0.006, -0.011, -0.009, 0.983, 0.047),
    },
    "logE/P": {
        "1963-1994": AR1Row(0.987, 0.008, -0.011, -0.010, 0.978, 0.049),
        "1963-2000": AR1Row(0.990, 0.007, -0.010, -0.009, 0.980, 0.049),
    },
}


PREDICTIVE: dict[str, dict[str, dict[str, PredictiveBlock]]] = {
    # ---------------- Table 2 & 3: dividend yield (pp. 221, 223) -----------
    "logDY": {
        "1946-2000": {
            "VWNY": PredictiveBlock(
                ols=EstimatorRow(0.917, 0.476, 0.027),
                stambaugh=EstimatorRow(0.196, 0.670, 0.308),
                rho1=EstimatorRow(0.663, 0.142, 0.000),
                adj_r2=0.004, sd_e=4.068, corr_em=-0.955),
            "EWNY": PredictiveBlock(
                ols=EstimatorRow(1.388, 0.558, 0.007),
                stambaugh=EstimatorRow(0.615, 0.758, 0.183),
                rho1=EstimatorRow(1.115, 0.268, 0.000),
                adj_r2=0.008, sd_e=4.773, corr_em=-0.878),
            "ExcVWNY": PredictiveBlock(
                ols=EstimatorRow(0.915, 0.478, 0.028),
                stambaugh=EstimatorRow(0.192, 0.673, 0.311),
                rho1=EstimatorRow(0.661, 0.145, 0.000),
                adj_r2=0.004, sd_e=4.087, corr_em=-0.953),
            "ExcEWNY": PredictiveBlock(
                ols=EstimatorRow(1.387, 0.560, 0.007),
                stambaugh=EstimatorRow(0.611, 0.760, 0.184),
                rho1=EstimatorRow(1.112, 0.268, 0.000),
                adj_r2=0.008, sd_e=4.788, corr_em=-0.878),
        },
        "1946-1972": {
            "VWNY": PredictiveBlock(
                ols=EstimatorRow(1.421, 0.723, 0.025),
                stambaugh=EstimatorRow(0.013, 1.400, 0.405),
                rho1=EstimatorRow(0.844, 0.257, 0.001),
                adj_r2=0.009, sd_e=3.649, corr_em=-0.935),
            "EWNY": PredictiveBlock(
                ols=EstimatorRow(1.077, 0.863, 0.107),
                stambaugh=EstimatorRow(-0.511, 1.637, 0.551),
                # The table prints 0.152; the discussion on p. 224 says 0.147
                # for the same regression. The discrepancy is in the paper.
                rho1=EstimatorRow(0.425, 0.405, 0.152),
                adj_r2=0.002, sd_e=4.359, corr_em=-0.884),
            "ExcVWNY": PredictiveBlock(
                ols=EstimatorRow(1.733, 0.724, 0.009),
                stambaugh=EstimatorRow(0.325, 1.402, 0.327),
                rho1=EstimatorRow(1.156, 0.261, 0.000),
                adj_r2=0.014, sd_e=3.656, corr_em=-0.933),
            "ExcEWNY": PredictiveBlock(
                ols=EstimatorRow(1.388, 0.864, 0.055),
                stambaugh=EstimatorRow(-0.199, 1.639, 0.460),
                rho1=EstimatorRow(0.736, 0.407, 0.037),
                adj_r2=0.005, sd_e=4.364, corr_em=-0.883),
        },
        "1973-2000": {
            "VWNY": PredictiveBlock(
                ols=EstimatorRow(0.732, 0.665, 0.136),
                stambaugh=EstimatorRow(-0.751, 1.254, 0.700),
                rho1=EstimatorRow(0.641, 0.167, 0.000),
                adj_r2=0.001, sd_e=4.441, corr_em=-0.968),
            "EWNY": PredictiveBlock(
                ols=EstimatorRow(1.703, 0.770, 0.014),
                stambaugh=EstimatorRow(0.152, 1.397, 0.367),
                rho1=EstimatorRow(1.608, 0.373, 0.000),
                adj_r2=0.012, sd_e=5.142, corr_em=-0.875),
            "ExcVWNY": PredictiveBlock(
                ols=EstimatorRow(0.395, 0.669, 0.278),
                stambaugh=EstimatorRow(-1.096, 1.261, 0.835),
                rho1=EstimatorRow(0.304, 0.169, 0.041),
                adj_r2=-0.002, sd_e=4.467, corr_em=-0.968),
            "ExcEWNY": PredictiveBlock(
                ols=EstimatorRow(1.366, 0.774, 0.039),
                stambaugh=EstimatorRow(-0.194, 1.406, 0.474),
                rho1=EstimatorRow(1.271, 0.376, 0.000),
                adj_r2=0.006, sd_e=5.174, corr_em=-0.875),
        },
    },
    # ---------------- Table 5: book-to-market (p. 227) ---------------------
    "logB/M": {
        "1963-1994": {
            "VWNY": PredictiveBlock(
                ols=EstimatorRow(1.801, 0.784, 0.011),
                stambaugh=EstimatorRow(0.772, 1.240, 0.220),
                rho1=EstimatorRow(0.731, 0.341, 0.017),
                adj_r2=0.011, sd_e=4.265, corr_em=-0.901),
            "EWNY": PredictiveBlock(
                ols=EstimatorRow(2.312, 0.963, 0.008),
                stambaugh=EstimatorRow(1.150, 1.493, 0.189),
                rho1=EstimatorRow(1.107, 0.545, 0.022),
                adj_r2=0.012, sd_e=5.239, corr_em=-0.826),
            "ExcVWNY": PredictiveBlock(
                ols=EstimatorRow(1.275, 0.790, 0.054),
                stambaugh=EstimatorRow(0.236, 1.250, 0.352),
                rho1=EstimatorRow(0.196, 0.342, 0.291),
                adj_r2=0.004, sd_e=4.298, corr_em=-0.902),
            "ExcEWNY": PredictiveBlock(
                ols=EstimatorRow(1.786, 0.969, 0.033),
                stambaugh=EstimatorRow(0.615, 1.503, 0.287),
                rho1=EstimatorRow(0.571, 0.547, 0.151),
                adj_r2=0.006, sd_e=5.274, corr_em=-0.827),
        },
        "1963-2000": {
            "VWNY": PredictiveBlock(
                ols=EstimatorRow(0.698, 0.594, 0.108),
                stambaugh=EstimatorRow(-0.222, 0.945, 0.515),
                rho1=EstimatorRow(0.276, 0.258, 0.149),
                adj_r2=0.001, sd_e=4.237, corr_em=-0.890),
            "EWNY": PredictiveBlock(
                ols=EstimatorRow(1.484, 0.670, 0.014),
                stambaugh=EstimatorRow(0.501, 1.089, 0.266),
                rho1=EstimatorRow(1.032, 0.400, 0.005),
                adj_r2=0.009, sd_e=5.032, corr_em=-0.802),
            "ExcVWNY": PredictiveBlock(
                ols=EstimatorRow(0.351, 0.567, 0.268),
                stambaugh=EstimatorRow(-0.576, 0.950, 0.698),
                rho1=EstimatorRow(-0.075, 0.257, 0.626),
                adj_r2=-0.001, sd_e=4.260, corr_em=-0.892),
            "ExcEWNY": PredictiveBlock(
                ols=EstimatorRow(1.136, 0.674, 0.046),
                stambaugh=EstimatorRow(0.147, 1.096, 0.367),
                rho1=EstimatorRow(0.681, 0.402, 0.047),
                adj_r2=0.004, sd_e=5.060, corr_em=-0.803),
        },
    },
    # ---------------- Table 6: earnings-price (p. 228) ---------------------
    "logE/P": {
        "1963-1994": {
            "VWNY": PredictiveBlock(
                ols=EstimatorRow(1.566, 0.660, 0.009),
                stambaugh=EstimatorRow(0.495, 1.071, 0.213),
                rho1=EstimatorRow(0.566, 0.332, 0.046),
                adj_r2=0.012, sd_e=4.263, corr_em=-0.866),
            "EWNY": PredictiveBlock(
                ols=EstimatorRow(1.950, 0.811, 0.008),
                stambaugh=EstimatorRow(0.655, 1.295, 0.194),
                rho1=EstimatorRow(0.820, 0.493, 0.049),
                adj_r2=0.012, sd_e=5.239, corr_em=-0.796),
            "ExcVWNY": PredictiveBlock(
                ols=EstimatorRow(1.119, 0.665, 0.047),
                stambaugh=EstimatorRow(0.039, 1.079, 0.332),
                rho1=EstimatorRow(0.108, 0.332, 0.380),
                adj_r2=0.005, sd_e=4.296, corr_em=-0.868),
            "ExcEWNY": PredictiveBlock(
                ols=EstimatorRow(1.503, 0.817, 0.033),
                stambaugh=EstimatorRow(0.198, 1.305, 0.282),
                rho1=EstimatorRow(0.363, 0.495, 0.236),
                adj_r2=0.005, sd_e=5.275, corr_em=-0.798),
        },
        "1963-2000": {
            "VWNY": PredictiveBlock(
                ols=EstimatorRow(1.121, 0.576, 0.026),
                stambaugh=EstimatorRow(0.205, 0.916, 0.275),
                rho1=EstimatorRow(0.403, 0.294, 0.088),
                adj_r2=0.006, sd_e=4.226, corr_em=-0.861),
            "EWNY": PredictiveBlock(
                ols=EstimatorRow(1.753, 0.685, 0.005),
                stambaugh=EstimatorRow(0.691, 1.062, 0.159),
                rho1=EstimatorRow(0.983, 0.432, 0.012),
                adj_r2=0.012, sd_e=5.023, corr_em=-0.778),
            "ExcVWNY": PredictiveBlock(
                ols=EstimatorRow(0.725, 0.580, 0.106),
                stambaugh=EstimatorRow(-0.198, 0.923, 0.424),
                rho1=EstimatorRow(0.000, 0.294, 0.510),
                adj_r2=0.001, sd_e=4.255, corr_em=-0.863),
            "ExcEWNY": PredictiveBlock(
                ols=EstimatorRow(1.358, 0.689, 0.025),
                stambaugh=EstimatorRow(0.288, 1.069, 0.247),
                rho1=EstimatorRow(0.580, 0.433, 0.093),
                adj_r2=0.006, sd_e=5.054, corr_em=-0.780),
        },
    },
}


# ==========================================================================
# Tolerances
# ==========================================================================
"""
How these were chosen
---------------------
A tolerance has to be loose enough to absorb every *legitimate* reason our
number can differ from a number published in 2004, and tight enough to fail
when the pipeline is actually wrong. The legitimate reasons, in rough order
of how much they can move a statistic:

  (a) CRSP data vintage. Twenty-two years of revisions and delisting-return
      backfills separate Lewellen's extract from ours.
  (b) CRSP schema. The CIZ (Flat File 2.0) format replaced SIZ in January
      2025 and changed return compounding: CIZ `mthret` compounds daily
      returns with dividends reinvested on the ex-date, where legacy `ret`
      was a month-to-month holding-period return reinvested at month end.
      Our dividend flow is reconstructed as (vwretd - vwretx) * totval_{t-1},
      so this propagates straight into DY.
  (c) Construction ambiguity. The paper says DY is "dividends paid over the
      prior year divided by the current level of the index" but does not
      publish the reconstruction; ours is an inference.
  (d) Compustat restatement and backfill, plus the gradual book-equity
      coverage ramp through ~1966 already documented in ISSUE1.md.
  (e) Universe ambiguity -- "NYSE" does not pin down share codes or whether
      the index came from CRSP's own series or from security-level aggregation.
  (f) Published rounding to 2-3 significant figures (+/- 0.005 on a 2dp value).

Monte Carlo noise is deliberately NOT in this list: it is eliminated by
seeding, and `test_estimators_are_reproducible_under_seed` asserts that.
That is what buys the room to keep the load-bearing tolerances tight.

The organising principle: tolerance follows PROPAGATION, not statistic type.
-----------------------------------------------------------------------------
Only three quantities actually flow into the estimates the paper is about:

    b_hat_adj = b_hat - gamma_hat * (rho_hat - 1),  gamma_hat = cov(e,m)/var(m)

so what matters is rho_hat, the dispersion of the log ratio (which sets
var(x) and hence b_hat), and corr(e,m) (which sets gamma_hat). Everything
else in Table 1 is descriptive.

rho_hat deserves emphasis because the sensitivity is violent. For VWNY
1946-2000 the paper has gamma_hat = -90.4 and rho_hat = 0.997, so the bias
correction is -(-90.4)(0.997-1) = -0.271, taking OLS 0.917 down to 0.663 --
which reproduces the published number. Move rho_hat by just 0.01, to 0.987,
and the correction becomes -1.18, giving b_hat_adj = -0.26: a sign flip and a
different paper. A tolerance on rho_hat of 0.01 is therefore far too loose to
be meaningful, even though 0.01 "looks" tight next to a value of 0.997. We
set it at 0.002, which back-implies about +/-0.18 on b_hat_adj -- consistent
with the slope tolerance below. This is a demanding bar, and it is the right
one: a highly persistent series' autocorrelation is very stable to small data
perturbations, so failing it means something is structurally wrong rather
than merely revised.

Tiers
-----
  HARD/exact  Sample length T. Free to check, cannot legitimately drift.
  HARD/tight  Load-bearing: rho1, SD of log ratios, corr(e,m).
  HARD/loose  Diagnostic: return means and SDs, ratio levels, slopes, SEs,
              rho12, rho24.
  INFO        Reported but never gated: skew, adj_r2, sd_e.

Only skewness is genuinely informational: it enters no estimator and is
dominated by single months such as October 1987, so it moves on data revisions
without telling us anything about the pipeline.

A correction worth recording, because it changed the design. rho12 and rho24
were initially demoted alongside skew on the grounds that the paper's own logDY
rho24 of 1.062 exceeds one -- impossible for a Pearson correlation -- so exact
agreement looked unachievable by construction. The premise was correct; the
conclusion was not. An out-of-range value does not mean the statistic is
irreproducible, it means the estimator has been misidentified: only an
unbounded statistic can exceed one, so the paper's autocorrelations must be
lag-k OLS slopes (cov / var(x_{t-k})) rather than Pearson correlations
(cov / sd * sd). Recomputing on that basis reproduces every published lag-12
and lag-24 cell to within 0.02, the 1.062 included. Both are now gated, which
recovered roughly two dozen of the paper's numbers that had been written off.
The general lesson: a statistic that cannot be reproduced with the obvious
estimator is evidence about which estimator was used.
"""

# Gate levels.
HARD = "hard"    # counts toward pass/fail
INFO = "info"    # computed and reported, never fails the gate


@dataclass(frozen=True)
class Tol:
    """A tolerance: pass if |ours - paper| <= abs_tol OR <= rel_tol * |paper|.

    The OR (rather than AND) matters near zero, where relative error explodes:
    VWNY's rho1 of 0.032 versus a computed 0.041 is a 28% relative miss but is
    two statistically indistinguishable near-zero numbers. ``abs_tol`` is the
    floor that lets such a case pass on its own terms. This is the same
    relative-or-absolute design already used by ``qa_table1.ABS_TOLERANCE``;
    only the numbers are tightened here.
    """

    abs_tol: float = 0.0
    rel_tol: float = 0.0
    gate: str = HARD
    why: str = ""

    def passes(self, ours: float, paper: float) -> bool:
        diff = abs(ours - paper)
        return diff <= self.abs_tol or (
            self.rel_tol > 0 and diff <= self.rel_tol * abs(paper)
        )


# -------------------------------------------------------------------------
# Table 1 tolerances, per (series-kind, statistic).
# -------------------------------------------------------------------------
# Series are classed as "vw" (value-weighted return), "ew" (equal-weighted
# return), "level" (a raw ratio: DY, B/M, E/P) or "log" (a log ratio).
TABLE_1_TOL: dict[tuple[str, str], Tol] = {
    # --- returns -----------------------------------------------------------
    ("vw", "mean"): Tol(abs_tol=0.05, why=(
        "Monthly mean in percent. Revisions touching a handful of months move "
        "this by ~0.01pp; 0.05pp still catches a wrong universe, since adding "
        "AMEX/NASDAQ or mis-setting the window shifts it by far more.")),
    ("ew", "mean"): Tol(abs_tol=0.10, why=(
        "Looser than value-weighted on purpose: equal weighting puts full "
        "weight on the small-cap tail, which is exactly where delisting "
        "returns and coverage backfills have been revised most since 2004.")),
    ("vw", "sd"): Tol(abs_tol=0.10, why=(
        "~2.5% of the 4.08 level. Dispersion is driven by the bulk of the "
        "distribution, so it is markedly more revision-stable than the mean.")),
    ("ew", "sd"): Tol(abs_tol=0.15, why="As above, widened for the small-cap tail."),
    # --- ratio levels ------------------------------------------------------
    ("level", "mean"): Tol(rel_tol=0.05, why=(
        "DY/B/M/E/P levels depend entirely on our reconstruction of the "
        "dividend flow and of book equity, neither of which the paper fully "
        "specifies. Note this statistic is diagnostic, not load-bearing: the "
        "regressions use the LOG ratio, so a constant scaling error in the "
        "level is absorbed by the intercept and does not bias b_hat at all. "
        "We gate it to catch a grossly wrong construction, not to certify it.")),
    ("level", "sd"): Tol(rel_tol=0.08, why=(
        "Inherits the level's construction uncertainty and adds sensitivity "
        "to the sample's own dispersion.")),
    ("log", "mean"): Tol(abs_tol=0.05, why=(
        "log of a percent-valued ratio. Purely a location parameter, absorbed "
        "by the regression intercept; diagnostic only.")),
    # --- the load-bearing ones --------------------------------------------
    ("log", "sd"): Tol(rel_tol=0.06, why=(
        "LOAD-BEARING. var(x) is the denominator of the OLS slope, so an "
        "error here scales b_hat directly. 6% is about the widest that still "
        "keeps the slope inside its own 20% tolerance. Expect logB/M to be "
        "the first failure: ISSUE1.md documents its SD running ~0.46 against "
        "the paper's 0.36 because of the Compustat coverage ramp through 1966.")),
    ("vw", "rho1"): Tol(abs_tol=0.03, why=(
        "Return autocorrelations are near zero (0.032) and are not used by any "
        "estimator; the absolute floor does the work.")),
    ("ew", "rho1"): Tol(abs_tol=0.03, why="As above."),
    ("level", "rho1"): Tol(abs_tol=0.003, why="LOAD-BEARING -- see ('log','rho1')."),
    ("log", "rho1"): Tol(abs_tol=0.002, why=(
        "THE load-bearing statistic. b_hat_adj = b_hat - gamma*(rho_hat-1) "
        "with gamma about -90, so an error of e in rho_hat moves the "
        "bias-adjusted slope by roughly 90*e. At 0.002 that is +/-0.18, which "
        "matches the slope tolerance; at 0.01 it would be 0.9 and could flip "
        "the sign of the paper's headline result. Tight by necessity, and "
        "achievable because a near-unit-root series' autocorrelation barely "
        "moves under small data revisions.")),
    # --- informational -----------------------------------------------------
    ("vw", "skew"): Tol(abs_tol=0.25, gate=INFO, why="Dominated by single months (Oct 1987)."),
    ("ew", "skew"): Tol(abs_tol=0.25, gate=INFO, why="As above."),
    ("level", "skew"): Tol(abs_tol=0.25, gate=INFO, why="As above."),
    ("log", "skew"): Tol(abs_tol=0.25, gate=INFO, why="As above."),
}

# rho12 and rho24 are GATED, having previously been written off as
# irreproducible. See the module docstring: the paper's autocorrelations are
# lag-k OLS slopes rather than Pearson correlations, and once computed that way
# (qa_table1.lag_k_slope) every published lag-12 and lag-24 value reproduces to
# within 0.02 -- including the logDY rho24 of 1.062 that exceeds one and was
# the clue identifying the estimator in the first place.
#
# 0.03 sits just above the largest observed miss (0.018, logDY rho24 for
# 1973-2000, the least stable cell in the table) while still being tight enough
# that a wrong lag or a wrong estimator fails loudly: Pearson misses these
# cells by up to 0.21 and the standard ACF by up to 0.38.
for _kind in ("vw", "ew", "level", "log"):
    for _lag in ("rho12", "rho24"):
        TABLE_1_TOL[(_kind, _lag)] = Tol(
            abs_tol=0.03,
            why="Reproducible once computed as a lag-k OLS slope, the paper's "
                "own convention. Diagnostic rather than load-bearing -- only "
                "lag 1 enters the estimators -- but a long-lag mismatch is a "
                "good early signal that the persistence structure is wrong.")
del _kind, _lag


# -------------------------------------------------------------------------
# Tolerances for the regression tables (2, 3, 5, 6).
# -------------------------------------------------------------------------
REGRESSION_TOL: dict[str, Tol] = {
    "rho": Tol(abs_tol=0.002, why="Same statistic and same reasoning as ('log','rho1')."),
    "corr_em": Tol(abs_tol=0.03, why=(
        "LOAD-BEARING, and the best canary in the suite. gamma = cov(e,m)/var(m) "
        "scales the entire bias correction, and corr(e,m) is mechanically near "
        "-0.9 in any correct construction. A near-zero value means returns and "
        "the ratio are misaligned in time -- which is precisely the symptom "
        "that exposed the `totval` lag bug (ISSUE1.md, eighth round) after "
        "Table 1 had passed cleanly.")),
    "b": Tol(abs_tol=0.10, rel_tol=0.20, why=(
        "Slopes are small and, on the paper's own standard errors, noisy: "
        "VWNY 1946-2000 has b = 0.917 with SE 0.476, so 20% is well under half "
        "a standard error. The 0.10 absolute floor prevents blowup where the "
        "published slope is near zero (Stambaugh's 0.013 for VWNY 1946-1972, "
        "or E/P's 0.000 for excess VWNY 1963-2000), where a relative test is "
        "meaningless. The scientific claim is the sign and the ordering across "
        "estimators, which the qualitative assertions test separately and far "
        "more robustly than any digit comparison.")),
    "se": Tol(abs_tol=0.10, rel_tol=0.20, why="Same reasoning as the slope."),
    "p": Tol(abs_tol=0.10, why=(
        "A loose absolute band, because the decision-relevant comparison is "
        "categorical, not numeric: p = 0.03 versus 0.045 is the same finding, "
        "while 0.03 versus 0.30 is the paper's entire argument. The categorical "
        "check at alpha = 0.05 is asserted separately and is the real test.")),
    "adj_r2": Tol(abs_tol=0.01, gate=INFO, why=(
        "Reported for completeness. These are ~0.004-0.014 -- economically "
        "tiny by construction in return prediction -- so gating on them would "
        "be noise-fitting.")),
    "sd_e": Tol(rel_tol=0.05, gate=INFO, why="Descriptive; tracks the return SD already gated in Table 1."),
}


def classify_series(name: str) -> str:
    """Map a panel column name to its tolerance class: vw / ew / level / log.

    Used to look up ``TABLE_1_TOL[(kind, stat)]``. Value- and equal-weighted
    returns are separated because equal weighting is materially more exposed
    to small-cap delisting revisions; raw and log ratios are separated because
    only the log series feed the regressions.
    """
    if name.startswith("log"):
        return "log"
    if name in ("DY", "B/M", "E/P"):
        return "level"
    if "EW" in name:
        return "ew"
    if "VW" in name:
        return "vw"
    raise KeyError(f"no tolerance class for series {name!r}")
