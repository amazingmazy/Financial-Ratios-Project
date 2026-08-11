"""
dashboard.py
============
Extension 1: interactive educational dashboard (README Issue 4).

Four required pieces, each implemented as specified:
  1. Dashboard shell -- current ratio level, rho_hat, and OLS/Stambaugh/rho~1
     estimates + p-values side by side, for a selected ratio and window.
  2. Bayesian rho dial (the core feature) -- a control over the PRIOR belief
     about rho, spanning point mass at rho=1, a flat prior on rho<=1, and a
     shifted-normal prior with adjustable spread, each producing a live
     posterior probability that b<=0 -- reproducing the paper's own Section
     4.1 worked example (nominal EWNY, 1946-72: 0.147 -> 0.017 -> 0.032).
  3. Power indicator -- flags whether the current rho_hat clears the paper's
     own rule-of-thumb threshold for the conditional test to have power.
  4. Drop-last-N-years sensitivity panel -- generalizes Table 4's single
     1994-vs-2000 comparison to an adjustable N.

A bonus, non-required section (kept from an earlier version of this
dashboard) lets you directly explore a single point value of rho and see a
continuous power curve -- useful for building intuition, but NOT the
Bayesian dial itself; don't confuse the two.

Stretch goal from the README (an embedded Table A.1/Fig. A.1 power
simulation, letting b/rho/T vary) is NOT built here -- explicitly marked
lowest priority in the README, and left for a future pass.

Run with:
    streamlit run src/dashboard.py
"""

from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from estimators import fit_ar1, fit_predictive_ols, conditional_rho_test, stambaugh_correction
from dashboard_data import (
    PRESET_WINDOWS, PREDICTOR_LABELS, RETURN_LABELS, RAW_LEVEL_COL,
    load_panel, filter_window, build_x_r, power_curve_grid,
    flat_prior_weights, shifted_normal_prior_weights, integrate_posterior,
    power_threshold, has_power, truncate_window_by_years,
)

st.set_page_config(page_title="Lewellen (2004) dashboard", layout="wide")


@st.cache_data
def _load(path: str) -> pd.DataFrame:
    return load_panel(path)


@st.cache_data
def _stambaugh_cached(r: np.ndarray, x: np.ndarray, n_sims: int, seed: int = 0):
    return stambaugh_correction(r, x, n_sims=n_sims, random_state=seed)


st.title("Lewellen (2004) — interactive educational dashboard")

panel_path = st.sidebar.text_input("Master panel CSV path", value="data/master_panel.csv")
try:
    df = _load(panel_path)
except FileNotFoundError:
    st.error(f"Could not find {panel_path}. Run Issue 1's pipeline first, or point this at your master panel CSV.")
    st.stop()

available_predictors = [c for c in PREDICTOR_LABELS if c in df.columns and df[c].notna().any()]
if not available_predictors:
    st.error("No usable predictor columns (logDY / logB/M / logE/P) found in this panel.")
    st.stop()

predictor = st.sidebar.selectbox("Predictor", available_predictors, format_func=lambda c: PREDICTOR_LABELS.get(c, c))
return_col = st.sidebar.selectbox("Return series", list(RETURN_LABELS), format_func=lambda c: RETURN_LABELS.get(c, c))
window_label = st.sidebar.selectbox("Sample window", list(PRESET_WINDOWS))
start, end = PRESET_WINDOWS[window_label]
n_sims = st.sidebar.slider("Stambaugh Monte Carlo draws", 1000, 20000, 5000, step=1000)

sub_df = filter_window(df, start, end)
x, r, sub = build_x_r(sub_df, predictor, return_col)
if x is None:
    st.warning("Fewer than 30 valid observations for this predictor/series/window combination.")
    st.stop()

ar1 = fit_ar1(x)
rho_hat, se_rho = ar1["rho_hat"], ar1["se_rho"]
x_lag = x[:-1]
ols = fit_predictive_ols(r, x_lag)
stam = _stambaugh_cached(r, x, n_sims=n_sims)
cond_at_1 = conditional_rho_test(r, x, rho_assumed=0.999999)

raw_col = RAW_LEVEL_COL.get(predictor)
current_level = sub[raw_col].iloc[-1] if raw_col in sub.columns else None

# ==========================================================================
# 1. Dashboard shell
# ==========================================================================

st.header("1 · Dashboard shell")
shell_cols = st.columns(4)
shell_cols[0].metric("Current level", f"{current_level:.2f}%" if current_level is not None else "n/a",
                      help=f"Most recent {raw_col} in the selected window ({sub['date'].iloc[-1].date()})")
shell_cols[1].metric("ρ̂ (AR(1))", f"{rho_hat:.4f}")
shell_cols[2].metric("T (months)", f"{len(r)}")
shell_cols[3].metric("SE(ρ̂)", f"{se_rho:.5f}")

st.caption("OLS / Stambaugh / ρ≈1 side by side, for this predictor, series, and window:")
side_by_side = pd.DataFrame([
    {"Estimator": "OLS (naive)", "b̂": ols["b_hat"], "SE": ols["se_b"], "p-value": ols["p"]},
    {"Estimator": "Stambaugh (1999)", "b̂": stam.b_hat_adj, "SE": stam.se, "p-value": stam.p},
    {"Estimator": "ρ≈1 (Lewellen)", "b̂": cond_at_1.b_hat_adj, "SE": cond_at_1.se, "p-value": cond_at_1.p},
])
st.dataframe(side_by_side.style.format({"b̂": "{:.4f}", "SE": "{:.4f}", "p-value": "{:.4f}"}),
             hide_index=True, use_container_width=True)

# ==========================================================================
# 2. Bayesian rho dial (core feature)
# ==========================================================================

st.header("2 · Bayesian ρ dial")
st.caption(
    "The paper's own equivalence: for a fixed, known ρ, the conditional test's one-sided "
    "p-value **is** the posterior probability that b≤0 (flat prior on b). For any other "
    "belief about ρ, that same p(ρ) gets averaged over a prior/posterior density on ρ."
)

prior_type = st.radio(
    "Prior belief about ρ", ["Point mass at ρ=1", "Flat prior on ρ≤1", "Shifted-normal prior"],
    horizontal=True,
)

rho_grid = power_curve_grid(0.90, 0.999999, 250)
p_grid = np.array([conditional_rho_test(r, x, rho_assumed=rg).p for rg in rho_grid])

if prior_type == "Point mass at ρ=1":
    posterior_p = float(p_grid[-1])
    weights = None
    st.caption("Certainty that ρ=1 exactly -- this recreates the paper's conditional test exactly "
               "(the same number as the ρ≈1 row in the shell above).")
elif prior_type == "Flat prior on ρ≤1":
    weights = flat_prior_weights(rho_grid, rho_hat, se_rho)
    posterior_p = integrate_posterior(rho_grid, p_grid, weights)
    st.caption("No preference among any ρ≤1 -- combined with ρ̂'s own sampling distribution, "
               "this is a truncated-normal posterior on ρ centered at ρ̂.")
else:
    shift_in_se = st.slider("Shift (in units of SE(ρ̂))", -2.0, 3.0, 1.0, step=0.1,
                             help="The paper's own example uses +1.0 (shift the belief about ρ "
                                  "upward by one standard deviation of ρ̂'s sampling distribution).")
    spread_mult = st.slider("Spread (× SE(ρ̂))", 0.2, 5.0, 1.0, step=0.1,
                             help="The paper's own example uses 1.0x (same spread as ρ̂'s own SE).")
    weights = shifted_normal_prior_weights(rho_grid, rho_hat, se_rho,
                                            shift_in_se=shift_in_se, spread_multiplier=spread_mult)
    posterior_p = integrate_posterior(rho_grid, p_grid, weights)
    st.caption(f"Normal prior centered at ρ̂ + {shift_in_se:.1f}×SE(ρ̂) = "
               f"{rho_hat + shift_in_se * se_rho:.4f}, truncated at ρ≤1.")

st.metric("Posterior probability that b ≤ 0", f"{posterior_p:.4f}",
          delta="reject b≤0 at 5%" if posterior_p < 0.05 else "cannot reject b≤0 at 5%",
          delta_color="normal" if posterior_p < 0.05 else "off")

fig_bayes = go.Figure()
fig_bayes.add_trace(go.Scatter(x=rho_grid, y=p_grid, mode="lines", name="p(ρ)",
                                line=dict(color="#1E2761", width=2), yaxis="y1"))
if weights is not None:
    fig_bayes.add_trace(go.Scatter(x=rho_grid, y=weights / weights.max(), mode="lines",
                                    name="prior density (scaled)", line=dict(color="#C9A24B", width=2, dash="dot"),
                                    yaxis="y2"))
fig_bayes.add_vline(x=rho_hat, line_dash="dash", line_color="#5A6478",
                     annotation_text=f"ρ̂ = {rho_hat:.4f}")
fig_bayes.update_layout(
    xaxis_title="ρ", yaxis=dict(title="p(ρ)", range=[0, 1]),
    yaxis2=dict(title="prior density (scaled)", overlaying="y", side="right", range=[0, 1.1], showgrid=False),
    height=380, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=1.1),
)
st.plotly_chart(fig_bayes, use_container_width=True)

with st.expander("Reproduce the paper's own worked example (Section 4.1)"):
    st.write(
        "Select **nominal EWNY**, window **1946-1972 (Table 3a)**, and step through all three "
        "prior types above. The paper reports 0.147 → 0.017 → 0.032 for point mass → flat prior → "
        "shifted-normal; your real data won't match those exact numbers (different data, different "
        "era), but the *ordering* should hold: point mass gives the most conservative (highest) "
        "posterior probability, flat prior the least conservative, shifted-normal in between -- "
        "because ρ≈1 is deliberately the most conservative single assumption in the paper's own "
        "framework (Section 2.3)."
    )

# ==========================================================================
# 3. Power indicator
# ==========================================================================

st.header("3 · Does the conditional test have power right now?")
threshold_monthly = power_threshold(len(r), "monthly")
threshold_annual = power_threshold(len(r), "annual")
powered = has_power(rho_hat, len(r))

p1, p2, p3 = st.columns(3)
p1.metric("ρ̂ (this sample)", f"{rho_hat:.4f}")
p2.metric("Paper's rule-of-thumb threshold", f"{threshold_monthly:.4f}",
          help=f"Interpolated from the paper's two calibration points (25yr: 0.98 monthly / "
               f"0.85 annual; 50yr: 0.99 / 0.90) for a {len(r)/12:.0f}-year sample.")
if powered:
    p3.success("✓ ρ̂ clears the threshold — the conditional test should have real power here.")
else:
    p3.warning("✗ ρ̂ is below the threshold — the conditional test is unlikely to add much "
               "power over the unconditional (Stambaugh) test in this sample.")

# ==========================================================================
# 4. Drop-last-N-years sensitivity panel
# ==========================================================================

st.header("4 · Sensitivity: what if we drop the last N years?")
st.caption(
    "Generalizes the paper's Table 4 (which compares exactly 1946-1994 vs. 1946-2000) to any N."
)

max_drop = max(1, int((pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25) - 5)
n_drop = st.slider("Years dropped from the end of the window", 0, min(max_drop, 15), 0)

trunc_end = truncate_window_by_years(end, n_drop)
trunc_df = filter_window(df, start, trunc_end)
x_t, r_t, sub_t = build_x_r(trunc_df, predictor, return_col)

if x_t is None:
    st.warning("Not enough data left after dropping that many years.")
else:
    ar1_t = fit_ar1(x_t)
    ols_t = fit_predictive_ols(r_t, x_t[:-1])
    stam_t = _stambaugh_cached(r_t, x_t, n_sims=min(n_sims, 5000))  # capped for slider responsiveness
    cond_t = conditional_rho_test(r_t, x_t, rho_assumed=0.999999)

    st.write(f"Truncated window: {start} to {trunc_end} ({len(r_t)} months, ρ̂ = {ar1_t['rho_hat']:.4f})")
    sens = pd.DataFrame([
        {"Estimator": "OLS", "b̂": ols_t["b_hat"], "SE": ols_t["se_b"], "p-value": ols_t["p"]},
        {"Estimator": "Stambaugh", "b̂": stam_t.b_hat_adj, "SE": stam_t.se, "p-value": stam_t.p},
        {"Estimator": "ρ≈1", "b̂": cond_t.b_hat_adj, "SE": cond_t.se, "p-value": cond_t.p},
    ])
    st.dataframe(sens.style.format({"b̂": "{:.4f}", "SE": "{:.4f}", "p-value": "{:.4f}"}),
                 hide_index=True, use_container_width=True)
    st.caption(
        f"Full window: OLS b̂={ols['b_hat']:.3f}, Stambaugh b̂={stam.b_hat_adj:.3f}, "
        f"ρ≈1 b̂={cond_at_1.b_hat_adj:.3f}. Compare against the truncated numbers above -- the "
        f"paper's own finding is that OLS/Stambaugh move a lot as the last few years are added "
        f"back in, while ρ≈1 barely moves."
    )

# ==========================================================================
# Bonus (not required by the README): direct point-value rho exploration
# ==========================================================================

with st.expander("Bonus: explore a single point value of ρ directly (not the Bayesian dial above)"):
    st.caption(
        "This is a simpler, complementary tool -- pick one exact value of ρ (not a prior/belief "
        "distribution) and see the conditional test's result at exactly that value, plus a "
        "continuous power curve across the full range."
    )
    rho_point = st.slider("Assumed ρ (point value)", 0.90, 0.9999, min(0.9999, max(0.90, float(rho_hat))),
                           step=0.0001, format="%.4f", key="point_rho_slider")
    cond_point = conditional_rho_test(r, x, rho_assumed=rho_point)
    bc1, bc2, bc3, bc4 = st.columns(4)
    bc1.metric("b̂_adj", f"{cond_point.b_hat_adj:.4f}")
    bc2.metric("SE", f"{cond_point.se:.4f}")
    bc3.metric("t", f"{cond_point.t:.4f}")
    bc4.metric("p", f"{cond_point.p:.4f}")

    fig_point = go.Figure()
    fig_point.add_trace(go.Scatter(x=rho_grid, y=p_grid, mode="lines", line=dict(color="#1E2761", width=2)))
    fig_point.add_hline(y=0.05, line_dash="dot", line_color="#B85042", annotation_text="5% threshold")
    fig_point.add_vline(x=rho_hat, line_dash="dash", line_color="#5A6478", annotation_text=f"ρ̂={rho_hat:.4f}")
    fig_point.add_vline(x=rho_point, line_color="#C9A24B", line_width=3, annotation_text="current ρ")
    fig_point.update_layout(xaxis_title="Assumed ρ", yaxis_title="p-value", height=350,
                             margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig_point, use_container_width=True)

with st.expander("Data used in this regression"):
    st.write(f"{len(sub)} months, {sub['date'].min().date()} to {sub['date'].max().date()}")
    st.dataframe(sub[["date", predictor, return_col]].tail(10), hide_index=True)
