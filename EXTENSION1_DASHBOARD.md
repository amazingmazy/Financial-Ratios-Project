# Extension 1 — Interactive Educational Dashboard (README Issue 4)

An interactive tool built to match the README's Issue 4 specification
directly, not a looser approximation of it.

## Alignment with the README spec

| README requirement | Status |
|---|---|
| Dashboard shell: ratio level, ρ̂, OLS/Stambaugh/ρ≈1 side by side | Built (section 1) |
| **Bayesian ρ dial** (core feature): point mass at ρ=1, flat prior ρ≤1, shifted-normal with adjustable spread, live posterior P(b≤0) | Built (section 2) |
| Power indicator vs. paper's rule-of-thumb threshold | Built (section 3) |
| Drop-last-N-years sensitivity panel | Built (section 4) |
| Stretch: embedded Table A.1/Fig. A.1 power simulation |

## Files

```
src/dashboard.py         Streamlit app (UI layer), four numbered sections
                          matching the README's four required bullets
src/dashboard_data.py    Pure data-handling functions: window definitions,
                          x/r construction, the Bayesian posterior
                          integration, the power rule-of-thumb, and the
                          drop-N-years window truncation
tests/test_dashboard.py  Unit tests for every function in dashboard_data.py,
                          plus a live Streamlit AppTest smoke test that
                          drives all four required sections and asserts
                          they behave correctly
```

## Running it

```bash
pip install streamlit plotly
streamlit run src/dashboard.py
```

## What each section does

**1 · Dashboard shell.** Current level of the selected ratio (e.g. "DY:
3.8%"), ρ̂, T, SE(ρ̂), and a three-row table showing OLS / Stambaugh / ρ≈1
side by side for the current predictor, series, and window.

**2 · Bayesian ρ dial (the core feature).** The paper's own equivalence
(Section 4.1) is what makes this possible without new machinery: for a
*fixed, known* ρ, the conditional test's one-sided p-value already **is**
the Bayesian posterior probability that b≤0 (a standard result for a
one-sided test with a flat prior on b). So a "Bayesian dial" over the prior
belief about ρ just means averaging that same p(ρ) curve over a
prior/posterior density on ρ, instead of evaluating it at one point:

- **Point mass at ρ=1** — certainty; no averaging needed, this exactly
  recreates the paper's conditional test.
- **Flat prior on ρ≤1** — combined with ρ̂'s own sampling distribution
  (Normal(ρ̂, SE(ρ̂))), this gives a truncated-normal posterior on ρ.
- **Shifted-normal prior** — centered at ρ̂ + (adjustable shift)×SE(ρ̂), with
  adjustable spread. The paper's own worked example uses shift=+1,
  spread=1×SE(ρ̂); both are sliders here, so that's one setting among many,
  not the only option.

An expander walks through reproducing the paper's exact worked example
(nominal EWNY, 1946-72 window: 0.147 → 0.017 → 0.032) and explains why your
real data won't match those exact numbers but the *ordering* should —
point mass gives the most conservative (highest) posterior, flat prior the
least, shifted-normal in between, because ρ≈1 is deliberately the most
conservative single assumption in the paper's own framework (Section 2.3).

**3 · Power indicator.** Compares ρ̂ against the paper's own two stated
calibration points (25 years: 0.98 monthly / 0.85 annual; 50 years: 0.99 /
0.90 — Section 2.4), linearly interpolated/extrapolated for the actual
sample length. Flags whether the conditional test should have real power in
the current selection, or whether it's unlikely to beat Stambaugh's
unconditional test here.

**4 · Drop-last-N-years sensitivity panel.** A slider for N years dropped
from the end of the current window, recomputing all three estimators on the
truncated sample — generalizing Table 4's one fixed 1994-vs-2000 comparison
to any N. At N=6 on the 1946-2000 window, this exactly reproduces Table 4's
own comparison (2000-12-31 → 1994-12-31).