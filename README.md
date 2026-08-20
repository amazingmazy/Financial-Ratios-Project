# Replicating Lewellen (2004)

Replication and extension of Jonathan Lewellen, *"Predicting returns with financial
ratios,"* **Journal of Financial Economics** 74(2): 209–235.

Anthony Mazy and Stefano Ramponi · FINM 32900

---

## What the paper claims

Regress next month's market return on this month's dividend yield and the slope is
positive — but the yield's innovations are almost perfectly negatively correlated
with return innovations, because a price rise raises the return and mechanically
lowers the yield in the same month. Stambaugh (1986, 1999) showed this biases the
slope upward in small samples, and the standard correction removes most of the
apparent predictability.

Lewellen's contribution is that the standard correction discards something known for
free: a dividend yield cannot explode, so its autocorrelation ρ must be below 1.
Conditioning on that bound rather than integrating over every possible ρ gives a
tighter estimator — and flips the conclusion back to "returns are predictable."

## What we did

- **Replicated** Tables 1–6 on CRSP and Compustat data, with unit tests asserting our
  numbers against the paper's published values within stated tolerances
- **Extended** the sample from the paper's 2000 cutoff through 2025
- **Added** our own exhibit: dividend yield's persistence peaked right around when the
  paper was written and has eroded since, so the condition the method depends on is
  weaker today than in any window Lewellen could observe

Findings, including where our numbers diverge and why, are in
[`reports/replication_report.tex`](reports/replication_report.tex) (build it below) and
in [ISSUE1.md](ISSUE1.md) / [ISSUE2.md](ISSUE2.md).

---

## Running it

Requires a WRDS account with CRSP and Compustat access.

```bash
conda env create -f environment.yml
conda activate financial-ratios

cp .env.example .env          # then set WRDS_USERNAME
doit
```

`doit` runs the whole pipeline: pulls both CRSP panels, builds every table, runs the
test suite. Add the presentation layer with:

```bash
doit exhibits schema_compare_macros run_notebooks compile_latex_docs
```

Your WRDS **password is never read by this project** — the `wrds` client resolves it
from `~/.pgpass` (on Windows, `%APPDATA%/postgresql/pgpass.conf`). Set it up once with
`python -c "import wrds; wrds.Connection()"` and answer the prompt.

Without WRDS you can still run everything that doesn't need a pull: `doit test`
(the suite skips the data-dependent assertions), and `python run_issue1.py
--source synthetic` builds an offline panel that exercises the pipeline without
producing real numbers.

### Useful tasks

| task | what it does |
|---|---|
| `doit` | the full chain: both panels, all tables, tests |
| `doit panel_siz` / `panel_ciz` | pull one CRSP panel |
| `doit test` | test suite (`test_all` includes the slow Monte Carlo tests) |
| `doit exhibits` | our own summary table and figure |
| `doit compile_latex_docs` | build the report PDF |
| `doit run_notebooks` | execute the walkthrough, export HTML |
| `doit dashboard` | launch the interactive ρ dashboard |
| `doit list` | everything available |

Configuration — data directories, sample dates, simulation count — lives in
[`src/settings.py`](src/settings.py) with every key documented in
[`.env.example`](.env.example). Resolution order is CLI > environment > `.env` >
default.

---

## Where things are

```
run_issue1.py       orchestrates the data pull and QA gate
dodo.py             the build; every task above is defined here

src/
  settings.py               configuration
  wrds_pull.py              CRSP and Compustat queries (SIZ and CIZ schemas)
  construct_panel.py        cleaning only — produces the tidy panel
  qa_table1.py              QA gate against the paper's Table 1
  estimators.py             OLS, Stambaugh, rho~1, modified Bonferroni
  replicate_paper_tables.py Tables 2-6 and the extension windows
  paper_values.py           the paper's published values + tolerance rationale
  analysis.py               our own exhibits
  dashboard.py              interactive rho dashboard
  02_walkthrough.ipynb.py   guided tour of the data and the analysis

tests/              unit tests, including the assertions against the paper
reports/            LaTeX write-up
_data/, _output/    generated — gitignored, and never committed
```

CRSP and Compustat are licensed, so **no pulled data is in this repository**.
Everything under `_data/` and `_output/` is reproducible from `doit`.

### Start here

New to the project? Read
[`src/02_walkthrough.ipynb.py`](src/02_walkthrough.ipynb.py) — a guided tour of the
panel, the correlation that causes the bias, and the three estimators side by side.
`doit run_notebooks` renders it to HTML.

---

## Notes on the data

Two CRSP panels, deliberately. The legacy **SIZ** tables stopped being updated after
2024-12-31; the current **CIZ** schema reaches the present but compounds returns
differently. We replicate on SIZ, whose convention is closer to the data Lewellen
used, and extend on CIZ. Both are built rather than spliced, and the two agree on
every inference — see ISSUE2.md.

Book equity is `CEQ − preferred stock`, excluding deferred taxes. The paper never
states its formula, and including deferred taxes puts our aggregate B/M about 10%
above the published mean. The choice is a flag
(`pull_compustat_be_and_earnings(include_deferred_taxes=...)`) rather than a
hard-coded constant.

Scaffolded from
[cookiecutter_chartbook](https://github.com/backofficedev/cookiecutter_chartbook)
via `cruft`; `cruft check` reports the link is current.

## Division of work

| | |
|---|---|
| **Stefano Ramponi** | data pipeline (Issue 1), estimator library and paper tables (Issue 2), interactive dashboard, LaTeX report |
| **Anthony Mazy** | replication tests against published values, autocorrelation-estimator fix, CIZ schema port, sample extension, environment and build automation, walkthrough notebook |

