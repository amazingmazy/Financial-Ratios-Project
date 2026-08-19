# Rubric tracker

Working reference against the final project rubric (202 points). Update the
status column as items land; keep the notes pointing at evidence rather than
intentions, so this stays a record of what is true rather than what we mean to
do.

`[x]` done &nbsp;·&nbsp; `[~]` partial &nbsp;·&nbsp; `[ ]` not started

Last reviewed: after the cruft retrofit (scaffold, settings.py, dodo.py, LaTeX chain).

## Deliverables

| # | Item | Pts | Status | Notes |
|---|---|---|---|---|
| 1 | Single LaTeX document: describes the replication, contains every table and chart the code produces, high-level discussion of successes, challenges and data sources. No code snippets. | 4 | `[~]` | `reports/replication_report.tex` exists and compiles to PDF via `doit compile_latex_docs`. Structure and build chain done; discussion sections still TODO. Prose can be lifted from `ISSUE1.md` / `ISSUE2.md`. |
| 2 | At least one Jupyter notebook touring the cleaned data and the analysis. Code snippets fine here. | 4 | `[~]` | Scaffold's jupytext notebook present and wired into `doit run_notebooks` (script -> .ipynb -> execute -> HTML). Still the template's example; needs rewriting as a tour of our panel. |
| 3 | Replicate the assigned tables, with a chosen tolerance and unit tests asserting the numbers match within it. | 20 | `[~]` | Tables 1-3 reproduce with zero failures (Table 2 VWNY 1946-2000: OLS 0.978 s.e. 0.477 against published 0.917/0.476; rho~1 0.686 t=4.56 against 0.663/4.67). Tables 5/6: the B/M gap was traced to deferred taxes in book equity and largely closed (mean 58.69 -> 52.08 against the paper's 53.13); the remaining ~5%, shared with E/P, is the firm-screen residual. `xfail`ed with reasons. 320 assertions in `tests/test_replication_vs_paper.py`; tolerances fixed before results were seen and justified per statistic in `src/paper_values.py`. |
| 4 | Reproduce the same tables with updated numbers, through the most recent data. | 20 | `[x]` | `replicate_paper_tables.py --extended`, two windows, both CRSP schemas for robustness. Paper's conclusion survives to 2025; post-2000 the conditional test loses power exactly as the paper's own Table A.1 predicts. See ISSUE2.md. |
| 5 | Our own summary-statistics table AND charts, typeset in LaTeX, with captions stating what the reader should take away. | 20 | `[ ]` | **Largest single gap.**  |

## Engineering

| # | Item | Pts | Status | Notes |
|---|---|---|---|---|
| 6 | Tidy dataset produced by files whose only job is cleaning; analysis kept separate. | 4 | `[x]` | `construct_panel.py` cleans, `estimators.py` / `replicate_paper_tables.py` analyse. Clean split already. |
| 7 | Statistics in all LaTeX tables auto-generated from code. | 4 | `[~]` | Mechanism proven end to end: `doit generated_tables` writes a `.tex` fragment to `_output/`, and the report `\input`s it -- nothing typed by hand. Currently a placeholder table; swaps for the real ones with item 5. |
| 8 | Project automated end-to-end with PyDoit. | 4 | `[x]` | `dodo.py`: 13 tasks. Pulls both CRSP schemas, builds all table sets, runs the suite, compiles LaTeX, executes notebooks, builds the chartbook site. Incremental via `file_dep`/`targets`; `check_credentials` fails fast if `WRDS_USERNAME` is unset rather than hanging on a prompt. |
| 9 | Unit tests exist and are well motivated. | 4 | `[x]` | ~1,050 lines across five files plus the replication suite. Includes regression tests tied to named bugs (totval lag, Stambaugh intercept, NaN-gap AR(1) seam). |
| 10 | Scaffolded with `cruft create https://github.com/backofficedev/cookiecutter_chartbook`. | 4 | `[x]` | Retrofitted: `.cruft.json` records the template and commit, and `cruft check` reports SUCCESS. Adopted `settings.py`, `.env.example`, `reports/`, `docs_src/`, `chartbook.toml`, `.latexmkrc`, `data_manual/` and the `_data`/`_output` conventions. Kept our verified `requirements.txt`/`environment.yml` and `tests/` layout over the template's. |
| 11 | Each unit test has a purpose; no unnecessary or repetitive tests. | 4 | `[x]` | Each maps to a specific failure mode; the parametrised replication tests are one assertion per published cell. |
| 12 | Repo and history free of copyrighted material / raw data. | 4 | `[x]` | Verified with `git log --all --diff-filter=A`. CRSP panels live in `data/`, gitignored, never committed. |
| 13 | Repo and history free of secrets. | 4 | `[x]` | Verified. WRDS password resolves from `~/.pgpass` outside the repo; only `WRDS_USERNAME` is read from the environment. |
| 14 | `.env` plus sensible `settings.py` defaults (data dir, keys, `START_DATE`, `END_DATE`), with the format documented in `.env.example`. | 4 | `[x]` | `src/settings.py` from the scaffold, extended with `SIZ_END_DATE` and `N_SIMS`. Resolution order is CLI > env > `.env` > default. `.env.example` documents every key and explains why the WRDS password is deliberately absent (it lives in `~/.pgpass`). `.env` gitignored, `.env.example` tracked -- both verified. |
| 15 | No trace of `.env` in the commit history. | 4 | `[x]` | Verified across every branch. |
| 16 | `requirements.txt` describing the packages needed to run the code. | 4 | `[x]` | Rewritten and verified from an empty environment: WRDS connects and queries, suite runs 217 passed / 0 skipped. Added `streamlit` (its absence was silently skipping `test_dashboard.py`) and dropped ~10 unused packages carried over from an unrelated FastAPI template. `wrds>=3.5` floor documented -- it is what stops a resolver satisfying a newer pandas with a 2023-era wrds that pandas 3 cannot drive. An earlier note here claimed the file was broken by a pandas 3 conflict; that was wrong, the file always resolved correctly, and the breakage came from installing packages ad-hoc without it. |
| 17 | Repo and history free of secrets. *(the rubric lists this twice)* | 4 | `[x]` | As item 13. |
| 20 | Every Python file has a top-level docstring describing what it does. | 4 | `[x]` | All modules. |
| 21 | Functions have descriptive names and docstrings where appropriate. | 4 | `[x]` | |

## Collaboration

| # | Item | Pts | Status | Notes |
|---|---|---|---|---|
| 18 | Each group member has made commits. | 4 | `[x]` |  |
| 19 | Each member has made **and** merged a GitHub pull request. | 4 | `[~]` |  |

## Individual

| # | Item | Pts | Status | Notes |
|---|---|---|---|---|
| 22 | Did I accomplish my assigned tasks and contribute a substantial part of the code, as evidenced by the commit history? | 50 | `[~]` |  |
| 23 | Oral defense: defend the analysis and design choices, and demonstrate running and modifying the project -- conda env, `doit` end-to-end, a live edit, SSH where relevant. | 20 | `[~]` | Both halves of the demo now exist and are verified: `conda env create -f environment.yml` builds the full dependency set, and `doit` runs the pipeline end to end. NOTE: `C:\WINDOWS\System32` is missing from this machine's PATH, which makes conda's pip step fail with "'chcp' is not recognized" -- fix before the defense. |

## Where the points are

| | Pts |
|---|---|
| Secured | 56 |
| Largely done (items 3, 4) | roughly 37 of 40 |
| Partially done (items 1, 2, 7) | 12 |
| Not started (item 5) | 20 |
| Individual (items 22, 23) | 70 |

## Suggested order

1. ~~`requirements.txt` + `environment.yml`~~ -- done, verified from a clean environment.
2. ~~cruft scaffold~~ (item 10) -- done, `cruft check` passes.
3. ~~`dodo.py`~~ (item 8) -- done, 13 tasks, incremental.
4. **Our own table and 2-3 figures** (item 5, 20 pts). The largest remaining item
   by a wide margin, and the only one still at zero. The build chain that carries
   it into the PDF already works, so this is analysis and design rather than
   plumbing.
5. **Fill in the report prose** (items 1, 7 -- 8 pts). The skeleton compiles;
   the discussion sections are TODO and much of the text can be lifted from
   `ISSUE1.md` and `ISSUE2.md`.
6. **Rewrite the notebook** (item 2, 4 pts) as a tour of our panel rather than
   the template's plotly example.

All that remains is presentation-layer work; the data pipeline, estimators,
tests and automation are done. One machine issue to fix before the defense: add
`C:\WINDOWS\System32` back to PATH, or conda's pip step fails.
