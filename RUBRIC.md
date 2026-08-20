# Rubric tracker

Working reference against the final project rubric (202 points). Update the
status column as items land; keep the notes pointing at evidence rather than
intentions, so this stays a record of what is true rather than what we mean to
do.

`[x]` done &nbsp;·&nbsp; `[~]` partial &nbsp;·&nbsp; `[ ]` not started

Last reviewed: after PR #7 merged and PR #8 (Stefano's LaTeX report) was reviewed.

**Read the two columns separately.** Status reflects what is on `main` -- what a
grader cloning today would actually see. Work sitting on an unmerged branch is
noted but not counted, because it cannot be built.

## Deliverables

| # | Item | Pts | Status | Notes |
|---|---|---|---|---|
| 1 | Single LaTeX document: describes the replication, contains every table and chart the code produces, high-level discussion of successes, challenges and data sources. No code snippets. | 4 | `[~]` | On `main`: skeleton compiles via `doit compile_latex_docs`, sections are TODO. PR #8 fills in Introduction, Data, Replication and Extension and renders a clean 12-page PDF -- blocked, see below. |
| 2 | At least one Jupyter notebook touring the cleaned data and the analysis. Code snippets fine here. | 4 | `[x]` | `src/02_walkthrough.ipynb.py` -- guided tour of the panel, the correlation that causes the bias, the three estimators side by side, the replication against published values, and where the method loses power post-2000. Stored as a jupytext script so it diffs cleanly; `doit run_notebooks` converts, executes and exports HTML. 19 cells, executes with zero errors. |
| 3 | Replicate the assigned tables, with a chosen tolerance and unit tests asserting the numbers match within it. | 20 | `[~]` | Tables 1-3 reproduce with zero failures (Table 2 VWNY 1946-2000: OLS 0.978 s.e. 0.477 against published 0.917/0.476; rho~1 0.686 t=4.56 against 0.663/4.67). Tables 5/6: the B/M gap was traced to deferred taxes in book equity and largely closed (mean 58.69 -> 52.08 against the paper's 53.13); the remaining ~5%, shared with E/P, is the firm-screen residual. `xfail`ed with reasons. 320 assertions in `tests/test_replication_vs_paper.py`; tolerances fixed before results were seen and justified per statistic in `src/paper_values.py`. |
| 4 | Reproduce the same tables with updated numbers, through the most recent data. | 20 | `[x]` | `replicate_paper_tables.py --extended`, two windows, both CRSP schemas for robustness. Paper's conclusion survives to 2025; post-2000 the conditional test loses power exactly as the paper's own Table A.1 predicts. See ISSUE2.md. |
| 5 | Our own summary-statistics table AND charts, typeset in LaTeX, with captions stating what the reader should take away. | 20 | `[ ]` | **Largest single gap on `main`.** Substantially built on PR #8 (`src/analysis.py`: decade summary table + rolling autocorrelation figure against the paper's own power thresholds) but **blocked** -- see the note below. |

## Engineering

| # | Item | Pts | Status | Notes |
|---|---|---|---|---|
| 6 | Tidy dataset produced by files whose only job is cleaning; analysis kept separate. | 4 | `[x]` | `construct_panel.py` cleans, `estimators.py` / `replicate_paper_tables.py` analyse. Clean split already. |
| 7 | Statistics in all LaTeX tables auto-generated from code. | 4 | `[~]` | Mechanism proven on `main` (`doit generated_tables` writes a `.tex` fragment, the report `\input`s it). PR #8 adds 12 real fragments. Caveat raised in review: that PR also carries 15 hand-typed statistics in its *prose*, one of which had already drifted from the data (`0.862` typed against `0.8764` computed). |
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
| 19 | Each member has made **and** merged a GitHub pull request. | 4 | `[x]` | Closed when PR #7 merged (10 commits, not squashed, individual attribution intact). Stefano authored #1, #3-#6; Anthony authored #7 and merged #3-#7. |

## Individual

| # | Item | Pts | Status | Notes |
|---|---|---|---|---|
| 22 | Did I accomplish my assigned tasks and contribute a substantial part of the code, as evidenced by the commit history? | 50 | `[~]` |  |
| 23 | Oral defense: defend the analysis and design choices, and demonstrate running and modifying the project -- conda env, `doit` end-to-end, a live edit, SSH where relevant. | 20 | `[~]` | Both halves of the demo now exist and are verified: `conda env create -f environment.yml` builds the full dependency set, and `doit` runs the pipeline end to end. NOTE: `C:\WINDOWS\System32` is missing from this machine's PATH, which makes conda's pip step fail with "'chcp' is not recognized" -- fix before the defense. |

## Blocked on PR #8

Roughly 28 points of finished work cannot be counted because it does not build
from a clean checkout. Three defects, all small, flagged in review
(CHANGES_REQUESTED):

1. `dodo.py` resolves `WRDS_USERNAME` at import time, so **every** doit task
   fails without a `.env` -- `doit list` included.
2. `START_DATE` is passed as a `datetime` rather than a string, so the panel
   tasks emit `--start 1926-01-01 00:00:00`, which argparse rejects.
3. The report hardcodes `../output` while the code writes to `_output`, so 11 of
   its 12 `\input` fragments are missing and LaTeX halts.

The LaTeX document itself is sound -- it compiles to 12 pages once the
fragments resolve -- so this is integration, not authoring. Also pending:
generated artefacts (`output/*.png`, `*.tex`) are tracked in git and need
`git rm --cached`.

## Where the points are

| | Pts |
|---|---|
| Secured | 64 |
| Largely done (items 3, 4) | roughly 37 of 40 |
| Partially done on `main` (items 1, 7) | ~2 of 8 |
| Not started on `main` (item 5) | 0 of 20 |
| *Blocked on PR #8 (items 1, 5, 7)* | *~28* |
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
6. ~~Rewrite the notebook~~ (item 2) -- done, `src/02_walkthrough.ipynb.py`.

All that remains is presentation-layer work; the data pipeline, estimators,
tests and automation are done. One machine issue to fix before the defense: add
`C:\WINDOWS\System32` back to PATH, or conda's pip step fails.
