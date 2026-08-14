# Rubric tracker

Working reference against the final project rubric (202 points). Update the
status column as items land; keep the notes pointing at evidence rather than
intentions, so this stays a record of what is true rather than what we mean to
do.

`[x]` done &nbsp;·&nbsp; `[~]` partial &nbsp;·&nbsp; `[ ]` not started

Last reviewed: after PR #7 (replication tests, CIZ pull, sample extension).

## Deliverables

| # | Item | Pts | Status | Notes |
|---|---|---|---|---|
| 1 | Single LaTeX document: describes the replication, contains every table and chart the code produces, high-level discussion of successes, challenges and data sources. No code snippets. | 4 | `[ ]` | No `reports/`, no `.tex` anywhere. Much of the prose already exists in `ISSUE1.md` / `ISSUE2.md` and can be lifted. |
| 2 | At least one Jupyter notebook touring the cleaned data and the analysis. Code snippets fine here. | 4 | `[ ]` | No notebooks in the repo, ever. `jupytext` is already a dependency. |
| 3 | Replicate the assigned tables, with a chosen tolerance and unit tests asserting the numbers match within it. | 20 | `[~]` | Tables 1-3 reproduce with zero failures (Table 2 VWNY 1946-2000: OLS 0.978 s.e. 0.477 against published 0.917/0.476; rho~1 0.686 t=4.56 against 0.663/4.67). Tables 5/6 diverge ~10% (B/M) and ~5% (E/P) on levels -- documented, `xfail`ed, defensible under the whole-Compustat instruction, but not a full reproduction. 320 assertions in `tests/test_replication_vs_paper.py`; tolerances fixed before results were seen and justified per statistic in `src/paper_values.py`. |
| 4 | Reproduce the same tables with updated numbers, through the most recent data. | 20 | `[x]` | `replicate_paper_tables.py --extended`, two windows, both CRSP schemas for robustness. Paper's conclusion survives to 2025; post-2000 the conditional test loses power exactly as the paper's own Table A.1 predicts. See ISSUE2.md. |
| 5 | Our own summary-statistics table AND charts, typeset in LaTeX, with captions stating what the reader should take away. | 20 | `[ ]` | **Largest single gap.**  |

## Engineering

| # | Item | Pts | Status | Notes |
|---|---|---|---|---|
| 6 | Tidy dataset produced by files whose only job is cleaning; analysis kept separate. | 4 | `[x]` | `construct_panel.py` cleans, `estimators.py` / `replicate_paper_tables.py` analyse. Clean split already. |
| 7 | Statistics in all LaTeX tables auto-generated from code. | 4 | `[ ]` | Blocked on item 1. Results already land in `output/*.csv` in tidy long format, so this is a rendering step. |
| 8 | Project automated end-to-end with PyDoit. | 4 | `[ ]` | No `dodo.py`. `doit` is already in `requirements.txt`. Also half of the oral-defense demo. |
| 9 | Unit tests exist and are well motivated. | 4 | `[x]` | ~1,050 lines across five files plus the replication suite. Includes regression tests tied to named bugs (totval lag, Stambaugh intercept, NaN-gap AR(1) seam). |
| 10 | Scaffolded with `cruft create https://github.com/backofficedev/cookiecutter_chartbook`. | 4 | `[ ]` | Not used. No `.cruft.json`. Current `Makefile`/`Dockerfile` are leftovers from an unrelated FastAPI template pointing at an `app/` directory that has never existed. |
| 11 | Each unit test has a purpose; no unnecessary or repetitive tests. | 4 | `[x]` | Each maps to a specific failure mode; the parametrised replication tests are one assertion per published cell. |
| 12 | Repo and history free of copyrighted material / raw data. | 4 | `[x]` | Verified with `git log --all --diff-filter=A`. CRSP panels live in `data/`, gitignored, never committed. |
| 13 | Repo and history free of secrets. | 4 | `[x]` | Verified. WRDS password resolves from `~/.pgpass` outside the repo; only `WRDS_USERNAME` is read from the environment. |
| 14 | `.env` plus sensible `settings.py` defaults (data dir, keys, `START_DATE`, `END_DATE`), with the format documented in `.env.example`. | 4 | `[ ]` | Neither file exists. Dates are currently CLI arguments. |
| 15 | No trace of `.env` in the commit history. | 4 | `[x]` | Verified across every branch. |
| 16 | `requirements.txt` describing the packages needed to run the code. | 4 | `[~]` | **Present but broken.** `pandas` is unpinned, so a fresh install resolves 3.0, which drops the raw DBAPI2 support `wrds` needs -- every `raw_sql` call fails. A bare `wrds` also resolves to 3.1.6 rather than the `>=3.2.0` the file asks for. `requests`, `streamlit` and `plotly` are imported by the code but absent. Working set is `pandas<3` with `wrds>=3.5`. |
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
| 23 | Oral defense: defend the analysis and design choices, and demonstrate running and modifying the project -- conda env, `doit` end-to-end, a live edit, SSH where relevant. | 20 | `[~]` |  |

## Where the points are

| | Pts |
|---|---|
| Secured | 40 |
| Largely done (items 3, 4) | roughly 37 of 40 |
| Not started | 44 |
| Cheap and broken (item 16) | 4 |
| Individual (items 22, 23) | 70 |

## Suggested order

1. **`requirements.txt` + `environment.yml`** (item 16, and half of 23). Fifteen minutes; without it a grader following the README cannot run the project at all.
2. **cruft scaffold** (item 10). Worth 4 directly, but it is also the container for `dodo.py`, `settings.py`, `.env.example` and `reports/` -- another 16 behind it.
3. **`dodo.py`** (item 8, and the other half of 23).
4. **Own table and 2-3 figures** (item 5). Largest single item.
5. **LaTeX document** (items 1, 7). The vehicle for item 5.
6. **Notebook** (item 2).

Items 1-3 are pure infrastructure and separable from the econometrics, so they
can run in parallel with any remaining work on Tables 5/6.
