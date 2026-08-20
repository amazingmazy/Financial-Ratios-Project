"""
dodo.py
=======
End-to-end build for the Lewellen (2004) replication, as a PyDoit pipeline.

    doit list          # what can be built
    doit               # build the default chain
    doit panel_siz     # one task, plus whatever it depends on
    doit clean         # remove generated artefacts
    doit forget        # re-run next time even if targets look current

Everything downstream of the WRDS pulls is incremental: doit compares each
task's `file_dep` against its `targets`, so editing `estimators.py` rebuilds the
tables without re-pulling a panel, while editing `wrds_pull.py` rebuilds from
the pull onward.

Structure. The first half is this project's own pipeline -- pull, tables, tests.
The second half is inherited from the cookiecutter_chartbook scaffold and covers
the presentation layer: jupytext notebooks, LaTeX compilation, and the chartbook
site. Kept in one file because doit expects one, and separated by banner so it
is obvious which half came from where.

Why the two panels are separate tasks
-------------------------------------
CRSP's legacy SIZ tables stop at 2024-12-31; the current CIZ schema runs a year
further. SIZ reproduces the paper's own numbers more closely -- it shares
Lewellen's month-end return-compounding convention -- while CIZ is the only one
that reaches the present. So the replication runs on SIZ and the extension on
CIZ, and both are built rather than one being derived from the other. See
ISSUE2.md, "Robustness: the schema choice does not move any conclusion."

`tables_ciz2024` truncates the CIZ panel at the SIZ end date so the schema
comparison is like-for-like rather than one side silently getting an extra year.

Configuration
-------------
Paths and dates come from `src/settings.py`, which reads `.env` and falls back
to documented defaults -- see `.env.example`. Nothing here hard-codes a path.

`WRDS_USERNAME` must be set for the pulls; the password comes from `~/.pgpass`
(on Windows `%APPDATA%/postgresql/pgpass.conf`) and is never read by this
project. Without the username a bare `wrds.Connection()` falls back to an
interactive prompt, which under doit raises EOFError with no useful message, so
`task_check_credentials` fails early with an actionable one instead.
"""

import os
import shutil
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / "src"))
from settings import config  # noqa: E402

try:
    _wrds_username = config("WRDS_USERNAME")
except Exception:
    _wrds_username = None
if _wrds_username:
    os.environ.setdefault("WRDS_USERNAME", _wrds_username)

BASE_DIR = config("BASE_DIR")
DATA_DIR = config("DATA_DIR")
OUTPUT_DIR = config("OUTPUT_DIR")
SRC = BASE_DIR / "src"
REPORTS = BASE_DIR / "reports"

PY = sys.executable  # the interpreter running doit, so the env is never ambiguous

START_DATE = datetime.strptime(config("START_DATE"), "%Y-%m-%d")
SIZ_END_DATE = datetime.strptime(config("SIZ_END_DATE"), "%Y-%m-%d")
N_SIMS = config("N_SIMS")

PANEL_SIZ = DATA_DIR / "master_panel.csv"
PANEL_CIZ = DATA_DIR / "master_panel_ciz.csv"

# Files whose contents genuinely change a panel, so touching one re-pulls.
PULL_DEPS = [BASE_DIR / "run_issue1.py", SRC / "wrds_pull.py",
             SRC / "construct_panel.py", SRC / "public_fallback.py",
             SRC / "qa_table1.py", SRC / "paper_values.py", SRC / "settings.py"]

# The table driver depends on the estimators but not on the pull.
TABLE_DEPS = [SRC / "replicate_paper_tables.py", SRC / "estimators.py"]

DOIT_CONFIG = {
    "default_tasks": ["panel_siz", "panel_ciz", "tables_siz", "tables_ciz",
                      "tables_ciz2024", "test"],
    "verbosity": 2,
}


###############################################################################
# This project's pipeline
###############################################################################

def task_check_credentials():
    """Fail early and clearly if WRDS_USERNAME is not set."""
    def check():
        username = os.environ.get("WRDS_USERNAME") or _from_env_file()
        if username:
            print(f"  WRDS_USERNAME = {username}")
            return True
        print(
            "\nWRDS_USERNAME is not set, so the pulls would hit an interactive\n"
            "prompt and fail with EOFError. Either add it to .env (copy\n"
            ".env.example to .env), or set it for this shell:\n\n"
            "    PowerShell:  $env:WRDS_USERNAME = 'your_wrds_id'\n"
            "    bash:        export WRDS_USERNAME=your_wrds_id\n\n"
            "The password is read from ~/.pgpass (Windows:\n"
            "%APPDATA%/postgresql/pgpass.conf), not from this project.\n"
        )
        return False

    def _from_env_file():
        try:
            return config("WRDS_USERNAME")
        except Exception:
            return None

    return {"actions": [check], "uptodate": [False]}


def task_panel_siz():
    """Build the master panel from the legacy CRSP SIZ schema (through 2024).

    The panel the replication runs on. SIZ's month-end return compounding is the
    convention Lewellen's own data used, and it reproduces his published numbers
    more closely than CIZ does.
    """
    return {
        "actions": [
            f'"{PY}" "{BASE_DIR / "run_issue1.py"}" --source wrds '
            f"--index-source exchcd_filtered "
            f"--start {START_DATE} --end {SIZ_END_DATE} "
            f'--out "{PANEL_SIZ}" --skip-qa'
        ],
        "file_dep": PULL_DEPS,
        "targets": [PANEL_SIZ],
        "task_dep": ["check_credentials"],
        "clean": True,
    }


def task_panel_ciz():
    """Build the master panel from the current CRSP CIZ schema (through present).

    Required for the extension: the legacy SIZ tables return a short panel rather
    than an error past 2024-12-31, so an extension built on them would silently
    end a year early.
    """
    return {
        "actions": [
            f'"{PY}" "{BASE_DIR / "run_issue1.py"}" --source wrds '
            f"--index-source ciz --start {START_DATE} "
            f'--out "{PANEL_CIZ}" --skip-qa'
        ],
        "file_dep": PULL_DEPS,
        "targets": [PANEL_CIZ],
        "task_dep": ["check_credentials"],
        "clean": True,
    }


# Names replicate_paper_tables.py writes one .tex fragment per, whenever run
# with --extended (which _tables_task always does). Kept as an explicit list
# rather than a glob so a silently-missing table fails the build instead of
# being missed.
REPLICATION_TABLE_NAMES = [
    "table2", "table3a", "table3b", "table4a", "table4b",
    "table5a", "table5b", "table6a", "table6b",
    "table_ext_1946", "table_ext_2001",
]

# Subset that also gets a macros_<name>_<tag>.tex written (see main() and
# run_extended() in replicate_paper_tables.py) -- these are the tables whose
# specific numbers the report's prose quotes by name, not all 11.
MACRO_TABLE_NAMES = ["table2", "table_ext_1946", "table_ext_2001"]


def _tables_task(tag, panel, extra_args=""):
    csv_target = OUTPUT_DIR / f"issue2_tables_2_3_4_5_6_extended_{tag}.csv"
    tex_targets = [OUTPUT_DIR / f"{name}_{tag}.tex" for name in REPLICATION_TABLE_NAMES]
    # Every macro-bearing table also gets a matching macros_<name>_<tag>.tex,
    # from the same run, so a table and the prose macros describing it can
    # never disagree.
    macro_targets = [OUTPUT_DIR / f"macros_{name}_{tag}.tex" for name in MACRO_TABLE_NAMES]
    return {
        "actions": [
            f'"{PY}" "{SRC / "replicate_paper_tables.py"}" "{panel}" '
            f"--extended --tag {tag} --n-sims {N_SIMS} {extra_args}".strip()
        ],
        "file_dep": TABLE_DEPS + [panel],
        "targets": [csv_target] + tex_targets + macro_targets,
        "clean": True,
    }


def task_tables_siz():
    """Tables 2-6 and the extension windows on the SIZ panel (the replication)."""
    return _tables_task("siz", PANEL_SIZ)


def task_tables_ciz():
    """Tables 2-6 and the extension windows on the CIZ panel (the extension)."""
    return _tables_task("ciz", PANEL_CIZ)


def task_tables_ciz2024():
    """CIZ truncated to the SIZ end date, for a like-for-like schema comparison.

    Without the truncation the schemas are compared over different windows, and
    any difference is confounded with sample length rather than attributable to
    the schema.
    """
    return _tables_task("ciz2024", PANEL_CIZ, f"--end {SIZ_END_DATE}")


def task_schema_compare_macros():
    """Emit the SIZ-vs-CIZ2024 corr(e,m) macros for the Extension section's
    schema-robustness paragraph.

    Reads the combined CSVs tables_siz and tables_ciz2024 already produce,
    rather than recomputing -- this is a pure re-export of two numbers that
    already exist, so the schema-comparison sentence can't disagree with the
    two runs that back it.
    """
    script = SRC / "schema_compare_macros.py"
    siz_csv = OUTPUT_DIR / "issue2_tables_2_3_4_5_6_extended_siz.csv"
    ciz2024_csv = OUTPUT_DIR / "issue2_tables_2_3_4_5_6_extended_ciz2024.csv"
    target = OUTPUT_DIR / "macros_schema_compare.tex"
    return {
        "actions": [f'"{PY}" "{script}" "{siz_csv}" "{ciz2024_csv}" "{target}"'],
        "file_dep": [script, SRC / "replicate_paper_tables.py", siz_csv, ciz2024_csv],
        "task_dep": ["tables_siz", "tables_ciz2024"],
        "targets": [target],
        "clean": True,
    }


def task_test():
    """Run the test suite, including the assertions against the paper's values.

    Depends on the SIZ panel because the replication tests compare against it;
    without one they skip rather than fail, which would let a broken build look
    green.
    """
    return {
        "actions": [f'"{PY}" -m pytest "{BASE_DIR / "tests"}" -q -m "not slow"'],
        "file_dep": [PANEL_SIZ] + TABLE_DEPS + PULL_DEPS,
        "uptodate": [False],  # cheap, and should never be skipped
    }


###############################################################################
# Presentation layer, from the cookiecutter_chartbook scaffold
###############################################################################

# Notebooks are stored as jupytext .py scripts so they diff cleanly in git; the
# .ipynb is generated, executed and converted to HTML at build time. Add an
# entry here to wire a new notebook into the build.
notebook_tasks = {
    "01_example_notebook_interactive.ipynb": {
        "path": SRC / "01_example_notebook_interactive.ipynb.py",
        "file_dep": [],
        "targets": [],
    },
    # The guided tour of the panel and the estimators (rubric item 2). It reads
    # both panels and imports the estimator modules, so it is declared against
    # them: editing estimators.py re-executes the notebook, and a missing panel
    # makes doit build one rather than letting the notebook fail mid-execution
    # with a FileNotFoundError.
    "02_walkthrough.ipynb": {
        "path": SRC / "02_walkthrough.ipynb.py",
        "file_dep": [PANEL_SIZ, PANEL_CIZ, SRC / "estimators.py",
                     SRC / "paper_values.py", SRC / "qa_table1.py"],
        "targets": [],
    },
}


def task_run_notebooks():
    """Convert each jupytext script to a notebook, execute it, and export HTML."""
    for name, spec in notebook_tasks.items():
        pyfile = Path(spec["path"])
        nb = pyfile.with_suffix("")  # strips .py, leaving .ipynb
        yield {
            "name": name,
            "actions": [
                f'"{PY}" -m jupytext --to notebook --output "{nb}" "{pyfile}"',
                f'"{PY}" -m nbconvert --execute --to notebook --inplace "{nb}"',
                f'"{PY}" -m nbconvert --to html --output-dir "{OUTPUT_DIR}" "{nb}"',
            ],
            "file_dep": [pyfile, *spec["file_dep"]],
            "targets": [OUTPUT_DIR / f"{nb.stem}.html", *spec["targets"]],
            "clean": True,
        }


def task_generated_tables():
    """Write the LaTeX table fragments the report includes.

    Separated from the report build so that editing a table's code does not
    force a full LaTeX recompile, and so a broken fragment fails here with a
    Python traceback rather than as an opaque LaTeX error.
    """
    script = SRC / "pandas_to_latex_demo.py"
    return {
        "actions": [f'"{PY}" "{script}"'],
        "file_dep": [script],
        "targets": [OUTPUT_DIR / "pandas_to_latex_simple_table1.tex"],
        "clean": True,
    }


def task_exhibits():
    """Build our own summary table and figure (rubric item 5).

    Runs on the CIZ panel specifically, not SIZ -- the whole point is to look
    at log(DY)'s persistence past the paper's 2000 cutoff, and SIZ does not
    reach past 2024. Depends on estimators.py and dashboard_data.py, since
    own_analysis.py calls fit_ar1 and power_threshold from them rather than
    duplicating that logic.
    """
    script = SRC / "analysis.py"
    return {
        "actions": [f'"{PY}" "{script}" "{PANEL_CIZ}"'],
        "file_dep": [script, PANEL_CIZ, SRC / "estimators.py", SRC / "dashboard_data.py",
                     SRC / "replicate_paper_tables.py"],
        "targets": [
            OUTPUT_DIR / "own_summary_table.tex",
            OUTPUT_DIR / "own_rho_evolution.png",
            OUTPUT_DIR / "macros_own_exhibits.tex",
        ],
        "clean": True,
    }


def task_compile_latex_docs():
    """Compile the write-up to PDF with latexmk (xelatex).

    Builds `reports/replication_report.tex`, which is a skeleton at this stage --
    structure and build chain in place, discussion and our own exhibits still to
    be written. It deliberately does not use the scaffold's
    `my_article_header.sty`: that header currently fails to compile, because
    `\\newtheorem{mycorollary}` in `my_common_header.sty` raises "Command
    \\c@mylemma already defined", and this document needs none of the theorem
    machinery it provides.

    The scaffold's own `report_example.tex` and `report_simple_example.tex` are
    kept as references but not built -- the first hits that header bug, and both
    expect artefacts from the FRED example scripts, which were removed since
    this project does not use FRED for its exhibits.
    """
    tex = REPORTS / "replication_report.tex"
    replication_tables = [OUTPUT_DIR / f"{name}_siz.tex" for name in REPLICATION_TABLE_NAMES
                           if name not in ("table_ext_1946", "table_ext_2001")]
    # Extension tables specifically need CIZ, not SIZ: SIZ stops at 2024-12-31
    # and cannot produce the 1946-2025 / 2001-2025 windows the Extension
    # section's prose describes (see ISSUE2.md, "Known gaps: legacy CRSP
    # tables are frozen at 2024"). Using the _siz-tagged version here would
    # silently show a different, shorter window than the text next to it.
    extension_tables = [OUTPUT_DIR / f"{name}_ciz.tex" for name in ("table_ext_1946", "table_ext_2001")]
    # Prose macros mirror the same siz/ciz split as their matching tables --
    # \TableTwoOurs* from the siz run, \ExtFull*/\ExtRecent* from the ciz run,
    # so a macro is always sourced from the same run as the table it appears
    # next to.
    macros = [
        OUTPUT_DIR / "macros_table2_siz.tex",
        OUTPUT_DIR / "macros_table_ext_1946_ciz.tex",
        OUTPUT_DIR / "macros_table_ext_2001_ciz.tex",
        OUTPUT_DIR / "macros_schema_compare.tex",
        OUTPUT_DIR / "macros_own_exhibits.tex",
    ]
    return {
        "actions": [
            f'latexmk -xelatex -halt-on-error -cd "{tex}"',
            f'latexmk -xelatex -halt-on-error -c -cd "{tex}"',  # clean aux files
        ],
        "file_dep": [
            tex,
            OUTPUT_DIR / "pandas_to_latex_simple_table1.tex",
            OUTPUT_DIR / "own_summary_table.tex",
            OUTPUT_DIR / "own_rho_evolution.png",
        ] + replication_tables + extension_tables + macros,
        "task_dep": ["generated_tables", "exhibits", "tables_siz", "tables_ciz",
                      "tables_ciz2024", "schema_compare_macros"],
        "targets": [REPORTS / "replication_report.pdf"],
        "clean": True,
    }


def task_build_chartbook_site():
    """Build the chartbook site. Not in the default chain."""
    return {
        "actions": ["chartbook build -f"],
        "file_dep": [BASE_DIR / "README.md", BASE_DIR / "chartbook.toml"],
        "task_dep": ["run_notebooks"],
        "uptodate": [False],
    }


###############################################################################
# Utilities, none in the default chain
###############################################################################

def task_test_all():
    """Run the full suite including the slow Monte Carlo tests."""
    return {
        "actions": [f'"{PY}" -m pytest "{BASE_DIR / "tests"}" -q'],
        "uptodate": [False],
    }


def task_dashboard():
    """Launch the Streamlit dashboard.

    Long-running and interactive, so deliberately excluded from the default
    chain -- it would never terminate.
    """
    return {
        "actions": [f'"{PY}" -m streamlit run "{SRC / "dashboard.py"}"'],
        "file_dep": [SRC / "dashboard.py", SRC / "dashboard_data.py"],
        "uptodate": [False],
    }


def task_clean_pycache():
    """Remove __pycache__ directories."""
    def wipe():
        for d in BASE_DIR.rglob("__pycache__"):
            shutil.rmtree(d, ignore_errors=True)
        print("  removed __pycache__ directories")

    return {"actions": [wipe], "uptodate": [False]}
