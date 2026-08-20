"""
schema_compare_macros.py
=========================
Emits the two macros used in the Extension section's schema-robustness
paragraph: corr(e,m) for VWNY/logDY on the 1946-to-SIZ's-end window, once
under each CRSP schema (SIZ, and CIZ truncated to the same end date via
`--end` -- see task_tables_ciz2024 in dodo.py).

This deliberately does not recompute anything. Both numbers already exist in
the combined CSVs that `tables_siz` and `tables_ciz2024` produce (via
run_extended's "1946-<end>" window); this script only reads the one row it
needs from each and writes it out as a \\newcommand, so the schema-comparison
sentence in the report can never disagree with the tables it is describing
the same run also produced.

Run as:
    python src/schema_compare_macros.py \
        _output/issue2_tables_2_3_4_5_6_extended_siz.csv \
        _output/issue2_tables_2_3_4_5_6_extended_ciz2024.csv \
        _output/macros_schema_compare.tex
"""

from __future__ import annotations

import os
import sys

import pandas as pd

from replicate_paper_tables import write_macros_file


# The paper's own fixed windows that also happen to start with "1946-",
# and must be excluded so only the true extended window's row matches.
# Enumerated explicitly rather than inferred, so adding a new "1946-..."
# window in main() later either has no effect here or trips the "found N != 1"
# check below loudly -- it can't silently pick the wrong row.
_FIXED_1946_WINDOWS = {"1946-2000", "1946-1972", "1946-1994", "1946-2000 (Table 4 comparison)"}


def extended_vwny_row(csv_path: str) -> pd.Series:
    """The 1946-onward extended window's VWNY/logDY row -- distinguished from
    the paper's own fixed 1946-* windows (Table 2/3/4) by exclusion, since
    only the extended window's end year varies with the panel's actual data."""
    df = pd.read_csv(csv_path)
    mask = (
        (df["predictor"] == "logDY")
        & (df["series"] == "VWNY")
        & df["window"].str.startswith("1946-")
        & (~df["window"].isin(_FIXED_1946_WINDOWS))
    )
    matches = df[mask]
    if len(matches) != 1:
        raise ValueError(
            f"{csv_path}: expected exactly one 1946-onward extended VWNY/logDY "
            f"row, found {len(matches)} (windows: {matches['window'].tolist()}). "
            f"Was this file built with --extended? If main() gained a new "
            f"1946-* window, add it to _FIXED_1946_WINDOWS above."
        )
    return matches.iloc[0]


def main(siz_csv: str, ciz2024_csv: str, out_path: str) -> None:
    siz_row = extended_vwny_row(siz_csv)
    ciz_row = extended_vwny_row(ciz2024_csv)

    if siz_row["window"] != ciz_row["window"]:
        # Both should say "1946-2024" (or whatever SIZ_END_DATE resolves to)
        # if --end truncation worked -- if the windows differ, the comparison
        # is confounded by sample length, not schema, and should not be
        # reported as a clean schema-only comparison.
        raise ValueError(
            f"SIZ window ({siz_row['window']}) and CIZ2024 window "
            f"({ciz_row['window']}) don't match -- check that "
            f"task_tables_ciz2024's --end argument actually equals SIZ's "
            f"true end date before trusting this comparison."
        )

    macros = {
        "SchemaWindow": siz_row["window"],
        "SchemaSIZCorrEM": f"{siz_row['corr_em']:.4f}",
        "SchemaCIZCorrEM": f"{ciz_row['corr_em']:.4f}",
    }
    out_dir, filename = os.path.split(out_path)
    name, _ext = os.path.splitext(filename)
    write_macros_file(macros, out_dir, name)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("usage: python src/schema_compare_macros.py <siz.csv> <ciz2024.csv> <out.tex>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])