#!/usr/bin/env python3

"""
EDGE — V0.14 RAW ODDS SEMANTIC AUDIT

Purpose:
    Determine whether the raw odds datasets contain opening and
    closing prices that were flattened during normalization.

This is a data-lineage audit.

No model training.
No betting strategy.
No threshold optimization.
No ROI analysis.
"""

from pathlib import Path
import pandas as pd


RAW_DIR = Path("data/raw")


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main():

    print("EDGE — V0.14 RAW ODDS SEMANTIC AUDIT")
    print("=" * 78)

    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {RAW_DIR}"
        )

    files = sorted(RAW_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {RAW_DIR}"
        )

    # ------------------------------------------------------------
    # 1. RAW FILE INVENTORY
    # ------------------------------------------------------------

    section("1. RAW ODDS FILE INVENTORY")

    print(f"Directory: {RAW_DIR}")
    print(f"CSV files found: {len(files)}")

    for f in files:
        print(f"  - {f.name}")

    # ------------------------------------------------------------
    # 2. SEARCH FOR ODDS FIELDS
    # ------------------------------------------------------------

    section("2. OPENING / CLOSING FIELD DETECTION")

    all_columns = {}

    for file in files:

        try:
            df = pd.read_csv(file, nrows=5)
        except Exception as exc:
            print(f"\n✗ Could not read {file.name}: {exc}")
            continue

        all_columns[file.name] = list(df.columns)

        relevant = []

        for col in df.columns:

            low = str(col).lower()

            if any(
                token in low
                for token in [
                    ">2.5",
                    "<2.5",
                    "c>2.5",
                    "c<2.5",
                    "b365",
                    "avg",
                    "max"
                ]
            ):
                relevant.append(col)

        print(f"\n{file.name}")

        if relevant:

            for col in relevant:
                print(f"  {col}")

        else:

            print("  No obvious Over/Under 2.5 odds fields.")

    # ------------------------------------------------------------
    # 3. EXPLICIT C-FIELD AUDIT
    # ------------------------------------------------------------

    section("3. CLOSING-FIELD AUDIT")

    closing_fields = set()
    opening_fields = set()

    for file, columns in all_columns.items():

        for col in columns:

            text = str(col)

            if (
                "C>2.5" in text
                or "C<2.5" in text
            ):
                closing_fields.add(text)

            if (
                ">2.5" in text
                or "<2.5" in text
            ) and not (
                "C>2.5" in text
                or "C<2.5" in text
            ):
                opening_fields.add(text)

    print("Potential closing fields:")

    for col in sorted(closing_fields):
        print(f"  ✓ {col}")

    if not closing_fields:
        print("  ✗ None found.")

    print("\nPotential opening fields:")

    for col in sorted(opening_fields):
        print(f"  ✓ {col}")

    if not opening_fields:
        print("  ✗ None found.")

    # ------------------------------------------------------------
    # 4. RAW VALUE COMPARISON
    # ------------------------------------------------------------

    section("4. OPENING VS CLOSING VALUE COMPARISON")

    comparisons = [
        ("B365>2.5", "B365C>2.5"),
        ("B365<2.5", "B365C<2.5"),
        ("Avg>2.5", "AvgC>2.5"),
        ("Avg<2.5", "AvgC<2.5"),
        ("Max>2.5", "MaxC>2.5"),
        ("Max<2.5", "MaxC<2.5"),
    ]

    found_comparison = False

    for file in files:

        try:
            df = pd.read_csv(file)
        except Exception:
            continue

        for opening, closing in comparisons:

            if opening not in df.columns:
                continue

            if closing not in df.columns:
                continue

            found_comparison = True

            a = pd.to_numeric(
                df[opening],
                errors="coerce"
            )

            b = pd.to_numeric(
                df[closing],
                errors="coerce"
            )

            valid = a.notna() & b.notna()

            if not valid.any():
                continue

            a = a[valid]
            b = b[valid]

            diff = b - a

            print(
                f"\n{file.name}: {opening} → {closing}"
            )

            print(
                f"  Observations: {len(a):,}"
            )

            print(
                f"  Opening mean: {a.mean():.4f}"
            )

            print(
                f"  Closing mean: {b.mean():.4f}"
            )

            print(
                f"  Mean change:  {diff.mean():+.4f}"
            )

            print(
                f"  Increased:    {(diff > 0).sum():,}"
            )

            print(
                f"  Decreased:    {(diff < 0).sum():,}"
            )

            print(
                f"  Unchanged:    {(diff == 0).sum():,}"
            )

    if not found_comparison:
        print(
            "✗ No opening/closing pairs found in raw files."
        )

    # ------------------------------------------------------------
    # 5. DATA LINEAGE CHECK
    # ------------------------------------------------------------

    section("5. DATA LINEAGE CHECK")

    normalized = Path(
        "data/processed/market_odds.csv"
    )

    if normalized.exists():

        normalized_df = pd.read_csv(
            normalized,
            nrows=100
        )

        print(
            "Normalized odds columns:"
        )

        for col in normalized_df.columns:
            print(f"  - {col}")

        print(
            "\nThe normalized dataset currently stores "
            "price_type and source_column."
        )

        if "source_column" in normalized_df.columns:

            print(
                "\nExample source columns:"
            )

            for value in (
                normalized_df["source_column"]
                .dropna()
                .astype(str)
                .unique()
            ):

                print(f"  {value}")

    else:

        print(
            "Normalized odds file not found."
        )

    # ------------------------------------------------------------
    # 6. FINAL ASSESSMENT
    # ------------------------------------------------------------

    section("6. V0.14 ASSESSMENT")

    if closing_fields and opening_fields:

        print(
            "✓ RAW DATA APPEARS TO CONTAIN BOTH "
            "OPENING AND CLOSING PRICES."
        )

        print(
            "\nThis means the next task should be:"
        )

        print(
            "1. Verify the field semantics."
        )

        print(
            "2. Preserve opening/closing status during normalization."
        )

        print(
            "3. Rebuild the odds dataset with explicit fields."
        )

        print(
            "4. Then perform CLV analysis."
        )

    else:

        print(
            "✗ Opening/closing semantics could not be "
            "confirmed from the raw data."
        )

        print(
            "\nDo NOT infer them automatically."
        )

        print(
            "Inspect the original dataset documentation/source."
        )

    print("\n" + "=" * 78)
    print("V0.14 COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
