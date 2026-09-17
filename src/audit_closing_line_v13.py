#!/usr/bin/env python3

"""
EDGE — V0.13 CLOSING-LINE / MARKET QUALITY AUDIT

Purpose:
    Audit the available odds data before further model development.

This script does NOT:
    - optimize betting thresholds
    - calculate a betting strategy
    - optimize stake sizing
    - claim profitability
    - use future information

It checks whether EDGE currently has the information required
for meaningful closing-line-value (CLV) analysis.
"""

from pathlib import Path
import pandas as pd
import numpy as np


ODDS_FILE = Path("data/processed/market_odds.csv")
MATCH_FILE = Path("data/processed/matches.csv")


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main():

    print("EDGE — V0.13 CLOSING-LINE / MARKET QUALITY AUDIT")
    print("=" * 78)

    # ------------------------------------------------------------
    # 1. LOAD ODDS
    # ------------------------------------------------------------

    if not ODDS_FILE.exists():
        raise FileNotFoundError(
            f"Missing odds file: {ODDS_FILE}"
        )

    odds = pd.read_csv(ODDS_FILE)

    section("1. ODDS DATASET")

    print(f"File: {ODDS_FILE}")
    print(f"Rows: {len(odds):,}")
    print(f"Columns: {len(odds.columns)}")

    print("\nColumns:")
    for i, col in enumerate(odds.columns):
        print(f"{i:2}: {col}")

    # ------------------------------------------------------------
    # 2. IDENTIFY ODDS COLUMNS
    # ------------------------------------------------------------

    section("2. ODDS FIELD AUDIT")

    columns_lower = {
        c.lower(): c for c in odds.columns
    }

    keywords = [
        "open",
        "opening",
        "close",
        "closing",
        "last",
        "current",
        "avg",
        "average",
        "market",
        "over",
        "under",
        "odds",
        "timestamp",
        "time",
        "bookmaker",
        "source"
    ]

    found = []

    for col in odds.columns:
        low = col.lower()

        if any(k in low for k in keywords):
            found.append(col)

    print("Potentially relevant fields:")

    for col in found:
        print(f"  - {col}")

    if not found:
        print("  None detected.")

    # ------------------------------------------------------------
    # 3. CLOSING-LINE DETECTION
    # ------------------------------------------------------------

    section("3. CLOSING-LINE DETECTION")

    closing_candidates = [
        c for c in odds.columns
        if any(
            term in c.lower()
            for term in [
                "closing",
                "close",
                "closing_odds",
                "close_odds",
                "closing_probability",
                "close_probability"
            ]
        )
    ]

    opening_candidates = [
        c for c in odds.columns
        if any(
            term in c.lower()
            for term in [
                "opening",
                "open_odds",
                "open_probability"
            ]
        )
    ]

    print("Closing candidates:")

    if closing_candidates:
        for c in closing_candidates:
            print(f"  ✓ {c}")
    else:
        print("  ✗ No obvious closing-line field found.")

    print("\nOpening candidates:")

    if opening_candidates:
        for c in opening_candidates:
            print(f"  ✓ {c}")
    else:
        print("  ✗ No obvious opening-line field found.")

    # ------------------------------------------------------------
    # 4. TIMESTAMP AUDIT
    # ------------------------------------------------------------

    section("4. TIMESTAMP AUDIT")

    timestamp_candidates = [
        c for c in odds.columns
        if any(
            term in c.lower()
            for term in [
                "timestamp",
                "datetime",
                "date_time",
                "time"
            ]
        )
    ]

    if timestamp_candidates:

        for col in timestamp_candidates:

            parsed = pd.to_datetime(
                odds[col],
                errors="coerce"
            )

            valid = parsed.notna().sum()

            print(
                f"{col}: "
                f"{valid:,}/{len(odds):,} "
                f"valid timestamps"
            )

            if valid > 0:
                print(f"  Earliest: {parsed.min()}")
                print(f"  Latest:   {parsed.max()}")

    else:

        print("✗ No timestamp field detected.")

    # ------------------------------------------------------------
    # 5. BOOKMAKER / SOURCE AUDIT
    # ------------------------------------------------------------

    section("5. BOOKMAKER / SOURCE AUDIT")

    source_candidates = [
        c for c in odds.columns
        if any(
            term in c.lower()
            for term in [
                "bookmaker",
                "book",
                "source",
                "provider"
            ]
        )
    ]

    if source_candidates:

        for col in source_candidates:

            print(f"\n{col}:")

            values = (
                odds[col]
                .dropna()
                .astype(str)
                .value_counts()
                .head(20)
            )

            print(values.to_string())

    else:

        print("✗ No bookmaker/source field detected.")

    # ------------------------------------------------------------
    # 6. DUPLICATE / MULTIPLE ODDS OBSERVATIONS
    # ------------------------------------------------------------

    section("6. MULTIPLE-OBSERVATION AUDIT")

    id_candidates = [
        c for c in odds.columns
        if any(
            term in c.lower()
            for term in [
                "match_id",
                "fixture_id",
                "event_id"
            ]
        )
    ]

    if id_candidates:

        id_col = id_candidates[0]

        counts = odds[id_col].value_counts()

        print(f"Identifier: {id_col}")

        print(
            f"Unique events: {counts.size:,}"
        )

        print(
            f"Events with >1 odds observation: "
            f"{(counts > 1).sum():,}"
        )

        print(
            f"Maximum observations/event: "
            f"{counts.max():,}"
        )

        print("\nObservation distribution:")

        print(
            counts.value_counts()
            .sort_index()
            .head(20)
            .to_string()
        )

    else:

        print("✗ No match/event identifier detected.")

    # ------------------------------------------------------------
    # 7. REQUIRED CLV STRUCTURE
    # ------------------------------------------------------------

    section("7. CLV REQUIREMENTS")

    requirements = {
        "Event identifier": bool(id_candidates),
        "Opening odds": bool(opening_candidates),
        "Closing odds": bool(closing_candidates),
        "Timestamp": bool(timestamp_candidates),
        "Bookmaker/source": bool(source_candidates),
    }

    for requirement, available in requirements.items():

        status = "✓ AVAILABLE" if available else "✗ MISSING"

        print(
            f"{status:15} {requirement}"
        )

    # ------------------------------------------------------------
    # 8. CURRENT MARKET ODDS SUMMARY
    # ------------------------------------------------------------

    section("8. CURRENT MARKET ODDS SUMMARY")

    numeric_cols = odds.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    if numeric_cols:

        print(
            odds[numeric_cols]
            .describe()
            .transpose()
            .to_string()
        )

    else:

        print("No numeric odds fields detected.")

    # ------------------------------------------------------------
    # 9. MISSING DATA
    # ------------------------------------------------------------

    section("9. MISSING DATA")

    missing = (
        odds.isna()
        .mean()
        .sort_values(ascending=False)
    )

    missing = missing[missing > 0]

    if len(missing):

        print(
            (missing * 100)
            .round(2)
            .astype(str)
            .add("%")
            .to_string()
        )

    else:

        print("No missing values detected.")

    # ------------------------------------------------------------
    # 10. MATCH COVERAGE
    # ------------------------------------------------------------

    section("10. MATCH COVERAGE")

    if MATCH_FILE.exists():

        matches = pd.read_csv(MATCH_FILE)

        print(
            f"Matches dataset: {len(matches):,}"
        )

        if id_candidates:

            id_col = id_candidates[0]

            if id_col in matches.columns:

                odds_ids = set(
                    odds[id_col]
                    .dropna()
                    .astype(str)
                )

                match_ids = set(
                    matches[id_col]
                    .dropna()
                    .astype(str)
                )

                overlap = odds_ids & match_ids

                print(
                    f"Odds events: {len(odds_ids):,}"
                )

                print(
                    f"Match events: {len(match_ids):,}"
                )

                print(
                    f"Matched events: {len(overlap):,}"
                )

                coverage = (
                    len(overlap) / len(match_ids)
                    if match_ids else np.nan
                )

                print(
                    f"Coverage: {coverage:.2%}"
                )

    else:

        print(
            f"Match file not found: {MATCH_FILE}"
        )

    # ------------------------------------------------------------
    # 11. FINAL ASSESSMENT
    # ------------------------------------------------------------

    section("11. V0.13 ASSESSMENT")

    missing_requirements = [
        name
        for name, available in requirements.items()
        if not available
    ]

    if not missing_requirements:

        print(
            "✓ Required CLV fields appear to be available."
        )

        print(
            "\nNEXT STEP:"
        )

        print(
            "Build a chronological closing-line-value diagnostic."
        )

    else:

        print(
            "✗ EDGE is NOT yet ready for proper CLV analysis."
        )

        print(
            "\nMissing:"
        )

        for item in missing_requirements:
            print(f"  - {item}")

        print(
            "\nRECOMMENDATION:"
        )

        print(
            "Do not build another betting model yet."
        )

        print(
            "Improve the odds dataset first."
        )

        print(
            "The most important missing information is "
            "opening/closing odds with timestamps and source."
        )

    print("\n" + "=" * 78)
    print("V0.13 COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
