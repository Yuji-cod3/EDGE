#!/usr/bin/env python3

"""
EDGE — V0.15 EXPLICIT OPENING/CLOSING ODDS BUILDER

Purpose
-------
Rebuild the odds dataset while preserving the semantic distinction between
opening and closing prices.

Important:
- Do NOT infer closing prices from arbitrary observations.
- Football-Data style fields ending in "C" are treated as closing prices.
- Fields without "C" are treated as opening/current prices according to
  the source convention.
- Historical seasons before closing fields were available remain opening-only.
"""

from pathlib import Path
import pandas as pd
import numpy as np


RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")

OUTPUT_FILE = OUT_DIR / "market_odds_v15.csv"
AUDIT_FILE = Path("reports/odds_v15_audit.csv")


# ---------------------------------------------------------------------
# MARKET DEFINITION
# ---------------------------------------------------------------------

MARKET = "OVER_2.5"

SELECTIONS = {
    "OVER": {
        "opening": [
            "B365>2.5",
            "Avg>2.5",
            "Max>2.5",
            "BFE>2.5",
            "BbAv>2.5",
            "BbMx>2.5",
            "P>2.5",
        ],
        "closing": [
            "B365C>2.5",
            "AvgC>2.5",
            "MaxC>2.5",
            "BFEC>2.5",
            "PC>2.5",
        ],
    },
    "UNDER": {
        "opening": [
            "B365<2.5",
            "Avg<2.5",
            "Max<2.5",
            "BFE<2.5",
            "BbAv<2.5",
            "BbMx<2.5",
            "P<2.5",
        ],
        "closing": [
            "B365C<2.5",
            "AvgC<2.5",
            "MaxC<2.5",
            "BFEC<2.5",
            "PC<2.5",
        ],
    },
}


# ---------------------------------------------------------------------
# SOURCE CLASSIFICATION
# ---------------------------------------------------------------------

def identify_source(column):
    """
    Identify bookmaker / market source from a raw odds column.
    """

    if column.startswith("B365"):
        return "BET365"

    if column.startswith("BFE"):
        return "BETFAIR"

    if column.startswith("BFEC"):
        return "BETFAIR"

    if column.startswith("Avg"):
        return "MARKET_AVERAGE"

    if column.startswith("Max"):
        return "MARKET_MAX"

    if column.startswith("BbAv"):
        return "BETBOOK_AVERAGE"

    if column.startswith("BbMx"):
        return "BETBOOK_MAX"

    if column.startswith("P"):
        return "Pinnacle"

    return "UNKNOWN"


def classify_price_type(column):
    """
    Explicitly classify opening vs closing.

    A trailing C in the Football-Data convention indicates closing odds.
    """

    if column.startswith(("B365C", "AvgC", "MaxC", "BFEC", "PC")):
        return "CLOSING"

    return "OPENING"


# ---------------------------------------------------------------------
# MATCH ID
# ---------------------------------------------------------------------

def build_match_id(df, season):
    """
    Build a stable match identifier.

    Prefer an existing Date/HomeTeam/AwayTeam combination.
    """

    required = {"Date", "HomeTeam", "AwayTeam"}

    if required.issubset(df.columns):

        date_values = (
            df["Date"]
            .astype(str)
            .str.strip()
        )

        home = (
            df["HomeTeam"]
            .astype(str)
            .str.strip()
        )

        away = (
            df["AwayTeam"]
            .astype(str)
            .str.strip()
        )

        return (
            season
            + "_"
            + date_values
            + "_"
            + home
            + "_"
            + away
        )

    # Fallback to row-based ID.
    return [
        f"{season}_ROW_{i:04d}"
        for i in range(len(df))
    ]


# ---------------------------------------------------------------------
# SELECT BEST SOURCE
# ---------------------------------------------------------------------

def select_source(row, columns):
    """
    Select a deterministic source.

    Priority:
    1. BET365
    2. MARKET_AVERAGE
    3. MARKET_MAX
    4. BETFAIR
    5. BETBOOK_AVERAGE
    6. Pinnacle

    This does NOT optimize for value.
    It simply provides a stable primary observation.
    """

    priority = [
        "B365",
        "Avg",
        "Max",
        "BFE",
        "BbAv",
        "BbMx",
        "P",
    ]

    for prefix in priority:

        for column in columns:

            if column.startswith(prefix):

                value = row[column]

                if pd.notna(value):

                    try:
                        value = float(value)

                        if value > 1.0:
                            return column, value

                    except (ValueError, TypeError):
                        pass

    return None, np.nan


# ---------------------------------------------------------------------
# PROCESS ONE SEASON
# ---------------------------------------------------------------------

def process_season(path):

    season = path.stem.replace("EPL_", "")

    print(f"\nProcessing {path.name}")

    df = pd.read_csv(path)

    print(f"Rows: {len(df)}")

    match_ids = build_match_id(df, season)

    records = []

    for idx, row in df.iterrows():

        match_id = match_ids[idx]

        for selection, definition in SELECTIONS.items():

            opening_columns = [
                c for c in definition["opening"]
                if c in df.columns
            ]

            closing_columns = [
                c for c in definition["closing"]
                if c in df.columns
            ]

            opening_column, opening_price = select_source(
                row,
                opening_columns
            )

            closing_column, closing_price = select_source(
                row,
                closing_columns
            )

            # ---------------------------------------------------------
            # PRIMARY ODDS RECORD
            # ---------------------------------------------------------

            if pd.notna(opening_price):

                records.append({
                    "match_id": match_id,
                    "season": season,
                    "market": MARKET,
                    "selection": selection,

                    "opening_price": opening_price,
                    "opening_source_column": opening_column,
                    "opening_source": (
                        identify_source(opening_column)
                        if opening_column
                        else None
                    ),

                    "closing_price": closing_price,
                    "closing_source_column": closing_column,
                    "closing_source": (
                        identify_source(closing_column)
                        if closing_column
                        else None
                    ),

                    "has_opening": True,
                    "has_closing": pd.notna(closing_price),

                    "raw_row": idx,
                })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    print("=" * 78)
    print("EDGE — V0.15 EXPLICIT OPENING/CLOSING ODDS BUILDER")
    print("=" * 78)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(RAW_DIR.glob("EPL_*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No EPL CSV files found in {RAW_DIR}"
        )

    print(f"\nRaw files found: {len(files)}")

    all_data = []
    audit_rows = []

    for path in files:

        data = process_season(path)

        all_data.append(data)

        season = path.stem.replace("EPL_", "")

        if len(data) == 0:
            continue

        audit_rows.append({
            "season": season,
            "records": len(data),
            "unique_matches": data["match_id"].nunique(),
            "opening_available": int(
                data["has_opening"].sum()
            ),
            "closing_available": int(
                data["has_closing"].sum()
            ),
            "closing_rate": (
                data["has_closing"].mean()
            ),
        })

    if not all_data:
        raise RuntimeError("No odds records were produced.")

    odds = pd.concat(
        all_data,
        ignore_index=True
    )

    # -------------------------------------------------------------
    # DATA CLEANING
    # -------------------------------------------------------------

    numeric_columns = [
        "opening_price",
        "closing_price",
    ]

    for column in numeric_columns:

        odds[column] = pd.to_numeric(
            odds[column],
            errors="coerce"
        )

        odds.loc[
            odds[column] <= 1.0,
            column
        ] = np.nan

    # -------------------------------------------------------------
    # CLV FIELDS
    # -------------------------------------------------------------

    odds["closing_implied_probability"] = (
        1.0 / odds["closing_price"]
    )

    odds["opening_implied_probability"] = (
        1.0 / odds["opening_price"]
    )

    odds["price_change"] = (
        odds["closing_price"]
        - odds["opening_price"]
    )

    odds["implied_probability_change"] = (
        odds["closing_implied_probability"]
        - odds["opening_implied_probability"]
    )

    odds["closing_available"] = (
        odds["closing_price"].notna()
    )

    odds["opening_available"] = (
        odds["opening_price"].notna()
    )

    # -------------------------------------------------------------
    # VALIDATION
    # -------------------------------------------------------------

    print("\n" + "=" * 78)
    print("V0.15 DATASET SUMMARY")
    print("=" * 78)

    print(f"Total records: {len(odds):,}")
    print(
        f"Unique matches: "
        f"{odds['match_id'].nunique():,}"
    )

    print(
        f"Opening prices: "
        f"{odds['opening_available'].sum():,}"
    )

    print(
        f"Closing prices: "
        f"{odds['closing_available'].sum():,}"
    )

    print("\nClosing availability by season:")

    season_summary = (
        odds
        .groupby("season")
        .agg(
            matches=("match_id", "nunique"),
            records=("match_id", "size"),
            opening=("opening_available", "sum"),
            closing=("closing_available", "sum"),
        )
        .reset_index()
    )

    season_summary["closing_rate"] = (
        season_summary["closing"]
        / season_summary["records"]
    )

    print(season_summary.to_string(index=False))

    # -------------------------------------------------------------
    # SOURCE AUDIT
    # -------------------------------------------------------------

    print("\n" + "=" * 78)
    print("SOURCE AUDIT")
    print("=" * 78)

    print("\nOpening sources:")
    print(
        odds["opening_source"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nClosing sources:")
    print(
        odds["closing_source"]
        .value_counts(dropna=False)
        .to_string()
    )

    # -------------------------------------------------------------
    # DATA QUALITY CHECK
    # -------------------------------------------------------------

    print("\n" + "=" * 78)
    print("DATA QUALITY")
    print("=" * 78)

    print(
        "Duplicate match/selection records:",
        odds.duplicated(
            subset=[
                "match_id",
                "selection"
            ]
        ).sum()
    )

    print(
        "Missing opening prices:",
        odds["opening_price"].isna().sum()
    )

    print(
        "Missing closing prices:",
        odds["closing_price"].isna().sum()
    )

    print(
        "Invalid opening prices:",
        (
            odds["opening_price"] <= 1.0
        ).sum()
    )

    print(
        "Invalid closing prices:",
        (
            odds["closing_price"] <= 1.0
        ).sum()
    )

    # -------------------------------------------------------------
    # SAVE
    # -------------------------------------------------------------

    odds.to_csv(
        OUTPUT_FILE,
        index=False
    )

    season_summary.to_csv(
        AUDIT_FILE,
        index=False
    )

    print("\n" + "=" * 78)
    print("V0.15 COMPLETE")
    print("=" * 78)

    print(f"\nSaved dataset:")
    print(f"  {OUTPUT_FILE}")

    print(f"\nSaved audit:")
    print(f"  {AUDIT_FILE}")


if __name__ == "__main__":
    main()
