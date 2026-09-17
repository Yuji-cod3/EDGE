import pandas as pd
from pathlib import Path

FILE = Path("data/processed/market_odds.csv")


def main():
    df = pd.read_csv(FILE)

    print("\nEDGE — NORMALIZED ODDS VALIDATION")
    print("=" * 75)

    print(f"Rows: {len(df):,}")
    print(f"Unique matches: {df['match_id'].nunique():,}")

    # ---------------------------------------------------------
    # Required columns
    # ---------------------------------------------------------

    required = [
        "match_id",
        "season",
        "market",
        "selection",
        "price",
        "price_type",
        "price_source",
        "source_column",
    ]

    missing = [c for c in required if c not in df.columns]

    print("\nRequired columns:")
    if missing:
        print(f"  MISSING: {missing}")
    else:
        print("  All present")

    # ---------------------------------------------------------
    # Nulls
    # ---------------------------------------------------------

    print("\nMissing values:")

    nulls = df.isna().sum()
    nulls = nulls[nulls > 0]

    if nulls.empty:
        print("  None")
    else:
        print(nulls.to_string())

    # ---------------------------------------------------------
    # Invalid prices
    # ---------------------------------------------------------

    invalid_prices = df[
        (df["price"] <= 1) |
        (~df["price"].apply(lambda x: isinstance(x, (int, float))))
    ]

    print("\nInvalid prices:")
    print(f"  {len(invalid_prices)}")

    # ---------------------------------------------------------
    # Exact duplicates
    # ---------------------------------------------------------

    duplicate_rows = df.duplicated().sum()

    print("\nExact duplicate rows:")
    print(f"  {duplicate_rows}")

    # ---------------------------------------------------------
    # Duplicate logical observations
    # ---------------------------------------------------------

    keys = [
        "match_id",
        "market",
        "selection",
        "price_type",
        "price_source",
    ]

    duplicate_logical = df.duplicated(subset=keys).sum()

    print("\nDuplicate logical observations:")
    print(f"  {duplicate_logical}")

    # ---------------------------------------------------------
    # Market/source/type coverage
    # ---------------------------------------------------------

    print("\nObservations by market / price type / source:")

    counts = (
        df.groupby(
            ["market", "price_type", "price_source"]
        )
        .size()
        .sort_index()
    )

    print(counts.to_string())

    # ---------------------------------------------------------
    # Selection balance
    # ---------------------------------------------------------

    print("\nSelections:")

    print(
        df["selection"]
        .value_counts()
        .to_string()
    )

    # ---------------------------------------------------------
    # Pair integrity
    # ---------------------------------------------------------

    pair_counts = (
        df.groupby(
            [
                "match_id",
                "market",
                "price_type",
                "price_source",
            ]
        )["selection"]
        .nunique()
    )

    complete_pairs = (pair_counts == 2).sum()
    incomplete_pairs = (pair_counts != 2).sum()

    print("\nO/U pair integrity:")
    print(f"  Complete pairs:   {complete_pairs:,}")
    print(f"  Incomplete pairs: {incomplete_pairs:,}")

    if incomplete_pairs:
        print("\nIncomplete pair examples:")

        print(
            pair_counts[pair_counts != 2]
            .head(20)
            .to_string()
        )

    # ---------------------------------------------------------
    # Season coverage
    # ---------------------------------------------------------

    print("\nRows by season:")

    season_counts = (
        df.groupby("season")
        .size()
        .sort_index()
    )

    print(season_counts.to_string())

    # ---------------------------------------------------------
    # Price ranges
    # ---------------------------------------------------------

    print("\nPrice range:")

    print(f"  Minimum: {df['price'].min():.3f}")
    print(f"  Maximum: {df['price'].max():.3f}")
    print(f"  Median:  {df['price'].median():.3f}")

    print("=" * 75)

    # ---------------------------------------------------------
    # Final status
    # ---------------------------------------------------------

    problems = []

    if missing:
        problems.append("missing required columns")

    if nulls.any():
        problems.append("missing values")

    if len(invalid_prices):
        problems.append("invalid prices")

    if duplicate_rows:
        problems.append("exact duplicates")

    if duplicate_logical:
        problems.append("duplicate logical observations")

    if incomplete_pairs:
        problems.append("incomplete O/U pairs")

    if problems:
        print("\nSTATUS: REVIEW REQUIRED")
        print("Problems:")
        for problem in problems:
            print(f"  - {problem}")
    else:
        print("\nSTATUS: PASS")


if __name__ == "__main__":
    main()
