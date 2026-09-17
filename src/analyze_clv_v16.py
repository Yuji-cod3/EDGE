#!/usr/bin/env python3

"""
EDGE — V0.16 CLV / MARKET MOVEMENT ANALYSIS

Purpose:
    Analyze explicit opening and closing odds from V0.15.

Rules:
    - No threshold optimization
    - No stake optimization
    - No model retraining
    - No ROI claims
    - Descriptive CLV / market-efficiency analysis only
    - Closing data only used where explicitly available
"""

from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIG
# ============================================================

INPUT = Path("data/processed/market_odds_v15.csv")

OUTPUT_PREDICTIONS = Path(
    "data/processed/clv_v16_observations.csv"
)

OUTPUT_SUMMARY = Path(
    "reports/clv_v16_summary.csv"
)

MARKET = "OVER_2.5"


# ============================================================
# HELPERS
# ============================================================

def clean_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def normalize_selection(value):
    if pd.isna(value):
        return None

    value = str(value).strip().upper()

    if value in {">2.5", "OVER", "OVER_2.5", "O2.5"}:
        return "OVER_2.5"

    if value in {"<2.5", "UNDER", "UNDER_2.5", "U2.5"}:
        return "UNDER_2.5"

    return value


def classify_movement(opening, closing, tolerance=1e-9):
    if pd.isna(opening) or pd.isna(closing):
        return "MISSING"

    if closing < opening - tolerance:
        return "SHORTENED"

    if closing > opening + tolerance:
        return "DRIFTED"

    return "UNCHANGED"


# ============================================================
# LOAD
# ============================================================

print("=" * 78)
print("EDGE — V0.16 CLV / MARKET MOVEMENT ANALYSIS")
print("=" * 78)

if not INPUT.exists():
    raise FileNotFoundError(f"Input file not found: {INPUT}")

df = pd.read_csv(INPUT)

print()
print("Input file:")
print(INPUT)

print()
print("Rows:", len(df))
print("Columns:", len(df.columns))

print()
print("Columns:")
for col in df.columns:
    print(f"  - {col}")


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required = [
    "match_id",
    "season",
    "market",
    "selection",
    "opening_price",
    "closing_price",
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(f"  - {c}" for c in missing)
    )


# ============================================================
# NORMALIZE
# ============================================================

df["opening_price"] = clean_numeric(df["opening_price"])
df["closing_price"] = clean_numeric(df["closing_price"])

df["selection"] = df["selection"].apply(normalize_selection)

df["market"] = (
    df["market"]
    .astype(str)
    .str.strip()
    .str.upper()
)


# ============================================================
# BASIC COVERAGE
# ============================================================

print()
print("=" * 78)
print("1. COVERAGE")
print("=" * 78)

total_rows = len(df)

opening_available = df["opening_price"].notna().sum()
closing_available = df["closing_price"].notna().sum()

print(f"Total records:       {total_rows:,}")
print(f"Opening available:   {opening_available:,}")
print(f"Closing available:   {closing_available:,}")
print(
    f"Closing coverage:    "
    f"{closing_available / total_rows:.2%}"
)


# ============================================================
# CREATE MATCHED OPENING/CLOSING OBSERVATIONS
# ============================================================

analysis = df[
    df["opening_price"].notna()
    & df["closing_price"].notna()
].copy()

analysis = analysis[
    (analysis["opening_price"] > 1.0)
    & (analysis["closing_price"] > 1.0)
].copy()

print()
print("=" * 78)
print("2. MATCHED OPENING/CLOSING OBSERVATIONS")
print("=" * 78)

print(f"Matched observations: {len(analysis):,}")
print(
    f"Unique matches:       "
    f"{analysis['match_id'].nunique():,}"
)


# ============================================================
# CLV / MOVEMENT CALCULATIONS
# ============================================================

analysis["odds_change"] = (
    analysis["closing_price"]
    - analysis["opening_price"]
)

analysis["odds_change_pct"] = (
    analysis["closing_price"]
    / analysis["opening_price"]
    - 1.0
)

analysis["opening_implied_probability"] = (
    1.0 / analysis["opening_price"]
)

analysis["closing_implied_probability"] = (
    1.0 / analysis["closing_price"]
)

analysis["probability_change"] = (
    analysis["closing_implied_probability"]
    - analysis["opening_implied_probability"]
)

analysis["clv"] = (
    analysis["closing_price"]
    / analysis["opening_price"]
    - 1.0
)

analysis["movement"] = analysis.apply(
    lambda row: classify_movement(
        row["opening_price"],
        row["closing_price"]
    ),
    axis=1,
)


# ============================================================
# OVERALL MARKET MOVEMENT
# ============================================================

print()
print("=" * 78)
print("3. OVERALL MARKET MOVEMENT")
print("=" * 78)

print(
    f"Mean opening odds:        "
    f"{analysis['opening_price'].mean():.6f}"
)

print(
    f"Mean closing odds:        "
    f"{analysis['closing_price'].mean():.6f}"
)

print(
    f"Mean odds change:         "
    f"{analysis['odds_change'].mean():+.6f}"
)

print(
    f"Mean odds change %:       "
    f"{analysis['odds_change_pct'].mean():+.4%}"
)

print(
    f"Mean CLV:                 "
    f"{analysis['clv'].mean():+.6f}"
)

print(
    f"Median CLV:               "
    f"{analysis['clv'].median():+.6f}"
)


# ============================================================
# MOVEMENT DISTRIBUTION
# ============================================================

movement_counts = (
    analysis["movement"]
    .value_counts()
    .reindex(
        ["SHORTENED", "DRIFTED", "UNCHANGED"],
        fill_value=0
    )
)

print()
print("=" * 78)
print("4. MOVEMENT DISTRIBUTION")
print("=" * 78)

for movement, count in movement_counts.items():
    pct = count / len(analysis)
    print(
        f"{movement:<12} "
        f"{count:>6,} "
        f"({pct:.2%})"
    )


# ============================================================
# SOURCE ANALYSIS
# ============================================================

print()
print("=" * 78)
print("5. SOURCE ANALYSIS")
print("=" * 78)

if "opening_source" in analysis.columns:
    source_group = (
        analysis
        .groupby("opening_source", dropna=False)
        .agg(
            observations=("match_id", "size"),
            mean_opening=("opening_price", "mean"),
            mean_closing=("closing_price", "mean"),
            mean_clv=("clv", "mean"),
            median_clv=("clv", "median"),
            shortened=(
                "movement",
                lambda x: (x == "SHORTENED").sum()
            ),
            drifted=(
                "movement",
                lambda x: (x == "DRIFTED").sum()
            ),
        )
        .reset_index()
    )

    print(source_group.to_string(index=False))

else:
    print("opening_source not available.")


# ============================================================
# SEASON ANALYSIS
# ============================================================

print()
print("=" * 78)
print("6. SEASON-BY-SEASON CLV")
print("=" * 78)

season_summary = (
    analysis
    .groupby("season")
    .agg(
        observations=("match_id", "size"),
        unique_matches=("match_id", "nunique"),
        mean_opening=("opening_price", "mean"),
        mean_closing=("closing_price", "mean"),
        mean_odds_change=("odds_change", "mean"),
        mean_clv=("clv", "mean"),
        median_clv=("clv", "median"),
        shortened=(
            "movement",
            lambda x: (x == "SHORTENED").sum()
        ),
        drifted=(
            "movement",
            lambda x: (x == "DRIFTED").sum()
        ),
        unchanged=(
            "movement",
            lambda x: (x == "UNCHANGED").sum()
        ),
    )
    .reset_index()
)

season_summary["shortened_rate"] = (
    season_summary["shortened"]
    / season_summary["observations"]
)

season_summary["drifted_rate"] = (
    season_summary["drifted"]
    / season_summary["observations"]
)

print(season_summary.to_string(index=False))


# ============================================================
# SELECTION ANALYSIS
# ============================================================

print()
print("=" * 78)
print("7. SELECTION ANALYSIS")
print("=" * 78)

selection_summary = (
    analysis
    .groupby("selection")
    .agg(
        observations=("match_id", "size"),
        mean_opening=("opening_price", "mean"),
        mean_closing=("closing_price", "mean"),
        mean_clv=("clv", "mean"),
        median_clv=("clv", "median"),
        mean_probability_change=(
            "probability_change",
            "mean"
        ),
        shortened=(
            "movement",
            lambda x: (x == "SHORTENED").sum()
        ),
        drifted=(
            "movement",
            lambda x: (x == "DRIFTED").sum()
        ),
    )
    .reset_index()
)

print(selection_summary.to_string(index=False))


# ============================================================
# SOURCE + SELECTION ANALYSIS
# ============================================================

print()
print("=" * 78)
print("8. SOURCE × SELECTION")
print("=" * 78)

if "opening_source" in analysis.columns:

    source_selection = (
        analysis
        .groupby(
            ["opening_source", "selection"],
            dropna=False
        )
        .agg(
            observations=("match_id", "size"),
            mean_opening=("opening_price", "mean"),
            mean_closing=("closing_price", "mean"),
            mean_clv=("clv", "mean"),
            median_clv=("clv", "median"),
        )
        .reset_index()
    )

    print(source_selection.to_string(index=False))


# ============================================================
# EXTREME MOVEMENTS
# ============================================================

print()
print("=" * 78)
print("9. EXTREME MARKET MOVEMENTS")
print("=" * 78)

print("\nLargest shortenings:")

largest_shortening = (
    analysis
    .sort_values("clv", ascending=True)
    .head(10)
)

print(
    largest_shortening[
        [
            "match_id",
            "season",
            "selection",
            "opening_price",
            "closing_price",
            "clv",
        ]
    ].to_string(index=False)
)

print("\nLargest drifts:")

largest_drift = (
    analysis
    .sort_values("clv", ascending=False)
    .head(10)
)

print(
    largest_drift[
        [
            "match_id",
            "season",
            "selection",
            "opening_price",
            "closing_price",
            "clv",
        ]
    ].to_string(index=False)
)


# ============================================================
# MARKET QUALITY CHECK
# ============================================================

print()
print("=" * 78)
print("10. MARKET QUALITY CHECK")
print("=" * 78)

same_price = (
    analysis["opening_price"]
    == analysis["closing_price"]
).sum()

print(
    f"Exact unchanged prices: "
    f"{same_price:,} "
    f"({same_price / len(analysis):.2%})"
)

print(
    f"Opening price range: "
    f"{analysis['opening_price'].min():.3f} "
    f"→ "
    f"{analysis['opening_price'].max():.3f}"
)

print(
    f"Closing price range: "
    f"{analysis['closing_price'].min():.3f} "
    f"→ "
    f"{analysis['closing_price'].max():.3f}"
)


# ============================================================
# SAVE OBSERVATIONS
# ============================================================

OUTPUT_PREDICTIONS.parent.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_SUMMARY.parent.mkdir(
    parents=True,
    exist_ok=True
)

analysis.to_csv(
    OUTPUT_PREDICTIONS,
    index=False
)

season_summary.to_csv(
    OUTPUT_SUMMARY,
    index=False
)


# ============================================================
# FINAL ASSESSMENT
# ============================================================

print()
print("=" * 78)
print("11. V0.16 ASSESSMENT")
print("=" * 78)

print("✓ Opening/closing observations analyzed.")
print("✓ No betting threshold optimized.")
print("✓ No stake sizing optimized.")
print("✓ No ROI optimization performed.")
print("✓ CLV treated as a market-quality diagnostic.")
print("✓ Results should not be interpreted as proof of profitability.")

if len(analysis) > 0:
    shortened_rate = (
        (analysis["movement"] == "SHORTENED").mean()
    )

    drifted_rate = (
        (analysis["movement"] == "DRIFTED").mean()
    )

    print()
    print(
        f"Shortened rate: {shortened_rate:.2%}"
    )

    print(
        f"Drifted rate:   {drifted_rate:.2%}"
    )

print()
print("Saved observations:")
print(f"  {OUTPUT_PREDICTIONS}")

print("Saved summary:")
print(f"  {OUTPUT_SUMMARY}")

print()
print("=" * 78)
print("V0.16 COMPLETE")
print("=" * 78)
