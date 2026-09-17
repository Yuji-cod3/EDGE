from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================================
# EDGE — V0.20 CLV CALCULATION VALIDATION
# ============================================================================
#
# Purpose:
#   Validate the repaired V0.19 CLV dataset before it is used for analysis.
#
# This script does NOT modify V0.19 outputs.
#
# It checks:
#   1. Price validity
#   2. Missing values
#   3. Duplicate observations
#   4. OVER/UNDER completeness
#   5. Implied probabilities
#   6. Multiple CLV formulations
#   7. Extreme CLV values
#   8. Season-level coverage
#   9. Selection-level consistency
#  10. Result consistency
#
# ============================================================================


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "data/processed/clv_match_mapping_v19.csv"
)

OUTPUT_FILE = (
    ROOT / "data/processed/clv_audit_v20.csv"
)

REPORT_FILE = (
    ROOT / "reports/clv_audit_v20_report.csv"
)

SEASON_FILE = (
    ROOT / "reports/clv_audit_v20_seasons.csv"
)


print("=" * 78)
print("EDGE — V0.20 CLV CALCULATION VALIDATION")
print("=" * 78)


# ============================================================================
# 1. LOAD
# ============================================================================

df = pd.read_csv(INPUT_FILE)

print("\nINPUT DATASET")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")

print("\nColumns:")
for c in df.columns:
    print(f"  - {c}")


# ============================================================================
# 2. REQUIRED COLUMNS
# ============================================================================

required = {
    "match_id",
    "season",
    "selection",
    "opening_price",
    "closing_price",
    "match_id_result",
    "home_goals",
    "away_goals",
    "total_goals",
    "over25_result",
    "matched",
}

missing = required - set(df.columns)

if missing:
    raise ValueError(
        f"Missing required columns: {sorted(missing)}"
    )


# ============================================================================
# 3. MATCHING VALIDATION
# ============================================================================

print("\n" + "=" * 78)
print("1. MATCHING VALIDATION")
print("=" * 78)

print(
    "Matched:",
    int(df["matched"].sum())
)

print(
    "Unmatched:",
    int((~df["matched"]).sum())
)

if not df["matched"].all():
    raise ValueError(
        "V0.19 contains unmatched observations."
    )


# ============================================================================
# 4. PRICE VALIDATION
# ============================================================================

print("\n" + "=" * 78)
print("2. PRICE VALIDATION")
print("=" * 78)

invalid_opening = (
    df["opening_price"].isna()
    | (df["opening_price"] <= 1.0)
)

invalid_closing = (
    df["closing_price"].isna()
    | (df["closing_price"] <= 1.0)
)

print(
    "Invalid opening prices:",
    int(invalid_opening.sum())
)

print(
    "Invalid closing prices:",
    int(invalid_closing.sum())
)

print(
    "Opening price range:",
    df["opening_price"].min(),
    "→",
    df["opening_price"].max()
)

print(
    "Closing price range:",
    df["closing_price"].min(),
    "→",
    df["closing_price"].max()
)


# ============================================================================
# 5. DUPLICATE VALIDATION
# ============================================================================

print("\n" + "=" * 78)
print("3. DUPLICATE VALIDATION")
print("=" * 78)

duplicate_keys = [
    "match_id",
    "selection",
]

duplicates = df.duplicated(
    subset=duplicate_keys,
    keep=False
)

print(
    "Duplicate match + selection observations:",
    int(duplicates.sum())
)

if duplicates.any():
    print("\nDuplicate examples:")
    print(
        df.loc[
            duplicates,
            duplicate_keys
            + ["opening_price", "closing_price"]
        ]
        .head(20)
        .to_string(index=False)
    )


# ============================================================================
# 6. SELECTION VALIDATION
# ============================================================================

print("\n" + "=" * 78)
print("4. SELECTION VALIDATION")
print("=" * 78)

print(
    df["selection"]
    .value_counts(dropna=False)
    .to_string()
)

unexpected_selection = ~df["selection"].isin(
    ["OVER", "UNDER"]
)

print(
    "\nUnexpected selections:",
    int(unexpected_selection.sum())
)


# ============================================================================
# 7. OVER / UNDER COMPLETENESS
# ============================================================================

print("\n" + "=" * 78)
print("5. OVER / UNDER COMPLETENESS")
print("=" * 78)

selection_counts = (
    df.groupby("match_id")["selection"]
    .nunique()
)

both_count = int(
    (selection_counts == 2).sum()
)

single_count = int(
    (selection_counts == 1).sum()
)

print(
    "Matches with both OVER and UNDER:",
    both_count
)

print(
    "Matches with only one selection:",
    single_count
)


# ============================================================================
# 8. IMPLIED PROBABILITIES
# ============================================================================

print("\n" + "=" * 78)
print("6. IMPLIED PROBABILITIES")
print("=" * 78)

df["opening_implied_probability"] = (
    1.0 / df["opening_price"]
)

df["closing_implied_probability"] = (
    1.0 / df["closing_price"]
)

df["implied_probability_change"] = (
    df["closing_implied_probability"]
    - df["opening_implied_probability"]
)

print(
    "Opening implied probability range:",
    round(df["opening_implied_probability"].min(), 6),
    "→",
    round(df["opening_implied_probability"].max(), 6)
)

print(
    "Closing implied probability range:",
    round(df["closing_implied_probability"].min(), 6),
    "→",
    round(df["closing_implied_probability"].max(), 6)
)


# ============================================================================
# 9. ODDS-BASED CLV
# ============================================================================

print("\n" + "=" * 78)
print("7. CLV CALCULATIONS")
print("=" * 78)

# Raw decimal-odds movement.
#
# Positive means the closing price is higher than the opening price.
#
# For a bettor who took the opening price:
#   higher closing odds generally means favorable movement.
#
df["odds_change"] = (
    df["closing_price"]
    / df["opening_price"]
    - 1.0
)


# Implied probability movement.
#
# Positive means the outcome became MORE likely according to the market.
# Negative means the outcome became LESS likely.
#
df["probability_change"] = (
    df["closing_implied_probability"]
    - df["opening_implied_probability"]
)


# A bettor-oriented CLV formulation:
#
#   opening price / closing price - 1
#
# Positive means the bettor obtained a better price than the closing market.
#
df["clv_decimal"] = (
    df["opening_price"]
    / df["closing_price"]
    - 1.0
)


print("\nCLV decimal statistics:")
print(
    df["clv_decimal"]
    .describe()
    .to_string()
)

print("\nOdds movement statistics:")
print(
    df["odds_change"]
    .describe()
    .to_string()
)

print("\nImplied probability movement:")
print(
    df["probability_change"]
    .describe()
    .to_string()
)


# ============================================================================
# 10. CLV SIGN CHECK
# ============================================================================

print("\n" + "=" * 78)
print("8. CLV SIGN CONSISTENCY")
print("=" * 78)

same_direction = (
    np.sign(df["clv_decimal"])
    == np.sign(df["odds_change"])
)

print(
    "CLV decimal and raw odds movement same sign:",
    int(same_direction.sum()),
    "/",
    len(df)
)

print(
    "CLV decimal and raw odds movement opposite sign:",
    int((~same_direction).sum()),
    "/",
    len(df)
)


# ============================================================================
# 11. EXTREME VALUES
# ============================================================================

print("\n" + "=" * 78)
print("9. EXTREME CLV VALUES")
print("=" * 78)

extreme = (
    df["clv_decimal"].abs() >= 0.25
)

print(
    "Observations with |CLV| >= 25%:",
    int(extreme.sum())
)

if extreme.any():
    print("\nExtreme examples:")

    print(
        df.loc[
            extreme,
            [
                "match_id",
                "season",
                "selection",
                "opening_price",
                "closing_price",
                "clv_decimal",
                "probability_change",
            ],
        ]
        .sort_values(
            "clv_decimal"
        )
        .head(20)
        .to_string(index=False)
    )


# ============================================================================
# 12. RESULT CONSISTENCY
# ============================================================================

print("\n" + "=" * 78)
print("10. RESULT CONSISTENCY")
print("=" * 78)


def expected_result(total_goals):
    if pd.isna(total_goals):
        return None

    if total_goals > 2.5:
        return "OVER"

    return "UNDER"


df["calculated_result"] = (
    df["total_goals"]
    .apply(expected_result)
)

result_mismatch = (
    df["calculated_result"]
    != df["over25_result"]
)

print(
    "Result mismatches:",
    int(result_mismatch.sum())
)

if result_mismatch.any():

    print("\nResult mismatch examples:")

    print(
        df.loc[
            result_mismatch,
            [
                "match_id",
                "home_goals",
                "away_goals",
                "total_goals",
                "over25_result",
                "calculated_result",
            ],
        ]
        .head(20)
        .to_string(index=False)
    )


# ============================================================================
# 13. CLV BY SEASON
# ============================================================================

print("\n" + "=" * 78)
print("11. SEASON CLV")
print("=" * 78)

season_summary = (
    df.groupby("season")
    .agg(
        observations=("match_id", "size"),
        matches=("match_id", "nunique"),
        mean_clv=("clv_decimal", "mean"),
        median_clv=("clv_decimal", "median"),
        mean_odds_change=("odds_change", "mean"),
        mean_probability_change=(
            "probability_change",
            "mean",
        ),
        positive_clv_rate=(
            "clv_decimal",
            lambda x: (x > 0).mean(),
        ),
    )
    .reset_index()
)

print(
    season_summary.to_string(index=False)
)

season_summary.to_csv(
    SEASON_FILE,
    index=False,
)


# ============================================================================
# 14. CLV BY SELECTION
# ============================================================================

print("\n" + "=" * 78)
print("12. CLV BY SELECTION")
print("=" * 78)

selection_summary = (
    df.groupby("selection")
    .agg(
        observations=("match_id", "size"),
        mean_clv=("clv_decimal", "mean"),
        median_clv=("clv_decimal", "median"),
        positive_clv_rate=(
            "clv_decimal",
            lambda x: (x > 0).mean(),
        ),
    )
)

print(
    selection_summary.to_string()
)


# ============================================================================
# 15. SAVE AUDIT DATASET
# ============================================================================

df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================================
# 16. SUMMARY REPORT
# ============================================================================

report = pd.DataFrame(
    [
        {
            "version": "v0.20",
            "observations": len(df),
            "unique_matches": df["match_id"].nunique(),
            "matched_observations": int(
                df["matched"].sum()
            ),
            "unmatched_observations": int(
                (~df["matched"]).sum()
            ),
            "invalid_opening_prices": int(
                invalid_opening.sum()
            ),
            "invalid_closing_prices": int(
                invalid_closing.sum()
            ),
            "duplicate_match_selection_rows": int(
                duplicates.sum()
            ),
            "unexpected_selections": int(
                unexpected_selection.sum()
            ),
            "matches_with_both_selections": both_count,
            "matches_with_single_selection": single_count,
            "result_mismatches": int(
                result_mismatch.sum()
            ),
            "mean_clv": df["clv_decimal"].mean(),
            "median_clv": df["clv_decimal"].median(),
            "positive_clv_rate": (
                df["clv_decimal"] > 0
            ).mean(),
            "extreme_clv_count": int(
                extreme.sum()
            ),
        }
    ]
)

report.to_csv(
    REPORT_FILE,
    index=False,
)


# ============================================================================
# 17. FINAL STATUS
# ============================================================================

print("\n" + "=" * 78)
print("13. OUTPUT")
print("=" * 78)

print("Saved audit dataset:")
print(OUTPUT_FILE)

print("\nSaved report:")
print(REPORT_FILE)

print("\nSaved season summary:")
print(SEASON_FILE)


issues = (
    int(invalid_opening.sum())
    + int(invalid_closing.sum())
    + int(duplicates.sum())
    + int(unexpected_selection.sum())
    + int(result_mismatch.sum())
)

if issues == 0:
    status = "PASS — NO STRUCTURAL ISSUES DETECTED"
else:
    status = (
        f"REVIEW REQUIRED — {issues} "
        "potential issues detected"
    )

print(f"\nV0.20 STATUS: {status}")

print("=" * 78)
