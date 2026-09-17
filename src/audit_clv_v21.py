from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================================
# EDGE — V0.21 VIG-ADJUSTED CLV AUDIT
# ============================================================================
#
# Purpose:
#   Evaluate bookmaker-margin-adjusted CLV for the repaired V0.19 dataset.
#
# This script:
#   1. Validates OVER/UNDER pairing
#   2. Calculates raw implied probabilities
#   3. Calculates opening/closing overround
#   4. Calculates vig-adjusted ("fair") probabilities
#   5. Measures fair-probability movement
#   6. Measures price-based CLV
#   7. Compares raw and vig-adjusted movement
#   8. Audits CLV by selection and season
#   9. Identifies extreme observations
#
# It does NOT modify V0.19 or V0.20 outputs.
# ============================================================================


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "data/processed/clv_audit_v20.csv"
)

OUTPUT_FILE = (
    ROOT / "data/processed/clv_audit_v21.csv"
)

REPORT_FILE = (
    ROOT / "reports/clv_audit_v21_report.csv"
)

SEASON_FILE = (
    ROOT / "reports/clv_audit_v21_seasons.csv"
)

MATCH_FILE = (
    ROOT / "reports/clv_audit_v21_matches.csv"
)


print("=" * 78)
print("EDGE — V0.21 VIG-ADJUSTED CLV AUDIT")
print("=" * 78)


# ============================================================================
# 1. LOAD
# ============================================================================

df = pd.read_csv(INPUT_FILE)

print("\nINPUT DATASET")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")


required = {
    "match_id",
    "season",
    "selection",
    "opening_price",
    "closing_price",
    "opening_implied_probability",
    "closing_implied_probability",
    "clv_decimal",
}

missing = required - set(df.columns)

if missing:
    raise ValueError(
        f"Missing required columns: {sorted(missing)}"
    )


# ============================================================================
# 2. BASIC PRICE VALIDATION
# ============================================================================

print("\n" + "=" * 78)
print("1. PRICE VALIDATION")
print("=" * 78)

invalid = (
    df["opening_price"].isna()
    | df["closing_price"].isna()
    | (df["opening_price"] <= 1.0)
    | (df["closing_price"] <= 1.0)
)

print("Invalid observations:", int(invalid.sum()))

if invalid.any():
    raise ValueError(
        "Invalid price observations detected."
    )


# ============================================================================
# 3. BUILD MATCH-LEVEL OVER/UNDER TABLE
# ============================================================================

print("\n" + "=" * 78)
print("2. OVER / UNDER PAIRING")
print("=" * 78)

pair_counts = (
    df.groupby("match_id")["selection"]
    .nunique()
)

print(
    "Unique matches:",
    df["match_id"].nunique()
)

print(
    "Matches with both selections:",
    int((pair_counts == 2).sum())
)

print(
    "Matches with incomplete selection pair:",
    int((pair_counts != 2).sum())
)

if (pair_counts != 2).any():
    raise ValueError(
        "Incomplete OVER/UNDER pairs detected."
    )


# ============================================================================
# 4. PIVOT PRICES
# ============================================================================

prices = (
    df.pivot(
        index=[
            "match_id",
            "season",
        ],
        columns="selection",
        values=[
            "opening_price",
            "closing_price",
        ],
    )
    .reset_index()
)

prices.columns = [
    "_".join(
        str(x) for x in col
        if str(x) != ""
    ).strip("_")
    for col in prices.columns
]

expected_price_columns = {
    "opening_price_OVER",
    "opening_price_UNDER",
    "closing_price_OVER",
    "closing_price_UNDER",
}

missing_price_columns = (
    expected_price_columns
    - set(prices.columns)
)

if missing_price_columns:
    raise ValueError(
        "Missing paired price columns: "
        f"{sorted(missing_price_columns)}"
    )


# ============================================================================
# 5. RAW IMPLIED PROBABILITIES
# ============================================================================

prices["opening_raw_over"] = (
    1.0 / prices["opening_price_OVER"]
)

prices["opening_raw_under"] = (
    1.0 / prices["opening_price_UNDER"]
)

prices["closing_raw_over"] = (
    1.0 / prices["closing_price_OVER"]
)

prices["closing_raw_under"] = (
    1.0 / prices["closing_price_UNDER"]
)


# ============================================================================
# 6. OVERROUND
# ============================================================================

print("\n" + "=" * 78)
print("3. BOOKMAKER OVERROUND")
print("=" * 78)

prices["opening_overround"] = (
    prices["opening_raw_over"]
    + prices["opening_raw_under"]
)

prices["closing_overround"] = (
    prices["closing_raw_over"]
    + prices["closing_raw_under"]
)

prices["opening_margin"] = (
    prices["opening_overround"] - 1.0
)

prices["closing_margin"] = (
    prices["closing_overround"] - 1.0
)

print("\nOpening overround:")
print(
    prices["opening_overround"]
    .describe()
    .to_string()
)

print("\nClosing overround:")
print(
    prices["closing_overround"]
    .describe()
    .to_string()
)

print("\nOpening margin:")
print(
    prices["opening_margin"]
    .describe()
    .to_string()
)

print("\nClosing margin:")
print(
    prices["closing_margin"]
    .describe()
    .to_string()
)


# ============================================================================
# 7. VIG-ADJUSTED / FAIR PROBABILITIES
# ============================================================================

print("\n" + "=" * 78)
print("4. VIG-ADJUSTED PROBABILITIES")
print("=" * 78)

prices["opening_fair_over"] = (
    prices["opening_raw_over"]
    / prices["opening_overround"]
)

prices["opening_fair_under"] = (
    prices["opening_raw_under"]
    / prices["opening_overround"]
)

prices["closing_fair_over"] = (
    prices["closing_raw_over"]
    / prices["closing_overround"]
)

prices["closing_fair_under"] = (
    prices["closing_raw_under"]
    / prices["closing_overround"]
)


# Verify probabilities sum to 1.

prices["opening_fair_sum"] = (
    prices["opening_fair_over"]
    + prices["opening_fair_under"]
)

prices["closing_fair_sum"] = (
    prices["closing_fair_over"]
    + prices["closing_fair_under"]
)

opening_probability_error = (
    prices["opening_fair_sum"] - 1.0
).abs()

closing_probability_error = (
    prices["closing_fair_sum"] - 1.0
).abs()

print(
    "Maximum opening fair-probability error:",
    opening_probability_error.max()
)

print(
    "Maximum closing fair-probability error:",
    closing_probability_error.max()
)


# ============================================================================
# 8. FAIR PROBABILITY MOVEMENT
# ============================================================================

prices["fair_probability_change_over"] = (
    prices["closing_fair_over"]
    - prices["opening_fair_over"]
)

prices["fair_probability_change_under"] = (
    prices["closing_fair_under"]
    - prices["opening_fair_under"]
)


# ============================================================================
# 9. PRICE-BASED CLV
# ============================================================================

prices["clv_over"] = (
    prices["opening_price_OVER"]
    / prices["closing_price_OVER"]
    - 1.0
)

prices["clv_under"] = (
    prices["opening_price_UNDER"]
    / prices["closing_price_UNDER"]
    - 1.0
)


# ============================================================================
# 10. FAIR-PROBABILITY CLV
# ============================================================================

# Positive fair-probability CLV means the market's closing fair probability
# is LOWER than the opening fair probability from the bettor's perspective.
#
# Therefore:
#
#   opening_fair_probability
#   --------------------------------
#   closing_fair_probability
#
# A positive value means the bettor captured a favorable probability move.
#
prices["fair_clv_over"] = (
    prices["opening_fair_over"]
    / prices["closing_fair_over"]
    - 1.0
)

prices["fair_clv_under"] = (
    prices["opening_fair_under"]
    / prices["closing_fair_under"]
    - 1.0
)


# ============================================================================
# 11. RETURN TO LONG FORMAT
# ============================================================================

over = prices[
    [
        "match_id",
        "season",
        "opening_price_OVER",
        "closing_price_OVER",
        "opening_raw_over",
        "closing_raw_over",
        "opening_fair_over",
        "closing_fair_over",
        "fair_probability_change_over",
        "clv_over",
        "fair_clv_over",
        "opening_overround",
        "closing_overround",
        "opening_margin",
        "closing_margin",
    ]
].copy()

over["selection"] = "OVER"

over = over.rename(
    columns={
        "opening_price_OVER": "opening_price",
        "closing_price_OVER": "closing_price",
        "opening_raw_over": "opening_raw_probability",
        "closing_raw_over": "closing_raw_probability",
        "opening_fair_over": "opening_fair_probability",
        "closing_fair_over": "closing_fair_probability",
        "fair_probability_change_over": "fair_probability_change",
        "clv_over": "price_clv",
        "fair_clv_over": "fair_probability_clv",
    }
)


under = prices[
    [
        "match_id",
        "season",
        "opening_price_UNDER",
        "closing_price_UNDER",
        "opening_raw_under",
        "closing_raw_under",
        "opening_fair_under",
        "closing_fair_under",
        "fair_probability_change_under",
        "clv_under",
        "fair_clv_under",
        "opening_overround",
        "closing_overround",
        "opening_margin",
        "closing_margin",
    ]
].copy()

under["selection"] = "UNDER"

under = under.rename(
    columns={
        "opening_price_UNDER": "opening_price",
        "closing_price_UNDER": "closing_price",
        "opening_raw_under": "opening_raw_probability",
        "closing_raw_under": "closing_raw_probability",
        "opening_fair_under": "opening_fair_probability",
        "closing_fair_under": "closing_fair_probability",
        "fair_probability_change_under": "fair_probability_change",
        "clv_under": "price_clv",
        "fair_clv_under": "fair_probability_clv",
    }
)


audit = pd.concat(
    [over, under],
    ignore_index=True,
)


# ============================================================================
# 12. CLV SUMMARY
# ============================================================================

print("\n" + "=" * 78)
print("5. RAW VS VIG-ADJUSTED CLV")
print("=" * 78)

print("\nPrice-based CLV:")
print(
    audit["price_clv"]
    .describe()
    .to_string()
)

print("\nFair-probability CLV:")
print(
    audit["fair_probability_clv"]
    .describe()
    .to_string()
)

print("\nFair probability movement:")
print(
    audit["fair_probability_change"]
    .describe()
    .to_string()
)


# ============================================================================
# 13. SELECTION SUMMARY
# ============================================================================

print("\n" + "=" * 78)
print("6. CLV BY SELECTION")
print("=" * 78)

selection_summary = (
    audit.groupby("selection")
    .agg(
        observations=("match_id", "size"),
        mean_price_clv=("price_clv", "mean"),
        median_price_clv=("price_clv", "median"),
        positive_price_clv_rate=(
            "price_clv",
            lambda x: (x > 0).mean(),
        ),
        mean_fair_clv=(
            "fair_probability_clv",
            "mean",
        ),
        median_fair_clv=(
            "fair_probability_clv",
            "median",
        ),
        positive_fair_clv_rate=(
            "fair_probability_clv",
            lambda x: (x > 0).mean(),
        ),
        mean_fair_probability_change=(
            "fair_probability_change",
            "mean",
        ),
    )
)

print(
    selection_summary.to_string()
)


# ============================================================================
# 14. SEASON SUMMARY
# ============================================================================

print("\n" + "=" * 78)
print("7. CLV BY SEASON")
print("=" * 78)

season_summary = (
    audit.groupby("season")
    .agg(
        observations=("match_id", "size"),
        matches=("match_id", "nunique"),
        mean_price_clv=("price_clv", "mean"),
        median_price_clv=("price_clv", "median"),
        positive_price_clv_rate=(
            "price_clv",
            lambda x: (x > 0).mean(),
        ),
        mean_fair_clv=(
            "fair_probability_clv",
            "mean",
        ),
        median_fair_clv=(
            "fair_probability_clv",
            "median",
        ),
        positive_fair_clv_rate=(
            "fair_probability_clv",
            lambda x: (x > 0).mean(),
        ),
        mean_opening_margin=(
            "opening_margin",
            "mean",
        ),
        mean_closing_margin=(
            "closing_margin",
            "mean",
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
# 15. MARGIN MOVEMENT
# ============================================================================

prices["margin_change"] = (
    prices["closing_margin"]
    - prices["opening_margin"]
)

print("\n" + "=" * 78)
print("8. BOOKMAKER MARGIN MOVEMENT")
print("=" * 78)

print(
    prices["margin_change"]
    .describe()
    .to_string()
)


# ============================================================================
# 16. CORRELATION
# ============================================================================

print("\n" + "=" * 78)
print("9. RAW VS VIG-ADJUSTED RELATIONSHIP")
print("=" * 78)

correlation = audit[
    [
        "price_clv",
        "fair_probability_clv",
        "fair_probability_change",
    ]
].corr()

print(
    correlation.to_string()
)


# ============================================================================
# 17. EXTREME OBSERVATIONS
# ============================================================================

print("\n" + "=" * 78)
print("10. EXTREME FAIR CLV")
print("=" * 78)

extreme = (
    audit["fair_probability_clv"]
    .abs()
    >= 0.25
)

print(
    "|Fair CLV| >= 25%:",
    int(extreme.sum())
)

if extreme.any():

    print("\nExtreme examples:")

    print(
        audit.loc[
            extreme,
            [
                "match_id",
                "season",
                "selection",
                "opening_price",
                "closing_price",
                "opening_fair_probability",
                "closing_fair_probability",
                "price_clv",
                "fair_probability_clv",
            ],
        ]
        .sort_values(
            "fair_probability_clv"
        )
        .head(20)
        .to_string(index=False)
    )


# ============================================================================
# 18. SAVE
# ============================================================================

audit.to_csv(
    OUTPUT_FILE,
    index=False,
)


report = pd.DataFrame(
    [
        {
            "version": "v0.21",
            "observations": len(audit),
            "unique_matches": audit["match_id"].nunique(),
            "mean_price_clv": audit["price_clv"].mean(),
            "median_price_clv": audit["price_clv"].median(),
            "positive_price_clv_rate": (
                audit["price_clv"] > 0
            ).mean(),
            "mean_fair_clv": (
                audit["fair_probability_clv"].mean()
            ),
            "median_fair_clv": (
                audit["fair_probability_clv"].median()
            ),
            "positive_fair_clv_rate": (
                audit["fair_probability_clv"] > 0
            ).mean(),
            "mean_opening_margin": (
                audit["opening_margin"].mean()
            ),
            "mean_closing_margin": (
                audit["closing_margin"].mean()
            ),
            "extreme_fair_clv_count": int(
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
# 19. FINAL
# ============================================================================

print("\n" + "=" * 78)
print("11. OUTPUT")
print("=" * 78)

print("Saved audit dataset:")
print(OUTPUT_FILE)

print("\nSaved report:")
print(REPORT_FILE)

print("\nSaved season summary:")
print(SEASON_FILE)

print("\nV0.21 STATUS: COMPLETE — REVIEW RESULTS BEFORE LOCKING CLV")

print("=" * 78)
