from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================================
# EDGE — V0.22 CLV CONVENTION VALIDATION
# ============================================================================
#
# Purpose:
#   Correct and validate the bettor-oriented CLV convention.
#
# OFFICIAL CANDIDATE DEFINITIONS
#
# 1. Price CLV
#
#       opening_price / closing_price - 1
#
#   Positive = favorable movement for the bettor.
#
#
# 2. Vig-adjusted probability movement
#
#       closing_fair_probability / opening_fair_probability - 1
#
#   Positive = favorable movement for the bettor.
#
#
# This script does NOT modify previous datasets.
# ============================================================================


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "data/processed/clv_audit_v20.csv"
)

OUTPUT_FILE = (
    ROOT / "data/processed/clv_audit_v22.csv"
)

REPORT_FILE = (
    ROOT / "reports/clv_audit_v22_report.csv"
)

SEASON_FILE = (
    ROOT / "reports/clv_audit_v22_seasons.csv"
)


print("=" * 78)
print("EDGE — V0.22 CLV CONVENTION VALIDATION")
print("=" * 78)


# ============================================================================
# 1. LOAD
# ============================================================================

df = pd.read_csv(INPUT_FILE)

required = {
    "match_id",
    "season",
    "selection",
    "opening_price",
    "closing_price",
}

missing = required - set(df.columns)

if missing:
    raise ValueError(
        f"Missing required columns: {sorted(missing)}"
    )

print("\nINPUT DATASET")
print(f"Rows: {len(df)}")
print(f"Unique matches: {df['match_id'].nunique()}")


# ============================================================================
# 2. PIVOT
# ============================================================================

prices = (
    df.pivot(
        index=["match_id", "season"],
        columns="selection",
        values=["opening_price", "closing_price"],
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


required_prices = {
    "opening_price_OVER",
    "opening_price_UNDER",
    "closing_price_OVER",
    "closing_price_UNDER",
}

missing_prices = (
    required_prices - set(prices.columns)
)

if missing_prices:
    raise ValueError(
        f"Missing price columns: {sorted(missing_prices)}"
    )


# ============================================================================
# 3. RAW IMPLIED PROBABILITIES
# ============================================================================

prices["opening_raw_over"] = (
    1 / prices["opening_price_OVER"]
)

prices["opening_raw_under"] = (
    1 / prices["opening_price_UNDER"]
)

prices["closing_raw_over"] = (
    1 / prices["closing_price_OVER"]
)

prices["closing_raw_under"] = (
    1 / prices["closing_price_UNDER"]
)


# ============================================================================
# 4. OVERROUND
# ============================================================================

prices["opening_overround"] = (
    prices["opening_raw_over"]
    + prices["opening_raw_under"]
)

prices["closing_overround"] = (
    prices["closing_raw_over"]
    + prices["closing_raw_under"]
)


# ============================================================================
# 5. FAIR PROBABILITIES
# ============================================================================

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


# ============================================================================
# 6. OFFICIAL CANDIDATE CLV METRICS
# ============================================================================

# --------------------------------------------------------------------------
# Price CLV
# --------------------------------------------------------------------------

prices["price_clv_over"] = (
    prices["opening_price_OVER"]
    / prices["closing_price_OVER"]
    - 1
)

prices["price_clv_under"] = (
    prices["opening_price_UNDER"]
    / prices["closing_price_UNDER"]
    - 1
)


# --------------------------------------------------------------------------
# Vig-adjusted probability movement
# --------------------------------------------------------------------------

prices["fair_probability_clv_over"] = (
    prices["closing_fair_over"]
    / prices["opening_fair_over"]
    - 1
)

prices["fair_probability_clv_under"] = (
    prices["closing_fair_under"]
    / prices["opening_fair_under"]
    - 1
)


# ============================================================================
# 7. CONSTRUCT LONG DATASET
# ============================================================================

over = pd.DataFrame({
    "match_id": prices["match_id"],
    "season": prices["season"],
    "selection": "OVER",

    "opening_price": prices["opening_price_OVER"],
    "closing_price": prices["closing_price_OVER"],

    "opening_raw_probability":
        prices["opening_raw_over"],

    "closing_raw_probability":
        prices["closing_raw_over"],

    "opening_fair_probability":
        prices["opening_fair_over"],

    "closing_fair_probability":
        prices["closing_fair_over"],

    "opening_overround":
        prices["opening_overround"],

    "closing_overround":
        prices["closing_overround"],

    "opening_margin":
        prices["opening_overround"] - 1,

    "closing_margin":
        prices["closing_overround"] - 1,

    "price_clv":
        prices["price_clv_over"],

    "fair_probability_clv":
        prices["fair_probability_clv_over"],
})


under = pd.DataFrame({
    "match_id": prices["match_id"],
    "season": prices["season"],
    "selection": "UNDER",

    "opening_price": prices["opening_price_UNDER"],
    "closing_price": prices["closing_price_UNDER"],

    "opening_raw_probability":
        prices["opening_raw_under"],

    "closing_raw_probability":
        prices["closing_raw_under"],

    "opening_fair_probability":
        prices["opening_fair_under"],

    "closing_fair_probability":
        prices["closing_fair_under"],

    "opening_overround":
        prices["opening_overround"],

    "closing_overround":
        prices["closing_overround"],

    "opening_margin":
        prices["opening_overround"] - 1,

    "closing_margin":
        prices["closing_overround"] - 1,

    "price_clv":
        prices["price_clv_under"],

    "fair_probability_clv":
        prices["fair_probability_clv_under"],
})


audit = pd.concat(
    [over, under],
    ignore_index=True,
)


# ============================================================================
# 8. SIGN CONSISTENCY
# ============================================================================

print("\n" + "=" * 78)
print("1. CLV SIGN CONSISTENCY")
print("=" * 78)

same_sign = (
    np.sign(audit["price_clv"])
    == np.sign(audit["fair_probability_clv"])
)

print(
    "Same sign:",
    int(same_sign.sum()),
    "/",
    len(audit)
)

print(
    "Different sign:",
    int((~same_sign).sum()),
    "/",
    len(audit)
)


# ============================================================================
# 9. CORRELATION
# ============================================================================

print("\n" + "=" * 78)
print("2. CLV CORRELATION")
print("=" * 78)

correlation = audit[
    [
        "price_clv",
        "fair_probability_clv",
    ]
].corr()

print(
    correlation.to_string()
)


# ============================================================================
# 10. BASIC STATISTICS
# ============================================================================

print("\n" + "=" * 78)
print("3. CLV STATISTICS")
print("=" * 78)

print("\nPrice CLV:")
print(
    audit["price_clv"]
    .describe()
    .to_string()
)

print("\nFair probability CLV:")
print(
    audit["fair_probability_clv"]
    .describe()
    .to_string()
)


# ============================================================================
# 11. SELECTION SUMMARY
# ============================================================================

print("\n" + "=" * 78)
print("4. CLV BY SELECTION")
print("=" * 78)

selection_summary = (
    audit.groupby("selection")
    .agg(
        observations=("match_id", "size"),

        mean_price_clv=(
            "price_clv",
            "mean",
        ),

        median_price_clv=(
            "price_clv",
            "median",
        ),

        positive_price_clv_rate=(
            "price_clv",
            lambda x: (x > 0).mean(),
        ),

        mean_fair_probability_clv=(
            "fair_probability_clv",
            "mean",
        ),

        median_fair_probability_clv=(
            "fair_probability_clv",
            "median",
        ),

        positive_fair_probability_clv_rate=(
            "fair_probability_clv",
            lambda x: (x > 0).mean(),
        ),
    )
)

print(
    selection_summary.to_string()
)


# ============================================================================
# 12. SEASON SUMMARY
# ============================================================================

print("\n" + "=" * 78)
print("5. CLV BY SEASON")
print("=" * 78)

season_summary = (
    audit.groupby("season")
    .agg(
        observations=("match_id", "size"),
        matches=("match_id", "nunique"),

        mean_price_clv=(
            "price_clv",
            "mean",
        ),

        median_price_clv=(
            "price_clv",
            "median",
        ),

        positive_price_clv_rate=(
            "price_clv",
            lambda x: (x > 0).mean(),
        ),

        mean_fair_probability_clv=(
            "fair_probability_clv",
            "mean",
        ),

        median_fair_probability_clv=(
            "fair_probability_clv",
            "median",
        ),

        positive_fair_probability_clv_rate=(
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
# 13. EXAMPLE SANITY CHECKS
# ============================================================================

print("\n" + "=" * 78)
print("6. MANUAL SANITY CHECKS")
print("=" * 78)

examples = audit[
    [
        "match_id",
        "selection",
        "opening_price",
        "closing_price",
        "opening_fair_probability",
        "closing_fair_probability",
        "price_clv",
        "fair_probability_clv",
    ]
].copy()

examples["abs_price_clv"] = (
    examples["price_clv"].abs()
)

examples = (
    examples
    .sort_values(
        "abs_price_clv",
        ascending=False,
    )
    .head(10)
    .drop(columns=["abs_price_clv"])
)

print(
    examples.to_string(index=False)
)


# ============================================================================
# 14. EXTREME CLV
# ============================================================================

print("\n" + "=" * 78)
print("7. EXTREME CLV")
print("=" * 78)

extreme = (
    audit["price_clv"].abs() >= 0.25
)

print(
    "Observations with |price CLV| >= 25%:",
    int(extreme.sum())
)

if extreme.any():

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
            "price_clv"
        )
        .to_string(index=False)
    )


# ============================================================================
# 15. DATA QUALITY
# ============================================================================

print("\n" + "=" * 78)
print("8. DATA QUALITY CHECKS")
print("=" * 78)

fair_sum_open = (
    prices["opening_fair_over"]
    + prices["opening_fair_under"]
)

fair_sum_close = (
    prices["closing_fair_over"]
    + prices["closing_fair_under"]
)

open_error = (
    fair_sum_open - 1
).abs()

close_error = (
    fair_sum_close - 1
).abs()

print(
    "Maximum opening fair-probability error:",
    open_error.max()
)

print(
    "Maximum closing fair-probability error:",
    close_error.max()
)

print(
    "Missing CLV values:",
    int(
        audit[
            [
                "price_clv",
                "fair_probability_clv",
            ]
        ]
        .isna()
        .any(axis=1)
        .sum()
    )
)


# ============================================================================
# 16. SAVE
# ============================================================================

audit.to_csv(
    OUTPUT_FILE,
    index=False,
)


report = pd.DataFrame([
    {
        "version": "v0.22",
        "observations": len(audit),
        "unique_matches": audit["match_id"].nunique(),

        "same_sign_count":
            int(same_sign.sum()),

        "different_sign_count":
            int((~same_sign).sum()),

        "sign_consistency_rate":
            same_sign.mean(),

        "price_clv_mean":
            audit["price_clv"].mean(),

        "price_clv_median":
            audit["price_clv"].median(),

        "positive_price_clv_rate":
            (audit["price_clv"] > 0).mean(),

        "fair_probability_clv_mean":
            audit["fair_probability_clv"].mean(),

        "fair_probability_clv_median":
            audit["fair_probability_clv"].median(),

        "positive_fair_probability_clv_rate":
            (
                audit["fair_probability_clv"] > 0
            ).mean(),

        "clv_correlation":
            correlation.loc[
                "price_clv",
                "fair_probability_clv",
            ],

        "max_opening_probability_error":
            open_error.max(),

        "max_closing_probability_error":
            close_error.max(),

        "extreme_clv_count":
            int(extreme.sum()),
    }
])

report.to_csv(
    REPORT_FILE,
    index=False,
)


# ============================================================================
# 17. FINAL STATUS
# ============================================================================

print("\n" + "=" * 78)
print("9. OUTPUT")
print("=" * 78)

print("Saved audit dataset:")
print(OUTPUT_FILE)

print("\nSaved report:")
print(REPORT_FILE)

print("\nSaved season summary:")
print(SEASON_FILE)


if (
    same_sign.all()
    and open_error.max() < 1e-12
    and close_error.max() < 1e-12
    and not audit[
        [
            "price_clv",
            "fair_probability_clv",
        ]
    ].isna().any().any()
):

    print(
        "\nV0.22 STATUS: PASS — "
        "CLV CONVENTION VALIDATED"
    )

else:

    print(
        "\nV0.22 STATUS: REVIEW REQUIRED"
    )


print("=" * 78)
