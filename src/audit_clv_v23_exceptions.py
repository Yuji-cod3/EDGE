from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data/processed/clv_audit_v22.csv"
OUTPUT_FILE = ROOT / "data/processed/clv_v23_exceptions.csv"
REPORT_FILE = ROOT / "reports/clv_v23_exceptions_report.csv"

print("=" * 78)
print("EDGE — V0.23 CLV SIGN EXCEPTION AUDIT")
print("=" * 78)

df = pd.read_csv(INPUT_FILE)

required = {
    "match_id",
    "season",
    "selection",
    "opening_price",
    "closing_price",
    "opening_fair_probability",
    "closing_fair_probability",
    "price_clv",
    "fair_probability_clv",
    "opening_overround",
    "closing_overround",
}

missing = required - set(df.columns)

if missing:
    raise ValueError(
        f"Missing required columns: {sorted(missing)}"
    )


# ============================================================================
# 1. IDENTIFY SIGN EXCEPTIONS
# ============================================================================

df["price_sign"] = np.sign(df["price_clv"])
df["fair_sign"] = np.sign(df["fair_probability_clv"])

exceptions = df[
    (df["price_sign"] != df["fair_sign"])
    &
    (df["price_clv"] != 0)
    &
    (df["fair_probability_clv"] != 0)
].copy()

print("\nTOTAL OBSERVATIONS:", len(df))
print("SIGN EXCEPTIONS:", len(exceptions))


# ============================================================================
# 2. CALCULATE MOVEMENT COMPONENTS
# ============================================================================

exceptions["odds_ratio"] = (
    exceptions["closing_price"]
    / exceptions["opening_price"]
)

exceptions["fair_probability_ratio"] = (
    exceptions["closing_fair_probability"]
    / exceptions["opening_fair_probability"]
)

exceptions["margin_change"] = (
    exceptions["closing_overround"]
    - exceptions["opening_overround"]
)

exceptions["margin_change_abs"] = (
    exceptions["margin_change"].abs()
)

exceptions["price_clv_abs"] = (
    exceptions["price_clv"].abs()
)

exceptions["fair_clv_abs"] = (
    exceptions["fair_probability_clv"].abs()
)


# ============================================================================
# 3. CLASSIFY EXCEPTIONS
# ============================================================================

def classify(row):

    # Very small movements are potentially numerical / rounding effects.
    if (
        abs(row["price_clv"]) < 0.001
        and
        abs(row["fair_probability_clv"]) < 0.001
    ):
        return "TINY_MOVEMENT"

    # Large change in bookmaker margin can reverse the normalized
    # probability direction.
    if row["margin_change_abs"] >= 0.005:
        return "MARGIN_EFFECT"

    return "GENUINE_DIRECTION_DISAGREEMENT"


exceptions["exception_type"] = exceptions.apply(
    classify,
    axis=1,
)


# ============================================================================
# 4. SUMMARY
# ============================================================================

print("\n" + "=" * 78)
print("1. EXCEPTION CLASSIFICATION")
print("=" * 78)

print(
    exceptions["exception_type"]
    .value_counts()
    .to_string()
)


# ============================================================================
# 5. MAGNITUDE
# ============================================================================

print("\n" + "=" * 78)
print("2. EXCEPTION MAGNITUDES")
print("=" * 78)

print(
    exceptions[
        [
            "price_clv",
            "fair_probability_clv",
            "margin_change",
        ]
    ]
    .describe()
    .to_string()
)


# ============================================================================
# 6. EXTREME EXCEPTIONS
# ============================================================================

print("\n" + "=" * 78)
print("3. LARGEST EXCEPTIONS")
print("=" * 78)

largest = (
    exceptions.assign(
        discrepancy=(
            exceptions["price_clv"]
            - exceptions["fair_probability_clv"]
        ).abs()
    )
    .sort_values(
        "discrepancy",
        ascending=False,
    )
    .head(20)
)

print(
    largest[
        [
            "match_id",
            "season",
            "selection",
            "opening_price",
            "closing_price",
            "opening_fair_probability",
            "closing_fair_probability",
            "opening_overround",
            "closing_overround",
            "price_clv",
            "fair_probability_clv",
            "margin_change",
            "exception_type",
        ]
    ].to_string(index=False)
)


# ============================================================================
# 7. SELECTION
# ============================================================================

print("\n" + "=" * 78)
print("4. EXCEPTIONS BY SELECTION")
print("=" * 78)

selection_summary = (
    exceptions
    .groupby("selection")
    .agg(
        exceptions=("match_id", "size"),
        mean_price_clv=("price_clv", "mean"),
        mean_fair_clv=("fair_probability_clv", "mean"),
        mean_margin_change=("margin_change", "mean"),
    )
)

print(
    selection_summary.to_string()
)


# ============================================================================
# 8. SEASON
# ============================================================================

print("\n" + "=" * 78)
print("5. EXCEPTIONS BY SEASON")
print("=" * 78)

season_summary = (
    exceptions
    .groupby("season")
    .agg(
        exceptions=("match_id", "size"),
        mean_price_clv=("price_clv", "mean"),
        mean_fair_clv=("fair_probability_clv", "mean"),
        mean_margin_change=("margin_change", "mean"),
    )
)

print(
    season_summary.to_string()
)


# ============================================================================
# 9. EXACT MATHEMATICAL CHECK
# ============================================================================

print("\n" + "=" * 78)
print("6. MATHEMATICAL CHECK")
print("=" * 78)

# Price CLV:
#
# opening / closing - 1
#
# Probability movement:
#
# closing_fair / opening_fair - 1
#
# Their signs should normally agree, but vig normalization can
# alter the direction when the opposing side / margin changes.

exceptions["raw_probability_movement"] = (
    exceptions["closing_price"].rdiv(1)
    - exceptions["opening_price"].rdiv(1)
)

# Recalculate directly to verify no implementation error.

exceptions["recomputed_price_clv"] = (
    exceptions["opening_price"]
    / exceptions["closing_price"]
    - 1
)

exceptions["recomputed_fair_clv"] = (
    exceptions["closing_fair_probability"]
    / exceptions["opening_fair_probability"]
    - 1
)

price_error = (
    exceptions["recomputed_price_clv"]
    - exceptions["price_clv"]
).abs().max()

fair_error = (
    exceptions["recomputed_fair_clv"]
    - exceptions["fair_probability_clv"]
).abs().max()

print(
    "Maximum price CLV recalculation error:",
    price_error
)

print(
    "Maximum fair CLV recalculation error:",
    fair_error
)


# ============================================================================
# 10. SAVE
# ============================================================================

exceptions.to_csv(
    OUTPUT_FILE,
    index=False,
)

report = pd.DataFrame([
    {
        "version": "v0.23",
        "total_observations": len(df),
        "sign_exceptions": len(exceptions),
        "exception_rate": len(exceptions) / len(df),

        "tiny_movements": int(
            (
                exceptions["exception_type"]
                == "TINY_MOVEMENT"
            ).sum()
        ),

        "margin_effects": int(
            (
                exceptions["exception_type"]
                == "MARGIN_EFFECT"
            ).sum()
        ),

        "genuine_direction_disagreements": int(
            (
                exceptions["exception_type"]
                == "GENUINE_DIRECTION_DISAGREEMENT"
            ).sum()
        ),

        "max_price_recalculation_error":
            price_error,

        "max_fair_recalculation_error":
            fair_error,
    }
])

report.to_csv(
    REPORT_FILE,
    index=False,
)


# ============================================================================
# 11. FINAL STATUS
# ============================================================================

print("\n" + "=" * 78)
print("7. OUTPUT")
print("=" * 78)

print("Saved exceptions:")
print(OUTPUT_FILE)

print("\nSaved report:")
print(REPORT_FILE)

print("\nV0.23 STATUS: COMPLETE — REVIEW EXCEPTIONS")
print("=" * 78)
