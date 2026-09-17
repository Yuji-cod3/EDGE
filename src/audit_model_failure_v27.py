#!/usr/bin/env python3

"""
EDGE — V0.27 MODEL DIAGNOSTIC & DIRECTIONAL FAILURE AUDIT

Purpose:
    Diagnose why the current Poisson model underperforms the market.

Inputs:
    data/processed/model_clv_join_v25.csv
    data/processed/matches.csv
    data/processed/clv_audit_v20.csv

Outputs:
    data/processed/model_failure_v27.csv
    reports/model_failure_v27_report.csv

Important:
    This script is diagnostic only.
    It does NOT modify the model or training data.
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_CLV_FILE = ROOT / "data/processed/model_clv_join_v25.csv"
MATCHES_FILE = ROOT / "data/processed/matches.csv"
CLV_FILE = ROOT / "data/processed/clv_audit_v20.csv"

OUTPUT_DATA = ROOT / "data/processed/model_failure_v27.csv"
OUTPUT_REPORT = ROOT / "reports/model_failure_v27_report.csv"

OUTPUT_DATA.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)


# ============================================================================
# HELPERS
# ============================================================================

def print_section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def safe_brier(probability, target):
    return np.mean((probability - target) ** 2)


def safe_log_loss(probability, target):
    p = np.clip(np.asarray(probability, dtype=float), 1e-15, 1 - 1e-15)
    y = np.asarray(target, dtype=float)

    return -np.mean(
        y * np.log(p) +
        (1 - y) * np.log(1 - p)
    )


def edge_bucket(value):
    if value < -0.10:
        return "< -10%"
    elif value < -0.05:
        return "-10 to -5%"
    elif value < -0.025:
        return "-5 to -2.5%"
    elif value < -0.01:
        return "-2.5 to -1%"
    elif value < 0:
        return "-1 to 0%"
    elif value < 0.01:
        return "0 to 1%"
    elif value < 0.025:
        return "1 to 2.5%"
    elif value < 0.05:
        return "2.5 to 5%"
    elif value < 0.10:
        return "5 to 10%"
    else:
        return ">10%"


def probability_bucket(value):
    if value < 0.30:
        return "20-30%"
    elif value < 0.35:
        return "30-35%"
    elif value < 0.40:
        return "35-40%"
    elif value < 0.45:
        return "40-45%"
    elif value < 0.50:
        return "45-50%"
    elif value < 0.55:
        return "50-55%"
    elif value < 0.60:
        return "55-60%"
    elif value < 0.65:
        return "60-65%"
    elif value < 0.70:
        return "65-70%"
    else:
        return "70%+"


# ============================================================================
# LOAD DATA
# ============================================================================

print_section("EDGE — V0.27 MODEL DIAGNOSTIC & DIRECTIONAL FAILURE AUDIT")

df = pd.read_csv(MODEL_CLV_FILE)
matches = pd.read_csv(MATCHES_FILE)
clv = pd.read_csv(CLV_FILE)

print("\nDATASETS")
print(f"Model/CLV observations: {len(df)}")
print(f"Matches bridge: {len(matches)}")
print(f"CLV observations: {len(clv)}")


# ============================================================================
# NORMALIZE TYPES
# ============================================================================

numeric_columns = [
    "model_probability",
    "opening_fair_probability",
    "closing_fair_probability",
    "fair_probability_clv",
    "model_vs_closing_fair_probability",
    "target_over25",
]

for column in numeric_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")


# The target used for selection-level evaluation.
df["target"] = df["target_over25"]


# ============================================================================
# 1. PROBABILITY VALIDATION
# ============================================================================

print_section("1. PROBABILITY VALIDATION")

invalid_model = (
    df["model_probability"].isna() |
    (df["model_probability"] < 0) |
    (df["model_probability"] > 1)
).sum()

invalid_opening = (
    df["opening_fair_probability"].isna() |
    (df["opening_fair_probability"] < 0) |
    (df["opening_fair_probability"] > 1)
).sum()

invalid_closing = (
    df["closing_fair_probability"].isna() |
    (df["closing_fair_probability"] < 0) |
    (df["closing_fair_probability"] > 1)
).sum()

invalid_target = (
    df["target"].isna() |
    ~df["target"].isin([0, 1])
).sum()

print(f"Invalid model_selection_probability: {invalid_model}")
print(f"Invalid opening_fair_probability: {invalid_opening}")
print(f"Invalid closing_fair_probability: {invalid_closing}")
print(f"Invalid selection targets: {invalid_target}")


# ============================================================================
# 2. EXACT-SAMPLE PERFORMANCE
# ============================================================================

print_section("2. EXACT-SAMPLE PERFORMANCE")

valid = df.dropna(
    subset=[
        "model_probability",
        "opening_fair_probability",
        "closing_fair_probability",
        "target",
    ]
).copy()

print(f"Valid observations: {len(valid)}")

model_brier = safe_brier(
    valid["model_probability"],
    valid["target"]
)

opening_brier = safe_brier(
    valid["opening_fair_probability"],
    valid["target"]
)

closing_brier = safe_brier(
    valid["closing_fair_probability"],
    valid["target"]
)

model_logloss = safe_log_loss(
    valid["model_probability"],
    valid["target"]
)

opening_logloss = safe_log_loss(
    valid["opening_fair_probability"],
    valid["target"]
)

closing_logloss = safe_log_loss(
    valid["closing_fair_probability"],
    valid["target"]
)

print(f"Model Brier: {model_brier:.6f}")
print(f"Opening Brier: {opening_brier:.6f}")
print(f"Closing Brier: {closing_brier:.6f}")

print(f"Model Log Loss: {model_logloss:.6f}")
print(f"Opening Log Loss: {opening_logloss:.6f}")
print(f"Closing Log Loss: {closing_logloss:.6f}")


# ============================================================================
# 3. CALIBRATION BY MODEL PROBABILITY
# ============================================================================

print_section("3. CALIBRATION BY MODEL PROBABILITY")

valid["probability_bin"] = valid["model_probability"].apply(
    probability_bucket
)

calibration_rows = []

for bucket, group in valid.groupby(
    "probability_bin",
    sort=False
):
    predicted = group["model_probability"].mean()
    actual = group["target"].mean()

    calibration_rows.append({
        "probability_bin": bucket,
        "observations": len(group),
        "predicted": predicted,
        "actual": actual,
        "error": predicted - actual,
    })

calibration = pd.DataFrame(calibration_rows)

print(
    calibration.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================================
# 4. EDGE DIRECTION TEST
# ============================================================================

print_section("4. EDGE DIRECTION TEST")

valid["model_edge"] = (
    valid["model_probability"] -
    valid["opening_fair_probability"]
)

valid["model_error"] = (
    valid["model_probability"] -
    valid["target"]
)

valid["edge_direction"] = np.where(
    valid["model_edge"] > 0,
    "MODEL_HIGHER",
    "MODEL_LOWER"
)

direction_rows = []

for direction, group in valid.groupby("edge_direction"):

    direction_rows.append({
        "edge_direction": direction,
        "observations": len(group),
        "mean_model_probability": group["model_probability"].mean(),
        "mean_opening_probability": group[
            "opening_fair_probability"
        ].mean(),
        "actual_rate": group["target"].mean(),
        "mean_model_error": group["model_error"].mean(),
        "mean_clv": group["fair_probability_clv"].mean(),
        "model_calibration_error": (
            group["model_probability"].mean()
            -
            group["target"].mean()
        ),
    })

direction_summary = pd.DataFrame(direction_rows)

print(
    direction_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================================
# 5. EDGE MAGNITUDE VS MODEL ERROR
# ============================================================================

print_section("5. EDGE MAGNITUDE VS MODEL ERROR")

valid["abs_edge"] = valid["model_edge"].abs()
valid["abs_model_error"] = valid["model_error"].abs()

if valid["abs_edge"].nunique() > 1:
    edge_error_corr = valid[
        ["abs_edge", "abs_model_error"]
    ].corr().iloc[0, 1]
else:
    edge_error_corr = np.nan

print(
    "Correlation |model edge| → |model outcome error|: "
    f"{edge_error_corr:.6f}"
)

valid["edge_bucket"] = valid["model_edge"].apply(edge_bucket)

bucket_rows = []

bucket_order = [
    "< -10%",
    "-10 to -5%",
    "-5 to -2.5%",
    "-2.5 to -1%",
    "-1 to 0%",
    "0 to 1%",
    "1 to 2.5%",
    "2.5 to 5%",
    "5 to 10%",
    ">10%",
]

for bucket in bucket_order:

    group = valid[
        valid["edge_bucket"] == bucket
    ]

    if len(group) == 0:
        continue

    model_error = (
        group["model_probability"].mean()
        -
        group["target"].mean()
    )

    market_error = (
        group["opening_fair_probability"].mean()
        -
        group["target"].mean()
    )

    bucket_rows.append({
        "edge_bucket": bucket,
        "observations": len(group),
        "model_probability": group["model_probability"].mean(),
        "opening_probability": group[
            "opening_fair_probability"
        ].mean(),
        "closing_probability": group[
            "closing_fair_probability"
        ].mean(),
        "actual_rate": group["target"].mean(),
        "mean_clv": group["fair_probability_clv"].mean(),
        "model_error": model_error,
        "market_error": market_error,
        "model_abs_error": abs(model_error),
        "market_abs_error": abs(market_error),
    })

edge_buckets = pd.DataFrame(bucket_rows)

print(
    edge_buckets.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================================
# 6. OVER / UNDER DIAGNOSTIC
# ============================================================================

print_section("6. OVER / UNDER DIAGNOSTIC")

selection_rows = []

for selection, group in valid.groupby("selection"):

    model_brier_selection = safe_brier(
        group["model_probability"],
        group["target"]
    )

    market_brier_selection = safe_brier(
        group["opening_fair_probability"],
        group["target"]
    )

    model_calibration_error = (
        group["model_probability"].mean()
        -
        group["target"].mean()
    )

    market_calibration_error = (
        group["opening_fair_probability"].mean()
        -
        group["target"].mean()
    )

    selection_rows.append({
        "selection": selection,
        "observations": len(group),
        "model_probability": group["model_probability"].mean(),
        "market_probability": group[
            "opening_fair_probability"
        ].mean(),
        "actual_rate": group["target"].mean(),
        "model_brier": model_brier_selection,
        "market_brier": market_brier_selection,
        "mean_clv": group["fair_probability_clv"].mean(),
        "model_calibration_error": model_calibration_error,
        "market_calibration_error": market_calibration_error,
    })

selection_summary = pd.DataFrame(selection_rows)

print(
    selection_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================================
# 7. SEASON DIAGNOSTIC
# ============================================================================

print_section("7. SEASON DIAGNOSTIC")

season_rows = []

for season, group in valid.groupby("season"):

    model_brier_season = safe_brier(
        group["model_probability"],
        group["target"]
    )

    market_brier_season = safe_brier(
        group["opening_fair_probability"],
        group["target"]
    )

    model_logloss_season = safe_log_loss(
        group["model_probability"],
        group["target"]
    )

    market_logloss_season = safe_log_loss(
        group["opening_fair_probability"],
        group["target"]
    )

    season_rows.append({
        "season": season,
        "observations": len(group),
        "model_brier": model_brier_season,
        "market_brier": market_brier_season,
        "model_logloss": model_logloss_season,
        "market_logloss": market_logloss_season,
        "brier_difference": (
            model_brier_season -
            market_brier_season
        ),
        "logloss_difference": (
            model_logloss_season -
            market_logloss_season
        ),
        "mean_edge": group["model_edge"].mean(),
        "mean_clv": group["fair_probability_clv"].mean(),
    })

season_summary = pd.DataFrame(season_rows)

print(
    season_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================================
# 8. CLV-ONLY EVENT AUDIT
# ============================================================================

print_section("8. CLV-ONLY EVENT AUDIT")

# V0.25 model/CLV join already contains the validated canonical IDs
model_ids = set(
    valid["canonical_id"]
    .dropna()
    .unique()
)

# V0.20 contains the validated canonical IDs.
clv_ids = set(
    clv["canonical_id"]
    .dropna()
    .unique()
)

clv_only_ids = sorted(clv_ids - model_ids)
model_only_ids = sorted(model_ids - clv_ids)

print(f"CLV unique matches: {len(clv_ids)}")
print(f"Model unique matches: {len(model_ids)}")
print(f"CLV-only matches: {len(clv_only_ids)}")
print(f"Model-only matches: {len(model_only_ids)}")

if clv_only_ids:

    clv_only = (
        clv[
            clv["canonical_id"].isin(clv_only_ids)
        ]
        [
            [
                "canonical_id",
                "season",
                "date",
                "home_team",
                "away_team",
                "selection",
                "opening_price",
                "closing_price",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "date",
                "canonical_id",
                "selection",
            ]
        )
    )

    print("\nCLV-only events:")
    print(
        clv_only.to_string(
            index=False
        )
    )

else:

    clv_only = pd.DataFrame()

    print("No CLV-only events found.")


# ============================================================================
# 9. MODEL PROBABILITY DISTRIBUTION
# ============================================================================

print_section("9. MODEL PROBABILITY DISTRIBUTION")

distribution = {
    "minimum": valid["model_probability"].min(),
    "maximum": valid["model_probability"].max(),
    "mean": valid["model_probability"].mean(),
    "median": valid["model_probability"].median(),
    "std": valid["model_probability"].std(),
    "p05": valid["model_probability"].quantile(0.05),
    "p25": valid["model_probability"].quantile(0.25),
    "p75": valid["model_probability"].quantile(0.75),
    "p95": valid["model_probability"].quantile(0.95),
}

for key, value in distribution.items():
    print(f"{key:>8}: {value:.6f}")


# ============================================================================
# 10. MODEL LAMBDA DIAGNOSTIC
# ============================================================================

print_section("10. MODEL LAMBDA DIAGNOSTIC")

if "model_lambda" in valid.columns:

    lambda_values = pd.to_numeric(
        valid["model_lambda"],
        errors="coerce"
    )

    print(
        f"Lambda minimum: {lambda_values.min():.6f}"
    )

    print(
        f"Lambda maximum: {lambda_values.max():.6f}"
    )

    print(
        f"Lambda mean:    {lambda_values.mean():.6f}"
    )

    print(
        f"Lambda median:  {lambda_values.median():.6f}"
    )

    print(
        f"Lambda std:     {lambda_values.std():.6f}"
    )

else:

    print("model_lambda column not available.")


# ============================================================================
# 11. PROBABILITY CONSISTENCY CHECK
# ============================================================================

print_section("11. PROBABILITY CONSISTENCY CHECK")

if "model_under_probability" in valid.columns:

    valid["model_under_probability"] = pd.to_numeric(
        valid["model_under_probability"],
        errors="coerce"
    )

    probability_sum_error = (
        valid["model_probability"] +
        valid["model_under_probability"] -
        1.0
    ).abs()

    print(
        "Maximum model OVER + UNDER probability error: "
        f"{probability_sum_error.max():.12f}"
    )

else:

    print(
        "model_under_probability column not available."
    )


# ============================================================================
# 12. MODEL VS MARKET CORRELATION
# ============================================================================

print_section("12. MODEL VS MARKET CORRELATION")

model_market_corr = valid[
    [
        "model_probability",
        "opening_fair_probability",
    ]
].corr().iloc[0, 1]

model_closing_corr = valid[
    [
        "model_probability",
        "closing_fair_probability",
    ]
].corr().iloc[0, 1]

print(
    f"Model vs opening probability correlation: "
    f"{model_market_corr:.6f}"
)

print(
    f"Model vs closing probability correlation: "
    f"{model_closing_corr:.6f}"
)


# ============================================================================
# 13. OVERALL DIAGNOSTIC SUMMARY
# ============================================================================

print_section("13. OVERALL DIAGNOSTIC SUMMARY")

print(
    f"Model Brier:       {model_brier:.6f}"
)

print(
    f"Opening Brier:     {opening_brier:.6f}"
)

print(
    f"Closing Brier:     {closing_brier:.6f}"
)

print(
    f"Model Log Loss:    {model_logloss:.6f}"
)

print(
    f"Opening Log Loss:  {opening_logloss:.6f}"
)

print(
    f"Closing Log Loss:  {closing_logloss:.6f}"
)

print(
    f"Brier difference:  "
    f"{model_brier - opening_brier:+.6f}"
)

print(
    f"Log-loss difference:"
    f" {model_logloss - opening_logloss:+.6f}"
)

print(
    f"Edge/error correlation:"
    f" {edge_error_corr:.6f}"
)

print(
    f"CLV-only matches: {len(clv_only_ids)}"
)


# ============================================================================
# 14. SAVE DIAGNOSTIC DATA
# ============================================================================

print_section("14. SAVING OUTPUTS")

valid.to_csv(
    OUTPUT_DATA,
    index=False
)

# Combine report sections into a single long-form CSV.

report_rows = []

for _, row in calibration.iterrows():

    report_rows.append({
        "section": "calibration",
        "metric": "probability_bin",
        "group": row["probability_bin"],
        "value": row["actual"],
        "observations": row["observations"],
        "secondary_value": row["predicted"],
        "tertiary_value": row["error"],
    })


for _, row in direction_summary.iterrows():

    report_rows.append({
        "section": "edge_direction",
        "metric": "actual_rate",
        "group": row["edge_direction"],
        "value": row["actual_rate"],
        "observations": row["observations"],
        "secondary_value": row["mean_model_probability"],
        "tertiary_value": row["mean_opening_probability"],
    })


for _, row in edge_buckets.iterrows():

    report_rows.append({
        "section": "edge_bucket",
        "metric": "actual_rate",
        "group": row["edge_bucket"],
        "value": row["actual_rate"],
        "observations": row["observations"],
        "secondary_value": row["model_probability"],
        "tertiary_value": row["opening_probability"],
    })


for _, row in selection_summary.iterrows():

    report_rows.append({
        "section": "selection",
        "metric": "actual_rate",
        "group": row["selection"],
        "value": row["actual_rate"],
        "observations": row["observations"],
        "secondary_value": row["model_probability"],
        "tertiary_value": row["market_probability"],
    })


for _, row in season_summary.iterrows():

    report_rows.append({
        "section": "season",
        "metric": "brier_difference",
        "group": row["season"],
        "value": row["brier_difference"],
        "observations": row["observations"],
        "secondary_value": row["model_brier"],
        "tertiary_value": row["market_brier"],
    })


summary_rows = [
    {
        "section": "overall",
        "metric": "model_brier",
        "group": "ALL",
        "value": model_brier,
        "observations": len(valid),
        "secondary_value": opening_brier,
        "tertiary_value": closing_brier,
    },
    {
        "section": "overall",
        "metric": "model_log_loss",
        "group": "ALL",
        "value": model_logloss,
        "observations": len(valid),
        "secondary_value": opening_logloss,
        "tertiary_value": closing_logloss,
    },
    {
        "section": "overall",
        "metric": "edge_error_correlation",
        "group": "ALL",
        "value": edge_error_corr,
        "observations": len(valid),
        "secondary_value": np.nan,
        "tertiary_value": np.nan,
    },
    {
        "section": "overall",
        "metric": "model_market_correlation",
        "group": "ALL",
        "value": model_market_corr,
        "observations": len(valid),
        "secondary_value": model_closing_corr,
        "tertiary_value": np.nan,
    },
    {
        "section": "coverage",
        "metric": "clv_only_matches",
        "group": "ALL",
        "value": len(clv_only_ids),
        "observations": len(clv_ids),
        "secondary_value": len(model_ids),
        "tertiary_value": np.nan,
    },
]

report_rows.extend(summary_rows)

report = pd.DataFrame(report_rows)

report.to_csv(
    OUTPUT_REPORT,
    index=False
)


# ============================================================================
# FINAL STATUS
# ============================================================================

print()
print("=" * 78)
print("V0.27 OUTPUTS")
print("=" * 78)

print(f"Diagnostic dataset : {OUTPUT_DATA}")
print(f"Diagnostic report  : {OUTPUT_REPORT}")

print()
print("STATUS: V0.27 COMPLETE")
print("=" * 78)
