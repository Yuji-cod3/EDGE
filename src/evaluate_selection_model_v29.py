#!/usr/bin/env python3

"""
EDGE — V0.29 CORRECTED SELECTION-LEVEL MODEL EVALUATION

Purpose:
    Correct the V0.27 evaluation-target problem.

IMPORTANT:
    model_probability is selection-specific.

    OVER:
        model_probability = P(OVER 2.5)
        target = target_over25

    UNDER:
        model_probability = P(UNDER 2.5)
        target = 1 - target_over25

This script evaluates the existing model.
It does NOT retrain or modify the model.
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================================
# CONFIG
# ============================================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data/processed/model_clv_join_v25.csv"

OUTPUT_DATA = (
    ROOT / "data/processed/model_selection_evaluation_v29.csv"
)

OUTPUT_REPORT = (
    ROOT / "reports/model_selection_evaluation_v29_report.csv"
)

OUTPUT_DATA.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)


# ============================================================================
# HELPERS
# ============================================================================

def brier(prob, target):
    return np.mean((prob - target) ** 2)


def log_loss(prob, target):
    p = np.clip(
        np.asarray(prob, dtype=float),
        1e-15,
        1 - 1e-15
    )

    y = np.asarray(target, dtype=float)

    return -np.mean(
        y * np.log(p) +
        (1 - y) * np.log(1 - p)
    )


def edge_bucket(x):

    if x < -0.10:
        return "< -10%"
    elif x < -0.05:
        return "-10 to -5%"
    elif x < -0.025:
        return "-5 to -2.5%"
    elif x < -0.01:
        return "-2.5 to -1%"
    elif x < 0:
        return "-1 to 0%"
    elif x < 0.01:
        return "0 to 1%"
    elif x < 0.025:
        return "1 to 2.5%"
    elif x < 0.05:
        return "2.5 to 5%"
    elif x < 0.10:
        return "5 to 10%"
    else:
        return ">10%"


# ============================================================================
# LOAD
# ============================================================================

print("=" * 78)
print("EDGE — V0.29 CORRECTED SELECTION-LEVEL MODEL EVALUATION")
print("=" * 78)

df = pd.read_csv(INPUT)

print()
print(f"Rows loaded: {len(df)}")


# ============================================================================
# VALIDATION
# ============================================================================

required = [
    "selection",
    "model_probability",
    "opening_fair_probability",
    "closing_fair_probability",
    "target_over25",
    "fair_probability_clv",
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:

    print("\nMISSING COLUMNS:")
    for c in missing:
        print(f"  - {c}")

    raise SystemExit(
        "V0.29 STOPPED."
    )


# ============================================================================
# NUMERIC
# ============================================================================

for c in [
    "model_probability",
    "opening_fair_probability",
    "closing_fair_probability",
    "target_over25",
    "fair_probability_clv",
]:

    df[c] = pd.to_numeric(
        df[c],
        errors="coerce"
    )


# ============================================================================
# CORRECT TARGET
# ============================================================================

print()
print("=" * 78)
print("1. CORRECT SELECTION TARGET")
print("=" * 78)

df["selection"] = (
    df["selection"]
    .astype(str)
    .str.upper()
)

df["selection_target"] = np.where(
    df["selection"] == "OVER",
    df["target_over25"],
    np.where(
        df["selection"] == "UNDER",
        1 - df["target_over25"],
        np.nan
    )
)

print(
    df["selection"]
    .value_counts()
    .to_string()
)

print()

print(
    "OVER target definition:"
)

print(
    "    target = target_over25"
)

print(
    "UNDER target definition:"
)

print(
    "    target = 1 - target_over25"
)

print()

print(
    f"Missing selection targets: "
    f"{df['selection_target'].isna().sum()}"
)

print(
    f"OVER actual rate: "
    f"{df.loc[df.selection == 'OVER', 'selection_target'].mean():.6f}"
)

print(
    f"UNDER actual rate: "
    f"{df.loc[df.selection == 'UNDER', 'selection_target'].mean():.6f}"
)


# ============================================================================
# VALID DATA
# ============================================================================

valid = df.dropna(
    subset=[
        "model_probability",
        "opening_fair_probability",
        "closing_fair_probability",
        "selection_target",
    ]
).copy()


# ============================================================================
# 2. EXACT SAMPLE PERFORMANCE
# ============================================================================

print()
print("=" * 78)
print("2. CORRECTED EXACT-SAMPLE PERFORMANCE")
print("=" * 78)

model_brier = brier(
    valid["model_probability"],
    valid["selection_target"]
)

opening_brier = brier(
    valid["opening_fair_probability"],
    valid["selection_target"]
)

closing_brier = brier(
    valid["closing_fair_probability"],
    valid["selection_target"]
)

model_ll = log_loss(
    valid["model_probability"],
    valid["selection_target"]
)

opening_ll = log_loss(
    valid["opening_fair_probability"],
    valid["selection_target"]
)

closing_ll = log_loss(
    valid["closing_fair_probability"],
    valid["selection_target"]
)

print(
    f"Model Brier:    {model_brier:.6f}"
)

print(
    f"Opening Brier:  {opening_brier:.6f}"
)

print(
    f"Closing Brier:  {closing_brier:.6f}"
)

print()

print(
    f"Model Log Loss:   {model_ll:.6f}"
)

print(
    f"Opening Log Loss: {opening_ll:.6f}"
)

print(
    f"Closing Log Loss: {closing_ll:.6f}"
)

print()

print(
    f"Model - Opening Brier: "
    f"{model_brier - opening_brier:+.6f}"
)

print(
    f"Model - Opening LogLoss: "
    f"{model_ll - opening_ll:+.6f}"
)


# ============================================================================
# 3. OVER / UNDER
# ============================================================================

print()
print("=" * 78)
print("3. OVER / UNDER PERFORMANCE")
print("=" * 78)

selection_rows = []

for selection, group in valid.groupby("selection"):

    model_b = brier(
        group["model_probability"],
        group["selection_target"]
    )

    market_b = brier(
        group["opening_fair_probability"],
        group["selection_target"]
    )

    model_l = log_loss(
        group["model_probability"],
        group["selection_target"]
    )

    market_l = log_loss(
        group["opening_fair_probability"],
        group["selection_target"]
    )

    selection_rows.append({

        "selection": selection,

        "observations": len(group),

        "model_probability":
            group["model_probability"].mean(),

        "market_probability":
            group["opening_fair_probability"].mean(),

        "actual_rate":
            group["selection_target"].mean(),

        "model_brier":
            model_b,

        "market_brier":
            market_b,

        "model_log_loss":
            model_l,

        "market_log_loss":
            market_l,

        "brier_difference":
            model_b - market_b,

        "log_loss_difference":
            model_l - market_l,

        "mean_clv":
            group["fair_probability_clv"].mean(),

    })


selection_summary = pd.DataFrame(
    selection_rows
)

print(
    selection_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================================
# 4. CALIBRATION
# ============================================================================

print()
print("=" * 78)
print("4. MODEL CALIBRATION")
print("=" * 78)

bins = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
]

labels = [
    "30-35%",
    "35-40%",
    "40-45%",
    "45-50%",
    "50-55%",
    "55-60%",
    "60-65%",
    "65-70%",
]

valid["probability_bin"] = pd.cut(
    valid["model_probability"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

calibration_rows = []

for bucket, group in valid.groupby(
    "probability_bin",
    observed=True
):

    predicted = group["model_probability"].mean()

    actual = group["selection_target"].mean()

    calibration_rows.append({

        "probability_bin": str(bucket),

        "observations": len(group),

        "predicted": predicted,

        "actual": actual,

        "calibration_error":
            predicted - actual,

    })

calibration = pd.DataFrame(
    calibration_rows
)

print(
    calibration.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================================
# 5. EDGE DIRECTION
# ============================================================================

print()
print("=" * 78)
print("5. EDGE DIRECTION")
print("=" * 78)

valid["model_edge"] = (
    valid["model_probability"] -
    valid["opening_fair_probability"]
)

valid["edge_direction"] = np.where(
    valid["model_edge"] > 0,
    "MODEL_HIGHER",
    "MODEL_LOWER"
)

valid["model_error"] = (
    valid["model_probability"] -
    valid["selection_target"]
)

direction_rows = []

for direction, group in valid.groupby(
    "edge_direction"
):

    direction_rows.append({

        "edge_direction": direction,

        "observations": len(group),

        "mean_model_probability":
            group["model_probability"].mean(),

        "mean_market_probability":
            group["opening_fair_probability"].mean(),

        "actual_rate":
            group["selection_target"].mean(),

        "mean_model_error":
            group["model_error"].mean(),

        "mean_clv":
            group["fair_probability_clv"].mean(),

        "brier":
            brier(
                group["model_probability"],
                group["selection_target"]
            ),

        "market_brier":
            brier(
                group["opening_fair_probability"],
                group["selection_target"]
            ),

    })

direction_summary = pd.DataFrame(
    direction_rows
)

print(
    direction_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================================
# 6. EDGE BUCKETS
# ============================================================================

print()
print("=" * 78)
print("6. EDGE BUCKET PERFORMANCE")
print("=" * 78)

valid["edge_bucket"] = (
    valid["model_edge"]
    .apply(edge_bucket)
)

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

bucket_rows = []

for bucket in bucket_order:

    group = valid[
        valid["edge_bucket"] == bucket
    ]

    if len(group) == 0:
        continue

    model_error = (
        group["model_probability"].mean()
        -
        group["selection_target"].mean()
    )

    market_error = (
        group["opening_fair_probability"].mean()
        -
        group["selection_target"].mean()
    )

    bucket_rows.append({

        "edge_bucket": bucket,

        "observations": len(group),

        "model_probability":
            group["model_probability"].mean(),

        "market_probability":
            group["opening_fair_probability"].mean(),

        "actual_rate":
            group["selection_target"].mean(),

        "model_error":
            model_error,

        "market_error":
            market_error,

        "model_abs_error":
            abs(model_error),

        "market_abs_error":
            abs(market_error),

        "mean_clv":
            group["fair_probability_clv"].mean(),

        "positive_clv_rate":
            (
                group["fair_probability_clv"] > 0
            ).mean(),

        "brier":
            brier(
                group["model_probability"],
                group["selection_target"]
            ),

    })

edge_summary = pd.DataFrame(
    bucket_rows
)

print(
    edge_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================================
# 7. EDGE → CLV
# ============================================================================

print()
print("=" * 78)
print("7. MODEL EDGE → CLV RELATIONSHIP")
print("=" * 78)

edge_clv_corr = valid[
    ["model_edge", "fair_probability_clv"]
].corr().iloc[0, 1]

abs_edge_clv_corr = valid[
    ["model_edge", "fair_probability_clv"]
].assign(
    abs_edge=lambda x: x["model_edge"].abs()
)[
    ["abs_edge", "fair_probability_clv"]
].corr().iloc[0, 1]

print(
    f"Model edge → fair CLV correlation: "
    f"{edge_clv_corr:.6f}"
)

print(
    f"|Model edge| → fair CLV correlation: "
    f"{abs_edge_clv_corr:.6f}"
)

print(
    f"Mean fair CLV: "
    f"{valid['fair_probability_clv'].mean():.6f}"
)

print(
    f"Median fair CLV: "
    f"{valid['fair_probability_clv'].median():.6f}"
)

print(
    f"Positive CLV rate: "
    f"{(valid['fair_probability_clv'] > 0).mean():.6f}"
)


# ============================================================================
# 8. SEASON
# ============================================================================

print()
print("=" * 78)
print("8. SEASON PERFORMANCE")
print("=" * 78)

season_rows = []

for season, group in valid.groupby("season"):

    model_b = brier(
        group["model_probability"],
        group["selection_target"]
    )

    market_b = brier(
        group["opening_fair_probability"],
        group["selection_target"]
    )

    model_l = log_loss(
        group["model_probability"],
        group["selection_target"]
    )

    market_l = log_loss(
        group["opening_fair_probability"],
        group["selection_target"]
    )

    season_rows.append({

        "season": season,

        "observations": len(group),

        "model_brier": model_b,

        "market_brier": market_b,

        "brier_difference":
            model_b - market_b,

        "model_log_loss": model_l,

        "market_log_loss": market_l,

        "log_loss_difference":
            model_l - market_l,

        "mean_clv":
            group["fair_probability_clv"].mean(),

    })

season_summary = pd.DataFrame(
    season_rows
)

print(
    season_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================================
# 9. SANITY CHECK
# ============================================================================

print()
print("=" * 78)
print("9. SANITY CHECK")
print("=" * 78)

# Each match should have exactly one OVER and one UNDER row.

counts = (
    valid
    .groupby("match_id")["selection"]
    .nunique()
)

print(
    f"Matches evaluated: "
    f"{len(counts)}"
)

print(
    f"Matches with exactly 2 selections: "
    f"{(counts == 2).sum()}"
)

print(
    f"Matches with selection anomaly: "
    f"{(counts != 2).sum()}"
)

# Probability complement check.

pair_check = (
    valid
    .pivot_table(
        index="match_id",
        columns="selection",
        values="model_probability",
        aggfunc="first"
    )
)

if (
    "OVER" in pair_check.columns
    and
    "UNDER" in pair_check.columns
):

    pair_check["sum"] = (
        pair_check["OVER"] +
        pair_check["UNDER"]
    )

    pair_check["sum_error"] = (
        pair_check["sum"] - 1
    ).abs()

    print(
        "Maximum model probability complement error: "
        f"{pair_check['sum_error'].max():.12f}"
    )


# ============================================================================
# 10. FINAL ASSESSMENT
# ============================================================================

print()
print("=" * 78)
print("10. V0.29 FINAL ASSESSMENT")
print("=" * 78)

if model_brier < opening_brier:
    brier_result = "MODEL BETTER"
else:
    brier_result = "MARKET BETTER"

if model_ll < opening_ll:
    logloss_result = "MODEL BETTER"
else:
    logloss_result = "MARKET BETTER"

print(
    f"Brier assessment: {brier_result}"
)

print(
    f"Log-loss assessment: {logloss_result}"
)

print(
    f"Model Brier advantage/disadvantage: "
    f"{opening_brier - model_brier:+.6f}"
)

print(
    f"Model Log-loss advantage/disadvantage: "
    f"{opening_ll - model_ll:+.6f}"
)

print()

print(
    "IMPORTANT:"
)

print(
    "These results evaluate predictive quality."
)

print(
    "They do NOT establish profitable betting edge."
)

print(
    "ROI, staking and prospective paper testing "
    "remain separate evaluations."
)


# ============================================================================
# SAVE
# ============================================================================

print()
print("=" * 78)
print("11. SAVING OUTPUTS")
print("=" * 78)

valid.to_csv(
    OUTPUT_DATA,
    index=False
)

report_rows = []

for _, row in selection_summary.iterrows():

    report_rows.append({
        "section": "selection",
        "group": row["selection"],
        "metric": "brier_difference",
        "value": row["brier_difference"],
        "observations": row["observations"],
    })

for _, row in season_summary.iterrows():

    report_rows.append({
        "section": "season",
        "group": row["season"],
        "metric": "brier_difference",
        "value": row["brier_difference"],
        "observations": row["observations"],
    })

report_rows.extend([
    {
        "section": "overall",
        "group": "ALL",
        "metric": "model_brier",
        "value": model_brier,
        "observations": len(valid),
    },
    {
        "section": "overall",
        "group": "ALL",
        "metric": "opening_brier",
        "value": opening_brier,
        "observations": len(valid),
    },
    {
        "section": "overall",
        "group": "ALL",
        "metric": "closing_brier",
        "value": closing_brier,
        "observations": len(valid),
    },
    {
        "section": "overall",
        "group": "ALL",
        "metric": "model_log_loss",
        "value": model_ll,
        "observations": len(valid),
    },
    {
        "section": "overall",
        "group": "ALL",
        "metric": "opening_log_loss",
        "value": opening_ll,
        "observations": len(valid),
    },
    {
        "section": "overall",
        "group": "ALL",
        "metric": "closing_log_loss",
        "value": closing_ll,
        "observations": len(valid),
    },
    {
        "section": "overall",
        "group": "ALL",
        "metric": "edge_clv_correlation",
        "value": edge_clv_corr,
        "observations": len(valid),
    },
])

pd.DataFrame(
    report_rows
).to_csv(
    OUTPUT_REPORT,
    index=False
)


print()
print(
    f"Evaluation dataset: {OUTPUT_DATA}"
)

print(
    f"Evaluation report:  {OUTPUT_REPORT}"
)

print()
print("=" * 78)
print("STATUS: V0.29 COMPLETE")
print("=" * 78)
