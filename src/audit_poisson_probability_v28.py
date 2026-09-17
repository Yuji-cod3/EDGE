#!/usr/bin/env python3

"""
EDGE — V0.28 POISSON PROBABILITY CONSTRUCTION AUDIT

Purpose:
    Audit the mathematical construction of the current Poisson model
    probabilities before any model modification or ML upgrade.

This is a DIAGNOSTIC ONLY script.

It checks:

1. Required columns
2. Lambda validity
3. Reconstructed Poisson Over 2.5 probability
4. Reconstructed Under 2.5 probability
5. Model probability consistency
6. OVER / UNDER mapping
7. Probability sum
8. Lambda vs probability relationship
9. Target construction
10. Feature timing / leakage indicators where available
11. Model probability distribution
12. Extreme probability errors

No model parameters are changed.
No predictions are retrained.
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_CLV_FILE = (
    ROOT / "data/processed/model_clv_join_v25.csv"
)

MODEL_DATASET_FILE = (
    ROOT / "data/processed/model_dataset.csv"
)

MODEL_READY_FILE = (
    ROOT / "data/processed/model_ready.csv"
)

OUTPUT_DATA = (
    ROOT / "data/processed/poisson_probability_audit_v28.csv"
)

OUTPUT_REPORT = (
    ROOT / "reports/poisson_probability_audit_v28_report.csv"
)

OUTPUT_DATA.parent.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_REPORT.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================================
# HELPERS
# ============================================================================

def section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def poisson_over_25(lam):
    """
    P(X > 2) for X ~ Poisson(lambda).

    P(X <= 2) =
        e^-lambda * (1 + lambda + lambda^2 / 2)

    Therefore:

        P(X > 2) = 1 - P(X <= 2)
    """

    return 1.0 - math.exp(-lam) * (
        1.0 +
        lam +
        (lam ** 2) / 2.0
    )


def poisson_under_25(lam):
    """
    P(X <= 2) for X ~ Poisson(lambda).
    """

    return math.exp(-lam) * (
        1.0 +
        lam +
        (lam ** 2) / 2.0
    )


def safe_corr(a, b):
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")

    valid = pd.concat(
        [x, y],
        axis=1
    ).dropna()

    if len(valid) < 2:
        return np.nan

    if valid.iloc[:, 0].nunique() < 2:
        return np.nan

    if valid.iloc[:, 1].nunique() < 2:
        return np.nan

    return valid.iloc[:, 0].corr(
        valid.iloc[:, 1]
    )


# ============================================================================
# LOAD DATA
# ============================================================================

print("=" * 78)
print("EDGE — V0.28 POISSON PROBABILITY CONSTRUCTION AUDIT")
print("=" * 78)

df = pd.read_csv(MODEL_CLV_FILE)

print()
print("PRIMARY DATASET")
print(f"Rows: {len(df)}")

print("\nColumns:")
for column in df.columns:
    print(f"  - {column}")


# ============================================================================
# 1. REQUIRED COLUMN CHECK
# ============================================================================

section("1. REQUIRED COLUMN CHECK")

required_columns = [
    "match_id",
    "season",
    "date",
    "target_over25",
    "model_lambda",
    "model_probability",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    print("MISSING REQUIRED COLUMNS:")
    for column in missing_columns:
        print(f"  - {column}")

    raise SystemExit(
        "\nV0.28 STOPPED: required columns are missing."
    )

print("All core columns present.")

optional_columns = [
    "selection",
    "model_over_probability",
    "model_under_probability",
    "market_over_probability",
    "opening_fair_probability",
    "closing_fair_probability",
]

print("\nOptional columns:")
for column in optional_columns:
    print(
        f"  {column}: "
        f"{'YES' if column in df.columns else 'NO'}"
    )


# ============================================================================
# 2. NUMERIC CONVERSION
# ============================================================================

section("2. NUMERIC VALIDATION")

numeric_columns = [
    "model_lambda",
    "model_probability",
    "target_over25",
    "model_over_probability",
    "model_under_probability",
    "market_over_probability",
    "opening_fair_probability",
    "closing_fair_probability",
]

for column in numeric_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

print(
    f"Missing lambda: "
    f"{df['model_lambda'].isna().sum()}"
)

print(
    f"Missing model probability: "
    f"{df['model_probability'].isna().sum()}"
)

print(
    f"Missing target: "
    f"{df['target_over25'].isna().sum()}"
)


# ============================================================================
# 3. LAMBDA VALIDITY
# ============================================================================

section("3. LAMBDA VALIDITY")

lambda_invalid = (
    df["model_lambda"].isna() |
    (df["model_lambda"] <= 0) |
    ~np.isfinite(df["model_lambda"])
)

print(
    f"Invalid lambda values: "
    f"{lambda_invalid.sum()}"
)

print(
    f"Lambda minimum: "
    f"{df['model_lambda'].min():.9f}"
)

print(
    f"Lambda maximum: "
    f"{df['model_lambda'].max():.9f}"
)

print(
    f"Lambda mean: "
    f"{df['model_lambda'].mean():.9f}"
)

print(
    f"Lambda median: "
    f"{df['model_lambda'].median():.9f}"
)

print(
    f"Lambda standard deviation: "
    f"{df['model_lambda'].std():.9f}"
)


# ============================================================================
# 4. RECONSTRUCT POISSON PROBABILITIES
# ============================================================================

section("4. RECONSTRUCT POISSON PROBABILITIES")

df["audit_over_probability"] = (
    df["model_lambda"]
    .apply(
        lambda x:
        poisson_over_25(x)
        if pd.notna(x)
        else np.nan
    )
)

df["audit_under_probability"] = (
    df["model_lambda"]
    .apply(
        lambda x:
        poisson_under_25(x)
        if pd.notna(x)
        else np.nan
    )
)

df["audit_probability_sum"] = (
    df["audit_over_probability"] +
    df["audit_under_probability"]
)

print(
    "Reconstructed probabilities using:"
)

print(
    "P(Over 2.5) = "
    "1 - exp(-lambda) * "
    "(1 + lambda + lambda²/2)"
)

print(
    "P(Under 2.5) = "
    "exp(-lambda) * "
    "(1 + lambda + lambda²/2)"
)

print()

print(
    f"Reconstructed OVER minimum: "
    f"{df['audit_over_probability'].min():.9f}"
)

print(
    f"Reconstructed OVER maximum: "
    f"{df['audit_over_probability'].max():.9f}"
)

print(
    f"Reconstructed UNDER minimum: "
    f"{df['audit_under_probability'].min():.9f}"
)

print(
    f"Reconstructed UNDER maximum: "
    f"{df['audit_under_probability'].max():.9f}"
)


# ============================================================================
# 5. MATHEMATICAL SUM CHECK
# ============================================================================

section("5. MATHEMATICAL PROBABILITY SUM CHECK")

df["audit_sum_error"] = (
    df["audit_probability_sum"] - 1.0
).abs()

print(
    "Maximum reconstructed OVER + UNDER error: "
    f"{df['audit_sum_error'].max():.15f}"
)

print(
    "Mean reconstructed OVER + UNDER error: "
    f"{df['audit_sum_error'].mean():.15f}"
)

if df["audit_sum_error"].max() < 1e-10:
    print("\nRESULT: Poisson reconstruction is mathematically consistent.")
else:
    print("\nRESULT: Poisson reconstruction has a mathematical inconsistency.")


# ============================================================================
# 6. EXISTING MODEL OVER PROBABILITY
# ============================================================================

section("6. EXISTING MODEL OVER PROBABILITY AUDIT")

if "model_over_probability" in df.columns:

    df["model_over_probability_error"] = (
        df["model_over_probability"] -
        df["audit_over_probability"]
    )

    print(
        "Maximum absolute difference between "
        "stored OVER probability and reconstructed OVER probability:"
    )

    print(
        f"{df['model_over_probability_error'].abs().max():.12f}"
    )

    print(
        "Mean absolute difference:"
    )

    print(
        f"{df['model_over_probability_error'].abs().mean():.12f}"
    )

    print(
        "Correlation:"
    )

    print(
        f"{safe_corr(df['model_over_probability'], df['audit_over_probability']):.9f}"
    )

else:

    print(
        "model_over_probability is not available."
    )


# ============================================================================
# 7. EXISTING MODEL UNDER PROBABILITY
# ============================================================================

section("7. EXISTING MODEL UNDER PROBABILITY AUDIT")

if "model_under_probability" in df.columns:

    df["model_under_probability_error"] = (
        df["model_under_probability"] -
        df["audit_under_probability"]
    )

    print(
        "Maximum absolute difference between "
        "stored UNDER probability and reconstructed UNDER probability:"
    )

    print(
        f"{df['model_under_probability_error'].abs().max():.12f}"
    )

    print(
        "Mean absolute difference:"
    )

    print(
        f"{df['model_under_probability_error'].abs().mean():.12f}"
    )

    print(
        "Correlation:"
    )

    print(
        f"{safe_corr(df['model_under_probability'], df['audit_under_probability']):.9f}"
    )

else:

    print(
        "model_under_probability is not available."
    )


# ============================================================================
# 8. STORED MODEL PROBABILITY VS POISSON
# ============================================================================

section("8. STORED MODEL PROBABILITY VS POISSON")

df["stored_probability_error_vs_audit"] = (
    df["model_probability"] -
    df["audit_over_probability"]
)

print(
    "Treating stored model_probability as OVER probability:"
)

print(
    f"Mean error: "
    f"{df['stored_probability_error_vs_audit'].mean():.9f}"
)

print(
    f"Mean absolute error: "
    f"{df['stored_probability_error_vs_audit'].abs().mean():.9f}"
)

print(
    f"Maximum absolute error: "
    f"{df['stored_probability_error_vs_audit'].abs().max():.9f}"
)

print(
    f"Correlation: "
    f"{safe_corr(df['model_probability'], df['audit_over_probability']):.9f}"
)


# ============================================================================
# 9. SELECTION MAPPING
# ============================================================================

section("9. OVER / UNDER SELECTION MAPPING")

if "selection" not in df.columns:

    print(
        "Selection column unavailable."
    )

else:

    print(
        "Selections found:"
    )

    print(
        df["selection"]
        .value_counts(dropna=False)
        .to_string()
    )

    # Expected selection probability:
    #
    # OVER  -> Poisson OVER probability
    # UNDER -> Poisson UNDER probability

    df["expected_selection_probability"] = np.where(
        df["selection"].astype(str).str.upper() == "OVER",
        df["audit_over_probability"],
        np.where(
            df["selection"].astype(str).str.upper() == "UNDER",
            df["audit_under_probability"],
            np.nan
        )
    )

    df["selection_mapping_error"] = (
        df["model_probability"] -
        df["expected_selection_probability"]
    )

    valid_mapping = df[
        df["expected_selection_probability"].notna()
    ]

    print()

    print(
        f"Valid selection rows: "
        f"{len(valid_mapping)}"
    )

    print(
        "Mean absolute mapping error: "
        f"{valid_mapping['selection_mapping_error'].abs().mean():.9f}"
    )

    print(
        "Maximum absolute mapping error: "
        f"{valid_mapping['selection_mapping_error'].abs().max():.9f}"
    )

    print(
        "Correlation stored probability vs expected selection probability: "
        f"{safe_corr(valid_mapping['model_probability'], valid_mapping['expected_selection_probability']):.9f}"
    )


# ============================================================================
# 10. MODEL PROBABILITY SUM
# ============================================================================

section("10. STORED MODEL OVER + UNDER CHECK")

if (
    "model_over_probability" in df.columns
    and
    "model_under_probability" in df.columns
):

    df["stored_probability_sum"] = (
        df["model_over_probability"] +
        df["model_under_probability"]
    )

    df["stored_sum_error"] = (
        df["stored_probability_sum"] - 1.0
    ).abs()

    print(
        "Maximum stored OVER + UNDER error:"
    )

    print(
        f"{df['stored_sum_error'].max():.12f}"
    )

    print(
        "Mean stored OVER + UNDER error:"
    )

    print(
        f"{df['stored_sum_error'].mean():.12f}"
    )

    print(
        "Rows with sum error > 0.000001:"
    )

    print(
        (
            df["stored_sum_error"] > 0.000001
        ).sum()
    )

else:

    print(
        "Both stored OVER and UNDER probabilities "
        "are not available."
    )


# ============================================================================
# 11. TARGET AUDIT
# ============================================================================

section("11. TARGET AUDIT")

target = df["target_over25"]

print(
    f"Invalid target values: "
    f"{(~target.isin([0, 1])).sum()}"
)

print(
    f"Actual OVER 2.5 rate: "
    f"{target.mean():.9f}"
)

print(
    f"Actual UNDER 2.5 rate: "
    f"{1 - target.mean():.9f}"
)

if "selection" in df.columns:

    for selection in ["OVER", "UNDER"]:

        group = df[
            df["selection"].astype(str).str.upper()
            == selection
        ]

        if len(group) == 0:
            continue

        if selection == "OVER":
            selection_target = group["target_over25"]
        else:
            selection_target = 1 - group["target_over25"]

        print(
            f"{selection} rows: "
            f"{len(group)}"
        )

        print(
            f"{selection} target rate: "
            f"{selection_target.mean():.9f}"
        )


# ============================================================================
# 12. LAMBDA → PROBABILITY RELATIONSHIP
# ============================================================================

section("12. LAMBDA → PROBABILITY RELATIONSHIP")

print(
    "Correlation lambda → reconstructed OVER probability:"
)

print(
    f"{safe_corr(df['model_lambda'], df['audit_over_probability']):.9f}"
)

print(
    "Correlation lambda → stored model probability:"
)

print(
    f"{safe_corr(df['model_lambda'], df['model_probability']):.9f}"
)

print()

print(
    "Selected lambda/probability checkpoints:"
)

checkpoints = [
    df["model_lambda"].min(),
    df["model_lambda"].quantile(0.25),
    df["model_lambda"].median(),
    df["model_lambda"].quantile(0.75),
    df["model_lambda"].max(),
]

for lam in checkpoints:

    over = poisson_over_25(lam)
    under = poisson_under_25(lam)

    print(
        f"lambda={lam:.6f}  "
        f"OVER={over:.6f}  "
        f"UNDER={under:.6f}"
    )


# ============================================================================
# 13. PROBABILITY DISTRIBUTION
# ============================================================================

section("13. MODEL PROBABILITY DISTRIBUTION")

print(
    f"Minimum: "
    f"{df['model_probability'].min():.9f}"
)

print(
    f"Maximum: "
    f"{df['model_probability'].max():.9f}"
)

print(
    f"Mean: "
    f"{df['model_probability'].mean():.9f}"
)

print(
    f"Median: "
    f"{df['model_probability'].median():.9f}"
)

print(
    f"Std: "
    f"{df['model_probability'].std():.9f}"
)

print(
    f"5th percentile: "
    f"{df['model_probability'].quantile(0.05):.9f}"
)

print(
    f"95th percentile: "
    f"{df['model_probability'].quantile(0.95):.9f}"
)


# ============================================================================
# 14. FEATURE / DATASET INSPECTION
# ============================================================================

section("14. MODEL DATASET INSPECTION")

if MODEL_DATASET_FILE.exists():

    model_dataset = pd.read_csv(
        MODEL_DATASET_FILE
    )

    print(
        f"model_dataset.csv rows: "
        f"{len(model_dataset)}"
    )

    print(
        f"model_dataset.csv columns: "
        f"{len(model_dataset.columns)}"
    )

    print("\nColumns:")
    for column in model_dataset.columns:
        print(f"  - {column}")

else:

    model_dataset = None

    print(
        "model_dataset.csv not found."
    )


# ============================================================================
# 15. POTENTIAL LEAKAGE INDICATORS
# ============================================================================

section("15. POTENTIAL LEAKAGE INDICATORS")

if model_dataset is not None:

    suspicious_keywords = [
        "target",
        "result",
        "home_goals",
        "away_goals",
        "total_goals",
        "over25_result",
        "closing",
        "outcome",
    ]

    suspicious = []

    for column in model_dataset.columns:

        lower = column.lower()

        if any(
            keyword in lower
            for keyword in suspicious_keywords
        ):

            suspicious.append(column)

    print(
        "Columns requiring timing/leakage review:"
    )

    for column in suspicious:
        print(f"  - {column}")

    if not suspicious:
        print(
            "No obvious outcome-related column names found."
        )

else:

    print(
        "Cannot inspect model_dataset.csv."
    )


# ============================================================================
# 16. SAMPLE ROWS FOR MANUAL VERIFICATION
# ============================================================================

section("16. SAMPLE PROBABILITY VERIFICATION")

sample_columns = [
    "match_id",
    "season",
    "selection",
    "model_lambda",
    "model_probability",
    "audit_over_probability",
    "audit_under_probability",
    "target_over25",
]

sample_columns = [
    column
    for column in sample_columns
    if column in df.columns
]

sample = df[
    sample_columns
].head(20)

print(
    sample.to_string(
        index=False
    )
)


# ============================================================================
# 17. EXTREME DISCREPANCIES
# ============================================================================

section("17. LARGEST PROBABILITY DISCREPANCIES")

if "expected_selection_probability" in df.columns:

    discrepancy = (
        df[
            [
                "match_id",
                "season",
                "selection",
                "model_lambda",
                "model_probability",
                "expected_selection_probability",
                "selection_mapping_error",
                "target_over25",
            ]
        ]
        .copy()
    )

    discrepancy["absolute_error"] = (
        discrepancy["selection_mapping_error"]
        .abs()
    )

    discrepancy = discrepancy.sort_values(
        "absolute_error",
        ascending=False
    )

    print(
        discrepancy.head(20).to_string(
            index=False
        )
    )

else:

    print(
        "Selection mapping unavailable."
    )


# ============================================================================
# 18. SAVE AUDIT DATA
# ============================================================================

section("18. SAVING AUDIT OUTPUT")

df.to_csv(
    OUTPUT_DATA,
    index=False
)

report_rows = []

def add_report(
    section_name,
    metric,
    value,
    observations=None,
    note=""
):
    report_rows.append({
        "section": section_name,
        "metric": metric,
        "value": value,
        "observations": observations,
        "note": note,
    })


add_report(
    "lambda",
    "minimum",
    df["model_lambda"].min(),
    len(df)
)

add_report(
    "lambda",
    "maximum",
    df["model_lambda"].max(),
    len(df)
)

add_report(
    "lambda",
    "mean",
    df["model_lambda"].mean(),
    len(df)
)

add_report(
    "probability",
    "stored_model_minimum",
    df["model_probability"].min(),
    len(df)
)

add_report(
    "probability",
    "stored_model_maximum",
    df["model_probability"].max(),
    len(df)
)

add_report(
    "probability",
    "stored_model_mean",
    df["model_probability"].mean(),
    len(df)
)

add_report(
    "poisson",
    "maximum_reconstruction_sum_error",
    df["audit_sum_error"].max(),
    len(df)
)

add_report(
    "poisson",
    "mean_reconstruction_sum_error",
    df["audit_sum_error"].mean(),
    len(df)
)

add_report(
    "probability",
    "lambda_to_stored_probability_correlation",
    safe_corr(
        df["model_lambda"],
        df["model_probability"]
    ),
    len(df)
)

add_report(
    "probability",
    "lambda_to_reconstructed_over_correlation",
    safe_corr(
        df["model_lambda"],
        df["audit_over_probability"]
    ),
    len(df)
)

if "model_over_probability_error" in df.columns:

    add_report(
        "mapping",
        "max_stored_over_reconstruction_error",
        df["model_over_probability_error"]
        .abs()
        .max(),
        len(df)
    )

if "model_under_probability_error" in df.columns:

    add_report(
        "mapping",
        "max_stored_under_reconstruction_error",
        df["model_under_probability_error"]
        .abs()
        .max(),
        len(df)
    )

if "stored_sum_error" in df.columns:

    add_report(
        "mapping",
        "max_stored_over_under_sum_error",
        df["stored_sum_error"].max(),
        len(df)
    )

if "selection_mapping_error" in df.columns:

    add_report(
        "mapping",
        "max_selection_mapping_error",
        df["selection_mapping_error"]
        .abs()
        .max(),
        len(df)
    )

report = pd.DataFrame(
    report_rows
)

report.to_csv(
    OUTPUT_REPORT,
    index=False
)


# ============================================================================
# FINAL STATUS
# ============================================================================

section("V0.28 OUTPUTS")

print(
    f"Audit dataset : {OUTPUT_DATA}"
)

print(
    f"Audit report  : {OUTPUT_REPORT}"
)

print()
print(
    "STATUS: V0.28 COMPLETE"
)

print("=" * 78)
