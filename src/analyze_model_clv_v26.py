#!/usr/bin/env python3

"""
EDGE — V0.26 MODEL → CLV SIGNAL ANALYSIS

Purpose:
    Determine whether disagreement between the EDGE model and the
    pre-event market contains information about subsequent market movement.

Important:
    - No model training or modification occurs here.
    - Closing prices are evaluation information, NOT betting inputs.
    - No ROI/profitability conclusion is made.
    - Analysis is restricted to the model/CLV overlapping sample.

Outputs:
    data/processed/model_clv_signal_v26.csv
    reports/model_clv_signal_v26_report.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data/processed/model_clv_join_v25.csv"
OUTPUT_DATA = ROOT / "data/processed/model_clv_signal_v26.csv"
OUTPUT_REPORT = ROOT / "reports/model_clv_signal_v26_report.csv"


def safe_corr(a, b):
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    mask = x.notna() & y.notna()

    if mask.sum() < 2:
        return np.nan

    if x[mask].nunique() < 2 or y[mask].nunique() < 2:
        return np.nan

    return x[mask].corr(y[mask])


def brier(y, p):
    y = pd.to_numeric(y, errors="coerce")
    p = pd.to_numeric(p, errors="coerce")

    mask = y.notna() & p.notna()

    if mask.sum() == 0:
        return np.nan

    return np.mean((p[mask] - y[mask]) ** 2)


def log_loss_binary(y, p):
    y = pd.to_numeric(y, errors="coerce")
    p = pd.to_numeric(p, errors="coerce")

    mask = y.notna() & p.notna()

    if mask.sum() == 0:
        return np.nan

    p = np.clip(p[mask], 1e-15, 1 - 1e-15)
    y = y[mask]

    return -np.mean(
        y * np.log(p) + (1 - y) * np.log(1 - p)
    )


def main():

    print("=" * 78)
    print("EDGE — V0.26 MODEL → CLV SIGNAL ANALYSIS")
    print("=" * 78)

    # ------------------------------------------------------------------
    # 1. LOAD DATA
    # ------------------------------------------------------------------

    if not INPUT.exists():
        raise FileNotFoundError(f"Missing input: {INPUT}")

    df = pd.read_csv(INPUT)

    print("\nDATASET")
    print(f"Rows: {len(df)}")
    print("Columns:")
    for c in df.columns:
        print(f"  - {c}")

    required = [
        "match_id",
        "season",
        "date",
        "target_over25",
        "model_over_probability",
        "opening_price",
        "closing_price",
        "opening_fair_probability",
        "closing_fair_probability",
        "price_clv",
        "fair_probability_clv",
        "selection",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            "Required columns missing from V0.25 dataset:\n"
            + "\n".join(f"  - {c}" for c in missing)
        )

    # ------------------------------------------------------------------
    # 2. BASIC VALIDATION
    # ------------------------------------------------------------------

    print("\n1. BASIC VALIDATION")

    invalid_model = (
        ~pd.to_numeric(df["model_over_probability"], errors="coerce")
        .between(0, 1)
    ).sum()

    invalid_opening_fair = (
        ~pd.to_numeric(df["opening_fair_probability"], errors="coerce")
        .between(0, 1)
    ).sum()

    invalid_closing_fair = (
        ~pd.to_numeric(df["closing_fair_probability"], errors="coerce")
        .between(0, 1)
    ).sum()

    missing_target = df["target_over25"].isna().sum()

    print(f"Invalid model probabilities: {invalid_model}")
    print(f"Invalid opening fair probabilities: {invalid_opening_fair}")
    print(f"Invalid closing fair probabilities: {invalid_closing_fair}")
    print(f"Missing targets: {missing_target}")

    # ------------------------------------------------------------------
    # 3. CONVERT NUMERIC FIELDS
    # ------------------------------------------------------------------

    numeric_cols = [
        "model_over_probability",
        "opening_price",
        "closing_price",
        "opening_fair_probability",
        "closing_fair_probability",
        "price_clv",
        "fair_probability_clv",
        "target_over25",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ------------------------------------------------------------------
    # 4. CREATE SELECTION-SPECIFIC MODEL PROBABILITY
    # ------------------------------------------------------------------
    #
    # The model directly estimates P(OVER 2.5).
    #
    # Therefore:
    #
    # OVER  = P(OVER)
    # UNDER = 1 - P(OVER)
    #
    # This puts model probability on the same side of the market
    # as the CLV observation.
    # ------------------------------------------------------------------

    df["model_selection_probability"] = np.where(
        df["selection"].str.upper() == "OVER",
        df["model_over_probability"],
        np.where(
            df["selection"].str.upper() == "UNDER",
            1.0 - df["model_over_probability"],
            np.nan,
        ),
    )

    # ------------------------------------------------------------------
    # 5. MODEL VS OPENING MARKET
    # ------------------------------------------------------------------

    df["model_vs_opening_fair"] = (
        df["model_selection_probability"]
        - df["opening_fair_probability"]
    )

    # ------------------------------------------------------------------
    # 6. MODEL VS CLOSING MARKET
    # ------------------------------------------------------------------
    #
    # This is NOT a betting edge.
    #
    # It measures how the model compares with the eventual market price.
    # ------------------------------------------------------------------

    df["model_vs_closing_fair"] = (
        df["model_selection_probability"]
        - df["closing_fair_probability"]
    )

    # ------------------------------------------------------------------
    # 7. MODEL EDGE DIRECTION
    # ------------------------------------------------------------------

    df["model_edge_direction"] = np.where(
        df["model_vs_opening_fair"] > 0,
        "MODEL_HIGHER",
        np.where(
            df["model_vs_opening_fair"] < 0,
            "MODEL_LOWER",
            "EQUAL",
        ),
    )

    # ------------------------------------------------------------------
    # 8. EDGE BUCKETS
    # ------------------------------------------------------------------

    bins = [
        -np.inf,
        -0.10,
        -0.05,
        -0.025,
        -0.01,
        0.0,
        0.01,
        0.025,
        0.05,
        0.10,
        np.inf,
    ]

    labels = [
        "< -10%",
        "-10 to -5%",
        "-5 to -2.5%",
        "-2.5 to -1%",
        "-1 to 0%",
        "0 to 1%",
        "1 to 2.5%",
        "2.5 to 5%",
        "5 to 10%",
        "> 10%",
    ]

    df["edge_bucket"] = pd.cut(
        df["model_vs_opening_fair"],
        bins=bins,
        labels=labels,
        right=False,
    )

    # ------------------------------------------------------------------
    # 9. CLV SIGNAL SUMMARY
    # ------------------------------------------------------------------

    print("\n2. MODEL → CLV RELATIONSHIP")

    valid_signal = df[
        df["model_vs_opening_fair"].notna()
        & df["fair_probability_clv"].notna()
    ].copy()

    print(f"Valid signal observations: {len(valid_signal)}")

    correlation = safe_corr(
        valid_signal["model_vs_opening_fair"],
        valid_signal["fair_probability_clv"],
    )

    print(
        "Correlation model-vs-opening edge → fair-probability CLV: "
        f"{correlation:.6f}"
        if pd.notna(correlation)
        else "Correlation: N/A"
    )

    # ------------------------------------------------------------------
    # 10. OVER / UNDER BREAKDOWN
    # ------------------------------------------------------------------

    print("\n3. OVER / UNDER SIGNAL")

    selection_summary = (
        valid_signal
        .groupby("selection", observed=False)
        .agg(
            observations=("match_id", "size"),
            mean_model_edge=("model_vs_opening_fair", "mean"),
            median_model_edge=("model_vs_opening_fair", "median"),
            mean_fair_clv=("fair_probability_clv", "mean"),
            median_fair_clv=("fair_probability_clv", "median"),
            positive_clv_rate=(
                "fair_probability_clv",
                lambda x: (x > 0).mean(),
            ),
            correlation=(
                "model_vs_opening_fair",
                lambda x: safe_corr(
                    x,
                    valid_signal.loc[x.index, "fair_probability_clv"],
                ),
            ),
        )
        .reset_index()
    )

    print(selection_summary.to_string(index=False))

    # ------------------------------------------------------------------
    # 11. EDGE BUCKET ANALYSIS
    # ------------------------------------------------------------------

    print("\n4. MODEL EDGE BUCKETS")

    bucket_summary = (
        valid_signal
        .groupby("edge_bucket", observed=False)
        .agg(
            observations=("match_id", "size"),
            mean_model_probability=(
                "model_selection_probability",
                "mean",
            ),
            mean_opening_fair_probability=(
                "opening_fair_probability",
                "mean",
            ),
            mean_closing_fair_probability=(
                "closing_fair_probability",
                "mean",
            ),
            mean_model_edge=(
                "model_vs_opening_fair",
                "mean",
            ),
            mean_closing_difference=(
                "model_vs_closing_fair",
                "mean",
            ),
            mean_fair_clv=(
                "fair_probability_clv",
                "mean",
            ),
            median_fair_clv=(
                "fair_probability_clv",
                "median",
            ),
            positive_clv_rate=(
                "fair_probability_clv",
                lambda x: (x > 0).mean(),
            ),
        )
        .reset_index()
    )

    print(bucket_summary.to_string(index=False))

    # ------------------------------------------------------------------
    # 12. ACTUAL OUTCOMES BY EDGE BUCKET
    # ------------------------------------------------------------------
    #
    # target_over25 is always the actual OVER outcome.
    #
    # For UNDER selections, the correct outcome is 1 - target_over25.
    # ------------------------------------------------------------------

    valid_signal["selection_target"] = np.where(
        valid_signal["selection"].str.upper() == "OVER",
        valid_signal["target_over25"],
        1.0 - valid_signal["target_over25"],
    )

    outcome_summary = (
        valid_signal
        .groupby("edge_bucket", observed=False)
        .agg(
            observations=("match_id", "size"),
            actual_selection_rate=(
                "selection_target",
                "mean",
            ),
            model_probability=(
                "model_selection_probability",
                "mean",
            ),
            brier_score=(
                "model_selection_probability",
                lambda p: brier(
                    valid_signal.loc[p.index, "selection_target"],
                    p,
                ),
            ),
        )
        .reset_index()
    )

    print("\n5. ACTUAL OUTCOMES BY EDGE BUCKET")
    print(outcome_summary.to_string(index=False))

    # ------------------------------------------------------------------
    # 13. MODEL CALIBRATION
    # ------------------------------------------------------------------

    print("\n6. MODEL CALIBRATION")

    calibration_bins = pd.cut(
        valid_signal["model_selection_probability"],
        bins=np.linspace(0, 1, 11),
        include_lowest=True,
    )

    calibration = (
        valid_signal
        .groupby(calibration_bins, observed=False)
        .agg(
            observations=("match_id", "size"),
            mean_predicted=(
                "model_selection_probability",
                "mean",
            ),
            actual_rate=(
                "selection_target",
                "mean",
            ),
        )
        .reset_index()
    )

    calibration["calibration_error"] = (
        calibration["mean_predicted"]
        - calibration["actual_rate"]
    )

    print(calibration.to_string(index=False))

    # ------------------------------------------------------------------
    # 14. MODEL VS MARKET PREDICTIVE PERFORMANCE
    # ------------------------------------------------------------------
    #
    # Evaluate both probabilities on the same exact sample.
    # ------------------------------------------------------------------

    print("\n7. SAME-SAMPLE MODEL VS MARKET")

    same_sample = valid_signal[
        valid_signal["selection_target"].notna()
        & valid_signal["model_selection_probability"].notna()
        & valid_signal["opening_fair_probability"].notna()
        & valid_signal["closing_fair_probability"].notna()
    ].copy()

    model_brier = brier(
        same_sample["selection_target"],
        same_sample["model_selection_probability"],
    )

    opening_brier = brier(
        same_sample["selection_target"],
        same_sample["opening_fair_probability"],
    )

    closing_brier = brier(
        same_sample["selection_target"],
        same_sample["closing_fair_probability"],
    )

    model_logloss = log_loss_binary(
        same_sample["selection_target"],
        same_sample["model_selection_probability"],
    )

    opening_logloss = log_loss_binary(
        same_sample["selection_target"],
        same_sample["opening_fair_probability"],
    )

    closing_logloss = log_loss_binary(
        same_sample["selection_target"],
        same_sample["closing_fair_probability"],
    )

    print(f"Observations: {len(same_sample)}")
    print(f"Model Brier:   {model_brier:.6f}")
    print(f"Opening Brier: {opening_brier:.6f}")
    print(f"Closing Brier: {closing_brier:.6f}")

    print(f"Model Log Loss:   {model_logloss:.6f}")
    print(f"Opening Log Loss: {opening_logloss:.6f}")
    print(f"Closing Log Loss: {closing_logloss:.6f}")

    # ------------------------------------------------------------------
    # 15. SEASON ANALYSIS
    # ------------------------------------------------------------------

    print("\n8. SEASON ANALYSIS")

    season_summary = (
        valid_signal
        .groupby("season", observed=False)
        .agg(
            observations=("match_id", "size"),
            mean_model_edge=(
                "model_vs_opening_fair",
                "mean",
            ),
            mean_fair_clv=(
                "fair_probability_clv",
                "mean",
            ),
            positive_clv_rate=(
                "fair_probability_clv",
                lambda x: (x > 0).mean(),
            ),
            correlation=(
                "model_vs_opening_fair",
                lambda x: safe_corr(
                    x,
                    valid_signal.loc[x.index, "fair_probability_clv"],
                ),
            ),
        )
        .reset_index()
    )

    print(season_summary.to_string(index=False))

    # ------------------------------------------------------------------
    # 16. OVERALL DIAGNOSTIC
    # ------------------------------------------------------------------

    print("\n9. OVERALL DIAGNOSTIC")

    mean_edge = valid_signal["model_vs_opening_fair"].mean()
    median_edge = valid_signal["model_vs_opening_fair"].median()
    mean_clv = valid_signal["fair_probability_clv"].mean()
    median_clv = valid_signal["fair_probability_clv"].median()

    positive_edge_rate = (
        valid_signal["model_vs_opening_fair"] > 0
    ).mean()

    positive_clv_rate = (
        valid_signal["fair_probability_clv"] > 0
    ).mean()

    print(f"Mean model edge:       {mean_edge:.6f}")
    print(f"Median model edge:     {median_edge:.6f}")
    print(f"Positive edge rate:    {positive_edge_rate:.6f}")
    print(f"Mean fair CLV:         {mean_clv:.6f}")
    print(f"Median fair CLV:       {median_clv:.6f}")
    print(f"Positive CLV rate:     {positive_clv_rate:.6f}")

    # ------------------------------------------------------------------
    # 17. SAVE ENRICHED OBSERVATION DATA
    # ------------------------------------------------------------------

    OUTPUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_DATA, index=False)

    # ------------------------------------------------------------------
    # 18. SAVE COMBINED REPORT
    # ------------------------------------------------------------------

    report_rows = []

    def add_section(section, data):
        temp = data.copy()
        temp.insert(0, "section", section)
        report_rows.append(temp)

    add_section(
        "selection_summary",
        selection_summary,
    )

    add_section(
        "edge_bucket_summary",
        bucket_summary,
    )

    add_section(
        "outcome_summary",
        outcome_summary,
    )

    add_section(
        "calibration",
        calibration,
    )

    add_section(
        "season_summary",
        season_summary,
    )

    overall = pd.DataFrame(
        {
            "metric": [
                "valid_observations",
                "correlation_model_edge_to_fair_clv",
                "mean_model_edge",
                "median_model_edge",
                "positive_edge_rate",
                "mean_fair_clv",
                "median_fair_clv",
                "positive_clv_rate",
                "model_brier",
                "opening_market_brier",
                "closing_market_brier",
                "model_log_loss",
                "opening_market_log_loss",
                "closing_market_log_loss",
            ],
            "value": [
                len(valid_signal),
                correlation,
                mean_edge,
                median_edge,
                positive_edge_rate,
                mean_clv,
                median_clv,
                positive_clv_rate,
                model_brier,
                opening_brier,
                closing_brier,
                model_logloss,
                opening_logloss,
                closing_logloss,
            ],
        }
    )

    add_section("overall_metrics", overall)

    report = pd.concat(
        report_rows,
        ignore_index=True,
        sort=False,
    )

    report.to_csv(OUTPUT_REPORT, index=False)

    # ------------------------------------------------------------------
    # 19. FINAL STATUS
    # ------------------------------------------------------------------

    print("\nOUTPUTS")
    print(f"Signal dataset : {OUTPUT_DATA}")
    print(f"Analysis report: {OUTPUT_REPORT}")

    print("\nIMPORTANT INTERPRETATION RULES")
    print("1. Positive CLV is evidence of market movement in the model's direction.")
    print("2. Correlation does NOT prove profitability.")
    print("3. Closing probability is an evaluation target, not a betting input.")
    print("4. No ROI conclusion is made in V0.26.")
    print("5. No model changes are made in V0.26.")
    print("6. Small buckets must not be overinterpreted.")

    print("\nSTATUS: V0.26 COMPLETE")


if __name__ == "__main__":
    main()
