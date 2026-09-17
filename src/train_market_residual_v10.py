import pandas as pd
import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error


INPUT_FILE = "data/processed/residual_v08_predictions.csv"
OUTPUT_FILE = "data/processed/market_residual_v10_predictions.csv"

PROB_COL = "market_over_probability"
TARGET_COL = "target_over25"

# Market probability buckets
BINS = np.array([
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    1.00
])

MIN_BUCKET_SAMPLES = 100

# Shrink historical residuals toward zero.
# This prevents small historical samples from producing
# extreme corrections.
SHRINKAGE = 0.50


def assign_bucket(prob):
    """
    Assign market probability to a fixed bucket.
    """
    idx = np.searchsorted(BINS, prob, side="right") - 1

    if idx < 0:
        return -1

    if idx >= len(BINS) - 1:
        return len(BINS) - 2

    return idx


def calculate_residual_table(train):
    """
    Calculate historical market residual by probability bucket.

    residual = actual outcome - market probability
    """

    data = train.copy()

    data["bucket"] = data[PROB_COL].apply(assign_bucket)

    grouped = (
        data[data["bucket"] >= 0]
        .groupby("bucket")
        .agg(
            samples=(TARGET_COL, "count"),
            actual_rate=(TARGET_COL, "mean"),
            mean_market_probability=(PROB_COL, "mean"),
        )
        .reset_index()
    )

    grouped["raw_residual"] = (
        grouped["actual_rate"]
        - grouped["mean_market_probability"]
    )

    # Reliability weighting.
    #
    # A bucket with 100 samples gets less trust than one
    # with thousands of samples.
    grouped["reliability"] = (
        grouped["samples"]
        / (grouped["samples"] + MIN_BUCKET_SAMPLES)
    )

    grouped["shrunk_residual"] = (
        grouped["raw_residual"]
        * grouped["reliability"]
        * SHRINKAGE
    )

    return grouped


def apply_correction(test, residual_table):
    """
    Apply historical residual correction to test data.
    """

    data = test.copy()

    data["bucket"] = data[PROB_COL].apply(assign_bucket)

    correction_map = dict(
        zip(
            residual_table["bucket"],
            residual_table["shrunk_residual"]
        )
    )

    data["historical_residual"] = (
        data["bucket"]
        .map(correction_map)
        .fillna(0.0)
    )

    data["v10_over_probability"] = (
        data[PROB_COL]
        + data["historical_residual"]
    )

    # Keep probabilities numerically valid.
    data["v10_over_probability"] = (
        data["v10_over_probability"]
        .clip(0.01, 0.99)
    )

    return data


def evaluate(y, p):
    return {
        "brier": brier_score_loss(y, p),
        "logloss": log_loss(y, p, labels=[0, 1]),
        "mae": mean_absolute_error(y, p),
    }


def main():

    print("EDGE — V0.10 EXPANDING MARKET RESIDUAL CORRECTION")
    print("=" * 70)

    df = pd.read_csv(INPUT_FILE)

    required = [
        PROB_COL,
        TARGET_COL,
        "season",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df = df.sort_values(
        ["date", "match_id"]
    ).reset_index(drop=True)

    seasons = sorted(df["season"].unique())

    all_predictions = []
    season_results = []

    # Need at least one historical season.
    for i in range(1, len(seasons)):

        test_season = seasons[i]
        train_seasons = seasons[:i]

        train = df[
            df["season"].isin(train_seasons)
        ].copy()

        test = df[
            df["season"] == test_season
        ].copy()

        print(
            f"{test_season}: "
            f"train={len(train)} "
            f"test={len(test)}"
        )

        # --------------------------------------------------
        # Learn residual structure ONLY from past seasons
        # --------------------------------------------------

        residual_table = calculate_residual_table(train)

        test_pred = apply_correction(
            test,
            residual_table
        )

        y = test_pred[TARGET_COL]

        market = test_pred[PROB_COL]
        v10 = test_pred["v10_over_probability"]

        market_metrics = evaluate(y, market)
        v10_metrics = evaluate(y, v10)

        season_results.append({
            "season": test_season,
            "matches": len(test_pred),

            "market_brier": market_metrics["brier"],
            "v10_brier": v10_metrics["brier"],

            "market_logloss": market_metrics["logloss"],
            "v10_logloss": v10_metrics["logloss"],

            "market_mae": market_metrics["mae"],
            "v10_mae": v10_metrics["mae"],

            "mean_correction":
                test_pred["historical_residual"].mean(),
        })

        test_pred["model_version"] = "V0.10"

        all_predictions.append(test_pred)

    results = pd.DataFrame(season_results)

    predictions = pd.concat(
        all_predictions,
        ignore_index=True
    )

    # ------------------------------------------------------
    # Overall metrics
    # ------------------------------------------------------

    y = predictions[TARGET_COL]

    market = predictions[PROB_COL]
    v10 = predictions["v10_over_probability"]

    market_metrics = evaluate(y, market)
    v10_metrics = evaluate(y, v10)

    print()
    print("=" * 70)
    print("OVERALL OUT-OF-SAMPLE RESULTS")
    print("=" * 70)

    print(f"Predictions: {len(predictions):,}")
    print()

    print("MARKET")
    print(f"Brier score: {market_metrics['brier']:.6f}")
    print(f"Log Loss:    {market_metrics['logloss']:.6f}")
    print(f"MAE:         {market_metrics['mae']:.6f}")

    print()

    print("V0.10 RESIDUAL CORRECTION")
    print(f"Brier score: {v10_metrics['brier']:.6f}")
    print(f"Log Loss:    {v10_metrics['logloss']:.6f}")
    print(f"MAE:         {v10_metrics['mae']:.6f}")

    print()
    print("=" * 70)
    print("V0.10 IMPROVEMENT VS MARKET")
    print("=" * 70)

    # Positive = improvement
    brier_improvement = (
        market_metrics["brier"]
        - v10_metrics["brier"]
    )

    logloss_improvement = (
        market_metrics["logloss"]
        - v10_metrics["logloss"]
    )

    mae_improvement = (
        market_metrics["mae"]
        - v10_metrics["mae"]
    )

    print(
        f"Brier improvement:   "
        f"{brier_improvement:+.6f}"
    )

    print(
        f"LogLoss improvement: "
        f"{logloss_improvement:+.6f}"
    )

    print(
        f"MAE improvement:     "
        f"{mae_improvement:+.6f}"
    )

    # ------------------------------------------------------
    # Season results
    # ------------------------------------------------------

    print()
    print("=" * 70)
    print("SEASON-BY-SEASON RESULTS")
    print("=" * 70)

    print(
        results.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    # ------------------------------------------------------
    # Residual correction diagnostics
    # ------------------------------------------------------

    print()
    print("=" * 70)
    print("CORRECTION DIAGNOSTICS")
    print("=" * 70)

    print(
        f"Mean correction: "
        f"{predictions['historical_residual'].mean():+.6f}"
    )

    print(
        f"Median correction: "
        f"{predictions['historical_residual'].median():+.6f}"
    )

    print(
        f"Minimum correction: "
        f"{predictions['historical_residual'].min():+.6f}"
    )

    print(
        f"Maximum correction: "
        f"{predictions['historical_residual'].max():+.6f}"
    )

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    predictions.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
