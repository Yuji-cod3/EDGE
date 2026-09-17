import os
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss


INPUT_FILE = "data/processed/logistic_v06_predictions.csv"
OUTPUT_FILE = "data/processed/value_v11_predictions.csv"
REPORT_FILE = "reports/value_v11_summary.csv"

MODEL_COL = "v06_over_probability"
MARKET_COL = "market_over_probability"
ODDS_COL = "market_over_odds"
TARGET_COL = "target_over25"


EDGE_BINS = [
    (-np.inf, -0.10, "< -10%"),
    (-0.10, -0.05, "-10% to -5%"),
    (-0.05, -0.025, "-5% to -2.5%"),
    (-0.025, 0.0, "-2.5% to 0%"),
    (0.0, 0.025, "0% to 2.5%"),
    (0.025, 0.05, "2.5% to 5%"),
    (0.05, 0.10, "5% to 10%"),
    (0.10, np.inf, "> 10%"),
]


def edge_bucket(edge):
    for low, high, label in EDGE_BINS:
        if low < edge <= high:
            return label
    return "UNKNOWN"


def evaluate_probability(y, p):
    return {
        "brier": brier_score_loss(y, p),
        "logloss": log_loss(y, p, labels=[0, 1]),
    }


def main():

    print("EDGE — V0.11 VALUE / EV DIAGNOSTIC")
    print("=" * 78)

    df = pd.read_csv(INPUT_FILE)

    required = [
        "match_id",
        "season",
        "date",
        MODEL_COL,
        MARKET_COL,
        ODDS_COL,
        TARGET_COL,
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    print(f"Predictions available: {len(df):,}")

    # ---------------------------------------------------------
    # DATA VALIDATION
    # ---------------------------------------------------------

    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])

    if df[required].isnull().any().any():
        raise ValueError(
            "Required columns contain missing values."
        )

    if (df[ODDS_COL] <= 1).any():
        raise ValueError(
            "Invalid decimal odds <= 1 detected."
        )

    if not df[TARGET_COL].isin([0, 1]).all():
        raise ValueError(
            "Target must contain only 0/1."
        )

    # ---------------------------------------------------------
    # MODEL / MARKET EDGE
    # ---------------------------------------------------------

    df["probability_edge"] = (
        df[MODEL_COL] - df[MARKET_COL]
    )

    # Expected value using model probability.
    #
    # EV = P(win) * odds - 1
    df["expected_value"] = (
        df[MODEL_COL] * df[ODDS_COL] - 1.0
    )

    # Historical $1 flat-stake return.
    #
    # Win: odds - 1
    # Loss: -1
    df["flat_stake_profit"] = np.where(
        df[TARGET_COL] == 1,
        df[ODDS_COL] - 1.0,
        -1.0,
    )

    df["edge_bucket"] = (
        df["probability_edge"]
        .apply(edge_bucket)
    )

    # ---------------------------------------------------------
    # BASIC DATASET METRICS
    # ---------------------------------------------------------

    y = df[TARGET_COL]

    model_metrics = evaluate_probability(
        y,
        df[MODEL_COL]
    )

    market_metrics = evaluate_probability(
        y,
        df[MARKET_COL]
    )

    print()
    print("=" * 78)
    print("1. PROBABILITY PERFORMANCE")
    print("=" * 78)

    print(
        f"Model Brier:  {model_metrics['brier']:.6f}"
    )

    print(
        f"Market Brier: {market_metrics['brier']:.6f}"
    )

    print(
        f"Model LogLoss:  {model_metrics['logloss']:.6f}"
    )

    print(
        f"Market LogLoss: {market_metrics['logloss']:.6f}"
    )

    # ---------------------------------------------------------
    # OVERALL VALUE DISTRIBUTION
    # ---------------------------------------------------------

    print()
    print("=" * 78)
    print("2. VALUE DISTRIBUTION")
    print("=" * 78)

    print(
        f"Mean probability edge:   "
        f"{df['probability_edge'].mean():+.6f}"
    )

    print(
        f"Median probability edge: "
        f"{df['probability_edge'].median():+.6f}"
    )

    print(
        f"Mean expected value:     "
        f"{df['expected_value'].mean():+.6f}"
    )

    print(
        f"Median expected value:   "
        f"{df['expected_value'].median():+.6f}"
    )

    print(
        f"Positive EV matches: "
        f"{(df['expected_value'] > 0).sum():,}"
        f" / {len(df):,}"
    )

    # ---------------------------------------------------------
    # EDGE BUCKET ANALYSIS
    # ---------------------------------------------------------

    print()
    print("=" * 78)
    print("3. EDGE BUCKET ANALYSIS")
    print("=" * 78)

    bucket_rows = []

    for _, _, label in EDGE_BINS:

        subset = df[
            df["edge_bucket"] == label
        ]

        if len(subset) == 0:
            continue

        wins = subset[TARGET_COL].sum()
        matches = len(subset)

        total_profit = (
            subset["flat_stake_profit"].sum()
        )

        roi = total_profit / matches

        mean_ev = subset["expected_value"].mean()

        actual_win_rate = (
            wins / matches
        )

        bucket_rows.append({
            "edge_bucket": label,
            "matches": matches,
            "mean_model_probability":
                subset[MODEL_COL].mean(),
            "mean_market_probability":
                subset[MARKET_COL].mean(),
            "mean_probability_edge":
                subset["probability_edge"].mean(),
            "mean_odds":
                subset[ODDS_COL].mean(),
            "actual_win_rate":
                actual_win_rate,
            "mean_expected_value":
                mean_ev,
            "total_flat_profit":
                total_profit,
            "roi":
                roi,
        })

    bucket_df = pd.DataFrame(bucket_rows)

    print(
        bucket_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    # ---------------------------------------------------------
    # POSITIVE EV DIAGNOSTIC
    # ---------------------------------------------------------

    print()
    print("=" * 78)
    print("4. POSITIVE EV DIAGNOSTIC")
    print("=" * 78)

    positive_ev = df[
        df["expected_value"] > 0
    ].copy()

    if len(positive_ev) > 0:

        profit = (
            positive_ev["flat_stake_profit"].sum()
        )

        roi = profit / len(positive_ev)

        print(
            f"Selections:       {len(positive_ev):,}"
        )

        print(
            f"Mean model prob:  "
            f"{positive_ev[MODEL_COL].mean():.6f}"
        )

        print(
            f"Mean odds:        "
            f"{positive_ev[ODDS_COL].mean():.6f}"
        )

        print(
            f"Actual win rate:  "
            f"{positive_ev[TARGET_COL].mean():.6f}"
        )

        print(
            f"Mean expected EV: "
            f"{positive_ev['expected_value'].mean():+.6f}"
        )

        print(
            f"Historical ROI:   "
            f"{roi:+.6f}"
        )

    else:

        print("No positive-EV observations.")

    # ---------------------------------------------------------
    # SEASON ANALYSIS
    # ---------------------------------------------------------

    print()
    print("=" * 78)
    print("5. SEASON-BY-SEASON VALUE")
    print("=" * 78)

    season_rows = []

    for season, subset in df.groupby("season"):

        positive = subset[
            subset["expected_value"] > 0
        ]

        if len(positive) > 0:

            profit = (
                positive["flat_stake_profit"].sum()
            )

            roi = profit / len(positive)

            actual_rate = (
                positive[TARGET_COL].mean()
            )

        else:

            roi = np.nan
            actual_rate = np.nan

        season_rows.append({
            "season": season,
            "matches": len(subset),
            "positive_ev_selections": len(positive),
            "mean_edge":
                subset["probability_edge"].mean(),
            "mean_ev":
                subset["expected_value"].mean(),
            "positive_ev_win_rate":
                actual_rate,
            "positive_ev_roi":
                roi,
        })

    season_df = pd.DataFrame(
        season_rows
    )

    print(
        season_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    # ---------------------------------------------------------
    # EDGE DIRECTION TEST
    # ---------------------------------------------------------

    print()
    print("=" * 78)
    print("6. EDGE DIRECTION")
    print("=" * 78)

    positive_edge = df[
        df["probability_edge"] > 0
    ]

    negative_edge = df[
        df["probability_edge"] < 0
    ]

    print(
        f"Positive-edge matches: "
        f"{len(positive_edge):,}"
    )

    print(
        f"Negative-edge matches: "
        f"{len(negative_edge):,}"
    )

    if len(positive_edge) > 0:

        print(
            f"Positive-edge actual rate: "
            f"{positive_edge[TARGET_COL].mean():.6f}"
        )

        print(
            f"Positive-edge mean EV: "
            f"{positive_edge['expected_value'].mean():+.6f}"
        )

    if len(negative_edge) > 0:

        print(
            f"Negative-edge actual rate: "
            f"{negative_edge[TARGET_COL].mean():.6f}"
        )

        print(
            f"Negative-edge mean EV: "
            f"{negative_edge['expected_value'].mean():+.6f}"
        )

    # ---------------------------------------------------------
    # IMPORTANT WARNING
    # ---------------------------------------------------------

    print()
    print("=" * 78)
    print("7. DIAGNOSTIC ASSESSMENT")
    print("=" * 78)

    print(
        "This is a historical value diagnostic."
    )

    print(
        "No betting threshold was optimized."
    )

    print(
        "No stake sizing was optimized."
    )

    print(
        "No ROI conclusion should be treated as proof of an edge."
    )

    print(
        "Positive historical ROI can occur by chance."
    )

    print(
        "Multiple-testing and selection bias must be controlled."
    )

    print(
        "Closing-line data is still required for stronger"
        " market-efficiency analysis."
    )

    # ---------------------------------------------------------
    # SAVE RESULTS
    # ---------------------------------------------------------

    os.makedirs(
        os.path.dirname(REPORT_FILE),
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    bucket_df.to_csv(
        REPORT_FILE,
        index=False
    )

    print()
    print(f"Saved predictions to: {OUTPUT_FILE}")
    print(f"Saved summary to:     {REPORT_FILE}")


if __name__ == "__main__":
    main()
