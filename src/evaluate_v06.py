import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss


INPUT_FILE = "data/processed/logistic_v06_predictions.csv"


def clip_probability(p):
    return np.clip(p, 1e-6, 1 - 1e-6)


def evaluate(y, p):
    p = clip_probability(p)

    return {
        "brier": brier_score_loss(y, p),
        "logloss": log_loss(y, p),
        "mae": np.mean(np.abs(y - p)),
    }


def main():

    print("\nEDGE — V0.6 RESIDUAL / EDGE DIAGNOSTIC")
    print("=" * 78)

    df = pd.read_csv(INPUT_FILE)

    print(f"Predictions: {len(df):,}")

    y = df["target_over25"].astype(int)

    market = df["market_over_probability"]
    model = df["v06_over_probability"]

    df["probability_edge"] = (
        model - market
    )

    # ---------------------------------------------------------
    # Overall
    # ---------------------------------------------------------

    market_metrics = evaluate(y, market)
    model_metrics = evaluate(y, model)

    print("\n" + "=" * 78)
    print("1. OVERALL PERFORMANCE")
    print("=" * 78)

    print(
        f"Market Brier: {market_metrics['brier']:.6f}"
    )
    print(
        f"V0.6 Brier:   {model_metrics['brier']:.6f}"
    )

    print(
        f"Market LogLoss: {market_metrics['logloss']:.6f}"
    )
    print(
        f"V0.6 LogLoss:   {model_metrics['logloss']:.6f}"
    )

    print(
        f"Market MAE: {market_metrics['mae']:.6f}"
    )
    print(
        f"V0.6 MAE:   {model_metrics['mae']:.6f}"
    )

    # ---------------------------------------------------------
    # Probability edge
    # ---------------------------------------------------------

    print("\n" + "=" * 78)
    print("2. MODEL VS MARKET PROBABILITY")
    print("=" * 78)

    print(
        f"Mean model probability:  "
        f"{model.mean():.6f}"
    )

    print(
        f"Mean market probability: "
        f"{market.mean():.6f}"
    )

    print(
        f"Mean probability edge:   "
        f"{df['probability_edge'].mean():+.6f}"
    )

    print(
        f"Median probability edge: "
        f"{df['probability_edge'].median():+.6f}"
    )

    print(
        f"Minimum edge: "
        f"{df['probability_edge'].min():+.6f}"
    )

    print(
        f"Maximum edge: "
        f"{df['probability_edge'].max():+.6f}"
    )

    # ---------------------------------------------------------
    # Edge buckets
    # ---------------------------------------------------------

    bins = [
        -np.inf,
        -0.10,
        -0.05,
        -0.025,
        0.0,
        0.025,
        0.05,
        0.10,
        np.inf,
    ]

    labels = [
        "< -10%",
        "-10% to -5%",
        "-5% to -2.5%",
        "-2.5% to 0%",
        "0% to 2.5%",
        "2.5% to 5%",
        "5% to 10%",
        "> 10%",
    ]

    df["edge_bin"] = pd.cut(
        df["probability_edge"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    edge_table = (
        df.groupby(
            "edge_bin",
            observed=False,
        )
        .agg(
            matches=("target_over25", "size"),
            mean_model_probability=(
                "v06_over_probability",
                "mean",
            ),
            mean_market_probability=(
                "market_over_probability",
                "mean",
            ),
            actual_over_rate=(
                "target_over25",
                "mean",
            ),
            mean_edge=(
                "probability_edge",
                "mean",
            ),
        )
        .reset_index()
    )

    print("\n" + "=" * 78)
    print("3. EDGE BUCKET ANALYSIS")
    print("=" * 78)

    print(
        edge_table.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # ---------------------------------------------------------
    # Odds buckets
    # ---------------------------------------------------------

    odds_bins = [
        1.0,
        1.40,
        1.60,
        1.80,
        2.00,
        2.25,
        2.50,
        3.00,
        np.inf,
    ]

    odds_labels = [
        "1.00-1.40",
        "1.40-1.60",
        "1.60-1.80",
        "1.80-2.00",
        "2.00-2.25",
        "2.25-2.50",
        "2.50-3.00",
        "3.00+",
    ]

    df["odds_bin"] = pd.cut(
        df["market_over_odds"],
        bins=odds_bins,
        labels=odds_labels,
        include_lowest=True,
    )

    odds_table = (
        df.groupby(
            "odds_bin",
            observed=False,
        )
        .agg(
            matches=("target_over25", "size"),
            actual_over_rate=(
                "target_over25",
                "mean",
            ),
            mean_model_probability=(
                "v06_over_probability",
                "mean",
            ),
            mean_market_probability=(
                "market_over_probability",
                "mean",
            ),
            mean_edge=(
                "probability_edge",
                "mean",
            ),
        )
        .reset_index()
    )

    print("\n" + "=" * 78)
    print("4. PERFORMANCE BY MARKET ODDS")
    print("=" * 78)

    print(
        odds_table.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # ---------------------------------------------------------
    # Season
    # ---------------------------------------------------------

    season_rows = []

    for season, group in df.groupby("season"):

        y_s = group["target_over25"]

        market_s = group[
            "market_over_probability"
        ]

        model_s = group[
            "v06_over_probability"
        ]

        m = evaluate(y_s, market_s)
        v = evaluate(y_s, model_s)

        season_rows.append(
            {
                "season": season,
                "matches": len(group),
                "market_brier": m["brier"],
                "v06_brier": v["brier"],
                "brier_difference":
                    v["brier"] - m["brier"],
                "market_logloss": m["logloss"],
                "v06_logloss": v["logloss"],
                "logloss_difference":
                    v["logloss"] - m["logloss"],
            }
        )

    season_table = pd.DataFrame(
        season_rows
    )

    print("\n" + "=" * 78)
    print("5. SEASON PERFORMANCE")
    print("=" * 78)

    print(
        season_table.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # ---------------------------------------------------------
    # Directional diagnostic
    # ---------------------------------------------------------

    df["model_direction_correct"] = (
        (
            df["probability_edge"] > 0
        )
        &
        (
            df["target_over25"] == 1
        )
    ) | (
        (
            df["probability_edge"] < 0
        )
        &
        (
            df["target_over25"] == 0
        )
    )

    directional_rate = (
        df["model_direction_correct"].mean()
    )

    print("\n" + "=" * 78)
    print("6. DIRECTIONAL DIAGNOSTIC")
    print("=" * 78)

    print(
        f"Directional agreement: "
        f"{directional_rate:.4%}"
    )

    print(
        "\nIMPORTANT:"
        "\nThis is NOT a betting strategy."
        "\nNo threshold has been optimized."
        "\nNo ROI conclusion should be drawn from this diagnostic."
    )


if __name__ == "__main__":
    main()
