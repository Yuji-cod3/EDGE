import pandas as pd
import numpy as np
from sklearn.metrics import brier_score_loss, log_loss


INPUT_FILE = "data/processed/logistic_v06_predictions.csv"
OUTPUT_FILE = "data/processed/shrinkage_v07_predictions.csv"

WEIGHTS = [0.00, 0.10, 0.25, 0.50, 0.75, 1.00]


def evaluate(y_true, probabilities):
    probabilities = np.clip(
        probabilities,
        1e-6,
        1 - 1e-6,
    )

    return {
        "brier": brier_score_loss(
            y_true,
            probabilities,
        ),
        "logloss": log_loss(
            y_true,
            probabilities,
        ),
        "mae": np.mean(
            np.abs(
                y_true - probabilities
            )
        ),
    }


def main():

    print("\nEDGE — V0.7 MARKET-ANCHORED SHRINKAGE")
    print("=" * 70)

    df = pd.read_csv(INPUT_FILE)

    required = [
        "match_id",
        "season",
        "date",
        "target_over25",
        "market_over_probability",
        "v06_over_probability",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df = df[required].dropna().copy()

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(
        ["date", "match_id"]
    ).reset_index(drop=True)

    print(
        f"Predictions available: {len(df):,}"
    )

    y = df["target_over25"].astype(int)

    # ---------------------------------------------------------
    # FIXED SHRINKAGE TEST
    # ---------------------------------------------------------

    results = []

    for alpha in WEIGHTS:

        market_weight = 1.0 - alpha

        probability = (
            alpha * df["v06_over_probability"]
            + market_weight
            * df["market_over_probability"]
        )

        metrics = evaluate(
            y,
            probability,
        )

        results.append(
            {
                "model_weight": alpha,
                "market_weight": market_weight,
                "brier": metrics["brier"],
                "logloss": metrics["logloss"],
                "mae": metrics["mae"],
            }
        )

    results_df = pd.DataFrame(results)

    print("\n")
    print("=" * 70)
    print("FIXED SHRINKAGE RESULTS")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # ---------------------------------------------------------
    # BEST WEIGHTS
    # ---------------------------------------------------------

    best_brier = results_df.loc[
        results_df["brier"].idxmin()
    ]

    best_logloss = results_df.loc[
        results_df["logloss"].idxmin()
    ]

    best_mae = results_df.loc[
        results_df["mae"].idxmin()
    ]

    print("\n")
    print("=" * 70)
    print("BEST FIXED WEIGHTS")
    print("=" * 70)

    print(
        f"Brier:   alpha={best_brier['model_weight']:.2f} "
        f"(Brier={best_brier['brier']:.6f})"
    )

    print(
        f"LogLoss: alpha={best_logloss['model_weight']:.2f} "
        f"(LogLoss={best_logloss['logloss']:.6f})"
    )

    print(
        f"MAE:     alpha={best_mae['model_weight']:.2f} "
        f"(MAE={best_mae['mae']:.6f})"
    )

    # ---------------------------------------------------------
    # SEASON-BY-SEASON ANALYSIS
    # ---------------------------------------------------------

    season_results = []

    for season, group in df.groupby(
        "season",
        sort=True,
    ):

        y_season = group[
            "target_over25"
        ].astype(int)

        market_probability = group[
            "market_over_probability"
        ].to_numpy()

        v06_probability = group[
            "v06_over_probability"
        ].to_numpy()

        for alpha in WEIGHTS:

            probability = (
                alpha * v06_probability
                + (1.0 - alpha)
                * market_probability
            )

            metrics = evaluate(
                y_season,
                probability,
            )

            season_results.append(
                {
                    "season": season,
                    "model_weight": alpha,
                    "market_weight": 1.0 - alpha,
                    "matches": len(group),
                    "brier": metrics["brier"],
                    "logloss": metrics["logloss"],
                    "mae": metrics["mae"],
                }
            )

    season_df = pd.DataFrame(
        season_results
    )

    print("\n")
    print("=" * 70)
    print("SEASON-BY-SEASON BRIER")
    print("=" * 70)

    pivot = season_df.pivot(
        index="season",
        columns="model_weight",
        values="brier",
    )

    print(
        pivot.to_string(
            float_format=lambda x: f"{x:.6f}"
        )
    )

    # ---------------------------------------------------------
    # MARKET COMPARISON
    # ---------------------------------------------------------

    market = results_df[
        results_df["model_weight"] == 0.0
    ].iloc[0]

    best = results_df.loc[
        results_df["brier"].idxmin()
    ]

    print("\n")
    print("=" * 70)
    print("DIAGNOSTIC ASSESSMENT")
    print("=" * 70)

    if best["model_weight"] == 0.0:

        print(
            "Market-only remains best among "
            "all tested shrinkage weights."
        )

        print(
            "V0.6 provides no measurable improvement "
            "through fixed shrinkage."
        )

    else:

        improvement = (
            market["brier"]
            - best["brier"]
        )

        print(
            f"Best fixed model weight: "
            f"alpha={best['model_weight']:.2f}"
        )

        if improvement > 0:

            print(
                f"Brier improvement over market: "
                f"{improvement:.6f}"
            )

        else:

            print(
                "Best shrinkage does not improve "
                "the market."
            )

    print("\nIMPORTANT:")
    print(
        "These are fixed-weight diagnostics only."
    )

    print(
        "No betting threshold has been optimized."
    )

    print(
        "No ROI conclusion should be drawn."
    )

    print(
        "The best fixed weight is NOT automatically "
        "a deployable model."
    )

    # ---------------------------------------------------------
    # SAVE ALL SHRINKAGE PREDICTIONS
    # ---------------------------------------------------------

    output = df.copy()

    for alpha in WEIGHTS:

        column = (
            f"v07_over_probability_{alpha:.2f}"
        )

        output[column] = np.clip(
            alpha
            * output["v06_over_probability"]
            + (1.0 - alpha)
            * output["market_over_probability"],
            1e-6,
            1 - 1e-6,
        )

    output.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
