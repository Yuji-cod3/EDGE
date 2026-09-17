import pandas as pd
import numpy as np
from sklearn.metrics import brier_score_loss, log_loss


INPUT_FILE = "data/processed/poisson_v02_predictions.csv"
OUTPUT_FILE = "data/processed/blended_v03_predictions.csv"


WEIGHTS = np.arange(0.0, 1.01, 0.10)


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

    print("\nEDGE — V0.3 MARKET + POISSON BLEND")
    print("=" * 70)

    df = pd.read_csv(INPUT_FILE)
    df["date"] = pd.to_datetime(df["date"])

    required = [
        "season",
        "target_over25",
        "market_over_probability",
        "model_over_probability",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=required).copy()

    df = df.sort_values(["date", "match_id"]).reset_index(drop=True)

    print(f"Predictions available: {len(df):,}")

    seasons = sorted(df["season"].unique())

    results = []

    # ---------------------------------------------------------
    # Chronological evaluation
    # ---------------------------------------------------------
    #
    # For each season, use ALL previous seasons to select
    # the blend weight, then evaluate that weight on the
    # current season.
    #
    # This prevents future-season leakage.
    # ---------------------------------------------------------

    for i, season in enumerate(seasons):

        if i == 0:
            continue

        train = df[df["season"].isin(seasons[:i])].copy()
        test = df[df["season"] == season].copy()

        if len(train) == 0 or len(test) == 0:
            continue

        print(
            f"{season}: "
            f"train={len(train):4d} "
            f"test={len(test):4d}"
        )

        y_train = train["target_over25"].values
        y_test = test["target_over25"].values

        # -----------------------------------------------------
        # Select weight using historical data only
        # -----------------------------------------------------

        weight_scores = []

        for weight in WEIGHTS:

            p_train = (
                weight * train["model_over_probability"]
                + (1 - weight) * train["market_over_probability"]
            )

            metrics = evaluate(y_train, p_train)

            weight_scores.append(
                {
                    "weight": weight,
                    "brier": metrics["brier"],
                    "logloss": metrics["logloss"],
                }
            )

        weight_table = pd.DataFrame(weight_scores)

        # Select using Brier score only.
        # Log loss remains an independent diagnostic.
        best_row = weight_table.loc[
            weight_table["brier"].idxmin()
        ]

        best_weight = float(best_row["weight"])

        # -----------------------------------------------------
        # Evaluate on completely unseen season
        # -----------------------------------------------------

        p_model = test["model_over_probability"].values
        p_market = test["market_over_probability"].values

        p_blend = (
            best_weight * p_model
            + (1 - best_weight) * p_market
        )

        p_blend = clip_probability(p_blend)

        season_result = test[
            [
                "match_id",
                "season",
                "date",
                "home_team_id",
                "away_team_id",
                "target_over25",
                "market_over_probability",
                "market_over_odds",
                "model_over_probability",
            ]
        ].copy()

        season_result["blend_model_weight"] = best_weight
        season_result["blend_market_weight"] = 1 - best_weight
        season_result["blend_over_probability"] = p_blend

        results.append(season_result)

        print(
            f"  selected model weight: {best_weight:.2f}"
        )

    if not results:
        raise RuntimeError("No out-of-sample predictions generated.")

    predictions = pd.concat(
        results,
        ignore_index=True
    )

    # ---------------------------------------------------------
    # Overall results
    # ---------------------------------------------------------

    y = predictions["target_over25"].values

    p_model = predictions["model_over_probability"].values
    p_market = predictions["market_over_probability"].values
    p_blend = predictions["blend_over_probability"].values

    model_metrics = evaluate(y, p_model)
    market_metrics = evaluate(y, p_market)
    blend_metrics = evaluate(y, p_blend)

    print("\n")
    print("=" * 70)
    print("OUT-OF-SAMPLE RESULTS")
    print("=" * 70)

    print(f"Predictions:              {len(predictions):,}")

    print("\nMODEL V0.2")
    print(f"Brier score:              {model_metrics['brier']:.6f}")
    print(f"Log Loss:                 {model_metrics['logloss']:.6f}")
    print(f"MAE:                      {model_metrics['mae']:.6f}")

    print("\nMARKET")
    print(f"Brier score:              {market_metrics['brier']:.6f}")
    print(f"Log Loss:                 {market_metrics['logloss']:.6f}")
    print(f"MAE:                      {market_metrics['mae']:.6f}")

    print("\nBLENDED V0.3")
    print(f"Brier score:              {blend_metrics['brier']:.6f}")
    print(f"Log Loss:                 {blend_metrics['logloss']:.6f}")
    print(f"MAE:                      {blend_metrics['mae']:.6f}")

    print("\nIMPROVEMENT VS MARKET")

    print(
        f"Brier improvement:       "
        f"{market_metrics['brier'] - blend_metrics['brier']:+.6f}"
    )

    print(
        f"Log Loss improvement:    "
        f"{market_metrics['logloss'] - blend_metrics['logloss']:+.6f}"
    )

    print(
        f"MAE improvement:         "
        f"{market_metrics['mae'] - blend_metrics['mae']:+.6f}"
    )

    print("\nSELECTED MODEL WEIGHTS")

    print(
        predictions[
            [
                "season",
                "blend_model_weight",
                "blend_market_weight",
            ]
        ]
        .drop_duplicates()
        .to_string(index=False)
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    predictions.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
