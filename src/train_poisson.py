import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import brier_score_loss, log_loss


INPUT_FILE = Path("data/processed/model_ready.csv")
OUTPUT_FILE = Path("data/processed/poisson_predictions.csv")


FEATURES = [
    "home_goals_scored_avg",
    "home_goals_conceded_avg",
    "away_goals_scored_avg",
    "away_goals_conceded_avg",
    "home_home_goals_scored_avg",
    "home_home_goals_conceded_avg",
    "away_away_goals_scored_avg",
    "away_away_goals_conceded_avg",
    "home_recent5_scored",
    "home_recent5_conceded",
    "away_recent5_scored",
    "away_recent5_conceded",
    "home_recent10_scored",
    "home_recent10_conceded",
    "away_recent10_scored",
    "away_recent10_conceded",
    "league_goals_avg",
]


def poisson_over25_probability(lam):
    """Return P(total goals > 2.5) for Poisson(lambda)."""

    p0 = math.exp(-lam)
    p1 = p0 * lam
    p2 = p1 * lam / 2.0

    return 1.0 - (p0 + p1 + p2)


def main():
    print("\nEDGE — POISSON BASELINE")
    print("=" * 70)

    df = pd.read_csv(INPUT_FILE)

    df["date"] = pd.to_datetime(df["date"])

    # ---------------------------------------------------------
    # Remove rows that cannot support the baseline at all.
    # Other feature gaps will be handled by training-only
    # median imputation inside each walk-forward split.
    # ---------------------------------------------------------

    required = [
        "home_goals_scored_avg",
        "home_goals_conceded_avg",
        "away_goals_scored_avg",
        "away_goals_conceded_avg",
        "league_goals_avg",
    ]

    before = len(df)

    df = df.dropna(subset=required).copy()

    print(f"Original matches: {before:,}")
    print(f"Modelable matches: {len(df):,}")
    print(f"Excluded cold-start/incomplete: {before - len(df):,}")

    # ---------------------------------------------------------
    # Chronological walk-forward evaluation.
    #
    # A season is predicted only from seasons that occurred
    # before it.
    # ---------------------------------------------------------

    seasons = sorted(df["season"].unique())

    predictions = []

    for i in range(1, len(seasons)):
        test_season = seasons[i]
        train_seasons = seasons[:i]

        train = df[
            df["season"].isin(train_seasons)
        ].copy()

        test = df[
            df["season"] == test_season
        ].copy()

        if train.empty or test.empty:
            continue

        X_train = train[FEATURES]
        y_train = train["target_total_goals"]

        X_test = test[FEATURES]

        # -----------------------------------------------------
        # Fit imputer ONLY on historical training data.
        # -----------------------------------------------------

        imputer = SimpleImputer(strategy="median")

        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test)

        # -----------------------------------------------------
        # Poisson regression baseline.
        # -----------------------------------------------------

        model = PoissonRegressor(
            alpha=1.0,
            max_iter=1000,
        )

        model.fit(
            X_train_imputed,
            y_train,
        )

        predicted_lambda = model.predict(
            X_test_imputed
        )

        predicted_lambda = np.maximum(
            predicted_lambda,
            1e-6,
        )

        model_probability = np.array([
            poisson_over25_probability(lam)
            for lam in predicted_lambda
        ])

        result = test[
            [
                "match_id",
                "season",
                "date",
                "home_team_id",
                "away_team_id",
                "target_total_goals",
                "target_over25",
                "market_over_probability",
                "market_over_odds",
            ]
        ].copy()

        result["model_lambda"] = predicted_lambda

        result["model_over_probability"] = (
            model_probability
        )

        result["model_edge_vs_market"] = (
            result["model_over_probability"]
            - result["market_over_probability"]
        )

        result["model_version"] = "poisson_v0.1"

        predictions.append(result)

        print(
            f"{test_season}: "
            f"train={len(train):4d} "
            f"test={len(test):3d}"
        )

    if not predictions:
        raise RuntimeError(
            "No out-of-sample predictions were generated."
        )

    pred = pd.concat(
        predictions,
        ignore_index=True,
    )

    # ---------------------------------------------------------
    # Evaluate probability quality.
    # ---------------------------------------------------------

    y = pred["target_over25"]

    model_p = pred["model_over_probability"]

    market_p = pred["market_over_probability"]

    model_brier = brier_score_loss(
        y,
        model_p,
    )

    market_brier = brier_score_loss(
        y,
        market_p,
    )

    model_logloss = log_loss(
        y,
        np.clip(model_p, 1e-6, 1 - 1e-6),
    )

    market_logloss = log_loss(
        y,
        np.clip(market_p, 1e-6, 1 - 1e-6),
    )

    print("\n")
    print("=" * 70)
    print("OUT-OF-SAMPLE RESULTS")
    print("=" * 70)

    print(
        f"Predictions:              {len(pred):,}"
    )

    print(
        f"Model Brier score:        {model_brier:.6f}"
    )

    print(
        f"Market Brier score:       {market_brier:.6f}"
    )

    print(
        f"Model Log Loss:           {model_logloss:.6f}"
    )

    print(
        f"Market Log Loss:          {market_logloss:.6f}"
    )

    print(
        f"Mean model probability:   "
        f"{model_p.mean():.4f}"
    )

    print(
        f"Mean market probability:  "
        f"{market_p.mean():.4f}"
    )

    print(
        f"Mean actual Over rate:    "
        f"{y.mean():.4f}"
    )

    print("\nLambda summary:")

    print(
        pred["model_lambda"]
        .describe()
        .round(4)
    )

    print("\nModel probability summary:")

    print(
        pred["model_over_probability"]
        .describe()
        .round(4)
    )

    # ---------------------------------------------------------
    # Save predictions.
    # ---------------------------------------------------------

    pred.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
