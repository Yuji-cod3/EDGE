import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, log_loss


INPUT_FILE = "data/processed/model_ready.csv"
OUTPUT_FILE = "data/processed/logistic_v06_predictions.csv"

TARGET = "target_over25"

MARKET_FEATURES = [
    "market_over_probability",
    "market_overround",
]

FOOTBALL_FEATURES = [
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

# V0.5-equivalent feature set
FEATURES_ONLY = FOOTBALL_FEATURES

# V0.6 = market + football information
COMBINED_FEATURES = MARKET_FEATURES + FOOTBALL_FEATURES


def clip_probability(p):
    return np.clip(p, 1e-6, 1 - 1e-6)


def market_logit(p):
    p = clip_probability(p)
    return np.log(p / (1.0 - p))


def build_model():
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    C=1.0,
                    random_state=42,
                ),
            ),
        ]
    )


def evaluate(y, probabilities):
    probabilities = clip_probability(probabilities)

    return {
        "brier": brier_score_loss(y, probabilities),
        "logloss": log_loss(y, probabilities),
        "mae": np.mean(np.abs(y - probabilities)),
    }


def main():

    print("\nEDGE — V0.6 MARKET + FEATURES LOGISTIC")
    print("=" * 70)

    df = pd.read_csv(INPUT_FILE)

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(
        ["date", "match_id"]
    ).reset_index(drop=True)

    required = (
        [TARGET]
        + FEATURES_ONLY
        + MARKET_FEATURES
        + [
            "season",
            "match_id",
            "date",
        ]
    )

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # Remove rows without a target.
    df = df.dropna(
        subset=[TARGET]
    ).copy()

    seasons = sorted(
        df["season"].unique()
    )

    all_predictions = []
    season_results = []

    # ---------------------------------------------------------
    # Walk-forward evaluation
    # ---------------------------------------------------------

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
            f"train={len(train):4d} "
            f"test={len(test):4d}"
        )

        y_train = train[TARGET].astype(int)
        y_test = test[TARGET].astype(int)

        # -----------------------------------------------------
        # 1. MARKET BASELINE
        # -----------------------------------------------------

        market_probability = clip_probability(
            test["market_over_probability"].to_numpy()
        )

        market_metrics = evaluate(
            y_test,
            market_probability,
        )

        # -----------------------------------------------------
        # 2. FEATURES ONLY
        # -----------------------------------------------------

        feature_model = build_model()

        feature_model.fit(
            train[FEATURES_ONLY],
            y_train,
        )

        feature_probability = feature_model.predict_proba(
            test[FEATURES_ONLY]
        )[:, 1]

        feature_metrics = evaluate(
            y_test,
            feature_probability,
        )

        # -----------------------------------------------------
        # 3. MARKET + FEATURES
        # -----------------------------------------------------

        combined_model = build_model()

        combined_model.fit(
            train[COMBINED_FEATURES],
            y_train,
        )

        combined_probability = combined_model.predict_proba(
            test[COMBINED_FEATURES]
        )[:, 1]

        combined_metrics = evaluate(
            y_test,
            combined_probability,
        )

        print(
            f"  Market Brier:   "
            f"{market_metrics['brier']:.6f}"
        )

        print(
            f"  Features Brier: "
            f"{feature_metrics['brier']:.6f}"
        )

        print(
            f"  V0.6 Brier:     "
            f"{combined_metrics['brier']:.6f}"
        )

        # -----------------------------------------------------
        # Store predictions
        # -----------------------------------------------------

        output = test[
            [
                "match_id",
                "season",
                "date",
                "home_team_id",
                "away_team_id",
                TARGET,
            ]
        ].copy()

        output["market_over_probability"] = (
            market_probability
        )

        output["feature_over_probability"] = (
            feature_probability
        )

        output["v06_over_probability"] = (
            combined_probability
        )

        output["market_over_odds"] = (
            test["market_over_odds"].to_numpy()
        )

        output["market_overround"] = (
            test["market_overround"].to_numpy()
        )

        output["model_version"] = "V0.6"

        all_predictions.append(output)

        season_results.append(
            {
                "season": test_season,
                "matches": len(test),

                "market_brier":
                    market_metrics["brier"],

                "feature_brier":
                    feature_metrics["brier"],

                "v06_brier":
                    combined_metrics["brier"],

                "market_logloss":
                    market_metrics["logloss"],

                "feature_logloss":
                    feature_metrics["logloss"],

                "v06_logloss":
                    combined_metrics["logloss"],

                "market_mae":
                    market_metrics["mae"],

                "feature_mae":
                    feature_metrics["mae"],

                "v06_mae":
                    combined_metrics["mae"],
            }
        )

    # ---------------------------------------------------------
    # Combine predictions
    # ---------------------------------------------------------

    predictions = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    predictions.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ---------------------------------------------------------
    # Overall evaluation
    # ---------------------------------------------------------

    y = predictions[TARGET].astype(int)

    market = predictions[
        "market_over_probability"
    ]

    features = predictions[
        "feature_over_probability"
    ]

    v06 = predictions[
        "v06_over_probability"
    ]

    market_metrics = evaluate(y, market)
    feature_metrics = evaluate(y, features)
    v06_metrics = evaluate(y, v06)

    print("\n")
    print("=" * 70)
    print("OVERALL OUT-OF-SAMPLE RESULTS")
    print("=" * 70)

    print(
        f"Predictions: {len(predictions):6d}"
    )

    print("\nMARKET")
    print(
        f"Brier score: {market_metrics['brier']:.6f}"
    )
    print(
        f"Log Loss:    {market_metrics['logloss']:.6f}"
    )
    print(
        f"MAE:         {market_metrics['mae']:.6f}"
    )

    print("\nFEATURES ONLY — V0.5")
    print(
        f"Brier score: {feature_metrics['brier']:.6f}"
    )
    print(
        f"Log Loss:    {feature_metrics['logloss']:.6f}"
    )
    print(
        f"MAE:         {feature_metrics['mae']:.6f}"
    )

    print("\nMARKET + FEATURES — V0.6")
    print(
        f"Brier score: {v06_metrics['brier']:.6f}"
    )
    print(
        f"Log Loss:    {v06_metrics['logloss']:.6f}"
    )
    print(
        f"MAE:         {v06_metrics['mae']:.6f}"
    )

    print("\n")
    print("=" * 70)
    print("V0.6 IMPROVEMENT VS MARKET")
    print("=" * 70)

    print(
        f"Brier improvement: "
        f"{market_metrics['brier'] - v06_metrics['brier']:+.6f}"
    )

    print(
        f"LogLoss improvement: "
        f"{market_metrics['logloss'] - v06_metrics['logloss']:+.6f}"
    )

    print(
        f"MAE improvement: "
        f"{market_metrics['mae'] - v06_metrics['mae']:+.6f}"
    )

    print("\n")
    print("=" * 70)
    print("SEASON-BY-SEASON RESULTS")
    print("=" * 70)

    season_df = pd.DataFrame(
        season_results
    )

    print(
        season_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
