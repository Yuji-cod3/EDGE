import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, log_loss


INPUT_FILE = "data/processed/model_ready.csv"
OUTPUT_FILE = "data/processed/logistic_v05_predictions.csv"


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


def clip_probability(p):
    return np.clip(p, 1e-6, 1 - 1e-6)


def evaluate(y, p):
    p = clip_probability(p)

    return {
        "brier": brier_score_loss(y, p),
        "logloss": log_loss(y, p),
        "mae": np.mean(np.abs(y - p)),
    }


def build_pipeline():

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    C=1.0,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def main():

    print("\nEDGE — V0.5 LOGISTIC REGRESSION")
    print("=" * 70)

    df = pd.read_csv(INPUT_FILE)

    df["date"] = pd.to_datetime(df["date"])

    required = [
        "match_id",
        "season",
        "date",
        "target_over25",
        "market_over_probability",
    ] + FEATURES

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df = df.sort_values(
        ["date", "match_id"]
    ).reset_index(drop=True)

    seasons = sorted(df["season"].unique())

    predictions = []

    for i, season in enumerate(seasons):

        if i == 0:
            continue

        train = df[
            df["season"].isin(seasons[:i])
        ].copy()

        test = df[
            df["season"] == season
        ].copy()

        if len(train) == 0 or len(test) == 0:
            continue

        print(
            f"{season}: "
            f"train={len(train):4d} "
            f"test={len(test):4d}"
        )

        X_train = train[FEATURES]
        y_train = train["target_over25"]

        X_test = test[FEATURES]
        y_test = test["target_over25"].values

        model = build_pipeline()

        model.fit(X_train, y_train)

        feature_probability = model.predict_proba(
            X_test
        )[:, 1]

        feature_probability = clip_probability(
            feature_probability
        )

        market_probability = clip_probability(
            test["market_over_probability"].values
        )

        feature_metrics = evaluate(
            y_test,
            feature_probability,
        )

        market_metrics = evaluate(
            y_test,
            market_probability,
        )

        print(
            f"  Feature Brier: "
            f"{feature_metrics['brier']:.6f}"
        )

        print(
            f"  Market Brier:  "
            f"{market_metrics['brier']:.6f}"
        )

        result = test[
            [
                "match_id",
                "season",
                "date",
                "home_team_id",
                "away_team_id",
                "target_over25",
                "market_over_probability",
                "market_over_odds",
            ]
        ].copy()

        result[
            "logistic_feature_probability"
        ] = feature_probability

        predictions.append(result)

    if not predictions:
        raise RuntimeError(
            "No out-of-sample predictions generated."
        )

    output = pd.concat(
        predictions,
        ignore_index=True,
    )

    y = output["target_over25"].values

    p_feature = output[
        "logistic_feature_probability"
    ].values

    p_market = output[
        "market_over_probability"
    ].values

    feature_metrics = evaluate(y, p_feature)
    market_metrics = evaluate(y, p_market)

    print("\n")
    print("=" * 70)
    print("OVERALL OUT-OF-SAMPLE RESULTS")
    print("=" * 70)

    print(f"Predictions: {len(output):,}")

    print("\nLOGISTIC — FEATURES ONLY")
    print(
        f"Brier score:  "
        f"{feature_metrics['brier']:.6f}"
    )
    print(
        f"Log Loss:     "
        f"{feature_metrics['logloss']:.6f}"
    )
    print(
        f"MAE:          "
        f"{feature_metrics['mae']:.6f}"
    )

    print("\nMARKET")
    print(
        f"Brier score:  "
        f"{market_metrics['brier']:.6f}"
    )
    print(
        f"Log Loss:     "
        f"{market_metrics['logloss']:.6f}"
    )
    print(
        f"MAE:          "
        f"{market_metrics['mae']:.6f}"
    )

    print("\nDIFFERENCE — LOGISTIC MINUS MARKET")

    print(
        f"Brier:        "
        f"{feature_metrics['brier'] - market_metrics['brier']:+.6f}"
    )

    print(
        f"Log Loss:     "
        f"{feature_metrics['logloss'] - market_metrics['logloss']:+.6f}"
    )

    print(
        f"MAE:          "
        f"{feature_metrics['mae'] - market_metrics['mae']:+.6f}"
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
