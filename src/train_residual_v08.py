import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error


INPUT_FILE = "data/processed/logistic_v06_predictions.csv"
OUTPUT_FILE = "data/processed/residual_v08_predictions.csv"

MIN_PROB = 0.001
MAX_PROB = 0.999

FEATURES = [
    "feature_over_probability",
    "v06_over_probability",
    "market_over_probability",
]


def clip_probability(p):
    return np.clip(p, MIN_PROB, MAX_PROB)


def metrics(y, p):
    p = clip_probability(p)

    return {
        "brier": brier_score_loss(y, p),
        "logloss": log_loss(y, p),
        "mae": mean_absolute_error(y, p),
    }


def main():

    print("\nEDGE — V0.8 MARKET RESIDUAL MODEL")
    print("=" * 70)

    df = pd.read_csv(INPUT_FILE)

    df = df.sort_values(["season", "date", "match_id"]).reset_index(drop=True)

    required = [
        "season",
        "target_over25",
        "market_over_probability",
        "feature_over_probability",
        "v06_over_probability",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    predictions = []

    seasons = sorted(df["season"].unique())

    # We need at least one previous season for training.
    for season in seasons[1:]:

        train = df[df["season"] < season].copy()
        test = df[df["season"] == season].copy()

        print(
            f"{season}: train={len(train):4d} "
            f"test={len(test):4d}"
        )

        X_train = train[FEATURES].copy()
        y_train = train["target_over25"].astype(int)

        X_test = test[FEATURES].copy()

        # Logistic model learns the relationship between
        # market/features and the actual outcome.
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "logistic",
                    LogisticRegression(
                        C=0.1,
                        max_iter=2000,
                        random_state=42,
                    ),
                ),
            ]
        )

        model.fit(X_train, y_train)

        feature_probability = model.predict_proba(X_test)[:, 1]

        market_probability = test["market_over_probability"].values

        # Residual correction.
        #
        # Rather than replacing the market probability,
        # estimate how much the model should move it.
        raw_delta = feature_probability - market_probability

        # Conservative correction.
        #
        # The residual model is deliberately prevented from
        # completely overriding the market.
        correction = 0.50 * raw_delta

        residual_probability = clip_probability(
            market_probability + correction
        )

        out = test[
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

        out["feature_probability"] = feature_probability
        out["raw_residual"] = raw_delta
        out["v08_over_probability"] = residual_probability
        out["model_version"] = "V0.8"

        predictions.append(out)

    result = pd.concat(predictions, ignore_index=True)

    y = result["target_over25"]

    market = result["market_over_probability"]
    v08 = result["v08_over_probability"]

    market_metrics = metrics(y, market)
    v08_metrics = metrics(y, v08)

    print("\n")
    print("=" * 70)
    print("OVERALL OUT-OF-SAMPLE RESULTS")
    print("=" * 70)

    print(f"Predictions: {len(result):,}")

    print("\nMARKET")
    print(f"Brier score: {market_metrics['brier']:.6f}")
    print(f"Log Loss:    {market_metrics['logloss']:.6f}")
    print(f"MAE:         {market_metrics['mae']:.6f}")

    print("\nV0.8 RESIDUAL")
    print(f"Brier score: {v08_metrics['brier']:.6f}")
    print(f"Log Loss:    {v08_metrics['logloss']:.6f}")
    print(f"MAE:         {v08_metrics['mae']:.6f}")

    print("\n")
    print("=" * 70)
    print("V0.8 IMPROVEMENT VS MARKET")
    print("=" * 70)

    print(
        f"Brier improvement: "
        f"{market_metrics['brier'] - v08_metrics['brier']:+.6f}"
    )

    print(
        f"LogLoss improvement: "
        f"{market_metrics['logloss'] - v08_metrics['logloss']:+.6f}"
    )

    print(
        f"MAE improvement: "
        f"{market_metrics['mae'] - v08_metrics['mae']:+.6f}"
    )

    print("\n")
    print("=" * 70)
    print("RESIDUAL SUMMARY")
    print("=" * 70)

    print(
        result[
            [
                "raw_residual",
                "market_over_probability",
                "feature_probability",
                "v08_over_probability",
            ]
        ].describe().to_string()
    )

    print("\n")
    print("=" * 70)
    print("SEASON-BY-SEASON RESULTS")
    print("=" * 70)

    rows = []

    for season, group in result.groupby("season"):

        y_s = group["target_over25"]

        m = metrics(
            y_s,
            group["market_over_probability"],
        )

        v = metrics(
            y_s,
            group["v08_over_probability"],
        )

        rows.append(
            {
                "season": season,
                "matches": len(group),
                "market_brier": m["brier"],
                "v08_brier": v["brier"],
                "brier_difference": v["brier"] - m["brier"],
                "market_logloss": m["logloss"],
                "v08_logloss": v["logloss"],
                "logloss_difference": v["logloss"] - m["logloss"],
            }
        )

    season_df = pd.DataFrame(rows)

    print(season_df.to_string(index=False))

    print("\n")
    print("=" * 70)
    print("DIAGNOSTIC ASSESSMENT")
    print("=" * 70)

    if v08_metrics["brier"] < market_metrics["brier"]:
        print("✓ V0.8 improves Brier score over market.")
    else:
        print("✗ V0.8 does NOT improve Brier score over market.")

    if v08_metrics["logloss"] < market_metrics["logloss"]:
        print("✓ V0.8 improves Log Loss over market.")
    else:
        print("✗ V0.8 does NOT improve Log Loss over market.")

    print("\nIMPORTANT:")
    print("This is a residual diagnostic, not a betting strategy.")
    print("No betting threshold has been optimized.")
    print("No ROI conclusion should be drawn.")
    print("Do not deploy V0.8 for real-money betting.")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    result.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
