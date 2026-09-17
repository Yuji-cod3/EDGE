import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss


INPUT_FILE = "data/processed/blended_v03_predictions.csv"
OUTPUT_FILE = "data/processed/market_calibration_v04_predictions.csv"


def clip_probability(p):
    return np.clip(p, 1e-6, 1 - 1e-6)


def evaluate(y, p):
    p = clip_probability(p)

    return {
        "brier": brier_score_loss(y, p),
        "logloss": log_loss(y, p),
        "mae": np.mean(np.abs(y - p)),
    }


def logistic_calibration(train_p, train_y, test_p):
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
    )

    X_train = train_p.reshape(-1, 1)
    X_test = test_p.reshape(-1, 1)

    model.fit(X_train, train_y)

    return model.predict_proba(X_test)[:, 1]


def isotonic_calibration(train_p, train_y, test_p):
    model = IsotonicRegression(
        y_min=1e-6,
        y_max=1 - 1e-6,
        out_of_bounds="clip",
    )

    model.fit(train_p, train_y)

    return model.predict(test_p)


def main():

    print("\nEDGE — V0.4 MARKET CALIBRATION")
    print("=" * 70)

    df = pd.read_csv(INPUT_FILE)

    df["date"] = pd.to_datetime(df["date"])

    required = [
        "season",
        "date",
        "match_id",
        "target_over25",
        "market_over_probability",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df = df.dropna(subset=required).copy()

    df = df.sort_values(
        ["date", "match_id"]
    ).reset_index(drop=True)

    print(f"Predictions available: {len(df):,}")

    seasons = sorted(df["season"].unique())

    results = []

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

        train_p = train[
            "market_over_probability"
        ].values

        train_y = train[
            "target_over25"
        ].values

        test_p = test[
            "market_over_probability"
        ].values

        test_y = test[
            "target_over25"
        ].values

        # -----------------------------------------------------
        # RAW MARKET
        # -----------------------------------------------------

        raw_p = clip_probability(test_p)

        # -----------------------------------------------------
        # LOGISTIC CALIBRATION
        # -----------------------------------------------------

        logistic_p = logistic_calibration(
            train_p,
            train_y,
            test_p,
        )

        # -----------------------------------------------------
        # ISOTONIC CALIBRATION
        # -----------------------------------------------------

        isotonic_p = isotonic_calibration(
            train_p,
            train_y,
            test_p,
        )

        raw_metrics = evaluate(
            test_y,
            raw_p,
        )

        logistic_metrics = evaluate(
            test_y,
            logistic_p,
        )

        isotonic_metrics = evaluate(
            test_y,
            isotonic_p,
        )

        print(
            f"  Raw market Brier:       "
            f"{raw_metrics['brier']:.6f}"
        )

        print(
            f"  Logistic Brier:         "
            f"{logistic_metrics['brier']:.6f}"
        )

        print(
            f"  Isotonic Brier:         "
            f"{isotonic_metrics['brier']:.6f}"
        )

        season_result = test[
            [
                "match_id",
                "season",
                "date",
                "target_over25",
                "market_over_probability",
            ]
        ].copy()

        season_result[
            "market_raw_probability"
        ] = raw_p

        season_result[
            "market_logistic_probability"
        ] = logistic_p

        season_result[
            "market_isotonic_probability"
        ] = isotonic_p

        results.append(season_result)

    if not results:
        raise RuntimeError(
            "No out-of-sample predictions generated."
        )

    predictions = pd.concat(
        results,
        ignore_index=True,
    )

    y = predictions[
        "target_over25"
    ].values

    raw_p = predictions[
        "market_raw_probability"
    ].values

    logistic_p = predictions[
        "market_logistic_probability"
    ].values

    isotonic_p = predictions[
        "market_isotonic_probability"
    ].values

    raw_metrics = evaluate(y, raw_p)
    logistic_metrics = evaluate(y, logistic_p)
    isotonic_metrics = evaluate(y, isotonic_p)

    print("\n")
    print("=" * 70)
    print("OUT-OF-SAMPLE RESULTS")
    print("=" * 70)

    print("\nRAW MARKET")
    print(
        f"Brier score:              "
        f"{raw_metrics['brier']:.6f}"
    )
    print(
        f"Log Loss:                 "
        f"{raw_metrics['logloss']:.6f}"
    )
    print(
        f"MAE:                      "
        f"{raw_metrics['mae']:.6f}"
    )

    print("\nLOGISTIC CALIBRATION")
    print(
        f"Brier score:              "
        f"{logistic_metrics['brier']:.6f}"
    )
    print(
        f"Log Loss:                 "
        f"{logistic_metrics['logloss']:.6f}"
    )
    print(
        f"MAE:                      "
        f"{logistic_metrics['mae']:.6f}"
    )

    print("\nISOTONIC CALIBRATION")
    print(
        f"Brier score:              "
        f"{isotonic_metrics['brier']:.6f}"
    )
    print(
        f"Log Loss:                 "
        f"{isotonic_metrics['logloss']:.6f}"
    )
    print(
        f"MAE:                      "
        f"{isotonic_metrics['mae']:.6f}"
    )

    print("\n")
    print("=" * 70)
    print("IMPROVEMENT VS RAW MARKET")
    print("=" * 70)

    print(
        f"Logistic Brier improvement: "
        f"{raw_metrics['brier'] - logistic_metrics['brier']:+.6f}"
    )

    print(
        f"Logistic LogLoss improvement: "
        f"{raw_metrics['logloss'] - logistic_metrics['logloss']:+.6f}"
    )

    print(
        f"Isotonic Brier improvement: "
        f"{raw_metrics['brier'] - isotonic_metrics['brier']:+.6f}"
    )

    print(
        f"Isotonic LogLoss improvement: "
        f"{raw_metrics['logloss'] - isotonic_metrics['logloss']:+.6f}"
    )

    predictions.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
