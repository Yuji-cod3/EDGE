import pandas as pd
import numpy as np


INPUT_FILE = "data/processed/poisson_v02_predictions.csv"


def calibration_table(df, probability_column, label):

    temp = df[
        [
            probability_column,
            "target_over25",
        ]
    ].dropna().copy()

    # Probability bins from 0.00 to 1.00
    bins = np.arange(0.0, 1.01, 0.05)

    temp["bin"] = pd.cut(
        temp[probability_column],
        bins=bins,
        include_lowest=True,
    )

    table = (
        temp
        .groupby(
            "bin",
            observed=False
        )
        .agg(
            predictions=(probability_column, "size"),
            mean_predicted=(
                probability_column,
                "mean",
            ),
            actual_over_rate=(
                "target_over25",
                "mean",
            ),
        )
        .reset_index()
    )

    table["calibration_error"] = (
        table["actual_over_rate"]
        - table["mean_predicted"]
    )

    table["abs_calibration_error"] = (
        table["calibration_error"].abs()
    )

    table["model"] = label

    return table


def overall_metrics(df, probability_column):

    temp = df[
        [
            probability_column,
            "target_over25",
        ]
    ].dropna()

    mae = np.mean(
        np.abs(
            temp["target_over25"]
            - temp[probability_column]
        )
    )

    brier = np.mean(
        (
            temp["target_over25"]
            - temp[probability_column]
        ) ** 2
    )

    return mae, brier


def main():

    print("\nEDGE — V0.2 CALIBRATION AUDIT")
    print("=" * 80)

    df = pd.read_csv(INPUT_FILE)

    print(f"Predictions: {len(df):,}")

    model = calibration_table(
        df,
        "model_over_probability",
        "POISSON_V0.2",
    )

    market = calibration_table(
        df,
        "market_over_probability",
        "MARKET",
    )

    combined = pd.concat(
        [model, market],
        ignore_index=True,
    )

    pd.set_option(
        "display.max_columns",
        None,
    )

    print("\nCALIBRATION BY PROBABILITY RANGE")
    print("=" * 80)

    print(
        combined[
            [
                "model",
                "bin",
                "predictions",
                "mean_predicted",
                "actual_over_rate",
                "calibration_error",
                "abs_calibration_error",
            ]
        ].to_string(index=False)
    )

    print("\nOVERALL PROBABILITY METRICS")
    print("=" * 80)

    results = []

    for label, probability_column in [
        ("POISSON_V0.2", "model_over_probability"),
        ("MARKET", "market_over_probability"),
    ]:

        mae, brier = overall_metrics(
            df,
            probability_column,
        )

        results.append(
            {
                "model": label,
                "MAE": mae,
                "Brier": brier,
            }
        )

        print(
            f"{label:<15} "
            f"MAE={mae:.6f}  "
            f"Brier={brier:.6f}"
        )

    print("\nCALIBRATION SUMMARY")
    print("=" * 80)

    for label in ["POISSON_V0.2", "MARKET"]:

        subset = combined[
            combined["model"] == label
        ]

        subset = subset[
            subset["predictions"] > 0
        ]

        weighted_calibration_error = np.average(
            subset["abs_calibration_error"],
            weights=subset["predictions"],
        )

        print(
            f"{label:<15} "
            f"Weighted absolute calibration error: "
            f"{weighted_calibration_error:.6f}"
        )

    print("\nSTATUS")
    print("=" * 80)
    print(
        "Calibration audit completed for V0.2."
    )


if __name__ == "__main__":
    main()
