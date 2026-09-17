import os
import numpy as np
import pandas as pd


INPUT_FILE = "data/processed/residual_v08_predictions.csv"
OUTPUT_FILE = "reports/market_residual_v09.csv"

Z = 1.96


def residual_stats(group):
    n = len(group)

    residual = group["market_residual"]

    mean_residual = residual.mean()
    mean_abs_residual = residual.abs().mean()

    if n > 1:
        std = residual.std(ddof=1)
        se = std / np.sqrt(n)
    else:
        std = np.nan
        se = np.nan

    if pd.notna(se):
        ci_low = mean_residual - Z * se
        ci_high = mean_residual + Z * se
    else:
        ci_low = np.nan
        ci_high = np.nan

    significant = (
        pd.notna(ci_low)
        and pd.notna(ci_high)
        and not (ci_low <= 0 <= ci_high)
    )

    return pd.Series(
        {
            "matches": n,
            "mean_market_probability":
                group["market_over_probability"].mean(),
            "actual_over_rate":
                group["target_over25"].mean(),
            "mean_residual": mean_residual,
            "mean_abs_residual": mean_abs_residual,
            "residual_std": std,
            "standard_error": se,
            "ci_95_low": ci_low,
            "ci_95_high": ci_high,
            "significant_95": significant,
        }
    )


def make_bucket_analysis(df, column, bins, labels=None):

    temp = df.copy()

    temp["bucket"] = pd.cut(
        temp[column],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    result = (
        temp.groupby("bucket", observed=False)
        .apply(residual_stats, include_groups=False)
        .reset_index()
    )

    result["variable"] = column

    return result


def print_section(title, table):

    print("\n")
    print("=" * 80)
    print(title)
    print("=" * 80)

    display_columns = [
        "bucket",
        "matches",
        "mean_market_probability",
        "actual_over_rate",
        "mean_residual",
        "mean_abs_residual",
        "ci_95_low",
        "ci_95_high",
        "significant_95",
    ]

    print(
        table[display_columns]
        .to_string(index=False)
    )


def main():

    print("\nEDGE — V0.9 MARKET RESIDUAL ANALYSIS")
    print("=" * 80)

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required = [
        "target_over25",
        "market_over_probability",
        "market_over_odds",
        "feature_probability",
        "v08_over_probability",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    print(f"Predictions available: {len(df):,}")

    # ---------------------------------------------------------
    # Market residual
    # ---------------------------------------------------------

    df["market_residual"] = (
        df["target_over25"]
        - df["market_over_probability"]
    )

    df["model_market_difference"] = (
        df["feature_probability"]
        - df["market_over_probability"]
    )

    # ---------------------------------------------------------
    # Model lambda is not present in V0.8 output.
    #
    # We therefore use the available probability difference
    # rather than inventing a lambda variable.
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # 1. MARKET PROBABILITY
    # ---------------------------------------------------------

    probability_bins = [
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
        1.00,
    ]

    probability_table = make_bucket_analysis(
        df,
        "market_over_probability",
        probability_bins,
    )

    print_section(
        "1. MARKET PROBABILITY RESIDUALS",
        probability_table,
    )

    # ---------------------------------------------------------
    # 2. MARKET ODDS
    # ---------------------------------------------------------

    odds_bins = [
        1.00,
        1.40,
        1.60,
        1.80,
        2.00,
        2.25,
        2.50,
        3.00,
        10.00,
    ]

    odds_table = make_bucket_analysis(
        df,
        "market_over_odds",
        odds_bins,
    )

    print_section(
        "2. MARKET ODDS RESIDUALS",
        odds_table,
    )

    # ---------------------------------------------------------
    # 3. MODEL-MARKET DISAGREEMENT
    # ---------------------------------------------------------

    disagreement_bins = [
        -1.00,
        -0.10,
        -0.05,
        -0.025,
        0.00,
        0.025,
        0.05,
        0.10,
        1.00,
    ]

    disagreement_table = make_bucket_analysis(
        df,
        "model_market_difference",
        disagreement_bins,
    )

    print_section(
        "3. MODEL-MARKET DISAGREEMENT RESIDUALS",
        disagreement_table,
    )

    # ---------------------------------------------------------
    # 4. V0.8 CORRECTION SIZE
    # ---------------------------------------------------------

    correction_bins = [
        -0.10,
        -0.05,
        -0.025,
        -0.01,
        0.00,
        0.01,
        0.025,
        0.05,
        0.10,
    ]

    correction_table = make_bucket_analysis(
        df,
        "v08_over_probability",
        correction_bins,
    )

    print_section(
        "4. V0.8 PROBABILITY RESIDUALS",
        correction_table,
    )

    # ---------------------------------------------------------
    # 5. SEASON
    # ---------------------------------------------------------

    season_table = (
        df.groupby("season")
        .apply(residual_stats, include_groups=False)
        .reset_index()
    )

    print("\n")
    print("=" * 80)
    print("5. SEASON-BY-SEASON MARKET RESIDUAL")
    print("=" * 80)

    print(
        season_table[
            [
                "season",
                "matches",
                "mean_market_probability",
                "actual_over_rate",
                "mean_residual",
                "mean_abs_residual",
                "ci_95_low",
                "ci_95_high",
                "significant_95",
            ]
        ].to_string(index=False)
    )

    # ---------------------------------------------------------
    # 6. OVERALL MARKET RESIDUAL
    # ---------------------------------------------------------

    overall = residual_stats(df)

    print("\n")
    print("=" * 80)
    print("6. OVERALL MARKET RESIDUAL")
    print("=" * 80)

    print(
        f"Matches:                 {int(overall['matches']):,}"
    )

    print(
        f"Mean market probability: "
        f"{overall['mean_market_probability']:.6f}"
    )

    print(
        f"Actual Over rate:        "
        f"{overall['actual_over_rate']:.6f}"
    )

    print(
        f"Mean residual:           "
        f"{overall['mean_residual']:+.6f}"
    )

    print(
        f"Mean absolute residual:  "
        f"{overall['mean_abs_residual']:.6f}"
    )

    print(
        f"95% CI:                  "
        f"[{overall['ci_95_low']:+.6f}, "
        f"{overall['ci_95_high']:+.6f}]"
    )

    print(
        f"Significant at 95%:      "
        f"{overall['significant_95']}"
    )

    # ---------------------------------------------------------
    # 7. FIND POTENTIAL STRUCTURAL BIASES
    # ---------------------------------------------------------

    all_tables = {
        "market_probability": probability_table,
        "market_odds": odds_table,
        "model_market_disagreement": disagreement_table,
        "v08_probability": correction_table,
        "season": season_table,
    }

    combined = []

    for name, table in all_tables.items():

        temp = table.copy()
        temp["analysis"] = name

        combined.append(temp)

    combined = pd.concat(
        combined,
        ignore_index=True,
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True,
    )

    combined.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ---------------------------------------------------------
    # Final diagnostic
    # ---------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("7. DIAGNOSTIC ASSESSMENT")
    print("=" * 80)

    significant = combined[
        combined["significant_95"] == True
    ]

    if len(significant) == 0:

        print(
            "No bucket shows statistically significant "
            "market residual at the 95% level."
        )

    else:

        print(
            f"Significant buckets detected: "
            f"{len(significant)}"
        )

        print(
            "\nThese are diagnostic findings only."
        )

        print(
            "A significant residual does NOT automatically "
            "represent betting value."
        )

        print(
            "Multiple-testing effects must be considered."
        )

    print("\nIMPORTANT:")
    print("This is a market-efficiency diagnostic.")
    print("No betting thresholds were optimized.")
    print("No ROI conclusion should be drawn.")
    print("No real-money deployment is justified.")

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
