"""
EDGE — V0.12 VALUE VALIDATION

Purpose:
    Validate whether model-derived positive EV survives
    predetermined edge thresholds and statistical uncertainty.

Important:
    - Diagnostic only.
    - No threshold optimization.
    - No stake optimization.
    - Chronological results only.
    - Historical ROI is not proof of future profitability.
"""

from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILE = Path("data/processed/value_v11_predictions.csv")
OUTPUT_FILE = Path("data/processed/value_validation_v12_predictions.csv")
SUMMARY_FILE = Path("reports/value_validation_v12_summary.csv")

EDGE_THRESHOLDS = [0.01, 0.025, 0.05, 0.075, 0.10]

N_BOOTSTRAP = 5000
RANDOM_SEED = 42


def bootstrap_ci(values, statistic=np.mean, n_bootstrap=N_BOOTSTRAP):
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(RANDOM_SEED)

    samples = rng.choice(
        values,
        size=(n_bootstrap, len(values)),
        replace=True,
    )

    stats = statistic(samples, axis=1)

    return (
        float(np.percentile(stats, 2.5)),
        float(np.percentile(stats, 97.5)),
    )


def max_drawdown(profits):
    profits = np.asarray(profits, dtype=float)

    if len(profits) == 0:
        return np.nan

    cumulative = np.cumsum(profits)
    running_max = np.maximum.accumulate(
        np.insert(cumulative, 0, 0)
    )[1:]

    drawdowns = cumulative - running_max

    return float(drawdowns.min())


def longest_losing_streak(profits):
    longest = 0
    current = 0

    for profit in profits:
        if profit < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    return longest


def evaluate_selection(df, threshold, label):
    selected = df[df["probability_edge"] >= threshold].copy()

    n = len(selected)

    if n == 0:
        return {
            "selection": label,
            "edge_threshold": threshold,
            "matches": 0,
            "mean_edge": np.nan,
            "mean_model_probability": np.nan,
            "mean_market_probability": np.nan,
            "mean_odds": np.nan,
            "required_break_even_rate": np.nan,
            "actual_win_rate": np.nan,
            "win_rate_ci_low": np.nan,
            "win_rate_ci_high": np.nan,
            "mean_expected_value": np.nan,
            "flat_profit": np.nan,
            "roi": np.nan,
            "roi_ci_low": np.nan,
            "roi_ci_high": np.nan,
            "max_drawdown": np.nan,
            "longest_losing_streak": np.nan,
            "status": "INSUFFICIENT DATA",
        }

    odds = selected["market_over_odds"].to_numpy(float)
    outcomes = selected["target_over25"].to_numpy(int)

    # Flat 1-unit stake.
    profits = np.where(
        outcomes == 1,
        odds - 1.0,
        -1.0,
    )

    roi_values = profits

    flat_profit = profits.sum()
    roi = profits.mean()

    win_rate = outcomes.mean()

    # Required break-even probability for each individual price.
    breakeven = 1.0 / odds
    mean_breakeven = breakeven.mean()

    win_low, win_high = bootstrap_ci(
        outcomes,
        statistic=np.mean,
    )

    roi_low, roi_high = bootstrap_ci(
        roi_values,
        statistic=np.mean,
    )

    # Conservative classification.
    if n < 100:
        status = "INSUFFICIENT DATA"
    elif roi_low > 0:
        status = "PROMISING — NEEDS FURTHER VALIDATION"
    elif roi_high < 0:
        status = "CONTRADICTED"
    else:
        status = "INCONCLUSIVE"

    return {
        "selection": label,
        "edge_threshold": threshold,
        "matches": n,
        "mean_edge": selected["probability_edge"].mean(),
        "mean_model_probability": selected[
            "model_probability"
        ].mean(),
        "mean_market_probability": selected[
            "market_probability"
        ].mean(),
        "mean_odds": odds.mean(),
        "required_break_even_rate": mean_breakeven,
        "actual_win_rate": win_rate,
        "win_rate_ci_low": win_low,
        "win_rate_ci_high": win_high,
        "mean_expected_value": selected[
            "expected_value"
        ].mean(),
        "flat_profit": flat_profit,
        "roi": roi,
        "roi_ci_low": roi_low,
        "roi_ci_high": roi_high,
        "max_drawdown": max_drawdown(profits),
        "longest_losing_streak": longest_losing_streak(profits),
        "status": status,
    }


def main():
    print("EDGE — V0.12 VALUE VALIDATION")
    print("=" * 78)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"Predictions available: {len(df):,}")

    required = [
        "match_id",
        "season",
        "date",
        "target_over25",
        "market_over_probability",
        "v06_over_probability",
        "market_over_odds",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # Standardize names so V12 does not depend on the
    # exact column naming used internally by V11.
    df["model_probability"] = df["v06_over_probability"]
    df["market_probability"] = df["market_over_probability"]

    df["probability_edge"] = (
        df["model_probability"]
        - df["market_probability"]
    )

    df["expected_value"] = (
        df["model_probability"]
        * df["market_over_odds"]
        - 1.0
    )

    df = df.sort_values(
        ["date", "match_id"]
    ).reset_index(drop=True)

    print()
    print("=" * 78)
    print("1. BASIC VALUE VALIDATION")
    print("=" * 78)

    print(
        f"Mean probability edge: "
        f"{df['probability_edge'].mean():+.6f}"
    )

    print(
        f"Median probability edge: "
        f"{df['probability_edge'].median():+.6f}"
    )

    print(
        f"Mean model EV: "
        f"{df['expected_value'].mean():+.6f}"
    )

    print(
        f"Positive model-EV matches: "
        f"{(df['expected_value'] > 0).sum():,}"
    )

    print()
    print("=" * 78)
    print("2. PREDETERMINED EDGE THRESHOLD VALIDATION")
    print("=" * 78)

    results = []

    for threshold in EDGE_THRESHOLDS:
        result = evaluate_selection(
            df,
            threshold,
            f"EDGE >= {threshold:.1%}",
        )

        results.append(result)

    # Also evaluate the raw positive-EV rule from V0.11.
    positive_ev = df[
        df["expected_value"] > 0
    ].copy()

    if len(positive_ev):
        odds = positive_ev[
            "market_over_odds"
        ].to_numpy(float)

        outcomes = positive_ev[
            "target_over25"
        ].to_numpy(int)

        profits = np.where(
            outcomes == 1,
            odds - 1,
            -1,
        )

        win_low, win_high = bootstrap_ci(
            outcomes,
            statistic=np.mean,
        )

        roi_low, roi_high = bootstrap_ci(
            profits,
            statistic=np.mean,
        )

        results.append({
            "selection": "MODEL EV > 0",
            "edge_threshold": np.nan,
            "matches": len(positive_ev),
            "mean_edge": positive_ev[
                "probability_edge"
            ].mean(),
            "mean_model_probability": positive_ev[
                "model_probability"
            ].mean(),
            "mean_market_probability": positive_ev[
                "market_probability"
            ].mean(),
            "mean_odds": odds.mean(),
            "required_break_even_rate": (
                1.0 / odds
            ).mean(),
            "actual_win_rate": outcomes.mean(),
            "win_rate_ci_low": win_low,
            "win_rate_ci_high": win_high,
            "mean_expected_value": positive_ev[
                "expected_value"
            ].mean(),
            "flat_profit": profits.sum(),
            "roi": profits.mean(),
            "roi_ci_low": roi_low,
            "roi_ci_high": roi_high,
            "max_drawdown": max_drawdown(profits),
            "longest_losing_streak":
                longest_losing_streak(profits),
            "status": (
                "INCONCLUSIVE"
                if roi_low <= 0 <= roi_high
                else (
                    "PROMISING — NEEDS FURTHER VALIDATION"
                    if roi_low > 0
                    else "CONTRADICTED"
                )
            ),
        })

    results_df = pd.DataFrame(results)

    pd.set_option(
        "display.max_columns",
        None,
    )
    pd.set_option(
        "display.width",
        200,
    )
    pd.set_option(
        "display.float_format",
        lambda x: f"{x:.6f}",
    )

    print(results_df.to_string(index=False))

    print()
    print("=" * 78)
    print("3. SEASON STABILITY")
    print("=" * 78)

    season_rows = []

    for season, season_df in df.groupby(
        "season",
        sort=True,
    ):
        for threshold in EDGE_THRESHOLDS:
            selected = season_df[
                season_df["probability_edge"] >= threshold
            ]

            n = len(selected)

            if n == 0:
                continue

            odds = selected[
                "market_over_odds"
            ].to_numpy(float)

            outcomes = selected[
                "target_over25"
            ].to_numpy(int)

            profits = np.where(
                outcomes == 1,
                odds - 1,
                -1,
            )

            season_rows.append({
                "season": season,
                "edge_threshold": threshold,
                "matches": n,
                "actual_win_rate": outcomes.mean(),
                "roi": profits.mean(),
                "mean_edge": selected[
                    "probability_edge"
                ].mean(),
            })

    season_df = pd.DataFrame(season_rows)

    if not season_df.empty:
        print(
            season_df.to_string(index=False)
        )

    print()
    print("=" * 78)
    print("4. VALIDATION ASSESSMENT")
    print("=" * 78)

    positive_roi = results_df[
        results_df["roi"] > 0
    ]

    statistically_positive = results_df[
        results_df["roi_ci_low"] > 0
    ]

    if len(statistically_positive):
        print(
            "Potential statistically positive "
            "selection detected."
        )
        print(
            "This is NOT sufficient for deployment."
        )
    elif len(positive_roi):
        print(
            "Some thresholds have positive "
            "historical ROI."
        )
        print(
            "However, confidence intervals cross zero."
        )
        print(
            "Result: INCONCLUSIVE."
        )
    else:
        print(
            "No tested threshold produced "
            "positive historical ROI."
        )
        print(
            "Result: CONTRADICTED."
        )

    print()
    print("IMPORTANT:")
    print("- Thresholds were predetermined.")
    print("- No stake sizing was optimized.")
    print("- No threshold was selected using ROI.")
    print("- Bootstrap intervals are historical uncertainty estimates.")
    print("- Multiple-testing effects still matter.")
    print("- Historical ROI is not evidence of future profitability.")
    print("- No real-money deployment is justified.")

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    results_df.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print()
    print(f"Saved predictions to: {OUTPUT_FILE}")
    print(f"Saved summary to:     {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
