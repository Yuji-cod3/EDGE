import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import brier_score_loss, log_loss


INPUT = Path("data/processed/poisson_v02_predictions.csv")


def safe_log_loss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return log_loss(y, p)


def max_drawdown(returns):
    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1
    return drawdown.min()


def evaluate_betting_rule(df, threshold):
    """
    Fixed diagnostic rule.

    Bet OVER only when:
        model probability - market probability >= threshold

    This is deliberately NOT optimized here.
    """

    work = df.copy()

    work["edge"] = (
        work["model_over_probability"]
        - work["market_over_probability"]
    )

    bets = work[work["edge"] >= threshold].copy()

    if len(bets) == 0:
        return {
            "bets": 0,
            "win_rate": np.nan,
            "roi": np.nan,
            "avg_odds": np.nan,
            "max_drawdown": np.nan,
        }

    bets["profit"] = np.where(
        bets["target_over25"] == 1,
        bets["market_over_odds"] - 1,
        -1.0,
    )

    returns = bets["profit"]

    return {
        "bets": len(bets),
        "win_rate": bets["target_over25"].mean(),
        "roi": returns.mean(),
        "avg_odds": bets["market_over_odds"].mean(),
        "max_drawdown": max_drawdown(returns),
    }


def main():

    print()
    print("EDGE — POISSON V0.2 DIAGNOSTIC")
    print("=" * 78)

    if not INPUT.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {INPUT}"
        )

    df = pd.read_csv(INPUT)

    required = [
        "match_id",
        "season",
        "date",
        "model_over_probability",
        "market_over_probability",
        "market_over_odds",
        "target_over25",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(["date", "match_id"]).reset_index(drop=True)

    df["model_over_probability"] = pd.to_numeric(
        df["model_over_probability"],
        errors="coerce",
    )

    df["market_over_probability"] = pd.to_numeric(
        df["market_over_probability"],
        errors="coerce",
    )

    df["market_over_odds"] = pd.to_numeric(
        df["market_over_odds"],
        errors="coerce",
    )

    df["target_over25"] = pd.to_numeric(
        df["target_over25"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "model_over_probability",
            "market_over_probability",
            "market_over_odds",
            "target_over25",
        ]
    )

    print(f"Predictions evaluated: {len(df):,}")

    # ------------------------------------------------------------
    # 1. OVERALL MODEL VS MARKET
    # ------------------------------------------------------------

    y = df["target_over25"].astype(int)

    model_p = df["model_over_probability"].clip(1e-6, 1 - 1e-6)

    market_p = df["market_over_probability"].clip(1e-6, 1 - 1e-6)

    model_brier = brier_score_loss(y, model_p)
    market_brier = brier_score_loss(y, market_p)

    model_ll = safe_log_loss(y, model_p)
    market_ll = safe_log_loss(y, market_p)

    print()
    print("=" * 78)
    print("1. OVERALL PERFORMANCE")
    print("=" * 78)

    print(f"Model Brier score:   {model_brier:.6f}")
    print(f"Market Brier score:  {market_brier:.6f}")
    print(f"Brier difference:    {model_brier - market_brier:+.6f}")

    print()

    print(f"Model Log Loss:      {model_ll:.6f}")
    print(f"Market Log Loss:     {market_ll:.6f}")
    print(f"Log Loss difference: {model_ll - market_ll:+.6f}")

    # ------------------------------------------------------------
    # 2. PER-SEASON PERFORMANCE
    # ------------------------------------------------------------

    print()
    print("=" * 78)
    print("2. PERFORMANCE BY SEASON")
    print("=" * 78)

    season_rows = []

    for season, group in df.groupby("season", sort=True):

        y_s = group["target_over25"].astype(int)

        mp = group["model_over_probability"].clip(
            1e-6, 1 - 1e-6
        )

        mkp = group["market_over_probability"].clip(
            1e-6, 1 - 1e-6
        )

        season_rows.append({
            "season": season,
            "matches": len(group),
            "model_brier": brier_score_loss(y_s, mp),
            "market_brier": brier_score_loss(y_s, mkp),
            "model_logloss": safe_log_loss(y_s, mp),
            "market_logloss": safe_log_loss(y_s, mkp),
            "actual_over_rate": y_s.mean(),
        })

    season_df = pd.DataFrame(season_rows)

    print(
        season_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ------------------------------------------------------------
    # 3. CALIBRATION
    # ------------------------------------------------------------

    print()
    print("=" * 78)
    print("3. MODEL CALIBRATION")
    print("=" * 78)

    bins = [
        0.00,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        1.00,
    ]

    labels = [
        "<=0.40",
        "0.40-0.45",
        "0.45-0.50",
        "0.50-0.55",
        "0.55-0.60",
        "0.60-0.65",
        "0.65-0.70",
        "0.70-0.75",
        ">0.75",
    ]

    df["model_bin"] = pd.cut(
        df["model_over_probability"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    calibration = (
        df.groupby(
            "model_bin",
            observed=False,
        )
        .agg(
            predictions=("target_over25", "size"),
            mean_prediction=("model_over_probability", "mean"),
            actual_over_rate=("target_over25", "mean"),
        )
        .reset_index()
    )

    calibration["calibration_error"] = (
        calibration["actual_over_rate"]
        - calibration["mean_prediction"]
    )

    print(
        calibration.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ------------------------------------------------------------
    # 4. MODEL VS MARKET
    # ------------------------------------------------------------

    df["edge"] = (
        df["model_over_probability"]
        - df["market_over_probability"]
    )

    print()
    print("=" * 78)
    print("4. MODEL VS MARKET PROBABILITY")
    print("=" * 78)

    print(
        f"Mean model probability:  "
        f"{df['model_over_probability'].mean():.4f}"
    )

    print(
        f"Mean market probability: "
        f"{df['market_over_probability'].mean():.4f}"
    )

    print(
        f"Mean probability edge:    "
        f"{df['edge'].mean():+.4f}"
    )

    print(
        f"Median probability edge:  "
        f"{df['edge'].median():+.4f}"
    )

    print(
        f"Minimum edge:             "
        f"{df['edge'].min():+.4f}"
    )

    print(
        f"Maximum edge:             "
        f"{df['edge'].max():+.4f}"
    )

    # ------------------------------------------------------------
    # 5. EDGE BUCKETS
    # ------------------------------------------------------------

    print()
    print("=" * 78)
    print("5. EDGE BUCKET ANALYSIS")
    print("=" * 78)

    edge_bins = [
        -1.0,
        -0.10,
        -0.05,
        -0.025,
        0.0,
        0.025,
        0.05,
        0.10,
        1.0,
    ]

    edge_labels = [
        "< -10%",
        "-10% to -5%",
        "-5% to -2.5%",
        "-2.5% to 0%",
        "0% to 2.5%",
        "2.5% to 5%",
        "5% to 10%",
        "> 10%",
    ]

    df["edge_bin"] = pd.cut(
        df["edge"],
        bins=edge_bins,
        labels=edge_labels,
        include_lowest=True,
    )

    edge_table = (
        df.groupby(
            "edge_bin",
            observed=False,
        )
        .agg(
            matches=("target_over25", "size"),
            mean_model_probability=(
                "model_over_probability",
                "mean",
            ),
            mean_market_probability=(
                "market_over_probability",
                "mean",
            ),
            actual_over_rate=("target_over25", "mean"),
        )
        .reset_index()
    )

    print(
        edge_table.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ------------------------------------------------------------
    # 6. ODDS RANGE
    # ------------------------------------------------------------

    print()
    print("=" * 78)
    print("6. PERFORMANCE BY OVER ODDS")
    print("=" * 78)

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

    odds_labels = [
        "1.00-1.40",
        "1.40-1.60",
        "1.60-1.80",
        "1.80-2.00",
        "2.00-2.25",
        "2.25-2.50",
        "2.50-3.00",
        "3.00+",
    ]

    df["odds_bin"] = pd.cut(
        df["market_over_odds"],
        bins=odds_bins,
        labels=odds_labels,
        include_lowest=True,
    )

    odds_table = (
        df.groupby(
            "odds_bin",
            observed=False,
        )
        .agg(
            matches=("target_over25", "size"),
            actual_over_rate=("target_over25", "mean"),
            mean_model_probability=(
                "model_over_probability",
                "mean",
            ),
            mean_market_probability=(
                "market_over_probability",
                "mean",
            ),
        )
        .reset_index()
    )

    print(
        odds_table.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ------------------------------------------------------------
    # 7. FIXED EDGE RULES
    # ------------------------------------------------------------

    print()
    print("=" * 78)
    print("7. FIXED EDGE SIMULATION")
    print("=" * 78)

    print(
        "IMPORTANT: thresholds are diagnostic only and "
        "are NOT optimized."
    )

    thresholds = [
        0.025,
        0.05,
        0.075,
        0.10,
    ]

    betting_rows = []

    for threshold in thresholds:

        result = evaluate_betting_rule(
            df,
            threshold,
        )

        betting_rows.append({
            "edge_threshold": threshold,
            **result,
        })

    betting_df = pd.DataFrame(betting_rows)

    print(
        betting_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ------------------------------------------------------------
    # 8. FINAL DIAGNOSTIC
    # ------------------------------------------------------------

    print()
    print("=" * 78)
    print("8. DIAGNOSTIC ASSESSMENT")
    print("=" * 78)

    if model_brier < market_brier:
        print("✓ Model beats market on Brier score.")
    else:
        print("✗ Model does NOT beat market on Brier score.")

    if model_ll < market_ll:
        print("✓ Model beats market on Log Loss.")
    else:
        print("✗ Model does NOT beat market on Log Loss.")

    positive_rules = betting_df[
        betting_df["roi"] > 0
    ]

    print()

    if len(positive_rules) == 0:
        print("No fixed diagnostic edge rule produced positive ROI.")
    else:
        print(
            f"{len(positive_rules)} fixed diagnostic rule(s) "
            "showed positive ROI."
        )

    print()
    print("EDGE DECISION:")
    print(
        "Do NOT deploy V0.2 for betting yet."
    )
    print(
        "Use this diagnostic to determine whether the next step "
        "should be calibration, market-model combination, or "
        "model redesign."
    )

    # ------------------------------------------------------------
    # SAVE REPORT DATA
    # ------------------------------------------------------------

    output = Path(
        "reports/poisson_v02_diagnostic.csv"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    season_df.to_csv(
        output,
        index=False,
    )

    print()
    print(f"Season diagnostic saved to: {output}")


if __name__ == "__main__":
    main()
