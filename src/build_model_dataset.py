import pandas as pd
from pathlib import Path


FEATURE_FILE = Path(
    "data/processed/model_dataset.csv"
)

ODDS_FILE = Path(
    "data/processed/market_odds.csv"
)

OUTPUT_FILE = Path(
    "data/processed/model_ready.csv"
)


def main():

    print("\nEDGE — BUILD MODEL DATASET")
    print("=" * 70)

    features = pd.read_csv(FEATURE_FILE)
    odds = pd.read_csv(ODDS_FILE)

    print(
        f"Feature rows: {len(features):,}"
    )

    print(
        f"Odds rows:    {len(odds):,}"
    )

    # ---------------------------------------------------------
    # Select the V0.1 market:
    #
    # O/U 2.5
    # PRE_CLOSING
    # MARKET_AVERAGE
    # ---------------------------------------------------------

    odds = odds[
        (odds["market"] == "OU25")
        &
        (odds["price_type"] == "PRE_CLOSING")
        &
        (odds["price_source"] == "MARKET_AVERAGE")
    ].copy()

    print(
        f"\nSelected odds rows: {len(odds):,}"
    )

    # ---------------------------------------------------------
    # Pivot OVER / UNDER into columns
    # ---------------------------------------------------------

    odds_pivot = (
        odds
        .pivot(
            index="match_id",
            columns="selection",
            values="price"
        )
        .reset_index()
    )

    odds_pivot = odds_pivot.rename(
        columns={
            "OVER": "market_over_odds",
            "UNDER": "market_under_odds",
        }
    )

    # ---------------------------------------------------------
    # Validate pair integrity
    # ---------------------------------------------------------

    missing_over = odds_pivot[
        "market_over_odds"
    ].isna().sum()

    missing_under = odds_pivot[
        "market_under_odds"
    ].isna().sum()

    print(
        f"Missing OVER prices:  {missing_over}"
    )

    print(
        f"Missing UNDER prices: {missing_under}"
    )

    # ---------------------------------------------------------
    # Merge
    # ---------------------------------------------------------

    df = features.merge(
        odds_pivot,
        on="match_id",
        how="left",
        validate="one_to_one"
    )

    # ---------------------------------------------------------
    # Calculate raw implied probabilities
    #
    # DO NOT normalize yet.
    #
    # The sum reveals the bookmaker/market margin.
    # ---------------------------------------------------------

    df["raw_implied_over"] = (
        1 / df["market_over_odds"]
    )

    df["raw_implied_under"] = (
        1 / df["market_under_odds"]
    )

    df["market_overround"] = (
        df["raw_implied_over"]
        +
        df["raw_implied_under"]
    )

    # ---------------------------------------------------------
    # Margin-adjusted market probability
    #
    # This is our initial fair-probability approximation.
    # ---------------------------------------------------------

    df["market_over_probability"] = (
        df["raw_implied_over"]
        /
        df["market_overround"]
    )

    df["market_under_probability"] = (
        df["raw_implied_under"]
        /
        df["market_overround"]
    )

    # ---------------------------------------------------------
    # Basic validation
    # ---------------------------------------------------------

    assert len(df) == len(features)

    assert df["match_id"].is_unique

    assert (
        df["market_over_probability"]
        .between(0, 1)
        .all()
    )

    assert (
        df["market_under_probability"]
        .between(0, 1)
        .all()
    )

    print(
        f"\nFinal rows: {len(df):,}"
    )

    print(
        f"Unique matches: "
        f"{df['match_id'].nunique():,}"
    )

    print(
        "\nMarket probability summary:"
    )

    print(
        df[
            [
                "market_over_probability",
                "market_under_probability",
                "market_overround",
            ]
        ].describe().round(4)
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
