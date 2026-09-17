import pandas as pd
from pathlib import Path

from team_registry import TEAM_REGISTRY


RAW_DIR = Path("data/raw")
MATCHES_FILE = Path("data/processed/matches.csv")
OUTPUT_DIR = Path("data/processed")


def season_from_filename(path):
    return path.stem.replace("EPL_", "")


def get_price(row, columns):
    for column in columns:
        if column in row.index:
            value = pd.to_numeric(row[column], errors="coerce")

            if pd.notna(value) and value > 1:
                return float(value), column

    return None, None


def main():
    matches = pd.read_csv(MATCHES_FILE)

    match_lookup = {
        row["match_id"]: row
        for _, row in matches.iterrows()
    }

    records = []
    errors = []

    for file in sorted(RAW_DIR.glob("EPL_*.csv")):

        season = season_from_filename(file)
        df = pd.read_csv(file)

        for index, row in df.iterrows():

            home_raw = row["HomeTeam"]
            away_raw = row["AwayTeam"]

            if home_raw not in TEAM_REGISTRY:
                errors.append(
                    f"{file.name} row {index}: "
                    f"unknown home team '{home_raw}'"
                )
                continue

            if away_raw not in TEAM_REGISTRY:
                errors.append(
                    f"{file.name} row {index}: "
                    f"unknown away team '{away_raw}'"
                )
                continue

            home_id = TEAM_REGISTRY[home_raw][0]
            away_id = TEAM_REGISTRY[away_raw][0]

            date = pd.to_datetime(
                row["Date"],
                dayfirst=True,
                errors="coerce"
            )

            if pd.isna(date):
                errors.append(
                    f"{file.name} row {index}: invalid date"
                )
                continue

            match_id = (
                f"{season}_"
                f"{date.strftime('%Y%m%d')}_"
                f"{home_id}_{away_id}"
            )

            if match_id not in match_lookup:
                errors.append(
                    f"{file.name} row {index}: "
                    f"match ID not found: {match_id}"
                )
                continue

            # -------------------------------------------------
            # PRE-CLOSING MARKET AVERAGE
            # -------------------------------------------------

            over_price, over_column = get_price(
                row,
                ["Avg>2.5", "BbAv>2.5"]
            )

            under_price, under_column = get_price(
                row,
                ["Avg<2.5", "BbAv<2.5"]
            )

            if over_price is not None:
                records.append(
                    {
                        "match_id": match_id,
                        "season": season,
                        "market": "OU25",
                        "selection": "OVER",
                        "price": over_price,
                        "price_type": "PRE_CLOSING",
                        "price_source": "MARKET_AVERAGE",
                        "source_column": over_column,
                    }
                )

            if under_price is not None:
                records.append(
                    {
                        "match_id": match_id,
                        "season": season,
                        "market": "OU25",
                        "selection": "UNDER",
                        "price": under_price,
                        "price_type": "PRE_CLOSING",
                        "price_source": "MARKET_AVERAGE",
                        "source_column": under_column,
                    }
                )

            # -------------------------------------------------
            # PRE-CLOSING MAXIMUM
            # -------------------------------------------------

            over_price, over_column = get_price(
                row,
                ["Max>2.5", "BbMx>2.5"]
            )

            under_price, under_column = get_price(
                row,
                ["Max<2.5", "BbMx<2.5"]
            )

            if over_price is not None:
                records.append(
                    {
                        "match_id": match_id,
                        "season": season,
                        "market": "OU25",
                        "selection": "OVER",
                        "price": over_price,
                        "price_type": "PRE_CLOSING",
                        "price_source": "MARKET_MAX",
                        "source_column": over_column,
                    }
                )

            if under_price is not None:
                records.append(
                    {
                        "match_id": match_id,
                        "season": season,
                        "market": "OU25",
                        "selection": "UNDER",
                        "price": under_price,
                        "price_type": "PRE_CLOSING",
                        "price_source": "MARKET_MAX",
                        "source_column": under_column,
                    }
                )

            # -------------------------------------------------
            # BET365 PRE-CLOSING
            # -------------------------------------------------

            over_price, over_column = get_price(
                row,
                ["B365>2.5"]
            )

            under_price, under_column = get_price(
                row,
                ["B365<2.5"]
            )

            if over_price is not None:
                records.append(
                    {
                        "match_id": match_id,
                        "season": season,
                        "market": "OU25",
                        "selection": "OVER",
                        "price": over_price,
                        "price_type": "PRE_CLOSING",
                        "price_source": "BET365",
                        "source_column": over_column,
                    }
                )

            if under_price is not None:
                records.append(
                    {
                        "match_id": match_id,
                        "season": season,
                        "market": "OU25",
                        "selection": "UNDER",
                        "price": under_price,
                        "price_type": "PRE_CLOSING",
                        "price_source": "BET365",
                        "source_column": under_column,
                    }
                )

            # -------------------------------------------------
            # CLOSING MARKET AVERAGE
            # -------------------------------------------------

            over_price, over_column = get_price(
                row,
                ["AvgC>2.5"]
            )

            under_price, under_column = get_price(
                row,
                ["AvgC<2.5"]
            )

            if over_price is not None:
                records.append(
                    {
                        "match_id": match_id,
                        "season": season,
                        "market": "OU25",
                        "selection": "OVER",
                        "price": over_price,
                        "price_type": "CLOSING",
                        "price_source": "MARKET_AVERAGE",
                        "source_column": over_column,
                    }
                )

            if under_price is not None:
                records.append(
                    {
                        "match_id": match_id,
                        "season": season,
                        "market": "OU25",
                        "selection": "UNDER",
                        "price": under_price,
                        "price_type": "CLOSING",
                        "price_source": "MARKET_AVERAGE",
                        "source_column": under_column,
                    }
                )

            # -------------------------------------------------
            # CLOSING MAXIMUM
            # -------------------------------------------------

            over_price, over_column = get_price(
                row,
                ["MaxC>2.5"]
            )

            under_price, under_column = get_price(
                row,
                ["MaxC<2.5"]
            )

            if over_price is not None:
                records.append(
                    {
                        "match_id": match_id,
                        "season": season,
                        "market": "OU25",
                        "selection": "OVER",
                        "price": over_price,
                        "price_type": "CLOSING",
                        "price_source": "MARKET_MAX",
                        "source_column": over_column,
                    }
                )

            if under_price is not None:
                records.append(
                    {
                        "match_id": match_id,
                        "season": season,
                        "market": "OU25",
                        "selection": "UNDER",
                        "price": under_price,
                        "price_type": "CLOSING",
                        "price_source": "MARKET_MAX",
                        "source_column": under_column,
                    }
                )

            # -------------------------------------------------
            # CLOSING BET365
            # -------------------------------------------------

            over_price, over_column = get_price(
                row,
                ["B365C>2.5"]
            )

            under_price, under_column = get_price(
                row,
                ["B365C<2.5"]
            )

            if over_price is not None:
                records.append(
                    {
                        "match_id": match_id,
                        "season": season,
                        "market": "OU25",
                        "selection": "OVER",
                        "price": over_price,
                        "price_type": "CLOSING",
                        "price_source": "BET365",
                        "source_column": over_column,
                    }
                )

            if under_price is not None:
                records.append(
                    {
                        "match_id": match_id,
                        "season": season,
                        "market": "OU25",
                        "selection": "UNDER",
                        "price": under_price,
                        "price_type": "CLOSING",
                        "price_source": "BET365",
                        "source_column": under_column,
                    }
                )

    if errors:
        print("\nERRORS DETECTED")
        print("=" * 70)

        for error in errors:
            print(error)

        raise SystemExit(
            f"\nNormalization stopped: {len(errors)} error(s)."
        )

    result = pd.DataFrame(records)

    # ---------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------

    required_columns = [
        "match_id",
        "season",
        "market",
        "selection",
        "price",
        "price_type",
        "price_source",
        "source_column",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in result.columns
    ]

    if missing_columns:
        raise SystemExit(
            f"Missing output columns: {missing_columns}"
        )

    if result["price"].isna().any():
        raise SystemExit("NULL prices detected.")

    if (result["price"] <= 1).any():
        raise SystemExit("Invalid odds <= 1 detected.")

    valid_selections = {"OVER", "UNDER"}

    if not set(result["selection"]).issubset(valid_selections):
        raise SystemExit("Invalid selection detected.")

    # Every market-average pre-closing observation should have
    # both sides available.
    avg_pre = result[
        (result["price_source"] == "MARKET_AVERAGE")
        & (result["price_type"] == "PRE_CLOSING")
    ]

    avg_counts = (
        avg_pre.groupby("match_id")["selection"]
        .nunique()
    )

    incomplete_avg = avg_counts[avg_counts != 2]

    if not incomplete_avg.empty:
        print(
            "\nWARNING:"
            f" {len(incomplete_avg)} matches have incomplete "
            "pre-closing market-average O/U prices."
        )

    output = OUTPUT_DIR / "market_odds.csv"

    result = result.sort_values(
        [
            "season",
            "match_id",
            "market",
            "price_type",
            "price_source",
            "selection",
        ]
    ).reset_index(drop=True)

    result.to_csv(output, index=False)

    print("\nEDGE ODDS NORMALIZATION")
    print("=" * 70)

    print(f"Odds observations: {len(result):,}")
    print(f"Matches represented: {result['match_id'].nunique():,}")

    print("\nPrice type:")
    print(result["price_type"].value_counts().to_string())

    print("\nPrice source:")
    print(result["price_source"].value_counts().to_string())

    print("\nSeason:")
    print(result["season"].value_counts().sort_index().to_string())

    print(f"\nSaved to: {output}")


if __name__ == "__main__":
    main()
