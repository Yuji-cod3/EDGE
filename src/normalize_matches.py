import pandas as pd
from pathlib import Path

from team_registry import TEAM_REGISTRY


RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def season_from_filename(path):
    return path.stem.replace("EPL_", "")


def make_match_id(season, date, home_team_id, away_team_id):
    date_string = date.strftime("%Y%m%d")
    return f"{season}_{date_string}_{home_team_id}_{away_team_id}"


def main():
    records = []
    errors = []

    files = sorted(RAW_DIR.glob("EPL_*.csv"))

    for file in files:
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

            home_team_id, home_team_name = TEAM_REGISTRY[home_raw]
            away_team_id, away_team_name = TEAM_REGISTRY[away_raw]

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

            home_goals = pd.to_numeric(
                row["FTHG"],
                errors="coerce"
            )

            away_goals = pd.to_numeric(
                row["FTAG"],
                errors="coerce"
            )

            if pd.isna(home_goals) or pd.isna(away_goals):
                errors.append(
                    f"{file.name} row {index}: missing goals"
                )
                continue

            total_goals = int(home_goals + away_goals)

            over25_result = (
                "OVER"
                if total_goals > 2
                else "UNDER"
            )

            kickoff_time = None

            if "Time" in df.columns:
                value = row["Time"]

                if pd.notna(value):
                    kickoff_time = str(value)

            match_id = make_match_id(
                season,
                date,
                home_team_id,
                away_team_id
            )

            records.append(
                {
                    "match_id": match_id,
                    "season": season,
                    "date": date.strftime("%Y-%m-%d"),
                    "kickoff_time": kickoff_time,
                    "home_team_id": home_team_id,
                    "away_team_id": away_team_id,
                    "home_team": home_team_name,
                    "away_team": away_team_name,
                    "home_goals": int(home_goals),
                    "away_goals": int(away_goals),
                    "total_goals": total_goals,
                    "over25_result": over25_result,
                }
            )

    if errors:
        print("\nERRORS DETECTED")
        print("=" * 60)

        for error in errors:
            print(error)

        raise SystemExit(
            f"\nNormalization stopped: {len(errors)} error(s)."
        )

    result = pd.DataFrame(records)

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    validation_errors = []

    # Match count
    season_counts = result.groupby("season").size()

    for season, count in season_counts.items():
        if count != 380:
            validation_errors.append(
                f"{season}: expected 380 matches, found {count}"
            )

    # Duplicate IDs
    duplicate_ids = result[
        result["match_id"].duplicated(keep=False)
    ]

    if not duplicate_ids.empty:
        validation_errors.append(
            f"Duplicate match IDs: {len(duplicate_ids)}"
        )

    # Goal consistency
    inconsistent_goals = result[
        result["total_goals"]
        != result["home_goals"] + result["away_goals"]
    ]

    if not inconsistent_goals.empty:
        validation_errors.append(
            f"Goal inconsistencies: {len(inconsistent_goals)}"
        )

    # Over/Under consistency
    expected_over25 = (
        result["total_goals"] > 2
    ).map({
        True: "OVER",
        False: "UNDER",
    })

    inconsistent_ou = result[
        result["over25_result"] != expected_over25
    ]

    if not inconsistent_ou.empty:
        validation_errors.append(
            f"Over/Under inconsistencies: {len(inconsistent_ou)}"
        )

    # ---------------------------------------------------------

    if validation_errors:
        print("\nVALIDATION FAILED")
        print("=" * 60)

        for error in validation_errors:
            print(f"- {error}")

        raise SystemExit(1)

    output = OUTPUT_DIR / "matches.csv"

    result.to_csv(output, index=False)

    print("\nEDGE MATCH NORMALIZATION")
    print("=" * 60)

    print(f"Matches: {len(result):,}")
    print(f"Seasons: {result['season'].nunique()}")
    print(f"Unique match IDs: {result['match_id'].nunique():,}")

    print("\nMatches per season:")

    for season, count in season_counts.items():
        print(f"  {season}: {count}")

    print("\nOver/Under 2.5 results:")

    print(
        result["over25_result"]
        .value_counts()
        .to_string()
    )

    print(f"\nSaved to: {output}")


if __name__ == "__main__":
    main()
