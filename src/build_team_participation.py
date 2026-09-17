import pandas as pd
from pathlib import Path

from team_registry import TEAM_REGISTRY


RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def season_from_filename(path):
    return path.stem.replace("EPL_", "")


def main():
    records = []
    unknown_names = set()

    files = sorted(RAW_DIR.glob("EPL_*.csv"))

    for file in files:
        season = season_from_filename(file)

        df = pd.read_csv(file)

        teams = set(df["HomeTeam"].dropna()) | set(
            df["AwayTeam"].dropna()
        )

        for raw_name in teams:
            if raw_name not in TEAM_REGISTRY:
                unknown_names.add(raw_name)
                continue

            team_id, canonical_name = TEAM_REGISTRY[raw_name]

            records.append(
                {
                    "season": season,
                    "team_id": team_id,
                    "team_name": canonical_name,
                    "raw_name": raw_name,
                    "competition": "EPL",
                }
            )

    if unknown_names:
        print("\nERROR: Unknown team names detected:")

        for name in sorted(unknown_names):
            print(f"  - {name}")

        raise SystemExit(1)

    result = pd.DataFrame(records)

    result = result.sort_values(
        ["season", "team_id"]
    ).reset_index(drop=True)

    output = OUTPUT_DIR / "season_teams.csv"

    result.to_csv(output, index=False)

    print("\nEDGE TEAM PARTICIPATION")
    print("=" * 60)

    print(f"Seasons: {result['season'].nunique()}")
    print(f"Participation records: {len(result)}")
    print(f"Unique teams: {result['team_id'].nunique()}")

    print("\nTeams per season:")

    counts = result.groupby("season")["team_id"].nunique()

    for season, count in counts.items():
        print(f"  {season}: {count}")

    print(f"\nSaved to: {output}")


if __name__ == "__main__":
    main()
