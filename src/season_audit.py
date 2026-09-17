import pandas as pd
from pathlib import Path

raw = Path("data/raw")

print("\nEDGE — PREMIER LEAGUE SEASON AUDIT")
print("=" * 70)

files = sorted(raw.glob("EPL_*.csv"))

all_teams = set()

for file in files:
    df = pd.read_csv(file)

    teams = set(df["HomeTeam"].dropna()) | set(df["AwayTeam"].dropna())
    all_teams.update(teams)

    missing_scores = df[["FTHG", "FTAG"]].isna().any(axis=1).sum()

    print(
        f"{file.name:<20} "
        f"matches={len(df):>3}  "
        f"teams={len(teams):>2}  "
        f"columns={len(df.columns):>3}  "
        f"missing_scores={missing_scores:>2}"
    )

print("=" * 70)

print(f"\nTotal unique raw team names across all seasons: {len(all_teams)}")

print("\nAll observed team names:")

for team in sorted(all_teams):
    print(f"  {team}")

print("\nExpected matches per complete EPL season: 380")
