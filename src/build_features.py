import pandas as pd
import numpy as np
from pathlib import Path

MATCHES_FILE = Path("data/processed/matches.csv")
OUTPUT_FILE = Path("data/processed/model_dataset.csv")


RECENT_WINDOWS = [5, 10]


def safe_mean(values):
    if not values:
        return np.nan
    return float(np.mean(values))


def team_stats(history, team_id, venue=None):
    """
    Return statistics using ONLY matches already present in history.

    venue:
        None   -> all matches
        HOME   -> matches where team was home
        AWAY   -> matches where team was away
    """

    if venue == "HOME":
        rows = [
            x for x in history
            if x["home_team_id"] == team_id
        ]

        scored = [x["home_goals"] for x in rows]
        conceded = [x["away_goals"] for x in rows]

    elif venue == "AWAY":
        rows = [
            x for x in history
            if x["away_team_id"] == team_id
        ]

        scored = [x["away_goals"] for x in rows]
        conceded = [x["home_goals"] for x in rows]

    else:
        rows = [
            x for x in history
            if (
                x["home_team_id"] == team_id
                or x["away_team_id"] == team_id
            )
        ]

        scored = []
        conceded = []

        for x in rows:
            if x["home_team_id"] == team_id:
                scored.append(x["home_goals"])
                conceded.append(x["away_goals"])
            else:
                scored.append(x["away_goals"])
                conceded.append(x["home_goals"])

    return {
        "matches": len(rows),
        "goals_scored_avg": safe_mean(scored),
        "goals_conceded_avg": safe_mean(conceded),
    }


def recent_stats(history, team_id, window):
    """
    Last N matches involving the team.

    Only historical matches already observed are used.
    """

    rows = [
        x for x in history
        if (
            x["home_team_id"] == team_id
            or x["away_team_id"] == team_id
        )
    ]

    rows = rows[-window:]

    scored = []
    conceded = []

    for x in rows:
        if x["home_team_id"] == team_id:
            scored.append(x["home_goals"])
            conceded.append(x["away_goals"])
        else:
            scored.append(x["away_goals"])
            conceded.append(x["home_goals"])

    return {
        "matches": len(rows),
        "goals_scored_avg": safe_mean(scored),
        "goals_conceded_avg": safe_mean(conceded),
    }


def league_stats(history):
    """
    League scoring environment using only prior matches.
    """

    if not history:
        return {
            "matches": 0,
            "goals_avg": np.nan,
        }

    total_goals = [
        x["home_goals"] + x["away_goals"]
        for x in history
    ]

    return {
        "matches": len(history),
        "goals_avg": safe_mean(total_goals),
    }


def main():

    df = pd.read_csv(MATCHES_FILE)

    df["date"] = pd.to_datetime(df["date"])

    # IMPORTANT:
    # Stable chronological ordering.
    df = df.sort_values(
        ["date", "match_id"]
    ).reset_index(drop=True)

    history = []
    records = []

    for _, row in df.iterrows():

        home = row["home_team_id"]
        away = row["away_team_id"]

        # -----------------------------------------------------
        # EVERYTHING BELOW IS CALCULATED BEFORE ADDING
        # THE CURRENT MATCH TO HISTORY.
        # -----------------------------------------------------

        home_overall = team_stats(
            history,
            home
        )

        away_overall = team_stats(
            history,
            away
        )

        home_home = team_stats(
            history,
            home,
            "HOME"
        )

        away_away = team_stats(
            history,
            away,
            "AWAY"
        )

        home_recent5 = recent_stats(
            history,
            home,
            5
        )

        away_recent5 = recent_stats(
            history,
            away,
            5
        )

        home_recent10 = recent_stats(
            history,
            home,
            10
        )

        away_recent10 = recent_stats(
            history,
            away,
            10
        )

        league = league_stats(history)

        record = {
            "match_id": row["match_id"],
            "season": row["season"],
            "date": row["date"].strftime("%Y-%m-%d"),
            "kickoff_time": row["kickoff_time"],

            "home_team_id": home,
            "away_team_id": away,

            # Overall
            "home_goals_scored_avg": home_overall[
                "goals_scored_avg"
            ],
            "home_goals_conceded_avg": home_overall[
                "goals_conceded_avg"
            ],

            "away_goals_scored_avg": away_overall[
                "goals_scored_avg"
            ],
            "away_goals_conceded_avg": away_overall[
                "goals_conceded_avg"
            ],

            "home_history_matches": home_overall[
                "matches"
            ],
            "away_history_matches": away_overall[
                "matches"
            ],

            # Venue-specific
            "home_home_goals_scored_avg": home_home[
                "goals_scored_avg"
            ],
            "home_home_goals_conceded_avg": home_home[
                "goals_conceded_avg"
            ],
            "home_home_matches": home_home[
                "matches"
            ],

            "away_away_goals_scored_avg": away_away[
                "goals_scored_avg"
            ],
            "away_away_goals_conceded_avg": away_away[
                "goals_conceded_avg"
            ],
            "away_away_matches": away_away[
                "matches"
            ],

            # Recent 5
            "home_recent5_scored": home_recent5[
                "goals_scored_avg"
            ],
            "home_recent5_conceded": home_recent5[
                "goals_conceded_avg"
            ],
            "home_recent5_matches": home_recent5[
                "matches"
            ],

            "away_recent5_scored": away_recent5[
                "goals_scored_avg"
            ],
            "away_recent5_conceded": away_recent5[
                "goals_conceded_avg"
            ],
            "away_recent5_matches": away_recent5[
                "matches"
            ],

            # Recent 10
            "home_recent10_scored": home_recent10[
                "goals_scored_avg"
            ],
            "home_recent10_conceded": home_recent10[
                "goals_conceded_avg"
            ],
            "home_recent10_matches": home_recent10[
                "matches"
            ],

            "away_recent10_scored": away_recent10[
                "goals_scored_avg"
            ],
            "away_recent10_conceded": away_recent10[
                "goals_conceded_avg"
            ],
            "away_recent10_matches": away_recent10[
                "matches"
            ],

            # League environment
            "league_matches_before": league[
                "matches"
            ],
            "league_goals_avg": league[
                "goals_avg"
            ],

            # Target — ONLY for evaluation/model training.
            "target_total_goals": (
                row["home_goals"] +
                row["away_goals"]
            ),

            "target_over25": int(
                row["total_goals"] > 2.5
            ),
        }

        records.append(record)

        # -----------------------------------------------------
        # NOW the current match becomes historical information.
        # -----------------------------------------------------

        history.append(
            {
                "match_id": row["match_id"],
                "date": row["date"],
                "home_team_id": home,
                "away_team_id": away,
                "home_goals": row["home_goals"],
                "away_goals": row["away_goals"],
            }
        )

    result = pd.DataFrame(records)

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nEDGE — FEATURE DATASET")
    print("=" * 70)

    print(f"Rows: {len(result):,}")
    print(f"Columns: {len(result.columns):,}")
    print(f"Matches: {result['match_id'].nunique():,}")

    print("\nTarget:")
    print(
        result["target_over25"]
        .value_counts()
        .rename(index={
            0: "UNDER",
            1: "OVER"
        })
        .to_string()
    )

    print("\nFeature availability:")

    feature_columns = [
        "home_goals_scored_avg",
        "home_goals_conceded_avg",
        "away_goals_scored_avg",
        "away_goals_conceded_avg",
        "home_home_goals_scored_avg",
        "home_home_goals_conceded_avg",
        "away_away_goals_scored_avg",
        "away_away_goals_conceded_avg",
        "home_recent5_scored",
        "home_recent5_conceded",
        "away_recent5_scored",
        "away_recent5_conceded",
        "league_goals_avg",
    ]

    for column in feature_columns:
        available = result[column].notna().sum()
        percentage = available / len(result) * 100

        print(
            f"  {column:<40}"
            f"{available:>5}/{len(result)} "
            f"({percentage:>5.1f}%)"
        )

    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
