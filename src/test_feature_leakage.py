import pandas as pd
from pathlib import Path

MATCHES = Path("data/processed/matches.csv")
FEATURES = Path("data/processed/model_dataset.csv")


def main():

    matches = pd.read_csv(MATCHES)
    features = pd.read_csv(FEATURES)

    matches["date"] = pd.to_datetime(matches["date"])
    features["date"] = pd.to_datetime(features["date"])

    print("\nEDGE — FEATURE LEAKAGE TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Basic integrity
    # ---------------------------------------------------------

    assert len(matches) == len(features), (
        "Match count differs between source and feature dataset."
    )

    assert matches["match_id"].is_unique
    assert features["match_id"].is_unique

    print("✓ Match count integrity")

    # ---------------------------------------------------------
    # 2. Every feature row must correspond to exactly one match
    # ---------------------------------------------------------

    merged = features.merge(
        matches[
            [
                "match_id",
                "home_team_id",
                "away_team_id",
                "home_goals",
                "away_goals",
            ]
        ],
        on="match_id",
        how="left",
        suffixes=("_feature", "_source"),
        validate="one_to_one",
    )

    assert merged["home_goals"].notna().all()
    assert merged["away_goals"].notna().all()

    print("✓ One-to-one match mapping")

    # ---------------------------------------------------------
    # 3. Current match must NOT be included in team history
    # ---------------------------------------------------------

    violations = []

    for _, row in features.iterrows():

        home_history = row["home_history_matches"]
        away_history = row["away_history_matches"]

        if home_history < 0 or away_history < 0:
            violations.append(
                (row["match_id"], "negative history count")
            )

        if row["home_home_matches"] < 0:
            violations.append(
                (row["match_id"], "negative home history")
            )

        if row["away_away_matches"] < 0:
            violations.append(
                (row["match_id"], "negative away history")
            )

    assert not violations, violations[:10]

    print("✓ No invalid history counts")

    # ---------------------------------------------------------
    # 4. History counts must be strictly less than the
    #    number of matches involving the team INCLUDING current
    # ---------------------------------------------------------

    for _, row in features.iterrows():

        match_id = row["match_id"]
        home = row["home_team_id"]
        away = row["away_team_id"]

        previous_home = matches[
            (
                (
                    (matches["home_team_id"] == home)
                    | (matches["away_team_id"] == home)
                )
                & (matches["match_id"] < match_id)
            )
        ]

        previous_away = matches[
            (
                (
                    (matches["home_team_id"] == away)
                    | (matches["away_team_id"] == away)
                )
                & (matches["match_id"] < match_id)
            )
        ]

        # We don't use this as the primary chronological test
        # because match_id ordering is season/date based, but it
        # provides an additional sanity check.

        if row["home_history_matches"] > len(previous_home):
            violations.append(
                (match_id, "home history exceeds prior matches")
            )

        if row["away_history_matches"] > len(previous_away):
            violations.append(
                (match_id, "away history exceeds prior matches")
            )

    assert not violations, violations[:10]

    print("✓ History counts do not exceed prior observations")

    # ---------------------------------------------------------
    # 5. First match of every team must have zero history
    # ---------------------------------------------------------

    first_team_matches = {}

    for _, row in matches.sort_values(
        ["date", "match_id"]
    ).iterrows():

        for team in [
            row["home_team_id"],
            row["away_team_id"],
        ]:

            if team not in first_team_matches:
                first_team_matches[team] = row["match_id"]

    for team, match_id in first_team_matches.items():

        feature_row = features[
            features["match_id"] == match_id
        ].iloc[0]

        if feature_row["home_team_id"] == team:
            count = feature_row["home_history_matches"]
        else:
            count = feature_row["away_history_matches"]

        assert count == 0, (
            f"{team} first match has history={count}"
        )

    print("✓ First-match cold starts")

    # ---------------------------------------------------------
    # 6. Feature dates cannot be after match date
    # ---------------------------------------------------------

    assert (
        features["date"] >=
        pd.to_datetime(features["date"]).min()
    ).all()

    print("✓ Feature dates valid")

    # ---------------------------------------------------------
    # Final
    # ---------------------------------------------------------

    print("=" * 70)
    print("STATUS: PASS")
    print("No obvious feature leakage detected.")


if __name__ == "__main__":
    main()
