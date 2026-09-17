import pandas as pd
import numpy as np

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))

from build_features import (
    team_stats,
    recent_stats,
    league_stats,
)


def assert_close(actual, expected, label):

    if pd.isna(expected):
        assert pd.isna(actual), (
            f"{label}: expected NaN, got {actual}"
        )
    else:
        assert np.isclose(actual, expected), (
            f"{label}: expected {expected}, got {actual}"
        )


def main():

    print("\nEDGE — DETERMINISTIC FEATURE TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # Synthetic historical matches
    #
    # These are ALL before the hypothetical target match.
    # ---------------------------------------------------------

    history = [
        {
            "match_id": "M1",
            "date": pd.Timestamp("2025-01-01"),
            "home_team_id": "A",
            "away_team_id": "B",
            "home_goals": 2,
            "away_goals": 1,
        },
        {
            "match_id": "M2",
            "date": pd.Timestamp("2025-01-08"),
            "home_team_id": "B",
            "away_team_id": "A",
            "home_goals": 0,
            "away_goals": 0,
        },
        {
            "match_id": "M3",
            "date": pd.Timestamp("2025-01-15"),
            "home_team_id": "A",
            "away_team_id": "C",
            "home_goals": 3,
            "away_goals": 2,
        },
    ]

    # ---------------------------------------------------------
    # TEST 1
    # Team A overall statistics
    #
    # A scored:
    #   M1 = 2
    #   M2 = 0
    #   M3 = 3
    #
    # Average = 5/3
    #
    # A conceded:
    #   M1 = 1
    #   M2 = 0
    #   M3 = 2
    #
    # Average = 1
    # ---------------------------------------------------------

    stats = team_stats(history, "A")

    assert stats["matches"] == 3

    assert_close(
        stats["goals_scored_avg"],
        5 / 3,
        "A overall goals scored"
    )

    assert_close(
        stats["goals_conceded_avg"],
        1.0,
        "A overall goals conceded"
    )

    print("✓ Overall team statistics")

    # ---------------------------------------------------------
    # TEST 2
    # Team A home statistics
    #
    # A was home in:
    # M1 = 2-1
    # M3 = 3-2
    #
    # Scored = 5 / 2 = 2.5
    # Conceded = 3 / 2 = 1.5
    # ---------------------------------------------------------

    stats = team_stats(
        history,
        "A",
        "HOME"
    )

    assert stats["matches"] == 2

    assert_close(
        stats["goals_scored_avg"],
        2.5,
        "A home goals scored"
    )

    assert_close(
        stats["goals_conceded_avg"],
        1.5,
        "A home goals conceded"
    )

    print("✓ Home-specific statistics")

    # ---------------------------------------------------------
    # TEST 3
    # Team A away statistics
    #
    # A was away in:
    # M2 = 0-0
    # ---------------------------------------------------------

    stats = team_stats(
        history,
        "A",
        "AWAY"
    )

    assert stats["matches"] == 1

    assert_close(
        stats["goals_scored_avg"],
        0.0,
        "A away goals scored"
    )

    assert_close(
        stats["goals_conceded_avg"],
        0.0,
        "A away goals conceded"
    )

    print("✓ Away-specific statistics")

    # ---------------------------------------------------------
    # TEST 4
    # Recent statistics
    # ---------------------------------------------------------

    stats = recent_stats(
        history,
        "A",
        2
    )

    # Last two A matches:
    #
    # M2 = 0-0
    # M3 = 3-2
    #
    # Scored = 3 / 2 = 1.5
    # Conceded = 2 / 2 = 1.0

    assert stats["matches"] == 2

    assert_close(
        stats["goals_scored_avg"],
        1.5,
        "A recent goals scored"
    )

    assert_close(
        stats["goals_conceded_avg"],
        1.0,
        "A recent goals conceded"
    )

    print("✓ Recent-window statistics")

    # ---------------------------------------------------------
    # TEST 5
    # Recent window larger than available history
    # ---------------------------------------------------------

    stats = recent_stats(
        history,
        "A",
        10
    )

    assert stats["matches"] == 3

    assert_close(
        stats["goals_scored_avg"],
        5 / 3,
        "A recent-10 goals scored"
    )

    print("✓ Large recent window handling")

    # ---------------------------------------------------------
    # TEST 6
    # League scoring average
    #
    # M1 = 3 goals
    # M2 = 0 goals
    # M3 = 5 goals
    #
    # Total = 8
    # Average = 8/3
    # ---------------------------------------------------------

    stats = league_stats(history)

    assert stats["matches"] == 3

    assert_close(
        stats["goals_avg"],
        8 / 3,
        "League goals average"
    )

    print("✓ League scoring environment")

    # ---------------------------------------------------------
    # TEST 7
    # Empty history
    # ---------------------------------------------------------

    stats = team_stats([], "A")

    assert stats["matches"] == 0
    assert pd.isna(stats["goals_scored_avg"])
    assert pd.isna(stats["goals_conceded_avg"])

    stats = recent_stats([], "A", 5)

    assert stats["matches"] == 0
    assert pd.isna(stats["goals_scored_avg"])
    assert pd.isna(stats["goals_conceded_avg"])

    stats = league_stats([])

    assert stats["matches"] == 0
    assert pd.isna(stats["goals_avg"])

    print("✓ Empty-history / cold-start handling")

    # ---------------------------------------------------------
    # TEST 8
    # Leakage sanity check
    #
    # Remove M3 and verify A's statistics change accordingly.
    # ---------------------------------------------------------

    history_before_m3 = history[:2]

    stats = team_stats(
        history_before_m3,
        "A"
    )

    # A before M3:
    #
    # scored = 2 + 0 = 2
    # conceded = 1 + 0 = 1
    #
    # averages:
    # scored = 1.0
    # conceded = 0.5

    assert stats["matches"] == 2

    assert_close(
        stats["goals_scored_avg"],
        1.0,
        "Pre-M3 A scoring"
    )

    assert_close(
        stats["goals_conceded_avg"],
        0.5,
        "Pre-M3 A conceding"
    )

    print("✓ Pre-match information isolation")

    print("=" * 70)
    print("STATUS: PASS")
    print("Deterministic feature calculations are correct.")


if __name__ == "__main__":
    main()
