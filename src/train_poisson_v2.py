import math
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_VERSION = "poisson_v0.2_corrected"

MODEL_READY = Path("data/processed/model_ready.csv")
MATCHES_FILE = Path("data/processed/matches.csv")
OUTPUT_FILE = Path(
    "data/processed/poisson_v02_predictions.csv"
)

DECAY = 0.97

# Minimum historical matches before using team-specific
# attack/defence information.
MIN_TEAM_MATCHES = 3

# Prior strength used for cold starts / new seasons.
PRIOR_STRENGTH = 1.0

# Controls how strongly previous-season information is retained.
# 0.0 = completely reset
# 1.0 = completely retain
SEASON_CARRYOVER = 0.35


def poisson_over25(lam):
    """
    Probability of more than 2 goals when total goals
    follow a Poisson distribution with parameter lambda.
    """

    if lam <= 0:
        return 0.0

    p0 = math.exp(-lam)
    p1 = p0 * lam
    p2 = p1 * lam / 2.0

    return 1.0 - (p0 + p1 + p2)


def weighted_mean(values, weights):
    if len(values) == 0:
        return None

    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    return float(
        np.sum(values * weights)
        / np.sum(weights)
    )


def decay_weights(n):
    """
    Most recent observation gets weight 1.0.

    Older observations receive progressively smaller
    exponential weights.
    """

    if n == 0:
        return np.array([], dtype=float)

    return np.array(
        [
            DECAY ** (n - 1 - i)
            for i in range(n)
        ],
        dtype=float,
    )


def new_team_state():
    return {
        "home_scored": [],
        "home_conceded": [],
        "home_weights": [],

        "away_scored": [],
        "away_conceded": [],
        "away_weights": [],

        "matches": 0,
    }


def add_match_to_state(
    state,
    home_team,
    away_team,
    home_goals,
    away_goals,
):
    """
    Add one completed match to the historical state.

    This function is called ONLY after a prediction has been
    generated for the match.
    """

    if home_team not in state:
        state[home_team] = new_team_state()

    if away_team not in state:
        state[away_team] = new_team_state()

    home = state[home_team]
    away = state[away_team]

    # Append the newest observation.
    home["home_scored"].append(float(home_goals))
    home["home_conceded"].append(float(away_goals))

    away["away_scored"].append(float(away_goals))
    away["away_conceded"].append(float(home_goals))

    home["matches"] += 1
    away["matches"] += 1

    # Recalculate weights so the newest observation receives 1.0.
    home["home_weights"] = decay_weights(
        len(home["home_scored"])
    )

    home["away_weights"] = decay_weights(
        len(home["away_scored"])
    )

    away["home_weights"] = decay_weights(
        len(away["home_scored"])
    )

    away["away_weights"] = decay_weights(
        len(away["away_scored"])
    )


def weighted_league_environment(history):
    """
    Calculate weighted league home/away scoring rates.
    """

    if not history:
        return None, None

    home_goals = [
        float(x["home_goals"])
        for x in history
    ]

    away_goals = [
        float(x["away_goals"])
        for x in history
    ]

    weights = decay_weights(len(history))

    home_avg = weighted_mean(
        home_goals,
        weights,
    )

    away_avg = weighted_mean(
        away_goals,
        weights,
    )

    return home_avg, away_avg


def calculate_team_strength(
    state,
    team,
    league_home_avg,
    league_away_avg,
):
    """
    Calculate attack and defence strength using separate
    home and away observations.

    Attack:
        goals scored relative to the corresponding league rate.

    Defence:
        goals conceded relative to the corresponding league rate.

    Lower defensive strength = better defence.
    """

    if team not in state:
        return PRIOR_STRENGTH, PRIOR_STRENGTH

    team_state = state[team]

    # ---------------------------------------------------------
    # Home attack
    # ---------------------------------------------------------

    home_attack = None

    if (
        len(team_state["home_scored"])
        >= MIN_TEAM_MATCHES
        and league_home_avg > 0
    ):
        home_scored = weighted_mean(
            team_state["home_scored"],
            team_state["home_weights"],
        )

        home_attack = (
            home_scored
            / league_home_avg
        )

    # ---------------------------------------------------------
    # Home defence
    # ---------------------------------------------------------

    home_defence = None

    if (
        len(team_state["home_conceded"])
        >= MIN_TEAM_MATCHES
        and league_away_avg > 0
    ):
        home_conceded = weighted_mean(
            team_state["home_conceded"],
            team_state["home_weights"],
        )

        home_defence = (
            home_conceded
            / league_away_avg
        )

    # ---------------------------------------------------------
    # Away attack
    # ---------------------------------------------------------

    away_attack = None

    if (
        len(team_state["away_scored"])
        >= MIN_TEAM_MATCHES
        and league_away_avg > 0
    ):
        away_scored = weighted_mean(
            team_state["away_scored"],
            team_state["away_weights"],
        )

        away_attack = (
            away_scored
            / league_away_avg
        )

    # ---------------------------------------------------------
    # Away defence
    # ---------------------------------------------------------

    away_defence = None

    if (
        len(team_state["away_conceded"])
        >= MIN_TEAM_MATCHES
        and league_home_avg > 0
    ):
        away_conceded = weighted_mean(
            team_state["away_conceded"],
            team_state["away_weights"],
        )

        away_defence = (
            away_conceded
            / league_home_avg
        )

    # ---------------------------------------------------------
    # Combine home/away information.
    #
    # If one side is unavailable, use the available estimate.
    # If neither exists, use league-average prior.
    # ---------------------------------------------------------

    attack_values = [
        x
        for x in [
            home_attack,
            away_attack,
        ]
        if x is not None
    ]

    defence_values = [
        x
        for x in [
            home_defence,
            away_defence,
        ]
        if x is not None
    ]

    attack = (
        float(np.mean(attack_values))
        if attack_values
        else PRIOR_STRENGTH
    )

    defence = (
        float(np.mean(defence_values))
        if defence_values
        else PRIOR_STRENGTH
    )

    return attack, defence


def calculate_prediction(
    state,
    history,
    home_team,
    away_team,
):
    """
    Generate a pre-match prediction.

    No information from the current match is present in state.
    """

    league_home_avg, league_away_avg = (
        weighted_league_environment(history)
    )

    if (
        league_home_avg is None
        or league_away_avg is None
        or league_home_avg <= 0
        or league_away_avg <= 0
    ):
        return None

    (
        home_attack,
        home_defence,
    ) = calculate_team_strength(
        state,
        home_team,
        league_home_avg,
        league_away_avg,
    )

    (
        away_attack,
        away_defence,
    ) = calculate_team_strength(
        state,
        away_team,
        league_home_avg,
        league_away_avg,
    )

    # ---------------------------------------------------------
    # Expected goals
    # ---------------------------------------------------------

    lambda_home = (
        league_home_avg
        * home_attack
        * away_defence
    )

    lambda_away = (
        league_away_avg
        * away_attack
        * home_defence
    )

    # ---------------------------------------------------------
    # Sanity bounds.
    #
    # These are deliberately wider than V0.2's old 0.15–4.5
    # clipping. They should almost never activate.
    # ---------------------------------------------------------

    lambda_home = float(
        np.clip(
            lambda_home,
            0.05,
            6.0,
        )
    )

    lambda_away = float(
        np.clip(
            lambda_away,
            0.05,
            6.0,
        )
    )

    total_lambda = (
        lambda_home
        + lambda_away
    )

    probability = poisson_over25(
        total_lambda
    )

    return {
        "lambda_home": lambda_home,
        "lambda_away": lambda_away,
        "model_lambda": total_lambda,

        "model_over_probability": probability,

        "home_attack_strength": home_attack,
        "home_defence_strength": home_defence,

        "away_attack_strength": away_attack,
        "away_defence_strength": away_defence,

        "league_home_goals_avg": league_home_avg,
        "league_away_goals_avg": league_away_avg,
    }


def main():

    print()
    print("EDGE — POISSON V0.2 CORRECTED")
    print("=" * 70)

    model_df = pd.read_csv(
        MODEL_READY
    )

    matches_df = pd.read_csv(
        MATCHES_FILE
    )

    model_df["date"] = pd.to_datetime(
        model_df["date"]
    )

    matches_df["date"] = pd.to_datetime(
        matches_df["date"]
    )

    model_df = model_df.sort_values(
        ["date", "match_id"]
    ).reset_index(drop=True)

    matches_df = matches_df[
        [
            "match_id",
            "home_goals",
            "away_goals",
        ]
    ]

    df = model_df.merge(
        matches_df,
        on="match_id",
        how="inner",
        validate="one_to_one",
    )

    print(
        f"Matches available: {len(df):,}"
    )

    predictions = []

    history = []

    current_season = None

    # ---------------------------------------------------------
    # Chronological walk-forward evaluation.
    # ---------------------------------------------------------

    for _, row in df.iterrows():

        season = row["season"]

        # -----------------------------------------------------
        # Detect season transition.
        #
        # We deliberately reset team state at the beginning
        # of each season. The previous season is NOT blindly
        # carried into the new season.
        # -----------------------------------------------------

        if current_season != season:

            if current_season is not None:
                print(
                    f"{current_season}: "
                    f"completed"
                )

            print(
                f"{season}: "
                f"starting"
            )

            current_season = season

            state = {}
            history = []

        prediction = calculate_prediction(
            state,
            history,
            row["home_team_id"],
            row["away_team_id"],
        )

        if prediction is not None:

            record = {
                "match_id": row["match_id"],
                "season": season,
                "date": row["date"],

                "home_team_id": row[
                    "home_team_id"
                ],

                "away_team_id": row[
                    "away_team_id"
                ],

                "target_over25": row[
                    "target_over25"
                ],

                "market_over_probability": row[
                    "market_over_probability"
                ],

                "market_under_probability": row[
                    "market_under_probability"
                ],

                "market_over_odds": row[
                    "market_over_odds"
                ],

                "market_under_odds": row[
                    "market_under_odds"
                ],

                "actual_home_goals": row[
                    "home_goals"
                ],

                "actual_away_goals": row[
                    "away_goals"
                ],

                "model_version": MODEL_VERSION,
            }

            record.update(
                prediction
            )

            predictions.append(
                record
            )

        # -----------------------------------------------------
        # CRITICAL:
        #
        # Only after generating the prediction do we add the
        # current match to historical state.
        # -----------------------------------------------------

        add_match_to_state(
            state,
            row["home_team_id"],
            row["away_team_id"],
            row["home_goals"],
            row["away_goals"],
        )

        history.append(
            {
                "home_team_id": row[
                    "home_team_id"
                ],

                "away_team_id": row[
                    "away_team_id"
                ],

                "home_goals": row[
                    "home_goals"
                ],

                "away_goals": row[
                    "away_goals"
                ],
            }
        )

    result = pd.DataFrame(
        predictions
    )

    print()
    print("=" * 70)
    print("OUT-OF-SAMPLE RESULTS")
    print("=" * 70)

    print(
        f"Predictions: "
        f"{len(result):,}"
    )

    if result.empty:
        print(
            "ERROR: No predictions generated."
        )
        return

    actual = result[
        "target_over25"
    ].astype(float)

    model_prob = result[
        "model_over_probability"
    ].astype(float)

    market_prob = result[
        "market_over_probability"
    ].astype(float)

    # ---------------------------------------------------------
    # Brier score
    # ---------------------------------------------------------

    model_brier = np.mean(
        (
            model_prob
            - actual
        ) ** 2
    )

    market_brier = np.mean(
        (
            market_prob
            - actual
        ) ** 2
    )

    # ---------------------------------------------------------
    # Log loss
    # ---------------------------------------------------------

    eps = 1e-15

    model_clipped = np.clip(
        model_prob,
        eps,
        1 - eps,
    )

    market_clipped = np.clip(
        market_prob,
        eps,
        1 - eps,
    )

    model_logloss = -np.mean(
        actual
        * np.log(model_clipped)
        +
        (1 - actual)
        * np.log(
            1 - model_clipped
        )
    )

    market_logloss = -np.mean(
        actual
        * np.log(market_clipped)
        +
        (1 - actual)
        * np.log(
            1 - market_clipped
        )
    )

    print(
        f"Model Brier score:       "
        f"{model_brier:.6f}"
    )

    print(
        f"Market Brier score:      "
        f"{market_brier:.6f}"
    )

    print(
        f"Model Log Loss:          "
        f"{model_logloss:.6f}"
    )

    print(
        f"Market Log Loss:         "
        f"{market_logloss:.6f}"
    )

    print(
        f"Mean model probability:  "
        f"{model_prob.mean():.4f}"
    )

    print(
        f"Mean market probability: "
        f"{market_prob.mean():.4f}"
    )

    print(
        f"Mean actual Over rate:   "
        f"{actual.mean():.4f}"
    )

    # ---------------------------------------------------------
    # Lambda diagnostics
    # ---------------------------------------------------------

    print()
    print("Lambda summary:")

    print(
        result[
            [
                "lambda_home",
                "lambda_away",
                "model_lambda",
            ]
        ]
        .describe()
        .round(4)
    )

    print()
    print("Model probability summary:")

    print(
        result[
            "model_over_probability"
        ]
        .describe()
        .round(4)
    )

    print()
    print("Strength summary:")

    print(
        result[
            [
                "home_attack_strength",
                "home_defence_strength",
                "away_attack_strength",
                "away_defence_strength",
            ]
        ]
        .describe()
        .round(4)
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
