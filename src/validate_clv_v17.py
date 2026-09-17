#!/usr/bin/env python3

"""
EDGE — V0.17 CORRECTED CLV / OUTCOME VALIDATION

Purpose:
    Correct the V0.16 CLV definition and test whether opening-to-closing
    market movement has a meaningful relationship with actual outcomes.

Important:
    This is a diagnostic study.

    No threshold optimization.
    No stake optimization.
    No model optimization.
    No real-money deployment.

Definitions:

    Market movement:
        closing_odds / opening_odds - 1

        Positive = odds drifted
        Negative = odds shortened

    Bettor-favorable CLV:
        opening_odds / closing_odds - 1

        Positive = bettor obtained a better price than closing market
        Negative = bettor obtained a worse price than closing market

    Probability movement:
        closing_implied_probability - opening_implied_probability

        Positive = market probability increased
        Negative = market probability decreased
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ODDS_FILE = Path(
    "data/processed/market_odds_v15.csv"
)

RAW_DIR = Path("data/raw")

OUTPUT_FILE = Path(
    "data/processed/clv_v17_validated.csv"
)

SEASON_OUTPUT = Path(
    "reports/clv_v17_season_summary.csv"
)

MOVEMENT_OUTPUT = Path(
    "reports/clv_v17_movement_summary.csv"
)


# ============================================================
# HELPERS
# ============================================================

def normalize_selection(value):
    if pd.isna(value):
        return None

    value = str(value).strip().upper()

    if value in {
        ">2.5",
        "OVER",
        "OVER_2.5",
        "O2.5",
    }:
        return "OVER_2.5"

    if value in {
        "<2.5",
        "UNDER",
        "UNDER_2.5",
        "U2.5",
    }:
        return "UNDER_2.5"

    return value


def classify_movement(opening, closing):
    if pd.isna(opening) or pd.isna(closing):
        return "MISSING"

    if closing < opening:
        return "SHORTENED"

    if closing > opening:
        return "DRIFTED"

    return "UNCHANGED"


def find_column(df, candidates):
    lookup = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for candidate in candidates:
        key = candidate.lower()

        if key in lookup:
            return lookup[key]

    return None


def load_raw_results():
    """
    Load FTHG/FTAG from all EPL raw files.
    """

    frames = []

    for path in sorted(RAW_DIR.glob("EPL_*.csv")):

        try:
            raw = pd.read_csv(path)
        except Exception as exc:
            print(
                f"WARNING: Could not read {path.name}: {exc}"
            )
            continue

        home_col = find_column(
            raw,
            ["HomeTeam"]
        )

        away_col = find_column(
            raw,
            ["AwayTeam"]
        )

        fthg_col = find_column(
            raw,
            ["FTHG"]
        )

        ftag_col = find_column(
            raw,
            ["FTAG"]
        )

        date_col = find_column(
            raw,
            ["Date"]
        )

        if not all(
            [
                home_col,
                away_col,
                fthg_col,
                ftag_col,
                date_col,
            ]
        ):
            print(
                f"WARNING: Missing result fields in "
                f"{path.name}"
            )
            continue

        season = path.stem.replace(
            "EPL_",
            ""
        )

        temp = pd.DataFrame()

        temp["season"] = season
        temp["date"] = (
            raw[date_col]
            .astype(str)
            .str.strip()
        )

        temp["home_team"] = (
            raw[home_col]
            .astype(str)
            .str.strip()
        )

        temp["away_team"] = (
            raw[away_col]
            .astype(str)
            .str.strip()
        )

        temp["FTHG"] = pd.to_numeric(
            raw[fthg_col],
            errors="coerce"
        )

        temp["FTAG"] = pd.to_numeric(
            raw[ftag_col],
            errors="coerce"
        )

        frames.append(temp)

    if not frames:
        raise RuntimeError(
            "No raw result datasets could be loaded."
        )

    results = pd.concat(
        frames,
        ignore_index=True
    )

    return results


def create_match_id(row):
    return (
        f"{row['season']}_"
        f"{row['date']}_"
        f"{row['home_team']}_"
        f"{row['away_team']}"
    )


# ============================================================
# START
# ============================================================

print("=" * 78)
print("EDGE — V0.17 CORRECTED CLV / OUTCOME VALIDATION")
print("=" * 78)


# ============================================================
# LOAD ODDS
# ============================================================

if not ODDS_FILE.exists():
    raise FileNotFoundError(
        f"Odds file not found: {ODDS_FILE}"
    )

odds = pd.read_csv(ODDS_FILE)

print()
print("Odds file:")
print(ODDS_FILE)

print(
    f"Odds records: {len(odds):,}"
)


# ============================================================
# VALIDATE ODDS
# ============================================================

required_odds = [
    "match_id",
    "season",
    "selection",
    "opening_price",
    "closing_price",
]

missing = [
    col
    for col in required_odds
    if col not in odds.columns
]

if missing:
    raise ValueError(
        "Missing odds columns:\n"
        + "\n".join(
            f"  - {x}"
            for x in missing
        )
    )


odds["selection"] = (
    odds["selection"]
    .apply(normalize_selection)
)

odds["opening_price"] = pd.to_numeric(
    odds["opening_price"],
    errors="coerce"
)

odds["closing_price"] = pd.to_numeric(
    odds["closing_price"],
    errors="coerce"
)


# ============================================================
# KEEP MATCHED OPENING/CLOSING DATA
# ============================================================

df = odds[
    odds["opening_price"].notna()
    & odds["closing_price"].notna()
].copy()

df = df[
    (df["opening_price"] > 1.0)
    & (df["closing_price"] > 1.0)
].copy()

print()
print("=" * 78)
print("1. CLV SAMPLE")
print("=" * 78)

print(
    f"Matched observations: {len(df):,}"
)

print(
    f"Unique matches: "
    f"{df['match_id'].nunique():,}"
)


# ============================================================
# CALCULATE CORRECTED METRICS
# ============================================================

df["market_movement"] = (
    df["closing_price"]
    / df["opening_price"]
    - 1.0
)

# Positive = bettor obtained a better price than close.
df["bettor_clv"] = (
    df["opening_price"]
    / df["closing_price"]
    - 1.0
)

df["opening_implied_probability"] = (
    1.0 / df["opening_price"]
)

df["closing_implied_probability"] = (
    1.0 / df["closing_price"]
)

df["probability_change"] = (
    df["closing_implied_probability"]
    - df["opening_implied_probability"]
)

df["movement"] = [
    classify_movement(o, c)
    for o, c in zip(
        df["opening_price"],
        df["closing_price"]
    )
]


# ============================================================
# LOAD MATCH RESULTS
# ============================================================

print()
print("=" * 78)
print("2. LOADING ACTUAL RESULTS")
print("=" * 78)

results = load_raw_results()

results["match_id"] = results.apply(
    create_match_id,
    axis=1
)

results = results.drop_duplicates(
    subset=["match_id"]
)

print(
    f"Raw result records: "
    f"{len(results):,}"
)


# ============================================================
# CREATE TARGET
# ============================================================

results["total_goals"] = (
    results["FTHG"]
    + results["FTAG"]
)

results["over_2_5_result"] = np.where(
    results["total_goals"] > 2.5,
    1,
    0
)

results["under_2_5_result"] = np.where(
    results["total_goals"] < 2.5,
    1,
    0
)


# ============================================================
# MERGE
# ============================================================

df = df.merge(
    results[
        [
            "match_id",
            "FTHG",
            "FTAG",
            "total_goals",
            "over_2_5_result",
            "under_2_5_result",
        ]
    ],
    on="match_id",
    how="left",
    validate="many_to_one",
)


# ============================================================
# RESULT MATCHING AUDIT
# ============================================================

df["result_available"] = (
    df["FTHG"].notna()
    & df["FTAG"].notna()
)

print()
print("=" * 78)
print("3. RESULT MATCHING")
print("=" * 78)

matched = df["result_available"].sum()

print(
    f"Results matched: "
    f"{matched:,} / {len(df):,}"
)

print(
    f"Match rate: "
    f"{matched / len(df):.2%}"
)


# ============================================================
# SELECTION OUTCOME
# ============================================================

df["selection_win"] = np.nan

over_mask = (
    df["selection"] == "OVER_2.5"
)

under_mask = (
    df["selection"] == "UNDER_2.5"
)

df.loc[
    over_mask & df["result_available"],
    "selection_win"
] = df.loc[
    over_mask & df["result_available"],
    "over_2_5_result"
]

df.loc[
    under_mask & df["result_available"],
    "selection_win"
] = df.loc[
    under_mask & df["result_available"],
    "under_2_5_result"
]


# ============================================================
# OVERALL CLV
# ============================================================

print()
print("=" * 78)
print("4. CORRECTED CLV")
print("=" * 78)

print(
    f"Mean bettor CLV: "
    f"{df['bettor_clv'].mean():+.6f}"
)

print(
    f"Median bettor CLV: "
    f"{df['bettor_clv'].median():+.6f}"
)

print(
    f"Mean market movement: "
    f"{df['market_movement'].mean():+.6f}"
)

print(
    f"Mean probability movement: "
    f"{df['probability_change'].mean():+.6f}"
)


# ============================================================
# MOVEMENT VS OUTCOME
# ============================================================

valid = df[
    df["selection_win"].notna()
].copy()

print()
print("=" * 78)
print("5. MARKET MOVEMENT VS ACTUAL OUTCOME")
print("=" * 78)

movement_summary = (
    valid
    .groupby("movement")
    .agg(
        observations=("match_id", "size"),
        mean_bettor_clv=(
            "bettor_clv",
            "mean"
        ),
        median_bettor_clv=(
            "bettor_clv",
            "median"
        ),
        mean_probability_change=(
            "probability_change",
            "mean"
        ),
        actual_win_rate=(
            "selection_win",
            "mean"
        ),
    )
    .reindex(
        [
            "SHORTENED",
            "DRIFTED",
            "UNCHANGED",
        ]
    )
    .reset_index()
)

print(
    movement_summary.to_string(
        index=False
    )
)


# ============================================================
# SELECTION ANALYSIS
# ============================================================

print()
print("=" * 78)
print("6. SELECTION × MARKET MOVEMENT")
print("=" * 78)

selection_summary = (
    valid
    .groupby(
        ["selection", "movement"]
    )
    .agg(
        observations=("match_id", "size"),
        mean_bettor_clv=(
            "bettor_clv",
            "mean"
        ),
        actual_win_rate=(
            "selection_win",
            "mean"
        ),
    )
    .reset_index()
)

print(
    selection_summary.to_string(
        index=False
    )
)


# ============================================================
# SEASON ANALYSIS
# ============================================================

print()
print("=" * 78)
print("7. SEASON-BY-SEASON CLV")
print("=" * 78)

season_summary = (
    valid
    .groupby("season")
    .agg(
        observations=("match_id", "size"),
        mean_bettor_clv=(
            "bettor_clv",
            "mean"
        ),
        median_bettor_clv=(
            "bettor_clv",
            "median"
        ),
        mean_probability_change=(
            "probability_change",
            "mean"
        ),
        actual_win_rate=(
            "selection_win",
            "mean"
        ),
        shortened=(
            "movement",
            lambda x:
            (x == "SHORTENED").sum()
        ),
        drifted=(
            "movement",
            lambda x:
            (x == "DRIFTED").sum()
        ),
        unchanged=(
            "movement",
            lambda x:
            (x == "UNCHANGED").sum()
        ),
    )
    .reset_index()
)

print(
    season_summary.to_string(
        index=False
    )
)


# ============================================================
# BET365 ONLY
# ============================================================

if "opening_source" in valid.columns:

    bet365 = valid[
        valid["opening_source"]
        == "BET365"
    ].copy()

    print()
    print("=" * 78)
    print("8. BET365-ONLY VALIDATION")
    print("=" * 78)

    print(
        f"BET365 observations: "
        f"{len(bet365):,}"
    )

    if len(bet365) > 0:

        print(
            f"Mean bettor CLV: "
            f"{bet365['bettor_clv'].mean():+.6f}"
        )

        print(
            f"Median bettor CLV: "
            f"{bet365['bettor_clv'].median():+.6f}"
        )

        print(
            f"Actual win rate: "
            f"{bet365['selection_win'].mean():.4%}"
        )


# ============================================================
# IMPORTANT MARKET DIRECTION TEST
# ============================================================

print()
print("=" * 78)
print("9. DIRECTIONAL MARKET TEST")
print("=" * 78)

shortened = valid[
    valid["movement"] == "SHORTENED"
]

drifted = valid[
    valid["movement"] == "DRIFTED"
]

if len(shortened) > 0:
    print(
        "Shortened selections:"
    )

    print(
        f"  N: "
        f"{len(shortened):,}"
    )

    print(
        f"  Win rate: "
        f"{shortened['selection_win'].mean():.4%}"
    )

if len(drifted) > 0:
    print()

    print(
        "Drifted selections:"
    )

    print(
        f"  N: "
        f"{len(drifted):,}"
    )

    print(
        f"  Win rate: "
        f"{drifted['selection_win'].mean():.4%}"
    )


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

SEASON_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

MOVEMENT_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

season_summary.to_csv(
    SEASON_OUTPUT,
    index=False
)

movement_summary.to_csv(
    MOVEMENT_OUTPUT,
    index=False
)


# ============================================================
# FINAL ASSESSMENT
# ============================================================

print()
print("=" * 78)
print("10. V0.17 ASSESSMENT")
print("=" * 78)

print(
    "✓ Corrected bettor-favorable CLV definition."
)

print(
    "✓ Market movement separated from CLV."
)

print(
    "✓ Opening/closing odds matched to actual results."
)

print(
    "✓ Movement direction evaluated against outcomes."
)

print(
    "✓ No threshold optimization."
)

print(
    "✓ No stake optimization."
)

print(
    "✓ No predictive model modification."
)

print()
print(
    "IMPORTANT:"
)

print(
    "This does NOT establish profitability."
)

print(
    "This is a market-efficiency diagnostic."
)

print()
print("Saved validated observations:")
print(f"  {OUTPUT_FILE}")

print("Saved season summary:")
print(f"  {SEASON_OUTPUT}")

print("Saved movement summary:")
print(f"  {MOVEMENT_OUTPUT}")

print()
print("=" * 78)
print("V0.17 COMPLETE")
print("=" * 78)
