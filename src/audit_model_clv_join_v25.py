from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================================
# EDGE — V0.25 MODEL / CLV JOIN REPAIR & VALIDATION
# ============================================================================

BASE = Path(__file__).resolve().parents[1]

MODEL_FILE = BASE / "data/processed/poisson_predictions.csv"
MATCHES_FILE = BASE / "data/processed/matches.csv"
CLV_V20_FILE = BASE / "data/processed/clv_audit_v20.csv"
CLV_V22_FILE = BASE / "data/processed/clv_audit_v22.csv"

OUTPUT_FILE = BASE / "data/processed/model_clv_join_v25.csv"
REPORT_FILE = BASE / "reports/model_clv_join_v25_report.csv"


print("=" * 78)
print("EDGE — V0.25 MODEL / CLV JOIN REPAIR & VALIDATION")
print("=" * 78)


# ============================================================================
# 1. LOAD DATA
# ============================================================================

model = pd.read_csv(MODEL_FILE)
matches = pd.read_csv(MATCHES_FILE)
clv20 = pd.read_csv(CLV_V20_FILE)
clv22 = pd.read_csv(CLV_V22_FILE)

print("\nDATASETS")
print("-" * 78)
print(f"Model predictions : {len(model)}")
print(f"Matches bridge    : {len(matches)}")
print(f"CLV v20           : {len(clv20)}")
print(f"CLV v22           : {len(clv22)}")


# ============================================================================
# 2. BASIC VALIDATION
# ============================================================================

required_model = [
    "match_id",
    "season",
    "date",
    "home_team_id",
    "away_team_id",
    "model_over_probability",
]

required_matches = [
    "match_id",
    "season",
    "date",
    "home_team_id",
    "away_team_id",
    "home_team",
    "away_team",
]

required_clv20 = [
    "match_id",
    "season",
    "canonical_id",
    "date",
    "home_team",
    "away_team",
    "selection",
]

required_clv22 = [
    "match_id",
    "season",
    "selection",
    "opening_price",
    "closing_price",
    "opening_fair_probability",
    "closing_fair_probability",
    "price_clv",
    "fair_probability_clv",
]

for name, df, columns in [
    ("model", model, required_model),
    ("matches", matches, required_matches),
    ("clv20", clv20, required_clv20),
    ("clv22", clv22, required_clv22),
]:
    missing = [c for c in columns if c not in df.columns]

    if missing:
        raise ValueError(
            f"{name} missing required columns: {missing}"
        )


# ============================================================================
# 3. DUPLICATE CHECKS
# ============================================================================

print("\nDUPLICATE CHECKS")
print("-" * 78)

model_dup = model["match_id"].duplicated().sum()
matches_dup = matches["match_id"].duplicated().sum()
clv20_dup = clv20.duplicated(["match_id", "selection"]).sum()
clv22_dup = clv22.duplicated(["match_id", "selection"]).sum()

print("Duplicate model match IDs:", model_dup)
print("Duplicate matches bridge IDs:", matches_dup)
print("Duplicate CLV v20 match+selection:", clv20_dup)
print("Duplicate CLV v22 match+selection:", clv22_dup)

if model_dup:
    raise ValueError("Duplicate model match IDs detected.")

if matches_dup:
    raise ValueError("Duplicate match bridge IDs detected.")

if clv20_dup:
    raise ValueError("Duplicate CLV v20 match+selection detected.")

if clv22_dup:
    raise ValueError("Duplicate CLV v22 match+selection detected.")


# ============================================================================
# 4. MODEL → MATCHES BRIDGE
# ============================================================================

print("\nMODEL → MATCHES BRIDGE")
print("-" * 78)

bridge = matches[
    [
        "match_id",
        "season",
        "date",
        "home_team_id",
        "away_team_id",
        "home_team",
        "away_team",
    ]
].copy()

# Rename bridge fields explicitly.
bridge = bridge.rename(
    columns={
        "season": "bridge_season",
        "date": "bridge_date",
        "home_team_id": "bridge_home_team_id",
        "away_team_id": "bridge_away_team_id",
        "home_team": "bridge_home_team",
        "away_team": "bridge_away_team",
    }
)

model_bridge = model.merge(
    bridge,
    on="match_id",
    how="left",
    indicator=True,
)

matched_bridge = (
    model_bridge["_merge"] == "both"
).sum()

print("Model rows:", len(model_bridge))
print("Successfully bridged:", matched_bridge)
print("Unmatched:", len(model_bridge) - matched_bridge)

if matched_bridge != len(model):
    raise ValueError(
        "Not all model matches could be bridged."
    )

model_bridge = model_bridge.drop(columns="_merge")


# ============================================================================
# 5. BUILD CANONICAL ID
# ============================================================================

print("\nCANONICAL ID GENERATION")
print("-" * 78)


def make_canonical_id(row):
    date = pd.to_datetime(
        row["date"],
        errors="coerce"
    )

    if pd.isna(date):
        return np.nan

    return (
        f"{str(row['season']).strip()}_"
        f"{date.strftime('%Y%m%d')}_"
        f"{str(row['home_team_id']).strip()}_"
        f"{str(row['away_team_id']).strip()}"
    )


model_bridge["canonical_id"] = model_bridge.apply(
    make_canonical_id,
    axis=1,
)

invalid_ids = model_bridge["canonical_id"].isna().sum()
duplicate_ids = model_bridge["canonical_id"].duplicated().sum()

print("Invalid canonical IDs:", invalid_ids)
print("Duplicate canonical IDs:", duplicate_ids)

if invalid_ids:
    raise ValueError(
        "Invalid model canonical IDs detected."
    )

if duplicate_ids:
    raise ValueError(
        "Duplicate model canonical IDs detected."
    )


# ============================================================================
# 6. CLV CANONICAL DATASET
# ============================================================================

print("\nCLV CANONICAL ID VALIDATION")
print("-" * 78)

clv_identity = clv20[
    [
        "canonical_id",
        "season",
        "match_id",
        "date",
        "home_team",
        "away_team",
    ]
].drop_duplicates()

clv_identity = clv_identity.rename(
    columns={
        "season": "clv_season",
        "match_id": "clv_match_id",
        "date": "clv_date",
        "home_team": "clv_home_team",
        "away_team": "clv_away_team",
    }
)

print(
    "Unique CLV canonical IDs:",
    clv_identity["canonical_id"].nunique()
)

print(
    "Unique CLV matches:",
    clv_identity["clv_match_id"].nunique()
)


# ============================================================================
# 7. MODEL → CLV CANONICAL JOIN
# ============================================================================

print("\nMODEL → CLV CANONICAL JOIN")
print("-" * 78)

joined = model_bridge.merge(
    clv_identity,
    on="canonical_id",
    how="inner",
)

print("Joined rows:", len(joined))
print(
    "Unique matched matches:",
    joined["canonical_id"].nunique()
)
print(
    "Model unique matches:",
    model_bridge["canonical_id"].nunique()
)

if joined.empty:
    raise ValueError(
        "No model/CLV canonical matches found."
    )


# ============================================================================
# 8. EVENT IDENTITY VALIDATION
# ============================================================================

print("\nEVENT IDENTITY VALIDATION")
print("-" * 78)

model_date = pd.to_datetime(
    joined["date"],
    errors="coerce"
)

clv_date = pd.to_datetime(
    joined["clv_date"],
    errors="coerce"
)

season_ok = (
    joined["season"].astype(str)
    ==
    joined["clv_season"].astype(str)
)

date_ok = model_date == clv_date

print(
    "Season mismatches:",
    (~season_ok).sum()
)

print(
    "Date mismatches:",
    (~date_ok).sum()
)

identity_ok = season_ok & date_ok

print(
    "Identity failures:",
    (~identity_ok).sum()
)

if (~identity_ok).any():
    raise ValueError(
        "Event identity validation failed."
    )


# ============================================================================
# 9. SEASON OVERLAP
# ============================================================================

print("\nSEASON OVERLAP")
print("-" * 78)

model_seasons = sorted(
    model["season"].astype(str).unique()
)

clv_seasons = sorted(
    clv20["season"].astype(str).unique()
)

overlap = sorted(
    set(model_seasons) & set(clv_seasons)
)

model_only = sorted(
    set(model_seasons) - set(clv_seasons)
)

clv_only = sorted(
    set(clv_seasons) - set(model_seasons)
)

print("Model seasons:")
print(model_seasons)

print("\nCLV seasons:")
print(clv_seasons)

print("\nExpected overlapping seasons:")
print(overlap)

print("\nModel-only seasons:")
print(model_only)

print("\nCLV-only seasons:")
print(clv_only)


# ============================================================================
# 10. COVERAGE BY SEASON
# ============================================================================

model_coverage = (
    model_bridge
    .groupby("season")["canonical_id"]
    .nunique()
    .reset_index(
        name="model_matches"
    )
)

clv_coverage = (
    joined
    .groupby("season")["canonical_id"]
    .nunique()
    .reset_index(
        name="clv_matched_matches"
    )

)

coverage = model_coverage.merge(
    clv_coverage,
    on="season",
    how="left",
)

coverage["clv_matched_matches"] = (
    coverage["clv_matched_matches"]
    .fillna(0)
    .astype(int)
)

coverage["match_rate"] = (
    coverage["clv_matched_matches"]
    /
    coverage["model_matches"]
)

print("\nCoverage by season:")
print(
    coverage.to_string(index=False)
)


# ============================================================================
# 11. IDENTIFY UNMATCHED MODEL EVENTS
# ============================================================================

print("\nUNMATCHED MODEL EVENTS")
print("-" * 78)

matched_canonical = set(
    joined["canonical_id"]
)

unmatched = model_bridge[
    ~model_bridge["canonical_id"].isin(
        matched_canonical
    )
].copy()

print(
    "Unmatched model matches:",
    len(unmatched)
)

if len(unmatched) > 0:
    print("\nFirst unmatched events:")
    print(
        unmatched[
            [
                "match_id",
                "season",
                "date",
                "home_team_id",
                "away_team_id",
                "canonical_id",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )


# ============================================================================
# 12. ATTACH CLV V22 VALUES
# ============================================================================

print("\nATTACHING VALIDATED CLV VALUES")
print("-" * 78)

clv_values = clv22[
    [
        "match_id",
        "season",
        "selection",
        "opening_price",
        "closing_price",
        "opening_fair_probability",
        "closing_fair_probability",
        "price_clv",
        "fair_probability_clv",
    ]
].copy()

clv_values = clv_values.rename(
    columns={
        "match_id": "clv_match_id",
        "season": "clv_season",
    }
)

final = joined.merge(
    clv_values,
    on=[
        "clv_match_id",
        "clv_season",
    ],
    how="inner",
)


# ============================================================================
# 13. MODEL PROBABILITY FOR EACH SELECTION
# ============================================================================

final["model_under_probability"] = (
    1.0 -
    final["model_over_probability"]
)

final["model_probability"] = np.where(
    final["selection"] == "OVER",
    final["model_over_probability"],
    final["model_under_probability"],
)


# ============================================================================
# 14. MODEL VS CLOSING FAIR PROBABILITY
# ============================================================================

final["model_vs_closing_fair_probability"] = (
    final["model_probability"]
    -
    final["closing_fair_probability"]
)


# ============================================================================
# 15. FINAL VALIDATION
# ============================================================================

print("\nFINAL VALIDATION")
print("-" * 78)

print(
    "Joined observations:",
    len(final)
)

print(
    "Unique matched matches:",
    final["canonical_id"].nunique()
)

print(
    "OVER observations:",
    (final["selection"] == "OVER").sum()
)

print(
    "UNDER observations:",
    (final["selection"] == "UNDER").sum()
)

print(
    "Missing CLV values:",
    final["fair_probability_clv"].isna().sum()
)

print(
    "Missing model probabilities:",
    final["model_probability"].isna().sum()
)

if final.empty:
    raise ValueError(
        "Final model/CLV dataset is empty."
    )

if final["fair_probability_clv"].isna().any():
    raise ValueError(
        "Missing CLV values detected."
    )

if final["model_probability"].isna().any():
    raise ValueError(
        "Missing model probabilities detected."
    )


# ============================================================================
# 16. SAVE
# ============================================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

final.to_csv(
    OUTPUT_FILE,
    index=False
)

coverage.to_csv(
    REPORT_FILE,
    index=False
)


# ============================================================================
# 17. SUMMARY
# ============================================================================

print("\nOUTPUTS")
print("-" * 78)
print("Joined dataset :", OUTPUT_FILE)
print("Coverage report:", REPORT_FILE)

print("\nSTATUS: V0.25 COMPLETE")
print("=" * 78)
