from pathlib import Path
import re
import pandas as pd


# ============================================================================
# EDGE — V0.19 CLV MATCHING REPAIR
# ============================================================================
#
# Purpose:
#   Repair the mismatch between the CLV dataset's human-readable match_id
#   and the canonical match_id used by matches.csv.
#
# Example:
#
#   CLV:
#       2019_20_09/08/2019_Liverpool_Norwich
#
#   MATCH:
#       2019_20_20190809_LIV_NOR
#
#   Canonical:
#       2019_20_20190809_LIV_NOR
#
# ============================================================================


ROOT = Path(__file__).resolve().parents[1]

ODDS_FILE = ROOT / "data/processed/market_odds_v15.csv"
MATCH_FILE = ROOT / "data/processed/matches.csv"

OUTPUT_MAPPING = (
    ROOT / "data/processed/clv_match_mapping_v19.csv"
)

OUTPUT_REPORT = (
    ROOT / "reports/clv_matching_v19_report.csv"
)

OUTPUT_DIAGNOSTIC = (
    ROOT / "reports/clv_matching_v19_diagnostic.csv"
)


print("=" * 78)
print("EDGE — V0.19 CLV MATCHING REPAIR")
print("=" * 78)


# ============================================================================
# 1. LOAD DATA
# ============================================================================

odds = pd.read_csv(ODDS_FILE)
matches = pd.read_csv(MATCH_FILE)

print("\nODDS DATASET")
print(f"Rows: {len(odds)}")

print("\nMATCH DATASET")
print(f"Rows: {len(matches)}")


# ============================================================================
# 2. VALIDATE REQUIRED COLUMNS
# ============================================================================

required_odds = {
    "match_id",
    "season",
    "selection",
    "opening_price",
    "closing_price",
}

required_matches = {
    "match_id",
    "season",
    "date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "total_goals",
    "over25_result",
}

missing_odds = required_odds - set(odds.columns)
missing_matches = required_matches - set(matches.columns)

if missing_odds:
    raise ValueError(
        f"Missing odds columns: {sorted(missing_odds)}"
    )

if missing_matches:
    raise ValueError(
        f"Missing match columns: {sorted(missing_matches)}"
    )


# ============================================================================
# 3. TEAM NAME → CANONICAL CODE
# ============================================================================

TEAM_MAP = {
    "Liverpool": "LIV",
    "Norwich": "NOR",
    "Norwich City": "NOR",

    "Man City": "MCI",
    "Manchester City": "MCI",

    "Man United": "MUN",
    "Manchester United": "MUN",

    "West Ham": "WHU",
    "West Ham United": "WHU",

    "Bournemouth": "BOU",
    "AFC Bournemouth": "BOU",

    "Sheffield United": "SHU",

    "Burnley": "BUR",
    "Southampton": "SOU",
    "Crystal Palace": "CRY",
    "Everton": "EVE",

    "Tottenham": "TOT",
    "Tottenham Hotspur": "TOT",

    "Arsenal": "ARS",
    "Chelsea": "CHE",

    "Leicester": "LEI",
    "Leicester City": "LEI",

    "Newcastle": "NEW",
    "Newcastle United": "NEW",

    "Brighton": "BRI",
    "Brighton & Hove Albion": "BRI",

    "Aston Villa": "AVL",

    "Wolves": "WOL",
    "Wolverhampton Wanderers": "WOL",

    "Watford": "WAT",

    "Stoke": "STK",
    "Stoke City": "STK",

    "Swansea": "SWA",
    "Swansea City": "SWA",

    "Sunderland": "SUN",

    "West Brom": "WBA",
    "West Bromwich Albion": "WBA",

    "Middlesbrough": "MID",

    "Hull": "HUL",
    "Hull City": "HUL",

    "QPR": "QPR",
    "Queens Park Rangers": "QPR",

    "Fulham": "FUL",

    "Brentford": "BRE",

    "Nott'm Forest": "NFO",
    "Nottingham Forest": "NFO",

    "Leeds": "LEE",
    "Leeds United": "LEE",

    "Ipswich": "IPS",
    "Ipswich Town": "IPS",

    "Luton": "LUT",
    "Luton Town": "LUT",

    "Cardiff": "CAR",
    "Cardiff City": "CAR",

    "Huddersfield": "HUD",
    "Huddersfield Town": "HUD",

    "Reading": "REA",

    "Wigan": "WIG",
    "Wigan Athletic": "WIG",

    "Blackburn": "BLB",
    "Blackburn Rovers": "BLB",

    "Bolton": "BOL",
    "Bolton Wanderers": "BOL",

    "Derby": "DER",
    "Derby County": "DER",

    "Portsmouth": "POR",

    "Coventry": "COV",
    "Coventry City": "COV",

    "Birmingham": "BIR",
    "Birmingham City": "BIR",
}


def normalize_team(name):
    if pd.isna(name):
        return None

    name = str(name).strip()

    return TEAM_MAP.get(name)


# ============================================================================
# 4. BUILD CANONICAL IDS FROM MATCH DATA
# ============================================================================

print("\n" + "=" * 78)
print("1. BUILDING CANONICAL MATCH IDS")
print("=" * 78)


def normalize_date(value):

    if pd.isna(value):
        return None

    value = str(value).strip()

    for fmt in (
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%Y%m%d",
    ):

        dt = pd.to_datetime(
            value,
            format=fmt,
            errors="coerce",
        )

        if pd.notna(dt):
            return dt.strftime("%Y%m%d")

    return None


matches["canonical_home"] = (
    matches["home_team"]
    .apply(normalize_team)
)

matches["canonical_away"] = (
    matches["away_team"]
    .apply(normalize_team)
)

matches["canonical_date"] = (
    matches["date"]
    .apply(normalize_date)
)

matches["canonical_id"] = (
    matches["season"].astype(str)
    + "_"
    + matches["canonical_date"].fillna("")
    + "_"
    + matches["canonical_home"].fillna("")
    + "_"
    + matches["canonical_away"].fillna("")
)


print(
    "Unresolved home teams:",
    matches["canonical_home"].isna().sum(),
)

print(
    "Unresolved away teams:",
    matches["canonical_away"].isna().sum(),
)

print(
    "Unresolved dates:",
    matches["canonical_date"].isna().sum(),
)


# ============================================================================
# 5. PARSE CLV MATCH IDENTIFIERS
# ============================================================================

print("\n" + "=" * 78)
print("2. PARSING CLV IDENTIFIERS")
print("=" * 78)


def parse_clv_match_id(match_id):

    if pd.isna(match_id):
        return None, None, None, None

    value = str(match_id).strip()

    # Expected:
    #
    # 2019_20_09/08/2019_Liverpool_Norwich
    #
    # Season = 2019_20
    # Date   = 09/08/2019
    # Home   = Liverpool
    # Away   = Norwich

    pattern = (
        r"^(\d{4}_\d{2})_"
        r"(\d{2}/\d{2}/\d{4})_"
        r"(.+?)_"
        r"(.+)$"
    )

    match = re.match(pattern, value)

    if not match:
        return None, None, None, None

    season = match.group(1)
    date_raw = match.group(2)
    home = match.group(3)
    away = match.group(4)

    return season, date_raw, home, away


parsed = odds["match_id"].apply(parse_clv_match_id)

odds[
    [
        "parsed_season",
        "parsed_date_raw",
        "parsed_home",
        "parsed_away",
    ]
] = pd.DataFrame(
    parsed.tolist(),
    index=odds.index,
)


odds["parsed_date"] = (
    odds["parsed_date_raw"]
    .apply(normalize_date)
)

odds["parsed_home_code"] = (
    odds["parsed_home"]
    .apply(normalize_team)
)

odds["parsed_away_code"] = (
    odds["parsed_away"]
    .apply(normalize_team)
)


odds["canonical_id"] = (
    odds["parsed_season"].fillna("")
    + "_"
    + odds["parsed_date"].fillna("")
    + "_"
    + odds["parsed_home_code"].fillna("")
    + "_"
    + odds["parsed_away_code"].fillna("")
)


# ============================================================================
# 6. CLV SAMPLE
# ============================================================================

print("\n" + "=" * 78)
print("3. CLV SAMPLE")
print("=" * 78)


clv = odds[
    odds["opening_price"].notna()
    & odds["closing_price"].notna()
].copy()


print("CLV observations:", len(clv))
print(
    "Unique matches:",
    clv["match_id"].nunique(),
)


# ============================================================================
# 7. MATCH
# ============================================================================

print("\n" + "=" * 78)
print("4. MATCHING")
print("=" * 78)


match_columns = [
    "canonical_id",
    "match_id",
    "season",
    "date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "total_goals",
    "over25_result",
]


match_reference = matches[
    match_columns
].copy()


duplicate_match_ids = (
    match_reference["canonical_id"]
    .duplicated(keep=False)
)

duplicate_count = (
    match_reference.loc[
        duplicate_match_ids,
        "canonical_id",
    ]
    .nunique()
)


print(
    "Duplicate canonical match IDs:",
    duplicate_count,
)


clv = clv.merge(
    match_reference,
    on="canonical_id",
    how="left",
    suffixes=("", "_result"),
)


clv["matched"] = (
    clv["match_id_result"].notna()
)


matched_count = int(
    clv["matched"].sum()
)

total_count = len(clv)

unmatched_count = (
    total_count - matched_count
)

match_rate = (
    matched_count / total_count * 100
    if total_count
    else 0
)


print("\nCLV observations:", total_count)
print("Matched observations:", matched_count)
print(f"Match rate: {match_rate:.2f}%")
print("Unmatched observations:", unmatched_count)


# ============================================================================
# 8. UNMATCHED DIAGNOSTIC
# ============================================================================

print("\n" + "=" * 78)
print("5. UNMATCHED DIAGNOSTIC")
print("=" * 78)


unmatched = clv[
    ~clv["matched"]
].copy()


diagnostic_columns = [
    "match_id",
    "season",
    "selection",
    "parsed_season",
    "parsed_date_raw",
    "parsed_home",
    "parsed_away",
    "parsed_date",
    "parsed_home_code",
    "parsed_away_code",
    "canonical_id",
]


diagnostic = unmatched[
    diagnostic_columns
].copy()


diagnostic.to_csv(
    OUTPUT_DIAGNOSTIC,
    index=False,
)


print(
    "Saved:",
    OUTPUT_DIAGNOSTIC,
)


if len(unmatched) > 0:

    print("\nFirst 30 unmatched:")

    print(
        diagnostic
        .head(30)
        .to_string(index=False)
    )


# ============================================================================
# 9. OUTPUT MAPPING
# ============================================================================

mapping_columns = [
    "match_id",
    "season",
    "selection",
    "opening_price",
    "closing_price",
    "canonical_id",
    "match_id_result",
    "date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "total_goals",
    "over25_result",
    "matched",
]


mapping = clv[
    mapping_columns
].copy()


mapping.to_csv(
    OUTPUT_MAPPING,
    index=False,
)


# ============================================================================
# 10. REPORT
# ============================================================================

report = pd.DataFrame(
    [
        {
            "version": "v0.19",
            "clv_observations": total_count,
            "unique_matches": clv["match_id"].nunique(),
            "matched_observations": matched_count,
            "unmatched_observations": unmatched_count,
            "match_rate_percent": match_rate,
            "duplicate_canonical_match_ids": duplicate_count,
            "unresolved_clv_dates": int(
                clv["parsed_date"].isna().sum()
            ),
            "unresolved_clv_home_teams": int(
                clv["parsed_home_code"].isna().sum()
            ),
            "unresolved_clv_away_teams": int(
                clv["parsed_away_code"].isna().sum()
            ),
        }
    ]
)


report.to_csv(
    OUTPUT_REPORT,
    index=False,
)


# ============================================================================
# 11. FINAL STATUS
# ============================================================================

print("\n" + "=" * 78)
print("6. OUTPUT")
print("=" * 78)

print(
    "Saved mapping:",
    OUTPUT_MAPPING,
)

print(
    "Saved report:",
    OUTPUT_REPORT,
)

print(
    "Saved diagnostic:",
    OUTPUT_DIAGNOSTIC,
)


if match_rate >= 99.0:
    status = "SUCCESS"

elif match_rate >= 95.0:
    status = "STRONG"

elif match_rate > 14.44:
    status = "IMPROVED — FURTHER REPAIR REQUIRED"

else:
    status = "NO SIGNIFICANT IMPROVEMENT"


print(
    f"\nV0.19 STATUS: {status}"
)

print("=" * 78)
