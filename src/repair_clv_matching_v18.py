from pathlib import Path
import pandas as pd
import re

ROOT = Path(__file__).resolve().parents[1]

ODDS_FILE = ROOT / "data/processed/market_odds_v15.csv"
MATCH_FILE = ROOT / "data/processed/matches.csv"

OUT_FILE = ROOT / "data/processed/clv_match_mapping_v18.csv"
REPORT_FILE = ROOT / "reports/clv_matching_v18_report.csv"


def norm_text(x):
    if pd.isna(x):
        return ""

    x = str(x).strip().lower()

    x = re.sub(r"[^a-z0-9]+", "", x)

    return x


def norm_date(x):
    if pd.isna(x):
        return ""

    dt = pd.to_datetime(x, errors="coerce")

    if pd.isna(dt):
        return ""

    return dt.strftime("%Y-%m-%d")


print("=" * 78)
print("EDGE — V0.18 CLV RESULT MATCHING REPAIR")
print("=" * 78)

# ----------------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------------

odds = pd.read_csv(ODDS_FILE)
matches = pd.read_csv(MATCH_FILE)

print("\nODDS DATASET")
print(f"Rows: {len(odds)}")
print("Columns:")
for c in odds.columns:
    print(f"  - {c}")

print("\nMATCH DATASET")
print(f"Rows: {len(matches)}")
print("Columns:")
for c in matches.columns:
    print(f"  - {c}")


# ----------------------------------------------------------------------
# 2. CLV SAMPLE
# ----------------------------------------------------------------------

clv = odds[
    odds["opening_price"].notna()
    & odds["closing_price"].notna()
].copy()

print("\n" + "=" * 78)
print("1. CLV SAMPLE")
print("=" * 78)

print(f"CLV observations: {len(clv)}")
print(f"Unique matches: {clv['match_id'].nunique()}")

print("\nExample odds identifiers:")
print(
    clv[
        [
            "match_id",
            "season",
            "selection",
            "opening_price",
            "closing_price",
        ]
    ].head(10).to_string(index=False)
)


# ----------------------------------------------------------------------
# 3. INSPECT MATCH IDENTIFIERS
# ----------------------------------------------------------------------

print("\n" + "=" * 78)
print("2. MATCH DATA IDENTIFIER INSPECTION")
print("=" * 78)

print("\nFirst result rows:")
print(matches.head(10).to_string(index=False))


# ----------------------------------------------------------------------
# 4. IDENTIFY RESULT COLUMNS
# ----------------------------------------------------------------------

def find_column(df, candidates):

    lower = {c.lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]

    return None


home_col = find_column(
    matches,
    [
        "HomeTeam",
        "home_team",
        "home",
        "Home",
    ],
)

away_col = find_column(
    matches,
    [
        "AwayTeam",
        "away_team",
        "away",
        "Away",
    ],
)

date_col = find_column(
    matches,
    [
        "Date",
        "date",
        "match_date",
    ],
)

season_col = find_column(
    matches,
    [
        "season",
        "Season",
    ],
)

result_col = find_column(
    matches,
    [
        "over25_result",
        "Over25Result",
        "FTR",
        "result",
        "Result",
        "outcome",
    ],
)
print("\nDetected columns:")
print(f"  Home:   {home_col}")
print(f"  Away:   {away_col}")
print(f"  Date:   {date_col}")
print(f"  Season: {season_col}")
print(f"  Result: {result_col}")


required = [
    home_col,
    away_col,
    date_col,
    season_col,
    result_col,
]

if any(x is None for x in required):
    print("\nERROR: Could not identify all required result columns.")
    print("Do not continue until the result schema is corrected.")
    raise SystemExit(1)


# ----------------------------------------------------------------------
# 5. BUILD NORMALIZED RESULT KEYS
# ----------------------------------------------------------------------

results = matches.copy()

results["_home_norm"] = results[home_col].map(norm_text)
results["_away_norm"] = results[away_col].map(norm_text)
results["_date_norm"] = results[date_col].map(norm_date)

results["_season_norm"] = (
    results[season_col]
    .astype(str)
    .str.strip()
)


# ----------------------------------------------------------------------
# 6. PARSE MATCH ID
# ----------------------------------------------------------------------

def parse_match_id(match_id):

    if pd.isna(match_id):
        return None, None, None, None

    s = str(match_id)

    # Expected EDGE format:
    # 2022_23_21/05/2023_Man City_Chelsea

    parts = s.split("_")

    if len(parts) < 5:
        return None, None, None, None

    season = parts[0] + "_" + parts[1]
    date = parts[2]
    home = parts[3]
    away = "_".join(parts[4:])

    return (
        season,
        norm_date(date),
        norm_text(home),
        norm_text(away),
    )


parsed = clv["match_id"].map(parse_match_id)

clv["_parsed_season"] = parsed.map(lambda x: x[0])
clv["_parsed_date"] = parsed.map(lambda x: x[1])
clv["_parsed_home"] = parsed.map(lambda x: x[2])
clv["_parsed_away"] = parsed.map(lambda x: x[3])


# ----------------------------------------------------------------------
# 7. BUILD RESULT KEY
# ----------------------------------------------------------------------

results["_result_key"] = (
    results["_season_norm"]
    + "|"
    + results["_date_norm"]
    + "|"
    + results["_home_norm"]
    + "|"
    + results["_away_norm"]
)

clv["_result_key"] = (
    clv["_parsed_season"]
    + "|"
    + clv["_parsed_date"]
    + "|"
    + clv["_parsed_home"]
    + "|"
    + clv["_parsed_away"]
)


# ----------------------------------------------------------------------
# 8. MATCH
# ----------------------------------------------------------------------

result_lookup = (
    results
    .drop_duplicates("_result_key")
    .set_index("_result_key")
)


clv["_result_found"] = clv["_result_key"].isin(result_lookup.index)

matched = clv[clv["_result_found"]].copy()

print("\n" + "=" * 78)
print("3. MATCHING RESULT")
print("=" * 78)

print(f"CLV observations: {len(clv)}")
print(f"Matched observations: {len(matched)}")
print(
    f"Match rate: "
    f"{len(matched) / len(clv) * 100:.2f}%"
)

print(
    f"Unmatched observations: "
    f"{len(clv) - len(matched)}"
)


# ----------------------------------------------------------------------
# 9. ATTACH RESULTS
# ----------------------------------------------------------------------

if len(matched) > 0:

    matched[result_col] = matched["_result_key"].map(
        result_lookup[result_col]
    )

    matched[home_col] = matched["_result_key"].map(
        result_lookup[home_col]
    )

    matched[away_col] = matched["_result_key"].map(
        result_lookup[away_col]
    )

    matched[date_col] = matched["_result_key"].map(
        result_lookup[date_col]
    )


# ----------------------------------------------------------------------
# 10. UNMATCHED EXAMPLES
# ----------------------------------------------------------------------

unmatched = clv[~clv["_result_found"]].copy()

print("\n" + "=" * 78)
print("4. UNMATCHED EXAMPLES")
print("=" * 78)

if len(unmatched) == 0:

    print("None.")

else:

    print(
        unmatched[
            [
                "match_id",
                "season",
                "selection",
            ]
        ]
        .drop_duplicates("match_id")
        .head(20)
        .to_string(index=False)
    )


# ----------------------------------------------------------------------
# 11. MATCH QUALITY
# ----------------------------------------------------------------------

print("\n" + "=" * 78)
print("5. MATCH QUALITY")
print("=" * 78)

if len(matched) > 0:

    print(
        "\nResult distribution:"
    )

    print(
        matched[result_col]
        .value_counts(dropna=False)
        .to_string()
    )

    print(
        "\nMatched seasons:"
    )

    print(
        matched["season"]
        .value_counts()
        .sort_index()
        .to_string()
    )


# ----------------------------------------------------------------------
# 12. SAVE CLEAN MAPPING
# ----------------------------------------------------------------------

keep = [
    "match_id",
    "season",
    "selection",
    "opening_price",
    "closing_price",
    "opening_source",
    "closing_source",
    result_col,
]

keep = [c for c in keep if c in matched.columns]

matched[keep].to_csv(
    OUT_FILE,
    index=False
)

print("\n" + "=" * 78)
print("6. OUTPUT")
print("=" * 78)

print(f"Saved mapping: {OUT_FILE}")


# ----------------------------------------------------------------------
# 13. REPORT
# ----------------------------------------------------------------------

report = pd.DataFrame(
    [
        {
            "clv_observations": len(clv),
            "matched_observations": len(matched),
            "unmatched_observations": len(unmatched),
            "match_rate": (
                len(matched) / len(clv)
                if len(clv)
                else 0
            ),
            "unique_clv_matches": clv["match_id"].nunique(),
            "unique_matched_matches": matched["match_id"].nunique(),
        }
    ]
)

report.to_csv(
    REPORT_FILE,
    index=False
)

print(f"Saved report: {REPORT_FILE}")

print("\n" + "=" * 78)

if len(matched) == len(clv):

    print("V0.18 STATUS: PASS")
    print("All CLV observations matched to actual results.")

elif len(matched) > 0:

    print("V0.18 STATUS: PARTIAL")
    print("Some CLV observations remain unmatched.")

else:

    print("V0.18 STATUS: FAIL")
    print("No CLV observations matched.")

print("=" * 78)
