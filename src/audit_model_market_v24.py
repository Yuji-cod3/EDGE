from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

MODEL_FILE = ROOT / "data/processed/poisson_predictions.csv"
CLV_FILE = ROOT / "data/processed/clv_audit_v22.csv"

OUT_DATA = ROOT / "data/processed/model_market_audit_v24.csv"
OUT_REPORT = ROOT / "reports/model_market_audit_v24_report.csv"
OUT_SEASON = ROOT / "reports/model_market_audit_v24_seasons.csv"


def log_loss(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()


def brier(y, p):
    return ((p - y) ** 2).mean()


def safe_corr(a, b):
    if len(a) < 2:
        return np.nan
    if a.nunique() < 2 or b.nunique() < 2:
        return np.nan
    return a.corr(b)


print("=" * 78)
print("EDGE — V0.24 MODEL VS MARKET / CLV AUDIT")
print("=" * 78)


# ---------------------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------------------

if not MODEL_FILE.exists():
    raise FileNotFoundError(f"Missing model file: {MODEL_FILE}")

if not CLV_FILE.exists():
    raise FileNotFoundError(f"Missing CLV file: {CLV_FILE}")

model = pd.read_csv(MODEL_FILE)
clv = pd.read_csv(CLV_FILE)

print("\nMODEL DATASET")
print(f"Rows: {len(model)}")
print("Columns:")
for c in model.columns:
    print(f"  - {c}")

print("\nCLV DATASET")
print(f"Rows: {len(clv)}")


# ---------------------------------------------------------------------------
# 2. MODEL VALIDATION
# ---------------------------------------------------------------------------

required_model = [
    "match_id",
    "season",
    "target_over25",
    "market_over_probability",
    "model_over_probability",
]

missing = [c for c in required_model if c not in model.columns]

if missing:
    raise ValueError(f"Missing model columns: {missing}")

print("\n" + "=" * 78)
print("1. MODEL PROBABILITY VALIDATION")
print("=" * 78)

model["model_over_probability"] = pd.to_numeric(
    model["model_over_probability"], errors="coerce"
)

model["market_over_probability"] = pd.to_numeric(
    model["market_over_probability"], errors="coerce"
)

model["target_over25"] = pd.to_numeric(
    model["target_over25"], errors="coerce"
)

print(
    "Invalid model probabilities:",
    (
        model["model_over_probability"].isna()
        | ~model["model_over_probability"].between(0, 1)
    ).sum()
)

print(
    "Invalid market probabilities:",
    (
        model["market_over_probability"].isna()
        | ~model["market_over_probability"].between(0, 1)
    ).sum()
)

print("Missing targets:", model["target_over25"].isna().sum())


# ---------------------------------------------------------------------------
# 3. PREPARE MODEL DATA
# ---------------------------------------------------------------------------

m = model[
    required_model
    + (
        ["date", "model_version"]
        if "date" in model.columns and "model_version" in model.columns
        else []
    )
].copy()

m = m.dropna(
    subset=[
        "match_id",
        "season",
        "target_over25",
        "market_over_probability",
        "model_over_probability",
    ]
)

m["target_over25"] = m["target_over25"].astype(int)

m["model_edge"] = (
    m["model_over_probability"]
    - m["market_over_probability"]
)

m["model_abs_edge"] = m["model_edge"].abs()

m["model_advantage"] = np.where(
    m["model_edge"] > 0,
    "MODEL_OVER",
    np.where(
        m["model_edge"] < 0,
        "MODEL_UNDER",
        "NEUTRAL"
    )
)


# ---------------------------------------------------------------------------
# 4. MODEL VS MARKET PREDICTIVE PERFORMANCE
# ---------------------------------------------------------------------------

print("\n" + "=" * 78)
print("2. MODEL VS MARKET PREDICTIVE PERFORMANCE")
print("=" * 78)

y = m["target_over25"]

model_brier = brier(
    y,
    m["model_over_probability"]
)

market_brier = brier(
    y,
    m["market_over_probability"]
)

model_ll = log_loss(
    y,
    m["model_over_probability"]
)

market_ll = log_loss(
    y,
    m["market_over_probability"]
)

print(f"\nModel Brier Score : {model_brier:.6f}")
print(f"Market Brier Score: {market_brier:.6f}")

print(f"\nModel Log Loss : {model_ll:.6f}")
print(f"Market Log Loss: {market_ll:.6f}")

print("\nBrier difference (Model - Market):")
print(f"{model_brier - market_brier:.6f}")

print("\nLog-loss difference (Model - Market):")
print(f"{model_ll - market_ll:.6f}")


# ---------------------------------------------------------------------------
# 5. MODEL / MARKET PROBABILITY RELATIONSHIP
# ---------------------------------------------------------------------------

print("\n" + "=" * 78)
print("3. MODEL / MARKET RELATIONSHIP")
print("=" * 78)

corr = safe_corr(
    m["model_over_probability"],
    m["market_over_probability"]
)

print(f"\nProbability correlation: {corr:.6f}")

print(
    "Mean model probability:",
    f"{m['model_over_probability'].mean():.6f}"
)

print(
    "Mean market probability:",
    f"{m['market_over_probability'].mean():.6f}"
)

print(
    "Mean model edge:",
    f"{m['model_edge'].mean():.6f}"
)

print(
    "Median model edge:",
    f"{m['model_edge'].median():.6f}"
)

print(
    "Positive model edge rate:",
    f"{(m['model_edge'] > 0).mean():.6f}"
)


# ---------------------------------------------------------------------------
# 6. MODEL EDGE BUCKETS
# ---------------------------------------------------------------------------

print("\n" + "=" * 78)
print("4. MODEL EDGE BUCKETS")
print("=" * 78)

bins = [
    -np.inf,
    -0.10,
    -0.05,
    -0.025,
    -0.01,
    0,
    0.01,
    0.025,
    0.05,
    0.10,
    np.inf,
]

labels = [
    "< -10%",
    "-10% to -5%",
    "-5% to -2.5%",
    "-2.5% to -1%",
    "-1% to 0%",
    "0% to 1%",
    "1% to 2.5%",
    "2.5% to 5%",
    "5% to 10%",
    "> 10%",
]

m["edge_bucket"] = pd.cut(
    m["model_edge"],
    bins=bins,
    labels=labels,
    right=False
)

bucket_rows = []

for bucket, g in m.groupby(
    "edge_bucket",
    observed=False
):
    if len(g) == 0:
        continue

    bucket_rows.append({
        "edge_bucket": str(bucket),
        "observations": len(g),
        "mean_model_probability":
            g["model_over_probability"].mean(),
        "mean_market_probability":
            g["market_over_probability"].mean(),
        "mean_edge":
            g["model_edge"].mean(),
        "actual_over_rate":
            g["target_over25"].mean(),
        "brier_model":
            brier(
                g["target_over25"],
                g["model_over_probability"]
            ),
        "brier_market":
            brier(
                g["target_over25"],
                g["market_over_probability"]
            ),
    })

bucket_df = pd.DataFrame(bucket_rows)

if len(bucket_df):
    print(bucket_df.to_string(index=False))


# ---------------------------------------------------------------------------
# 7. MATCH MODEL TO CLV
# ---------------------------------------------------------------------------

print("\n" + "=" * 78)
print("5. MODEL → CLV MATCHING")
print("=" * 78)

# CLV file contains one row per selection.
# Build an OVER and UNDER CLV view so each match can be compared
# against the model's corresponding probability.

clv_needed = [
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

missing_clv = [
    c for c in clv_needed
    if c not in clv.columns
]

if missing_clv:
    raise ValueError(f"Missing CLV columns: {missing_clv}")

clv_small = clv[clv_needed].copy()

over = clv_small[
    clv_small["selection"].eq("OVER")
].copy()

under = clv_small[
    clv_small["selection"].eq("UNDER")
].copy()

over = over.rename(
    columns={
        "opening_price": "opening_over_odds",
        "closing_price": "closing_over_odds",
        "opening_fair_probability": "opening_over_fair_probability",
        "closing_fair_probability": "closing_over_fair_probability",
        "price_clv": "over_price_clv",
        "fair_probability_clv": "over_fair_clv",
    }
).drop(columns=["selection"])

under = under.rename(
    columns={
        "opening_price": "opening_under_odds",
        "closing_price": "closing_under_odds",
        "opening_fair_probability": "opening_under_fair_probability",
        "closing_fair_probability": "closing_under_fair_probability",
        "price_clv": "under_price_clv",
        "fair_probability_clv": "under_fair_clv",
    }
).drop(columns=["selection"])

clv_match = over.merge(
    under,
    on=["match_id", "season"],
    how="inner"
)

print(f"CLV matches: {len(clv_match)}")


# ---------------------------------------------------------------------------
# 8. JOIN MODEL TO CLV
# ---------------------------------------------------------------------------

analysis = m.merge(
    clv_match,
    on=["match_id", "season"],
    how="inner",
    validate="one_to_one"
)

print(f"Model/CLV matched matches: {len(analysis)}")

if len(analysis) == 0:
    raise ValueError("No model/CLV matches found.")

analysis["model_under_probability"] = (
    1.0 - analysis["model_over_probability"]
)

analysis["opening_model_edge_over"] = (
    analysis["model_over_probability"]
    - analysis["opening_over_fair_probability"]
)

analysis["opening_model_edge_under"] = (
    analysis["model_under_probability"]
    - analysis["opening_under_fair_probability"]
)

analysis["closing_model_edge_over"] = (
    analysis["model_over_probability"]
    - analysis["closing_over_fair_probability"]
)

analysis["closing_model_edge_under"] = (
    analysis["model_under_probability"]
    - analysis["closing_under_fair_probability"]
)


# ---------------------------------------------------------------------------
# 9. DOES MODEL EDGE PREDICT CLV?
# ---------------------------------------------------------------------------

print("\n" + "=" * 78)
print("6. MODEL EDGE VS SUBSEQUENT CLV")
print("=" * 78)

print(
    "\nCorrelation between model edge and OVER fair CLV:",
    f"{safe_corr(analysis['opening_model_edge_over'], analysis['over_fair_clv']):.6f}"
)

print(
    "Correlation between model edge and UNDER fair CLV:",
    f"{safe_corr(analysis['opening_model_edge_under'], analysis['under_fair_clv']):.6f}"
)

# The model's preferred side:
analysis["selected_side"] = np.where(
    analysis["opening_model_edge_over"] > 0,
    "OVER",
    np.where(
        analysis["opening_model_edge_under"] > 0,
        "UNDER",
        "NONE"
    )
)

analysis["selected_model_edge"] = np.where(
    analysis["selected_side"].eq("OVER"),
    analysis["opening_model_edge_over"],
    np.where(
        analysis["selected_side"].eq("UNDER"),
        analysis["opening_model_edge_under"],
        np.nan
    )
)

analysis["selected_fair_clv"] = np.where(
    analysis["selected_side"].eq("OVER"),
    analysis["over_fair_clv"],
    np.where(
        analysis["selected_side"].eq("UNDER"),
        analysis["under_fair_clv"],
        np.nan
    )
)

selected = analysis[
    analysis["selected_side"].ne("NONE")
].copy()

print(
    f"\nMatches with a positive model edge: {len(selected)} / {len(analysis)}"
)

if len(selected):
    print(
        "Mean selected model edge:",
        f"{selected['selected_model_edge'].mean():.6f}"
    )

    print(
        "Mean selected fair CLV:",
        f"{selected['selected_fair_clv'].mean():.6f}"
    )

    print(
        "Positive selected fair CLV rate:",
        f"{(selected['selected_fair_clv'] > 0).mean():.6f}"
    )

    print(
        "Correlation model edge → selected fair CLV:",
        f"{safe_corr(selected['selected_model_edge'], selected['selected_fair_clv']):.6f}"
    )


# ---------------------------------------------------------------------------
# 10. MODEL EDGE DECILES
# ---------------------------------------------------------------------------

print("\n" + "=" * 78)
print("7. MODEL EDGE DECILE ANALYSIS")
print("=" * 78)

if len(selected) >= 20:
    selected["edge_decile"] = pd.qcut(
        selected["selected_model_edge"],
        q=10,
        duplicates="drop"
    )

    decile_rows = []

    for decile, g in selected.groupby(
        "edge_decile",
        observed=False
    ):
        decile_rows.append({
            "edge_decile": str(decile),
            "observations": len(g),
            "mean_model_edge":
                g["selected_model_edge"].mean(),
            "mean_fair_clv":
                g["selected_fair_clv"].mean(),
            "positive_fair_clv_rate":
                (g["selected_fair_clv"] > 0).mean(),
        })

    decile_df = pd.DataFrame(decile_rows)

    print(decile_df.to_string(index=False))

else:
    decile_df = pd.DataFrame()


# ---------------------------------------------------------------------------
# 11. SEASON ANALYSIS
# ---------------------------------------------------------------------------

print("\n" + "=" * 78)
print("8. SEASON ANALYSIS")
print("=" * 78)

season_rows = []

for season, g in analysis.groupby("season"):

    model_b = brier(
        g["target_over25"],
        g["model_over_probability"]
    )

    market_b = brier(
        g["target_over25"],
        g["market_over_probability"]
    )

    model_l = log_loss(
        g["target_over25"],
        g["model_over_probability"]
    )

    market_l = log_loss(
        g["target_over25"],
        g["market_over_probability"]
    )

    sg = g[g["selected_side"].ne("NONE")]

    season_rows.append({
        "season": season,
        "matches": len(g),
        "model_brier": model_b,
        "market_brier": market_b,
        "model_brier_minus_market": model_b - market_b,
        "model_log_loss": model_l,
        "market_log_loss": market_l,
        "model_log_loss_minus_market": model_l - market_l,
        "positive_edge_matches": len(sg),
        "positive_edge_rate":
            len(sg) / len(g) if len(g) else np.nan,
        "mean_selected_edge":
            sg["selected_model_edge"].mean()
            if len(sg) else np.nan,
        "mean_selected_fair_clv":
            sg["selected_fair_clv"].mean()
            if len(sg) else np.nan,
    })

season_df = pd.DataFrame(season_rows)

print(season_df.to_string(index=False))


# ---------------------------------------------------------------------------
# 12. FINAL INTERPRETATION FLAGS
# ---------------------------------------------------------------------------

print("\n" + "=" * 78)
print("9. V0.24 INTERPRETATION")
print("=" * 78)

if model_brier < market_brier:
    brier_verdict = "MODEL_BETTER"
else:
    brier_verdict = "MARKET_BETTER"

if model_ll < market_ll:
    ll_verdict = "MODEL_BETTER"
else:
    ll_verdict = "MARKET_BETTER"

selected_clv_mean = (
    selected["selected_fair_clv"].mean()
    if len(selected)
    else np.nan
)

if np.isnan(selected_clv_mean):
    clv_verdict = "INSUFFICIENT_DATA"
elif selected_clv_mean > 0:
    clv_verdict = "POSITIVE_SELECTED_CLV"
else:
    clv_verdict = "NON_POSITIVE_SELECTED_CLV"

print(f"\nBrier verdict : {brier_verdict}")
print(f"Log-loss verdict: {ll_verdict}")
print(f"Selected CLV verdict: {clv_verdict}")

print("\nIMPORTANT:")
print(
    "This audit does NOT establish profitability, "
    "causal predictive power, or out-of-sample superiority."
)
print(
    "It establishes whether the Poisson baseline adds measurable "
    "information relative to the market and whether its selections "
    "are associated with subsequent market movement."
)


# ---------------------------------------------------------------------------
# 13. OUTPUT
# ---------------------------------------------------------------------------

OUT_DATA.parent.mkdir(parents=True, exist_ok=True)
OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)

analysis.to_csv(OUT_DATA, index=False)

report_rows = [
    {
        "metric": "model_brier",
        "value": model_brier,
    },
    {
        "metric": "market_brier",
        "value": market_brier,
    },
    {
        "metric": "brier_difference_model_minus_market",
        "value": model_brier - market_brier,
    },
    {
        "metric": "model_log_loss",
        "value": model_ll,
    },
    {
        "metric": "market_log_loss",
        "value": market_ll,
    },
    {
        "metric": "log_loss_difference_model_minus_market",
        "value": model_ll - market_ll,
    },
    {
        "metric": "probability_correlation",
        "value": corr,
    },
    {
        "metric": "mean_model_edge",
        "value": m["model_edge"].mean(),
    },
    {
        "metric": "positive_model_edge_rate",
        "value": (m["model_edge"] > 0).mean(),
    },
    {
        "metric": "model_clv_matched_matches",
        "value": len(analysis),
    },
    {
        "metric": "positive_edge_matches",
        "value": len(selected),
    },
    {
        "metric": "mean_selected_model_edge",
        "value": (
            selected["selected_model_edge"].mean()
            if len(selected) else np.nan
        ),
    },
    {
        "metric": "mean_selected_fair_clv",
        "value": selected_clv_mean,
    },
    {
        "metric": "positive_selected_fair_clv_rate",
        "value": (
            (selected["selected_fair_clv"] > 0).mean()
            if len(selected) else np.nan
        ),
    },
    {
        "metric": "model_edge_selected_clv_correlation",
        "value": (
            safe_corr(
                selected["selected_model_edge"],
                selected["selected_fair_clv"]
            )
            if len(selected) > 1 else np.nan
        ),
    },
]

pd.DataFrame(report_rows).to_csv(
    OUT_REPORT,
    index=False
)

season_df.to_csv(
    OUT_SEASON,
    index=False
)

print("\n" + "=" * 78)
print("10. OUTPUT")
print("=" * 78)

print(f"\nSaved audit dataset:")
print(OUT_DATA)

print("\nSaved report:")
print(OUT_REPORT)

print("\nSaved season summary:")
print(OUT_SEASON)

print("\n" + "=" * 78)
print("V0.24 STATUS: COMPLETE — REVIEW RESULTS")
print("=" * 78)
