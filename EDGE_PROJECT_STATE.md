# EDGE — Project State

## Current Version
V0.30

## Project Purpose
EDGE is a sports analytics research system focused initially on English Premier League football and the Over/Under 2.5 goals market.

The objective is to estimate outcome probabilities, compare them with market-implied probabilities and closing-line information, investigate potential value, and evaluate the system through rigorous out-of-sample testing.

EDGE is a research and paper-testing project. It does not autonomously place real-money bets.

---

## Current Scope

Sport:
- Football / Soccer

Competition:
- English Premier League

Primary Market:
- Over/Under 2.5 total goals

Historical Dataset:
- EPL seasons 2015/16 through 2025/26
- 11 seasons
- 4,180 matches

---

## Current Pipeline

DATA
→ VALIDATION
→ NORMALIZATION
→ FEATURE CONSTRUCTION
→ PROBABILITY MODEL
→ MARKET COMPARISON
→ CLV ANALYSIS
→ VALUE ANALYSIS
→ OUT-OF-SAMPLE EVALUATION
→ LEAKAGE AUDIT
→ PAPER TESTING

---

## Current Model Work

Models/experiments currently present include:

- Historical baselines
- Poisson
- Logistic regression
- Market calibration
- Residual models
- Blended model experiments
- Selection-level evaluation

The current baseline under evaluation is the chronological Poisson model.

---

## V0.29 Selection-Level Evaluation

Dataset:
- 5,308 selection observations
- 2,654 OVER
- 2,654 UNDER
- Missing selection targets: 0

Selection target:

OVER:
target = target_over25

UNDER:
target = 1 - target_over25

The correction ensures that each selection is evaluated against its own binary outcome.

Current reported probability metrics:

Poisson:
- Brier Score: approximately 0.246
- Log Loss: approximately 0.685

Market:
- Brier Score: approximately 0.241
- Log Loss: approximately 0.674

Current interpretation:
The market currently outperforms the Poisson baseline on these probability-scoring metrics.

This does NOT establish that the Poisson model is useless or that no exploitable signal exists. It means additional validation is required before claiming an edge.

---

## CLV Work

EDGE has completed multiple iterations of:

- CLV matching
- Opening/closing odds audits
- Market residual analysis
- Model/CLV joins
- Model failure analysis
- Selection-level evaluation

The current CLV work should be treated as research evidence rather than proof of profitability.

---

## V0.30 Leakage Audit

Status:
PASS WITH DOCUMENTED LIMITATION

### Feature leakage
PASS.

The feature construction pipeline is chronological and generates features from historical information before adding the current match to historical state.

### Model chronology
PASS.

The Poisson pipeline generates predictions before incorporating the current match result into team/history state.

### Selection target
PASS.

The V0.29 evaluation correctly maps OVER and UNDER selections to their corresponding binary targets.

### Odds semantics
PASS.

The V0.15 odds builder explicitly distinguishes opening/current observations from closing observations using the source column convention.

Columns ending in C are treated as closing prices.

### Market timestamp limitation
LIMITATION.

The processed odds dataset does not contain an exact timestamp showing when each PRE_CLOSING observation became available.

Therefore the semantic distinction between opening/current and closing prices is established, but exact information availability relative to kickoff cannot be independently proven from the processed odds dataset.

Do not describe this as proof that no market timing leakage exists.

---

## Existing Leakage Test

`src/test_feature_leakage.py`

Current tests include:

- Match count integrity
- One-to-one match mapping
- Historical count sanity
- First-match cold starts
- Feature-date sanity

The test currently reports:

STATUS: PASS
No obvious feature leakage detected.

However, the date test should eventually be strengthened to test true information timestamps rather than only date-range sanity.

---

## Important Methodological Rules

1. Data before opinions.
2. Estimate probabilities rather than simply predicting winners.
3. Price matters.
4. NO BET is valid.
5. Avoid hindsight and look-ahead bias.
6. Track predictions, probabilities, odds, reasoning, results and model versions.
7. A winning streak does not prove an edge.
8. Compare simple models against complex models.
9. Never chase losses.
10. Treat model probabilities as estimates with uncertainty.

---

## Current Research Position

EDGE has NOT established a reliable betting edge.

The current evidence shows that the market remains difficult to outperform using the baseline Poisson probability model.

The correct next objective is not to force a better-looking backtest.

The objective is to establish whether a reproducible, out-of-sample probability or market signal exists.

---

## Next Development Stage

After repository migration:

1. Preserve the current V0.30 state.
2. Complete repository documentation.
3. Establish reproducible environment/setup instructions.
4. Review the existing feature pipeline.
5. Strengthen leakage tests where possible.
6. Define a clean chronological backtesting framework.
7. Compare baseline models.
8. Continue model research only after methodological integrity is established.

---

## Current Model Version

V0.30

Status:
Research / validation

Real-money automation:
NOT IMPLEMENTED

Autonomous wagering:
EXCLUDED
