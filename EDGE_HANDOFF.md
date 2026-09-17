# EDGE — ChatGPT / Analyst Handoff

## What is EDGE?

EDGE is a sports analytics research project.

The initial research scope is:

- English Premier League
- Over/Under 2.5 goals
- Historical seasons 2015/16–2025/26
- Probability estimation
- Market comparison
- CLV analysis
- Out-of-sample validation

The project is intended to identify whether measurable, reproducible discrepancies exist between model probabilities and market prices.

Do not assume an edge exists.

---

## Current State

Current version:
V0.30

The latest completed major evaluation is V0.29 selection-level model evaluation.

The current Poisson baseline has not outperformed the market on the primary probability-scoring metrics.

Current research conclusion:

NO ESTABLISHED EDGE.

---

## Immediate Next Task

The repository has now reached the point where it should be preserved and transferred between development environments/accounts.

The next work should focus on:

1. Repository reproducibility.
2. Documentation.
3. Leakage-test improvements.
4. Chronological backtesting infrastructure.
5. Model comparison.

Do not immediately jump to complex ML.

---

## Important Files

### Feature construction

`src/build_features.py`

Creates historical features used by the model.

### Model dataset

`src/build_model_dataset.py`

Builds the model-ready dataset and joins relevant market information.

### Poisson baseline

`src/train_poisson.py`
`src/train_poisson_v2.py`

### V0.29 evaluation

`src/evaluate_selection_model_v29.py`

### Leakage testing

`src/test_feature_leakage.py`

### Odds construction

`src/build_odds_v15.py`

### CLV work

Relevant files include:

- `src/audit_clv_v20.py`
- `src/audit_clv_v21.py`
- `src/audit_clv_v22.py`
- `src/audit_clv_v23_exceptions.py`
- `src/repair_clv_matching_v18.py`
- `src/repair_clv_matching_v19.py`

---

## Data

Raw EPL data is located in:

`data/raw/`

Processed datasets are located in:

`data/processed/`

Reports are located in:

`reports/`

Models directory:

`models/`

Backtests directory:

`backtests/`

---

## Important Data Limitation

The processed odds data distinguishes PRE_CLOSING and CLOSING prices.

The odds builder explicitly treats source columns ending in C as closing prices.

However, the processed dataset does not preserve an exact timestamp for every market observation.

Therefore:

- opening/closing semantic separation: established
- exact market information availability relative to kickoff: not independently proven

This limitation must remain documented.

---

## Research Rules

Do not:

- claim profitability from a short sample
- optimize against the full historical dataset and call it OOS
- use future information
- silently change target definitions
- remove losing selections because they hurt ROI
- treat CLV alone as proof of profitability
- jump to complex ML without baseline comparison
- automate real-money wagering

Always:

- use chronological testing
- preserve model versions
- document assumptions
- report uncertainty
- compare against market baselines
- retain failed experiments
- accept NO BET / INSUFFICIENT DATA

---

## Transfer Instructions

A new analyst or ChatGPT instance should:

1. Read this file.
2. Read `EDGE_PROJECT_STATE.md`.
3. Inspect `README.md` when available.
4. Inspect the current source code before modifying it.
5. Verify the current dataset and results.
6. Continue from the stated next task.
7. Do not rewrite the project from scratch.

The repository is the source of truth for the technical project state.
