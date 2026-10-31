# German Credit Risk Prediction

[![CI](https://github.com/kartikeyatripathi1254-design/german-credit-risk-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/kartikeyatripathi1254-design/german-credit-risk-prediction/actions/workflows/ci.yml)

Predicts whether a loan applicant is a good or bad credit risk using the UCI Statlog German Credit dataset
(1,000 applicants, 20 attributes, 30% bad loans), and serves the model in an interactive Streamlit app.

The focus is on evaluating the model the way a lender would: missing a bad loan is far more expensive than
turning away a good customer, so the model is judged on ranking quality and the cost of its decisions,
not on accuracy alone.

## Results (held-out test set, 200 applicants)

| Model | ROC-AUC (test) | CV ROC-AUC (5-fold) | Bad-loan recall | Bad-loan precision | Accuracy | Expected cost / applicant |
|---|---|---|---|---|---|---|
| **Random Forest** | **0.81** | 0.80 ± 0.04 | 88% | 44% | 63% | 0.51 |
| Logistic regression (baseline) | 0.79 | 0.78 ± 0.06 | 90% | 43% | 62% | 0.50 |

- **Cost-based threshold.** The UCI documentation gives a cost matrix in which approving a customer who
  defaults costs 5 and rejecting a good customer costs 1. The decision threshold is tuned on out-of-fold
  training predictions (no test-set leakage) to minimise that cost. The tuned threshold (about 0.35 on P(bad))
  lowers the Random Forest's expected cost from 0.63 to 0.51 per applicant compared with the default 0.5,
  at the price of lower accuracy (74% → 63%). That trade is intended: it catches 88% of bad loans instead of 70%.
- **Honest comparison.** The Random Forest ranks applicants slightly better than a scaled logistic regression
  (AUC 0.81 vs 0.79), but at the chosen thresholds both have a similar expected cost. On a dataset this small,
  the simpler model is a reasonable choice.
- **Most influential features:** checking-account status, credit amount, loan duration, age and savings.

Exact numbers are recomputed live in the app (*ML Model* page) and may move slightly between library versions.

## The app

- **Overview** – dataset summary and class balance computed from the data
- **Data Explorer** – raw data, statistics and distributions
- **ML Model** – test and cross-validation metrics, Random Forest vs baseline, confusion matrix, feature importance
- **Predict Risk** – enter an applicant's loan and financial profile and get P(bad) and a decision at the
  cost-based threshold
- **About** – data source and method

## Project structure

| File | What it does |
|---|---|
| `credit_model.py` | Data loading, encoding, training, threshold tuning and evaluation (no UI code, unit-tested) |
| `app.py` | Streamlit interface |
| `data/german.data` | Copy of the UCI dataset, used when the UCI server can't be reached |
| `notebooks/` | Exploratory analysis notebook |
| `tests/` | pytest suite for the model module |
| `Dockerfile`, `.github/workflows/ci.yml` | Container image and CI (lint, tests, Docker build) |

## Run locally

```bash
git clone https://github.com/kartikeyatripathi1254-design/german-credit-risk-prediction.git
cd german-credit-risk-prediction
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

With Docker:

```bash
docker build -t credit-risk .
docker run -p 8501:8501 credit-risk
```

## Tests

```bash
python -m pytest -q
```

## Method details

- Stratified 80/20 train/test split (random_state 42); categorical codes encoded with an `OrdinalEncoder`
  fitted on the training data only, with unknown categories handled at prediction time.
- Random Forest: 300 trees, max depth 8, min 3 samples per leaf, balanced class weights.
- Baseline: standardised logistic regression with balanced class weights.
- 5-fold stratified cross-validation on the training set for ROC-AUC and for the out-of-fold predictions used
  to choose the threshold.

## Limitations

- 1,000 applicants from 1970s–90s Germany: useful for method, not for real lending decisions.
- The prediction form fills some rarely changed attributes (e.g. personal status, telephone) with fixed values.
- Feature importances are global; they don't explain an individual prediction.

## Data and license

Dataset: Hofmann, H. (1994). *Statlog (German Credit Data)*. UCI Machine Learning Repository.
https://doi.org/10.24432/C5NC77 – licensed CC BY 4.0. Code: MIT.
