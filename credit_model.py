"""
credit_model.py - End-to-end German Credit Risk Modeling Pipeline.
Includes data loading, encoding, training, cost-based threshold tuning,
LightGBM integration, SHAP explainability, and regulatory fairness auditing.
"""

import os
import urllib.request
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    recall_score,
    precision_score,
    accuracy_score
)

# ---------------------------------------------------------------------------
# Constants & Dataset Schema
# ---------------------------------------------------------------------------
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
LOCAL_PATH = os.path.join(os.path.dirname(__file__), "data", "german.data")

COLUMN_NAMES = [
    "status", "duration", "credit_history", "purpose", "credit_amount",
    "savings", "employment_duration", "installment_rate", "personal_status_sex",
    "other_debtors", "present_residence", "property", "age",
    "other_installment_plans", "housing", "number_credits", "job",
    "people_liable", "telephone", "foreign_worker", "credit_risk"
]

CATEGORICAL_COLS = [
    "status", "credit_history", "purpose", "savings", "employment_duration",
    "personal_status_sex", "other_debtors", "property", "other_installment_plans",
    "housing", "job", "telephone", "foreign_worker"
]

NUMERIC_COLS = [
    "duration", "credit_amount", "installment_rate", "present_residence",
    "age", "number_credits", "people_liable"
]

# Cost matrix: approving bad loan (FN) costs 5, rejecting good loan (FP) costs 1
COST_FN = 5.0
COST_FP = 1.0


# ---------------------------------------------------------------------------
# 1. Data Loading (Required by app.py)
# ---------------------------------------------------------------------------
def load_data():
    """
    Loads German Credit dataset from UCI ML repo with local fallback.
    Returns: (df, source_str)
    """
    try:
        df = pd.read_csv(
            DATA_URL,
            sep=r"\s+",
            header=None,
            names=COLUMN_NAMES,
            engine="python"
        )
        source = "UCI Machine Learning Repository"
    except Exception:
        df = pd.read_csv(
            LOCAL_PATH,
            sep=r"\s+",
            header=None,
            names=COLUMN_NAMES,
            engine="python"
        )
        source = "Local fallback (data/german.data)"

    # Original UCI coding: 1 = Good, 2 = Bad -> map to 0 = Good, 1 = Bad
    if 2 in df["credit_risk"].values:
        df["credit_risk"] = df["credit_risk"].map({1: 0, 2: 1})

    return df, source


# ---------------------------------------------------------------------------
# 2. Data Preparation & Preprocessing
# ---------------------------------------------------------------------------
def prepare_data(df):
    """
    Splits into train/test (80/20 stratified) and applies ordinal encoding.
    """
    X = df[CATEGORICAL_COLS + NUMERIC_COLS].copy()
    y = df["credit_risk"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()

    X_train_encoded[CATEGORICAL_COLS] = encoder.fit_transform(X_train[CATEGORICAL_COLS])
    X_test_encoded[CATEGORICAL_COLS] = encoder.transform(X_test[CATEGORICAL_COLS])

    return X_train_encoded, X_test_encoded, y_train, y_test, encoder


# ---------------------------------------------------------------------------
# 3. Model Training & Threshold Optimization
# ---------------------------------------------------------------------------
def compute_expected_cost(y_true, y_pred_proba, threshold):
    """Calculates expected financial cost per applicant given threshold."""
    y_pred = (y_pred_proba >= threshold).astype(int)
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return (fn * COST_FN + fp * COST_FP) / len(y_true)


def tune_threshold(y_true, y_pred_proba):
    """Finds decision threshold minimizing expected loan loss."""
    thresholds = np.linspace(0.1, 0.9, 81)
    costs = [compute_expected_cost(y_true, y_pred_proba, t) for t in thresholds]
    best_idx = np.argmin(costs)
    return thresholds[best_idx], costs[best_idx]


def train_models(X_train, y_train):
    """Trains baseline Logistic Regression, Random Forest, and LightGBM."""
    weights = {0: 1.0, 1: 5.0}

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=3,
            class_weight=weights, random_state=42
        ),
        "Logistic Regression": LogisticRegression(
            class_weight=weights, max_iter=1000, random_state=42
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=6,
            scale_pos_weight=5.0, random_state=42, verbosity=-1
        )
    }

    for model in models.values():
        model.fit(X_train, y_train)

    return models


# ---------------------------------------------------------------------------
# 4. SHAP Local Explainability
# ---------------------------------------------------------------------------
def get_shap_explanation(model, sample_df):
    """Computes SHAP values for tree-based models."""
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer(sample_df)

    if len(shap_vals.values.shape) == 3:
        return explainer, shap_vals[:, :, 1]
    return explainer, shap_vals


# ---------------------------------------------------------------------------
# 5. Algorithmic Fairness & Bias Audit
# ---------------------------------------------------------------------------
def compute_fairness_audit(y_true, y_pred, sensitive_series, privileged_group, unprivileged_group):
    """Audits predictions for disparate impact and equal opportunity."""
    df_audit = pd.DataFrame({
        "y_true": y_true.values if hasattr(y_true, "values") else y_true,
        "y_pred": y_pred,
        "group": sensitive_series.values if hasattr(sensitive_series, "values") else sensitive_series
    })

    rate_priv = (df_audit[df_audit["group"] == privileged_group]["y_pred"] == 0).mean()
    rate_unpriv = (df_audit[df_audit["group"] == unprivileged_group]["y_pred"] == 0).mean()

    disparate_impact = rate_unpriv / (rate_priv + 1e-9)

    return {
        "Privileged Group": privileged_group,
        "Unprivileged Group": unprivileged_group,
        "Approval Rate (Privileged)": round(float(rate_priv), 3),
        "Approval Rate (Unprivileged)": round(float(rate_unpriv), 3),
        "Disparate Impact Ratio": round(float(disparate_impact), 3),
        "Passes 80% Rule": "Yes" if 0.80 <= disparate_impact <= 1.25 else "No"
    }
