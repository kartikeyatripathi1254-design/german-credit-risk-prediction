"""
Credit risk model: data loading, encoding, training and evaluation.

Kept separate from the Streamlit UI (app.py) so it can be unit-tested.

Target: 1 = bad credit risk, 0 = good credit risk.
The UCI documentation gives a cost matrix in which predicting "good" for a
customer who is actually bad costs 5, and predicting "bad" for a good customer
costs 1. The model therefore picks a decision threshold that minimises this
expected cost instead of using the default 0.5.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
LOCAL_DATA = Path(__file__).parent / "data" / "german.data"

COLUMNS = [
    "checking_account", "duration", "credit_history", "purpose",
    "credit_amount", "savings_account", "employment", "installment_rate",
    "personal_status", "other_debtors", "residence_since", "property",
    "age", "other_installments", "housing", "existing_credits",
    "job", "liable_people", "telephone", "foreign_worker", "credit_risk",
]
TARGET = "credit_risk"
COST_FALSE_GOOD = 5.0  # approving a customer who defaults
COST_FALSE_BAD = 1.0   # rejecting a customer who would have repaid
RANDOM_STATE = 42


def load_data(prefer_local: bool = True) -> tuple[pd.DataFrame, str]:
    """Return (raw dataframe, source). Uses the bundled copy first, then UCI."""
    sources = [str(LOCAL_DATA), UCI_URL] if prefer_local else [UCI_URL, str(LOCAL_DATA)]
    last_error = None
    for src in sources:
        try:
            if src == str(LOCAL_DATA) and not LOCAL_DATA.exists():
                continue
            df = pd.read_csv(src, sep=" ", header=None, names=COLUMNS)
            validate_raw(df)
            return df, src
        except Exception as exc:  # network errors, missing file, bad format
            last_error = exc
    raise RuntimeError(f"Could not load the German Credit data: {last_error}")


def validate_raw(df: pd.DataFrame) -> None:
    if list(df.columns) != COLUMNS:
        raise ValueError("Unexpected columns in German Credit data.")
    if not set(df[TARGET].unique()) <= {1, 2}:
        raise ValueError("Target column must contain only 1 (good) and 2 (bad).")


def split_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Map target 1/2 to 0 (good) / 1 (bad) and return X, y."""
    y = df[TARGET].map({1: 0, 2: 1}).astype(int)
    X = df.drop(columns=TARGET)
    return X, y


@dataclass
class CreditModel:
    """A fitted encoder + classifier with a cost-based decision threshold."""

    encoder: OrdinalEncoder
    model: object
    feature_names: list[str]
    categorical: list[str]
    threshold: float = 0.5
    metrics: dict = field(default_factory=dict)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[self.feature_names].copy()
        X[self.categorical] = self.encoder.transform(X[self.categorical].astype(str))
        return X

    def predict_proba_bad(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self.transform(X))[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba_bad(X) >= self.threshold).astype(int)

    @property
    def feature_importances(self) -> pd.Series:
        est = self.model
        if hasattr(est, "feature_importances_"):
            values = est.feature_importances_
        else:  # logistic regression pipeline
            values = np.abs(est[-1].coef_[0])
        return pd.Series(values, index=self.feature_names).sort_values(ascending=False)


def _make_encoder(X: pd.DataFrame, categorical: list[str]) -> OrdinalEncoder:
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    enc.fit(X[categorical].astype(str))
    return enc


def expected_cost(y_true, y_pred) -> float:
    """Average misclassification cost per customer under the UCI cost matrix."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float((fn * COST_FALSE_GOOD + fp * COST_FALSE_BAD) / len(y_true))


def best_threshold(y_true, proba_bad) -> float:
    """Threshold on P(bad) that minimises expected cost."""
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    costs = [expected_cost(y_true, (proba_bad >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmin(costs))])


def evaluate(y_true, proba_bad, threshold: float) -> dict:
    y_pred = (proba_bad >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, proba_bad),
        "accuracy": accuracy_score(y_true, y_pred),
        "bad_recall": recall_score(y_true, y_pred, zero_division=0),
        "bad_precision": precision_score(y_true, y_pred, zero_division=0),
        "expected_cost": expected_cost(y_true, y_pred),
        "threshold": threshold,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]),
    }


def _random_forest():
    return RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=3,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
    )


def _logistic_baseline():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))


def train(df: pd.DataFrame, test_size: float = 0.2) -> dict:
    """
    Train a Random Forest and a logistic-regression baseline.

    The decision threshold is tuned on cross-validated training predictions only,
    then both models are scored once on the held-out test set.
    Returns {"rf": CreditModel, "baseline": CreditModel, "test": (X_test, y_test)}.
    """
    X, y = split_features(df)
    categorical = X.select_dtypes(include="object").columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
    encoder = _make_encoder(X_train, categorical)

    def encode(frame):
        frame = frame.copy()
        frame[categorical] = encoder.transform(frame[categorical].astype(str))
        return frame

    Xtr, Xte = encode(X_train), encode(X_test)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for name, factory in (("rf", _random_forest), ("baseline", _logistic_baseline)):
        # Out-of-fold probabilities on the training set -> threshold (no test leakage).
        oof = np.zeros(len(Xtr))
        for tr_idx, va_idx in cv.split(Xtr, y_train):
            m = factory().fit(Xtr.iloc[tr_idx], y_train.iloc[tr_idx])
            oof[va_idx] = m.predict_proba(Xtr.iloc[va_idx])[:, 1]
        threshold = best_threshold(y_train.values, oof)
        cv_auc = cross_val_score(factory(), Xtr, y_train, cv=cv, scoring="roc_auc")

        model = factory().fit(Xtr, y_train)
        metrics = evaluate(y_test.values, model.predict_proba(Xte)[:, 1], threshold)
        metrics["cv_auc_mean"] = float(cv_auc.mean())
        metrics["cv_auc_std"] = float(cv_auc.std())
        results[name] = CreditModel(encoder, model, X.columns.tolist(), categorical, threshold, metrics)
    results["test"] = (X_test, y_test)
    return results
