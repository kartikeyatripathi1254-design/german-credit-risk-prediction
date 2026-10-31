import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score

# ---------------------------------------------------------------------------
# 1. Model Training with LightGBM Support
# ---------------------------------------------------------------------------
def train_models(X_train, y_train, class_weight_ratio=5.0):
    """
    Trains Logistic Regression, Random Forest, and LightGBM with cost-sensitive weighting.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    weights = {0: 1.0, 1: class_weight_ratio}

    models = {
        "Logistic Regression": LogisticRegression(
            class_weight=weights, max_iter=1000, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight=weights, random_state=42
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=6,
            scale_pos_weight=class_weight_ratio,
            random_state=42,
            verbosity=-1
        )
    }

    for name, model in models.items():
        model.fit(X_train, y_train)

    return models

# ---------------------------------------------------------------------------
# 2. Local Explainability with SHAP
# ---------------------------------------------------------------------------
def get_shap_explanation(model, transformed_applicant_df):
    """
    Computes SHAP values for a single applicant or batch using TreeExplainer.
    Returns: explainer, shap_values
    """
    # TreeExplainer is compatible with Random Forest and LightGBM
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(transformed_applicant_df)

    # For binary classification, pull values for the positive class (Bad Credit = 1)
    if len(shap_values.values.shape) == 3:
        # Some RF versions output (n_samples, n_features, n_classes)
        return explainer, shap_values[:, :, 1]
    
    return explainer, shap_values

# ---------------------------------------------------------------------------
# 3. Algorithmic Fairness & Bias Audit
# ---------------------------------------------------------------------------
def compute_fairness_audit(y_true, y_pred, sensitive_series, privileged_group, unprivileged_group):
    """
    Computes key Fair Lending compliance metrics:
      - Disparate Impact Ratio (80% rule compliance)
      - Statistical Parity Difference
      - Equal Opportunity Difference (Recall difference)
    """
    df_audit = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "group": sensitive_series
    })

    # Selection rate = Rate of predicting "Good Credit" (0)
    # (In loan approval context, Good Credit is the favorable outcome)
    rate_priv = (df_audit[df_audit["group"] == privileged_group]["y_pred"] == 0).mean()
    rate_unpriv = (df_audit[df_audit["group"] == unprivileged_group]["y_pred"] == 0).mean()

    disparate_impact = rate_unpriv / (rate_priv + 1e-9)
    statistical_parity_diff = rate_unpriv - rate_priv

    # True Positive Rate for favorable outcome (Good credit)
    tpr_priv = recall_score(
        df_audit[df_audit["group"] == privileged_group]["y_true"] == 0,
        df_audit[df_audit["group"] == privileged_group]["y_pred"] == 0,
        zero_division=0
    )
    tpr_unpriv = recall_score(
        df_audit[df_audit["group"] == unprivileged_group]["y_true"] == 0,
        df_audit[df_audit["group"] == unprivileged_group]["y_pred"] == 0,
        zero_division=0
    )
    equal_opportunity_diff = tpr_unpriv - tpr_priv

    return {
        "Privileged Group": privileged_group,
        "Unprivileged Group": unprivileged_group,
        "Approval Rate (Privileged)": round(rate_priv, 3),
        "Approval Rate (Unprivileged)": round(rate_unpriv, 3),
        "Disparate Impact Ratio": round(disparate_impact, 3),
        "Passes 80% Rule": "Yes" if 0.80 <= disparate_impact <= 1.25 else "No",
        "Statistical Parity Diff": round(statistical_parity_diff, 3),
        "Equal Opportunity Diff": round(equal_opportunity_diff, 3)
    }
