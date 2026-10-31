import pytest
from fastapi.testclient import TestClient
from api import app
from credit_model import compute_fairness_audit
import numpy as np
import pandas as pd

client = TestClient(app)

def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_fairness_audit_calculation():
    # Synthetic test data
    y_true = np.array([0, 1, 0, 1, 0, 0, 1, 0])
    y_pred = np.array([0, 0, 0, 1, 0, 0, 1, 1])
    groups = pd.Series(["A", "A", "A", "A", "B", "B", "B", "B"])

    result = compute_fairness_audit(y_true, y_pred, groups, "A", "B")
    
    assert "Disparate Impact Ratio" in result
    assert "Equal Opportunity Diff" in result
    assert isinstance(result["Disparate Impact Ratio"], float)
