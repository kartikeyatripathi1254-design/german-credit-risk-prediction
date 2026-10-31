import numpy as np
import pandas as pd
import pytest

import credit_model as cm


@pytest.fixture(scope="module")
def data():
    df, _ = cm.load_data()
    return df


@pytest.fixture(scope="module")
def trained(data):
    return cm.train(data)


def test_bundled_data_shape(data):
    assert data.shape == (1000, 21)
    assert set(data[cm.TARGET].unique()) == {1, 2}


def test_target_mapping(data):
    _, y = cm.split_features(data)
    assert set(y.unique()) == {0, 1}
    assert y.sum() == 300  # 300 bad credits in the UCI data


def test_expected_cost_weights_missed_defaults_5x():
    y_true = np.array([1, 0])
    assert cm.expected_cost(y_true, np.array([0, 0])) == pytest.approx(5 / 2)  # missed default
    assert cm.expected_cost(y_true, np.array([1, 1])) == pytest.approx(1 / 2)  # rejected good customer


def test_models_beat_chance_and_threshold_is_tuned(trained):
    for name in ("rf", "baseline"):
        m = trained[name].metrics
        assert m["roc_auc"] > 0.70
        assert 0.05 <= trained[name].threshold <= 0.95


def test_threshold_lowers_cost_vs_default(trained):
    X_test, y_test = trained["test"]
    rf = trained["rf"]
    p = rf.predict_proba_bad(X_test)
    tuned = cm.evaluate(y_test.values, p, rf.threshold)["expected_cost"]
    default = cm.evaluate(y_test.values, p, 0.5)["expected_cost"]
    assert tuned <= default


def test_predict_single_applicant_with_unseen_category(trained, data):
    row = data.drop(columns=cm.TARGET).iloc[[0]].copy()
    row["purpose"] = "A410"  # 'other' purpose, rare in the data
    proba = trained["rf"].predict_proba_bad(row)
    assert 0.0 <= proba[0] <= 1.0
    row["purpose"] = "UNKNOWN"
    assert trained["rf"].predict(row).shape == (1,)
