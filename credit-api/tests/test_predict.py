import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from predictor import SCHEMA_COLUMNS, load_model, predict_probability
from schemas import BorrowerInput

# Kaggle test borrower 1 (cs-test.csv) and its probability in submission_2.csv, the submission Kaggle scored 0.86670
BORROWER_1 = {"revolving_utilization": 0.88551908, "age": 43, "times_30_59_days_late": 0,
              "debt_ratio": 0.177512717, "monthly_income": 5700, "open_credit_lines": 4, "times_90_days_late": 0,
              "real_estate_loans": 0, "times_60_89_days_late": 0, "dependents": 0}
SUBMITTED_PROBABILITY = 0.06459152202678682


# --- Through the API, with the real model ---

def test_predict_matches_the_kaggle_submission(client):
    # Guards against training-serving skew: the API must give the probability that was validated and submitted
    response = client.post("/predict", json=BORROWER_1)
    assert response.status_code == 200
    assert response.json()["probability"] == pytest.approx(SUBMITTED_PROBABILITY, abs=1e-12)


def test_field_order_does_not_matter(client):
    reversed_order = dict(reversed(list(BORROWER_1.items())))
    assert client.post("/predict", json=reversed_order).json() == client.post("/predict", json=BORROWER_1).json()


def test_missing_income_is_scored_as_not_reported(client):
    # left out and null are the same: both become NaN, like the 29,731 blank incomes in training
    left_out = {k: v for k, v in BORROWER_1.items() if k != "monthly_income"}
    as_null = {**BORROWER_1, "monthly_income": None}
    response = client.post("/predict", json=left_out)
    assert response.status_code == 200
    assert response.json() == client.post("/predict", json=as_null).json()
    assert response.json() != client.post("/predict", json=BORROWER_1).json()


@pytest.mark.parametrize("change", [
    {"age": None},                  # a required field set to null
    {"age": 12},                    # out of range
    {"open_credit_lines": -1},      # impossible
    {"monthly_incom": 5700},        # a typo'd field name (extra="forbid")
])
def test_invalid_input_is_422(client, change):
    assert client.post("/predict", json={**BORROWER_1, **change}).status_code == 422


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model": "model.joblib"}


# --- predictor.py on its own, without the model file ---

class RecordingModel:
    # A stand-in model that records what it was given (a hand-written mock)
    feature_names_in_ = np.array(list(reversed(SCHEMA_COLUMNS)))     # a different order from the schema

    def predict_proba(self, X):
        self.X = X
        return np.array([[0.75, 0.25]])


def test_row_is_built_in_the_models_column_order_with_nan_for_none():
    model = RecordingModel()
    borrower = BorrowerInput(**{**BORROWER_1, "monthly_income": None})
    assert predict_probability(model, borrower) == 0.25
    assert list(model.X.columns) == list(model.feature_names_in_)
    assert np.isnan(model.X.loc[0, "MonthlyIncome"])


def test_load_model_rejects_a_model_with_other_columns(tmp_path):
    # e.g. a model retrained with an engineered column the schema doesn't have: the app must not start
    X = pd.DataFrame({column: [0.0, 1.0] for column in SCHEMA_COLUMNS[:-1] + ["DebtToIncome"]})
    path = tmp_path / "other.joblib"
    joblib.dump(HistGradientBoostingClassifier(max_iter=1).fit(X, [0, 1]), path)
    with pytest.raises(RuntimeError, match="DebtToIncome"):
        load_model(path)
