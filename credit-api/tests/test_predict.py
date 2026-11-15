import uuid
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sqlmodel import select

from predictor import SCHEMA_COLUMNS, load_model, model_version, predict_probability
from schemas import BorrowerInput, PredictionLog

# These tests run on a stand-in model (conftest.py), so they check the API's code and pass whichever model is the
# champion. Which model is served, and whether it's good enough, is settled by promote() and the registry.

# Kaggle test borrower 1 (cs-test.csv): a realistic request
BORROWER_1 = {"revolving_utilization": 0.88551908, "age": 43, "times_30_59_days_late": 0,
              "debt_ratio": 0.177512717, "monthly_income": 5700, "open_credit_lines": 4, "times_90_days_late": 0,
              "real_estate_loans": 0, "times_60_89_days_late": 0, "dependents": 0}

# A borrower with a different value in every column, and the same borrower the way training saw it: the Kaggle
# column names, written out here rather than taken from the schema's aliases, so a field mapped to the wrong
# column shows up as a different probability.
DISTINCT_BORROWER = {"revolving_utilization": 12.5, "age": 25, "times_30_59_days_late": 37, "debt_ratio": 50.5,
                     "monthly_income": 62, "open_credit_lines": 75, "times_90_days_late": 88,
                     "real_estate_loans": 5, "times_60_89_days_late": 95, "dependents": 20}
DISTINCT_BORROWER_AS_TRAINED = {
    "RevolvingUtilizationOfUnsecuredLines": 12.5, "age": 25, "NumberOfTime30-59DaysPastDueNotWorse": 37,
    "DebtRatio": 50.5, "MonthlyIncome": 62, "NumberOfOpenCreditLinesAndLoans": 75, "NumberOfTimes90DaysLate": 88,
    "NumberRealEstateLoansOrLines": 5, "NumberOfTime60-89DaysPastDueNotWorse": 95, "NumberOfDependents": 20}


def probability_for(client, borrower):
    # compares probabilities only: every response has its own request id
    return client.post("/predict", json=borrower).json()["probability"]


# --- Through the API ---

def test_api_returns_the_models_own_probability(client):
    # Training-serving skew: from JSON to row, the API must give the model exactly what training gave it.
    # The model's columns are in another order than the schema (conftest.py), so the order is checked too.
    model = client.app.state.model
    as_trained = pd.DataFrame([DISTINCT_BORROWER_AS_TRAINED])[model.feature_names_in_]
    response = client.post("/predict", json=DISTINCT_BORROWER)
    assert response.status_code == 200
    assert response.json()["probability"] == model.predict_proba(as_trained)[0, 1]


def test_the_check_above_catches_two_mixed_up_columns(client):
    # The test above is only as good as the stand-in model's sensitivity: swapping any two columns of the
    # distinct borrower must change the probability, or a mix-up between them would pass unnoticed
    model = client.app.state.model
    as_trained = pd.DataFrame([DISTINCT_BORROWER_AS_TRAINED])[model.feature_names_in_]
    original = model.predict_proba(as_trained)[0, 1]
    columns = list(model.feature_names_in_)
    for i, first in enumerate(columns):
        for second in columns[i + 1:]:
            swapped = as_trained.rename(columns={first: second, second: first})[columns]
            assert model.predict_proba(swapped)[0, 1] != original, f"{first} and {second}"


def test_field_order_does_not_matter(client):
    reversed_order = dict(reversed(list(BORROWER_1.items())))
    assert probability_for(client, reversed_order) == probability_for(client, BORROWER_1)


def test_missing_income_is_scored_as_not_reported(client):
    # left out and null are the same: both become NaN, like the 29,731 blank incomes in training
    left_out = {k: v for k, v in BORROWER_1.items() if k != "monthly_income"}
    as_null = {**BORROWER_1, "monthly_income": None}
    assert client.post("/predict", json=left_out).status_code == 200
    assert probability_for(client, left_out) == probability_for(client, as_null)
    assert probability_for(client, left_out) != probability_for(client, BORROWER_1)


@pytest.mark.parametrize("change", [
    {"age": None},                  # a required field set to null
    {"age": 12},                    # out of range
    {"open_credit_lines": -1},      # impossible
    {"monthly_incom": 5700},        # a typo'd field name (extra="forbid")
])
def test_invalid_input_is_422(client, change):
    assert client.post("/predict", json={**BORROWER_1, **change}).status_code == 422


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_version": "credit-risk/7"}     # the stand-in's


# --- Prediction logging ---

def test_prediction_is_logged_under_the_returned_id(client, session):
    before = datetime.now(timezone.utc)
    response = client.post("/predict", json=BORROWER_1).json()
    record = session.get(PredictionLog, uuid.UUID(response["id"]))     # find by primary key
    assert record is not None
    assert record.probability == response["probability"]
    assert record.model_dump(include=set(BORROWER_1)) == BORROWER_1     # the inputs as received
    assert record.model_version == "credit-risk/7"
    assert record.latency > 0
    # SQLite gives the timestamp back without its time zone (it stores text); it was written in UTC
    assert before - timedelta(seconds=1) <= record.timestamp.replace(tzinfo=timezone.utc) <= datetime.now(timezone.utc)


def test_rejected_requests_are_not_logged(client, session):
    client.post("/predict", json={**BORROWER_1, "age": 12})     # 422: never reaches the model
    client.get("/health")
    assert session.exec(select(PredictionLog)).all() == []


# --- predictor.py on its own ---

class RecordingModel:
    # A stand-in model that records what it was given (a hand-written mock)
    feature_names_in_ = np.array(list(reversed(SCHEMA_COLUMNS)))     # a different order from the schema

    def predict_proba(self, X):
        self.X = X
        return np.array([[0.75, 0.25]])


def test_row_is_built_in_the_models_column_order_with_nan_for_none():
    model = RecordingModel()
    borrower = BorrowerInput(**{**BORROWER_1, "monthly_income": None})
    probability, latency = predict_probability(model, borrower)
    assert probability == 0.25 and latency >= 0
    assert list(model.X.columns) == list(model.feature_names_in_)
    assert np.isnan(model.X.loc[0, "MonthlyIncome"])


# Startup checks: the API checks the model against its own schema and scikit-learn, which only it knows
# (promote() checks the model against the contract when it's published)

def tiny_model(columns):
    return HistGradientBoostingClassifier(max_iter=1).fit(pd.DataFrame({c: [0.0, 1.0] for c in columns}), [0, 1])


def test_load_model_and_version_from_a_model_folder(make_model_folder):
    model_dir = make_model_folder(tiny_model(SCHEMA_COLUMNS))
    assert set(load_model(model_dir).feature_names_in_) == set(SCHEMA_COLUMNS)
    assert model_version(model_dir) == "credit-risk/7"


def test_load_model_rejects_a_model_with_other_columns(make_model_folder):
    # e.g. a field renamed in the schema, or a model with an engineered column the schema doesn't have
    model_dir = make_model_folder(tiny_model(SCHEMA_COLUMNS[:-1] + ["DebtToIncome"]))
    with pytest.raises(RuntimeError, match="DebtToIncome"):
        load_model(model_dir)


def test_load_model_rejects_another_scikit_learn_version(make_model_folder):
    # e.g. scikit-learn bumped in requirements.txt: a model saved by another version may predict differently
    model_dir = make_model_folder(tiny_model(SCHEMA_COLUMNS), sklearn_version="1.0.0")
    with pytest.raises(RuntimeError, match="1.0.0"):
        load_model(model_dir)
