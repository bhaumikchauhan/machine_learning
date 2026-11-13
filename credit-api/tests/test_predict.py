import uuid
from datetime import datetime, timedelta, timezone

import json

import numpy as np
import pandas as pd
import pytest
import sklearn
import skops.io
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sqlmodel import select

from app import app
from predictor import SCHEMA_COLUMNS, load_model, model_version, predict_probability
from schemas import BorrowerInput, PredictionLog

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


def probability_for(client, borrower):
    # compares probabilities only: every response has its own request id
    return client.post("/predict", json=borrower).json()["probability"]


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
    health = client.get("/health").json()
    assert health["status"] == "ok"
    assert health["model_version"] == app.state.model_version
    assert health["model_version"].startswith("credit-risk/")      # the registered name and version


# --- Prediction logging ---

def test_prediction_is_logged_under_the_returned_id(client, session):
    before = datetime.now(timezone.utc)
    response = client.post("/predict", json=BORROWER_1).json()
    record = session.get(PredictionLog, uuid.UUID(response["id"]))     # find by primary key
    assert record is not None
    assert record.probability == response["probability"]
    assert record.model_dump(include=set(BORROWER_1)) == BORROWER_1     # the inputs as received
    assert record.model_version == app.state.model_version
    assert record.latency > 0
    # SQLite gives the timestamp back without its time zone (it stores text); it was written in UTC
    assert before - timedelta(seconds=1) <= record.timestamp.replace(tzinfo=timezone.utc) <= datetime.now(timezone.utc)


def test_rejected_requests_are_not_logged(client, session):
    client.post("/predict", json={**BORROWER_1, "age": 12})     # 422: never reaches the model
    client.get("/health")
    assert session.exec(select(PredictionLog)).all() == []


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
    probability, latency = predict_probability(model, borrower)
    assert probability == 0.25 and latency >= 0
    assert list(model.X.columns) == list(model.feature_names_in_)
    assert np.isnan(model.X.loc[0, "MonthlyIncome"])


def make_model_dir(tmp_path, columns, sklearn_version=sklearn.__version__):
    # A small model folder shaped like fetch_champion.py's: model.skops, MLmodel, registered_model.json
    X = pd.DataFrame({column: [0.0, 1.0] for column in columns})
    skops.io.dump(HistGradientBoostingClassifier(max_iter=1).fit(X, [0, 1]), tmp_path / "model.skops")
    mlmodel = {"flavors": {"sklearn": {"sklearn_version": sklearn_version}}}
    (tmp_path / "MLmodel").write_text(yaml.safe_dump(mlmodel))
    (tmp_path / "registered_model.json").write_text(json.dumps({"name": "credit-risk", "version": 7}))
    return tmp_path


def test_load_model_and_version_from_a_model_folder(tmp_path):
    model_dir = make_model_dir(tmp_path, SCHEMA_COLUMNS)
    assert set(load_model(model_dir).feature_names_in_) == set(SCHEMA_COLUMNS)
    assert model_version(model_dir) == "credit-risk/7"


def test_load_model_rejects_a_model_with_other_columns(tmp_path):
    # e.g. a model retrained with an engineered column the schema doesn't have: the app must not start
    model_dir = make_model_dir(tmp_path, SCHEMA_COLUMNS[:-1] + ["DebtToIncome"])
    with pytest.raises(RuntimeError, match="DebtToIncome"):
        load_model(model_dir)


def test_load_model_rejects_another_scikit_learn_version(tmp_path):
    # A model saved by another scikit-learn may load and predict differently: the app must not start
    model_dir = make_model_dir(tmp_path, SCHEMA_COLUMNS, sklearn_version="1.0.0")
    with pytest.raises(RuntimeError, match="1.0.0"):
        load_model(model_dir)
