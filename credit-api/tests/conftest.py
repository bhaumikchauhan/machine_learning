import json

import numpy as np
import pandas as pd
import pytest
import sklearn
import skops.io
import yaml
from fastapi.testclient import TestClient
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, create_engine

import app as api
import db
import predictor
from predictor import SCHEMA_COLUMNS

# The tests check the API's code, not a particular model: they run on a small stand-in model, so they need no
# model/ folder, no MLflow and no training data, and run the same on any machine (CI included). Whether a model is
# good enough is decided when it's promoted (give-me-some-credit, promote()), not here.


def save_model_folder(path, model, version=7, sklearn_version=sklearn.__version__):
    # A folder shaped like fetch_champion.py's output: model.skops, MLmodel, registered_model.json
    path.mkdir(parents=True, exist_ok=True)
    skops.io.dump(model, path / "model.skops")
    (path / "MLmodel").write_text(yaml.safe_dump({"flavors": {"sklearn": {"sklearn_version": sklearn_version}}}))
    (path / "registered_model.json").write_text(json.dumps({"name": "credit-risk", "version": version}))
    return path


@pytest.fixture
def make_model_folder(tmp_path):
    # For tests that need their own folder, e.g. a model with other columns
    return lambda model, **kwargs: save_model_folder(tmp_path / "model", model, **kwargs)


def fit_stand_in_model(columns):
    # A linear model on random data where every column has its own weight. With a different value in every column
    # (DISTINCT_BORROWER in test_predict.py), swapping any two columns changes the score, so a column mixed up
    # anywhere between the JSON and the model changes the probability. A boosting model doesn't guarantee that:
    # its trees can ignore a column in part of the range (a first version missed a swap of two late-payment counts).
    # Values from 0 to 100; some incomes and dependents are blank, as in the Kaggle data, and get the mean.
    # The real model's type (boosting, TreePredictor in TRUSTED_TYPES) is covered by the load_model tests.
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.uniform(0, 100, size=(2000, len(columns))), columns=columns)
    X.loc[rng.random(2000) < 0.2, ["MonthlyIncome", "NumberOfDependents"]] = np.nan
    weights = np.arange(1, len(columns) + 1)
    y = (X.fillna(50).to_numpy() @ weights + rng.normal(0, 200, 2000) > 50 * weights.sum()).astype(int)
    return make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression()).fit(X, y)


@pytest.fixture(scope="session")
def stand_in_model_dir(tmp_path_factory):
    # Fitted once per test run. Its columns are in reverse schema order, so the API has to build each row in the
    # model's order (feature_names_in_), not the JSON's or the schema's.
    model = fit_stand_in_model(list(reversed(SCHEMA_COLUMNS)))
    return save_model_folder(tmp_path_factory.mktemp("stand_in") / "model", model)


@pytest.fixture
def engine(monkeypatch):
    # A fresh in-memory database per test, so tests never write to data/predictions.db and don't see each other's
    # records. StaticPool keeps one connection: each new connection to "sqlite://" would be a new, empty database,
    # and FastAPI runs the endpoint on another thread than the test.
    test_engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    # Swap the engine in db.py for the duration of the test (monkeypatch puts the original back afterwards):
    # create_tables() at startup and get_session() per request both use it. The alternative,
    # app.dependency_overrides[db.get_session], would cover the sessions but not the startup.
    monkeypatch.setattr(db, "engine", test_engine)
    return test_engine


@pytest.fixture
def session(engine):
    # For tests that read the log table directly
    with Session(engine) as session:
        yield session


@pytest.fixture
def client(engine, stand_in_model_dir, monkeypatch):
    # The app with the stand-in model: lifespan loads from MODEL_DIR, swapped like the engine.
    # `with` runs the lifespan (startup and shutdown); a TestClient used without `with` skips it and has no model.
    monkeypatch.setattr(api, "MODEL_DIR", stand_in_model_dir)
    # The stand-in's SimpleImputer stores a numpy.dtype, which skops doesn't trust by default. Trusted for the
    # tests only: the API's own TRUSTED_TYPES stays as narrow as the real model needs.
    monkeypatch.setattr(predictor, "TRUSTED_TYPES", predictor.TRUSTED_TYPES + ["numpy.dtype"])
    with TestClient(api.app) as client:
        yield client
