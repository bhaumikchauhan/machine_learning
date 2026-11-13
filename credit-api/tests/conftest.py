import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, create_engine

import db
from app import MODEL_DIR, app


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
def client(engine):
    # The real model (the champion in model/), so these tests check the predictions the API actually serves.
    # `with` runs the lifespan (startup and shutdown); a TestClient used without `with` skips it and has no model.
    if not (MODEL_DIR / "registered_model.json").exists():
        pytest.skip(f"No model in {MODEL_DIR}/: run fetch_champion.py (see README.md)")
    with TestClient(app) as client:
        yield client
