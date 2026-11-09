import pytest
from fastapi.testclient import TestClient

from app import MODEL_PATH, app


@pytest.fixture
def client():
    # The real model file, so these tests check the predictions the API actually serves.
    # `with` runs the lifespan (startup and shutdown); a TestClient used without `with` skips it and has no model.
    if not MODEL_PATH.exists():
        pytest.skip(f"{MODEL_PATH} not found: run save_model() in give-me-some-credit first")
    with TestClient(app) as client:
        yield client
