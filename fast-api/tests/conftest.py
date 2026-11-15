# pytest fixtures, shared by every test file in this folder (pytest finds conftest.py by name).
# A fixture is a function whose result is passed to any test that names it as a parameter: like a @BeforeEach
# that hands the test its objects. `yield` splits setup from teardown.
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from carsharing import app
from db import get_session
from routers.auth import get_current_user
from schemas import UserOutput


@pytest.fixture
def session():
    # A fresh in-memory SQLite database per test, so tests never touch carsharing.db or each other's data.
    # StaticPool keeps the one connection alive (an in-memory database lives only as long as its connection).
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


@pytest.fixture
def client(session):
    # app.dependency_overrides swaps a dependency for the whole app, like @MockBean or @TestConfiguration:
    # every endpoint that asks for get_session now gets the test session, and get_current_user always returns a
    # test user (so tests don't need to log in). This is why dependencies make endpoints easy to test.
    app.dependency_overrides[get_session] = lambda: session
    app.dependency_overrides[get_current_user] = lambda: UserOutput(id=1, username="tester")
    yield TestClient(app)          # sends requests to the app in-process, no server (like MockMvc)
    app.dependency_overrides.clear()
