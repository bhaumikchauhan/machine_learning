# Database setup: one engine for the whole app, one session per request.
# Spring Boot: the engine is the DataSource, the session is the per-request EntityManager / unit of work.
from sqlmodel import Session, create_engine

DATABASE_URL = "sqlite:///carsharing.db"

# check_same_thread=False: SQLite by default refuses a connection used from another thread, and FastAPI runs
# plain `def` endpoints in a thread pool. echo=True would log every SQL statement (like spring.jpa.show-sql).
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False}, echo=False)


def get_session():
    # A dependency: endpoints declare they need a Session (see SessionDep in routers/cars.py) and FastAPI calls
    # this once per request. `yield` hands the session to the endpoint and pauses here; when the request is done,
    # the `with` block closes the session (rolling back anything not committed, e.g. if the endpoint raised).
    # It's a request-scoped bean with a cleanup hook, and tests can swap it out (see tests/conftest.py).
    with Session(engine) as session:
        yield session
