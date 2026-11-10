# The prediction log database: the engine and the per-request session (as in fast-api/db.py).
# The URL comes from PREDICTIONS_DB_URL: a SQLite file by default, a Postgres URL in production
# (e.g. postgresql+psycopg://user:password@host/credit). Only the URL changes, not the code.
import os
from pathlib import Path

from sqlmodel import Session, SQLModel, create_engine

PREDICTIONS_DB_URL = os.environ.get("PREDICTIONS_DB_URL", "sqlite:///data/predictions.db")

# check_same_thread is a SQLite-only setting (FastAPI may use a session on another thread than the one that opened
# the connection); Postgres would reject it
connect_args = {"check_same_thread": False} if PREDICTIONS_DB_URL.startswith("sqlite") else {}
engine = create_engine(PREDICTIONS_DB_URL, connect_args=connect_args, echo=False)


def create_tables():
    # SQLite creates the database file, but not its folder. In Docker the folder is the mounted volume.
    if engine.url.drivername == "sqlite" and engine.url.database:      # a file, not an in-memory database
        Path(engine.url.database).parent.mkdir(parents=True, exist_ok=True)
    SQLModel.metadata.create_all(engine)      # creates missing tables, never changes existing ones (no migrations)


def get_session():
    with Session(engine) as session:
        yield session
