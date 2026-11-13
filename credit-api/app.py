# The credit risk API: one endpoint that scores a borrower with the Give Me Some Credit boosting model, and logs
# every prediction. FastAPI basics (routing, schemas, Depends, lifespan) are explained in fast-api/; comments here
# cover serving a model.
#
# Run (from this folder):   fastapi dev app.py      then http://localhost:8000/docs
# Configuration from environment variables (externalised config, like Spring properties):
#   CREDIT_MODEL_DIR     the model folder, default model (the registered champion, downloaded by fetch_champion.py)
#   PREDICTIONS_DB_URL   the prediction log database, default sqlite:///data/predictions.db (see db.py)
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from sqlmodel import Session

import db
from predictor import load_model, model_version, predict_probability
from schemas import BorrowerInput, PredictionLog, PredictionOutput

MODEL_DIR = Path(os.environ.get("CREDIT_MODEL_DIR", "model"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once at startup: loading takes far longer than a prediction. app.state holds app-wide objects
    # (a singleton bean). If loading fails, the app doesn't start, so a broken model never takes traffic.
    db.create_tables()
    app.state.model = load_model(MODEL_DIR)
    # The model version in each log record: the registered version, e.g. "credit-risk/5" (it replaced a hash of
    # model.joblib). The image holds one fixed version; a new champion means fetch_champion.py and a new image.
    app.state.model_version = model_version(MODEL_DIR)
    yield


app = FastAPI(title="Credit risk", lifespan=lifespan)


def get_model(request: Request):
    # The model as a dependency instead of a global: tests can swap it via app.dependency_overrides
    return request.app.state.model


Model = Annotated[object, Depends(get_model)]
SessionDep = Annotated[Session, Depends(db.get_session)]


@app.post("/predict")
def predict(borrower: BorrowerInput, model: Model, session: SessionDep, request: Request) -> PredictionOutput:
    # A plain def, not async def: predict_proba is blocking CPU work. FastAPI runs a def endpoint in its thread
    # pool, so other requests keep being served; inside an async def it would block the event loop for all of them.
    probability, latency = predict_probability(model, borrower)

    # Logged here, not in middleware: only the endpoint has the validated input and the probability, and only
    # requests that reached the model are logged (no /health, no 422s). id and timestamp get their defaults.
    prediction_log = PredictionLog(**borrower.model_dump(), probability=probability, latency=latency,
                                   model_version=request.app.state.model_version)
    session.add(prediction_log)
    session.commit()        # INSERT. If it fails, the client gets a 500: no prediction is served without its record

    return PredictionOutput(id=prediction_log.id, probability=probability)


@app.get("/health")
def health(model: Model, request: Request) -> dict:
    # For a load balancer or container platform to check the app is up (like Spring Boot Actuator's /health).
    # Reaching this line means the model loaded, since the app doesn't start otherwise.
    return {"status": "ok", "model_version": request.app.state.model_version}
