# The credit risk API: one endpoint that scores a borrower with the Give Me Some Credit boosting model.
# FastAPI basics (routing, schemas, Depends, lifespan) are explained in fast-api/; comments here cover serving a model.
#
# Run (from this folder):   fastapi dev app.py      then http://localhost:8000/docs
# The model file comes from give_me_some_credit.save_model() (see README.md); another file can be given with the
# CREDIT_MODEL_PATH environment variable (externalised config, like a Spring property).
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Request

from predictor import load_model, predict_probability
from schemas import BorrowerInput, PredictionOutput

MODEL_PATH = Path(os.environ.get("CREDIT_MODEL_PATH",
                                 Path(__file__).parent.parent / "give-me-some-credit" / "model.joblib"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once at startup: loading takes far longer than a prediction. app.state holds app-wide objects
    # (a singleton bean). If loading fails, the app doesn't start, so a broken model never takes traffic.
    app.state.model = load_model(MODEL_PATH)
    yield


app = FastAPI(title="Credit risk", lifespan=lifespan)


def get_model(request: Request):
    # The model as a dependency instead of a global: tests can swap it via app.dependency_overrides
    return request.app.state.model


Model = Annotated[object, Depends(get_model)]


@app.post("/predict")
def predict(borrower: BorrowerInput, model: Model) -> PredictionOutput:
    # A plain def, not async def: predict_proba is blocking CPU work. FastAPI runs a def endpoint in its thread
    # pool, so other requests keep being served; inside an async def it would block the event loop for all of them.
    return PredictionOutput(probability=predict_probability(model, borrower))


@app.get("/health")
def health(model: Model) -> dict:
    # For a load balancer or container platform to check the app is up (like Spring Boot Actuator's /health).
    # Reaching this line means the model loaded, since the app doesn't start otherwise.
    return {"status": "ok", "model": MODEL_PATH.name}
