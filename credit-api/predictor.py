# Loading the model and turning one request into one prediction. Plain Python, no FastAPI, so it can be tested on
# its own (like a @Service that the controller calls).
#
# The model is the registered champion, downloaded into model/ by fetch_champion.py before the image is built:
#   model/model.skops             the fitted model, saved by MLflow in the skops format
#   model/MLmodel                 MLflow's description of it, including the scikit-learn version it was saved with
#   model/registered_model.json   which registered version it is (written by fetch_champion.py)
# MLflow itself isn't needed here: skops loads the file on its own.
import json
import time
from pathlib import Path

import pandas as pd
import sklearn
import skops.io
import yaml

from schemas import BorrowerInput

# The column names the schema produces, in field order (order doesn't matter: see predict_probability)
SCHEMA_COLUMNS = [field.serialization_alias for field in BorrowerInput.model_fields.values()]

# skops loads only types it trusts, so a model file can't run code while loading (pickle and joblib can).
# The boosting model needs one type skops doesn't trust by default. The list is fixed here, in the API's code,
# not read from the model folder: trusting whatever a file asks for would undo the protection.
TRUSTED_TYPES = ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"]


def load_model(model_dir: Path):
    # Fail at startup, not on the first request (like a Spring context that won't start on a missing bean):
    # 1. a model saved by another scikit-learn version may load, then predict differently. MLflow records the
    #    version in MLmodel; the fix for a mismatch is the pin in requirements.txt.
    # 2. a type outside TRUSTED_TYPES: skops.io.load raises UntrustedTypesFoundException.
    # 3. the schema must produce exactly the columns the model was fitted on, or every request would fail.
    mlmodel = yaml.safe_load((model_dir / "MLmodel").read_text())
    saved_with = mlmodel["flavors"]["sklearn"]["sklearn_version"]
    if saved_with != sklearn.__version__:
        raise RuntimeError(f"The model was saved with scikit-learn {saved_with}, this is {sklearn.__version__}: "
                           f"pin scikit-learn=={saved_with} in requirements.txt")

    model = skops.io.load(model_dir / "model.skops", trusted=TRUSTED_TYPES)

    expected = set(model.feature_names_in_)       # the column names remembered from fit
    if set(SCHEMA_COLUMNS) != expected:
        raise RuntimeError(f"Schema and model columns differ. Missing from the schema: "
                           f"{sorted(expected - set(SCHEMA_COLUMNS))}; unknown to the model: "
                           f"{sorted(set(SCHEMA_COLUMNS) - expected)}")
    return model


def model_version(model_dir: Path) -> str:
    # The registered name and version, e.g. "workspace.default.credit_risk/1": the same as the models:/ URI it came from, so a log
    # record leads straight back to the run in MLflow (its score, params and data hash).
    registered = json.loads((model_dir / "registered_model.json").read_text())
    return f"{registered['name']}/{registered['version']}"


def predict_probability(model, borrower: BorrowerInput) -> tuple[float, float]:
    # One borrower → a one-row DataFrame:
    # - columns=model.feature_names_in_ puts the columns in the order of fit, whatever order the JSON had
    #   (diagnostics.py, column_order_check: a numpy row in the wrong order gives wrong probabilities silently)
    # - dtype=float turns None into NaN, the blank the model was trained with
    row = pd.DataFrame([borrower.model_dump(by_alias=True)], columns=model.feature_names_in_, dtype=float)
    # The model's own time, in milliseconds. perf_counter is a monotonic clock made for timing; the wall clock
    # can jump when the system adjusts it.
    start_time = time.perf_counter()
    result = float(model.predict_proba(row)[0, 1])      # row 0, column 1 = probability of class 1 (serious late payment)
    end_time = time.perf_counter()
    latency = (end_time - start_time) * 1000  # Convert to milliseconds
    return result, latency
