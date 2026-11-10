# Loading the model and turning one request into one prediction. Plain Python, no FastAPI, so it can be tested on
# its own (like a @Service that the controller calls).
import time
import warnings

import joblib
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

from schemas import BorrowerInput

# The column names the schema produces, in field order (order doesn't matter: see predict_probability)
SCHEMA_COLUMNS = [field.serialization_alias for field in BorrowerInput.model_fields.values()]


def load_model(path):
    # Fail at startup, not on the first request (like a Spring context that won't start on a missing bean):
    # 1. a file saved by another scikit-learn version may load, then predict differently; scikit-learn only
    #    warns about it, so the warning is turned into an error. The fix is to pin the version in requirements.txt.
    # 2. the schema must produce exactly the columns the model was fitted on, or every request would fail.
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        model = joblib.load(path)
    expected = set(model.feature_names_in_)       # the column names remembered from fit
    if set(SCHEMA_COLUMNS) != expected:
        raise RuntimeError(f"Schema and model columns differ. Missing from the schema: "
                           f"{sorted(expected - set(SCHEMA_COLUMNS))}; unknown to the model: "
                           f"{sorted(set(SCHEMA_COLUMNS) - expected)}")
    return model


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
