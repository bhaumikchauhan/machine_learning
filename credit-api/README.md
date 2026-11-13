# Credit risk API

Serves the [Give Me Some Credit](../give-me-some-credit/feature_notes.md) model (gradient boosting on the 10 raw columns, Kaggle private AUC 0.86670): one borrower in, the probability of a serious late payment (90+ days) within 2 years out. Phase 1 of the [roadmap](../roadmap.md). FastAPI itself is explained in [fast-api/](../fast-api/README.md); the comments here cover what's new about serving a model.

Read in this order:

1. [schemas.py](schemas.py): the contract; snake_case JSON mapped to the model's column names, required vs optional fields, unknown fields rejected; the prediction log table
2. [predictor.py](predictor.py): loading the champion from `model/` with startup checks (scikit-learn version, trusted types, columns), and turning a request into a one-row DataFrame
3. [db.py](db.py): the log database from an environment variable (SQLite file or Postgres)
4. [app.py](app.py): loading once in `lifespan`, the model as a dependency, `POST /predict` as a plain `def` that logs each prediction, `GET /health`
5. [tests/](tests/conftest.py): the API's code on a stand-in model (no `model/` needed): the API gives the model exactly the row training would, input validation, the log records (in an in-memory database), the startup checks

## Run it

```
# 1. The model: download the current champion from the MLflow registry into model/
#    (from credit-api, with the training venv, which has mlflow; see "The model from the registry" below)
..\give-me-some-creditenv\Scripts\python fetch_champion.py

# 2. The API
python -m venv venv
venv\Scriptsctivate
pip install -r requirements.txt
fastapi dev app.py      # then http://localhost:8000/docs: POST /predict, "Try it out" has borrower 1 prefilled
pytest                  # 15 tests; a stand-in model and an in-memory database, so no model/ and never data/predictions.db
```

From PowerShell without /docs:

```
Invoke-RestMethod -Method Post http://localhost:8000/predict -ContentType application/json -Body '{"revolving_utilization": 0.8855, "age": 43, "times_30_59_days_late": 0, "debt_ratio": 0.1775, "monthly_income": 5700, "open_credit_lines": 4, "times_90_days_late": 0, "real_estate_loans": 0, "times_60_89_days_late": 0, "dependents": 0}'
```

`GET /health` shows the model version being served, e.g. `{"status": "ok", "model_version": "credit-risk/5"}`.

## The model from the registry

The model is trained, registered and promoted in MLflow (see [give-me-some-credit/feature_notes.md](../give-me-some-credit/feature_notes.md), "Tracking with MLflow"). `fetch_champion.py` downloads the current champion into `model/` before an image is built, so the image holds one fixed model version and the API never talks to MLflow. A new model goes live in four steps:

```
# 1. Train and register a new version (from give-me-some-credit, with its venv, which has mlflow)
cd ..\give-me-some-credit
venv\Scripts\python -c "import give_me_some_credit as g; g.train_and_log()"   # prints "Created version 'N' of model 'credit-risk'"

# 2. Promote it: moves credit-risk@champion to version N only if the checks pass (signature, fold AUCs, data hash, keep rule)
venv\Scripts\python -c "import give_me_some_credit as g; g.promote(N)"

# 3. Download the champion into credit-api/model/ (model.skops, MLmodel, ..., registered_model.json with the version)
..\give-me-some-creditenv\Scripts\python ..\credit-apietch_champion.py

# 4. Rebuild the image (see Docker below), or restart `fastapi dev`

# Optional: look at the runs and the registry in a browser, http://localhost:5000
venv\Scripts\mlflow ui --backend-store-uri sqlite:///mlflow.db
```

- **No MLflow server needed:** with the default tracking URI (`sqlite:///` + give-me-some-credit/mlflow.db), the scripts read the database file directly; `mlflow ui` is only a viewer on that file. To use a tracking server instead, set `CREDIT_MLFLOW_TRACKING_URI=http://...`.
- **Other settings:** `CREDIT_REGISTERED_MODEL` (default `credit-risk`) and `CREDIT_ALIAS` (default `champion`) for the fetch; `CREDIT_MODEL_DIR` (default `model`) for the API.
- **No champion → exit code 1**, so a build step after it stops instead of packaging an old model.
- **`mlflow` isn't in requirements.txt:** the API image doesn't need it, only the fetch step does, so that step runs with the training venv. The API loads `model.skops` with `skops` alone.
- **Fetched at build time, not at startup:** the API starts without the registry being reachable (restarts, more copies, the cloud), and an image is one fixed, reproducible model version; rolling back means running the previous image.

## Docker

[Dockerfile](Dockerfile), written by the user. From credit-api, with the champion fetched into `model/` (step 1 above):

```
docker build -t credit-api .                  # -t names the image; "." is the build context
docker run --rm -p 8000:8000 -v credit-data:/app/data credit-api
# -p host:container port; --rm deletes the container when it stops;
# -v mounts the volume credit-data at /app/data, where the prediction log is written, so the records outlive the container
```

Then http://localhost:8000/docs, as without Docker. To read the log: `docker exec <container> python -c "..."`, or mount a folder instead of a named volume (`-v ${PWD}/data:/app/data`) and open the file locally. The image is about 560 MB (Python, scikit-learn, pandas, FastAPI).

- **Requirements before code:** Docker reruns a step only when it or a step above it changed, so `pip install` comes from the cache after a code edit.
- **`fastapi run`, not `fastapi dev`:** `dev` listens on 127.0.0.1, which inside a container is only the container itself; `run` listens on 0.0.0.0, and doesn't reload.
- **`CMD [...]`, the exec form:** starts the server directly, so it gets the stop signal. Anything after the `]` makes Docker fall back to running the line through a shell, and the container fails at startup.
- **Only the files the API needs:** the code, `requirements.txt` and `model/`. Not `venv/` (Windows binaries; the image installs its own), the tests or the training data.
- **Data goes in a volume.** A container's own file system is deleted with it (`--rm`), and a new container starts from the unchanged image. Checked: two containers in a row on the same volume, and the second found the first one's record.

## Prediction logging

Every prediction is a row in the `predictionlog` table, written by the user (step 4, the tests, by Claude):

| Column | Used for |
|---|---|
| `id` (UUID, returned to the client) | Matching the real outcome later, up to 2 years on, to measure live AUC |
| `timestamp` (UTC) | Every drift check compares a time window with training |
| `model_version` (the registered version, e.g. `credit-risk/5`; records from before MLflow have 12 characters of the model file's SHA-256) | Telling which model made a prediction; replaying inputs through a new model and comparing |
| the 10 inputs, as received | Drift monitoring: are the inputs still distributed as in training? Replays. |
| `probability` | Prediction drift; the comparison with a new model |
| `latency` (ms in `predict_proba`) | Is the model fast enough? |

- **Logged in the endpoint, not in middleware:** only the endpoint has the validated input and the probability, and only requests that reached the model get a record (no `/health`, no 422s).
- **The log write is part of the request:** if it fails, the client gets a 500, so no prediction is served without its record. The cost is a database write per prediction; at high traffic, records go to a queue or stdout instead and are loaded in batches.
- **`create_all` never changes an existing table.** After adding an input field, delete `data/predictions.db` locally, or use a migration tool (Alembic) once the records matter.

## What the tests check

The tests check the API's code, not a particular model, so they pass whichever version is the champion and run without `model/`, MLflow or the training data (CI included). Which model is served, and whether it's good enough, is settled by `promote()` and the registry.

- **A stand-in model** (conftest.py): a linear model fitted on random data, with its columns in another order than the schema. Every column has its own weight, so with a different value in every column, swapping any two changes the probability. A boosting stand-in missed a swap of two late-payment counts (its trees ignored one of them in that range); `test_the_check_above_catches_two_mixed_up_columns` checks the stand-in for this.
- **No training-serving skew:** the API's probability for a borrower equals the model's own `predict_proba` on the same borrower written with the Kaggle column names, the way training built rows. The Kaggle-submission test this replaced pinned one model version and would have failed at the next promotion.
- **Startup checks stay in the API:** the scikit-learn version is the API image's own, and the schema is the API's code, so only the API can check the model against them. `promote()` checks the model against the contract when it's published; the API checks it still speaks that contract when it starts.

## What the API must match from training

| Must match | How |
|---|---|
| The fitted model | The registered champion, fetched by `fetch_champion.py`; `lifespan` loads it once |
| Column names and order | The row is built in `model.feature_names_in_` order; the JSON's key order doesn't matter |
| scikit-learn version | Pinned in requirements.txt; a model saved by another version (recorded in `model/MLmodel`) stops the app at startup |
| The raw inputs | No feature engineering; `monthly_income` and `dependents` may be null or left out (NaN, as in training) |

## Decisions

- **Probability only, no approve/decline.** The cut-off is a business choice and needs calibrated probabilities (a roadmap leftover).
- **Bounds reject impossible values** (age 18 to 120, no negative counts). Within them the model scores anything; trees treat a value beyond the training range like the largest one seen.
- **The 96/98 late-count codes** in the Kaggle file were data-entry codes, not real counts; the API accepts them as numbers, so they still score like in training.
- **`model/` isn't in git.** It's a build output, downloaded by `fetch_champion.py`; the versions live in the MLflow registry.

## CI

[.github/workflows/credit-api.yml](../.github/workflows/credit-api.yml) runs on every push that changes `credit-api/` (or the workflow itself). A fresh Ubuntu runner checks out the repo, sets up Python 3.11 (as in the Dockerfile), installs requirements.txt and runs `pytest` in `credit-api/`. The tests need no `model/`, MLflow or training data, so nothing else is needed.

Not yet: building the image. The Dockerfile copies `model/`, which isn't in git, and the runner can't reach the registry (`mlflow.db` on one machine). The build job comes after an MLflow registry in the cloud.

## Next in phase 1

A cloud MLflow registry, then the CI build job (fetch the champion, `docker build`), then a cloud deployment.
