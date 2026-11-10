# Credit risk API

Serves the [Give Me Some Credit](../give-me-some-credit/feature_notes.md) model (gradient boosting on the 10 raw columns, Kaggle private AUC 0.86670): one borrower in, the probability of a serious late payment (90+ days) within 2 years out. Phase 1 of the [roadmap](../roadmap.md). FastAPI itself is explained in [fast-api/](../fast-api/README.md); the comments here cover what's new about serving a model.

Read in this order:

1. [schemas.py](schemas.py): the contract; snake_case JSON mapped to the model's column names, required vs optional fields, unknown fields rejected
2. [predictor.py](predictor.py): loading the model with startup checks, and turning a request into a one-row DataFrame
3. [app.py](app.py): loading once in `lifespan`, the model as a dependency, `POST /predict` as a plain `def`, `GET /health`
4. [tests/test_predict.py](tests/test_predict.py): the API against the Kaggle submission, input validation, `predictor.py` with a stand-in model

## Run it

```
# 1. Save the model (from give-me-some-credit, with the repo's .venv; writes give-me-some-credit/model.joblib)
cd ..\give-me-some-credit
..\.venv\Scripts\python -c "import give_me_some_credit as g; g.save_model()"

# 2. Copy it next to the API (the app loads model.joblib from the folder it starts in; MLflow replaces this later)
copy model.joblib ..\credit-api\

# 3. The API (from credit-api)
cd ..\credit-api
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
fastapi dev app.py      # then http://localhost:8000/docs: POST /predict, "Try it out" has borrower 1 prefilled
pytest                  # 10 tests
```

From PowerShell without /docs:

```
Invoke-RestMethod -Method Post http://localhost:8000/predict -ContentType application/json -Body '{"revolving_utilization": 0.8855, "age": 43, "times_30_59_days_late": 0, "debt_ratio": 0.1775, "monthly_income": 5700, "open_credit_lines": 4, "times_90_days_late": 0, "real_estate_loans": 0, "times_60_89_days_late": 0, "dependents": 0}'
```

## Docker

[Dockerfile](Dockerfile), written by the user. From credit-api, with `model.joblib` copied in (step 2 above):

```
docker build -t credit-api .                  # -t names the image; "." is the build context
docker run --rm -p 8000:8000 credit-api       # -p host:container port; --rm deletes the container when it stops
```

Then http://localhost:8000/docs, as without Docker. The image is about 560 MB (Python, scikit-learn, pandas, FastAPI).

- **Requirements before code:** Docker reruns a step only when it or a step above it changed, so `pip install` comes from the cache after a code edit.
- **`fastapi run`, not `fastapi dev`:** `dev` listens on 127.0.0.1, which inside a container is only the container itself; `run` listens on 0.0.0.0, and doesn't reload.
- **`CMD [...]`, the exec form:** starts the server directly, so it gets the stop signal. Anything after the `]` makes Docker fall back to running the line through a shell, and the container fails at startup.
- **Only the files the API needs:** the code, `requirements.txt` and the model. Not `venv/` (Windows binaries; the image installs its own), the tests or the training data.

## What the API must match from training

| Must match | How |
|---|---|
| The fitted model | `save_model()` fits it once; `lifespan` loads it once |
| Column names and order | The row is built in `model.feature_names_in_` order; the JSON's key order doesn't matter |
| scikit-learn version | Pinned in requirements.txt; a file from another version stops the app at startup |
| The raw inputs | No feature engineering; `monthly_income` and `dependents` may be null or left out (NaN, as in training) |

## Decisions

- **Probability only, no approve/decline.** The cut-off is a business choice and needs calibrated probabilities (a roadmap leftover).
- **Bounds reject impossible values** (age 18 to 120, no negative counts). Within them the model scores anything; trees treat a value beyond the training range like the largest one seen.
- **The 96/98 late-count codes** in the Kaggle file were data-entry codes, not real counts; the API accepts them as numbers, so they still score like in training.
- **`model.joblib` isn't in git.** It's a build output, rebuilt by `save_model()`; MLflow stores versions later in phase 1.

## Next in phase 1

Prediction logging, MLflow (the API loads a registered model instead of a file), CI with GitHub Actions, then a cloud deployment.
