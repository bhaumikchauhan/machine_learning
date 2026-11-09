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

# 2. The API (from credit-api)
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

Docker image, prediction logging, MLflow (the API loads a registered model instead of a file), CI with GitHub Actions, then a cloud deployment.
