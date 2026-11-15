# ML engineer roadmap

The plan from here to an ML engineer job. It starts from 13 years of Java/C# backend work, which counts most in deployment and the capstone. Agreed 2026-11-06, written down 2026-11-09.

Status: ✅ done · 🟡 in progress · ⬜ to do

## Done so far

| Area | Problem | Model and result | What it taught |
|---|---|---|---|
| Supervised, classification | [Titanic](titanic/feature_notes.md) | Logistic regression, 14 features including group fate on a shared ticket; Kaggle 0.78708 | Feature engineering one column at a time, cross-validation, a keep rule, leakage from a target-based feature |
| Supervised, regression | [House Prices](house-prices/feature_notes.md) | Ridge; CV RMSE 0.1118, Kaggle 0.12412 | The leftover check, feature engineering depends on the model, trusting CV over the leaderboard, blending |
| Supervised, imbalanced classification | [Give Me Some Credit](give-me-some-credit/feature_notes.md) | Gradient boosting on the raw columns; CV AUC 0.865, Kaggle private 0.86670 | Choosing the score (AUC, not accuracy), impossible values, boosting learns shapes on its own |
| Unsupervised | [Credit card clustering](credit-card-clustering/feature_notes.md) | k-means, k = 6, six named segments; checked with PCA, Ward and DBSCAN | Stating the goal without a target, scaling defines "similar", choosing k, profiles as the real check |
| Deep learning basics | [Digit Recognizer](digit-recognizer/feature_notes.md) | PyTorch CNN; 98.84% held-out over 5 seeds, Kaggle 0.98960 | Training loops, seeding and determinism, the noise of a difference, augmentation, error analysis |

The habits behind them, written up in [concepts.md](concepts.md):

- **Validation:** cross-validation or a held-out set chosen before looking, the test set scored once, a prediction written down before each Kaggle submission.
- **Keep rules:** written before running, sized from the noise, never moved after seeing the result.
- **Error analysis:** look at the leftovers and the mistakes before guessing a fix.
- **scikit-learn Pipelines:** fills, encodings and scaling fitted inside each fold, so nothing leaks.
- **PyTorch:** networks, the training loop, early stopping, GPU training.

## What's left

### Phase 1: Deployment ✅

**Goal:** serve one model end to end, the way a production team would: the Give Me Some Credit boosting model (chosen 2026-11-09).

| Step | What | Status |
|---|---|---|
| FastAPI | Endpoints, schemas, dependencies, tests. Studied with a condensed course project in [fast-api/](fast-api/README.md). | ✅ |
| Model endpoint | Load the model once at startup; a `predict` endpoint with validated input and output schemas. Done in [credit-api/](credit-api/README.md) for the Give Me Some Credit model. | ✅ |
| pytest | Tests for the endpoint, the schemas and the preprocessing; 15 tests in credit-api, on a stand-in model: they test the API's code, and whether a model is good enough is decided at promotion. | ✅ |
| Docker | An image for the API with pinned dependencies; it runs the same on any machine. Dockerfile in credit-api. | ✅ |
| Prediction logging | Log each request, prediction and latency. This is the raw material for monitoring later. A `predictionlog` table in credit-api, kept in a Docker volume. | ✅ |
| MLflow | Track the training runs (parameters, scores, files) and register the model; the API loads a registered version. `train_and_log()` and `promote()` in give-me-some-credit; `credit-api/fetch_champion.py` downloads the champion into the image before it's built. The registry is MLflow on Databricks Free Edition (`workspace.default.credit_risk`), reachable from CI. | ✅ |
| CI with GitHub Actions | Every push runs the tests and builds the image. [.github/workflows/credit-api.yml](.github/workflows/credit-api.yml): a test job, then a build job that fetches the champion from Databricks (token as a GitHub secret) and runs `docker build`. | ✅ |
| Cloud | Deploy the container to one cloud and call it over HTTPS. Live on Azure Container Apps (centralindia), image in Azure Container Registry, pulled with a managed identity. A push to a release branch (`R_yy_mm_dd_hh_mm_ss`) tests, builds and deploys it, signing in to Azure with OIDC; see [credit-api/azure_deployment_plan.md](credit-api/azure_deployment_plan.md). | ✅ |

Refresher courses (Pluralsight): FastAPI Foundations (done), Python: Testing Strategies, Docker Foundations, Introduction to MLOps.

**Done when** a public URL returns predictions from a registered model, and a push to GitHub tests and rebuilds it. Done 2026-11-14: https://credit-api.lemonocean-12436fd6.centralindia.azurecontainerapps.io, first release `R_26_11_14_12_38_36`.
the /docs give the API
https://credit-api.lemonocean-12436fd6.centralindia.azurecontainerapps.io

### Phase 2: Transformers and LLMs ⬜

**Study first** (the transformer is new; the order goes from intuition to code to the library):

| # | Resource | What it gives |
|---|---|---|
| 1 | 3Blue1Brown, "Neural networks" series, the chapters on GPT and attention (YouTube) | The intuition: embeddings, attention, what a transformer layer does |
| 2 | Jay Alammar, "The Illustrated Transformer" (blog) | The architecture drawn step by step |
| 3 | Andrej Karpathy, "Let's build GPT: from scratch, in code, spelled out" (YouTube, about 2 hours) | A small transformer written in PyTorch, building on the Digit Recognizer training loops |
| 4 | Hugging Face LLM Course, chapters 1 to 3 (free, huggingface.co/learn) | Tokenizers, pretrained models, and fine-tuning with the `transformers` library: exactly part 1 |
| Later | The rest of the Hugging Face course, and DeepLearning.AI's short courses on RAG | Part 2 |
| Optional | *Natural Language Processing with Transformers* (Tunstall, von Werra, Wolf; O'Reilly) or *Hands-On Large Language Models* (Alammar, Grootendorst; O'Reilly) | A book to go deeper |


**Part 1, a transformer classifier:** Kaggle "Natural Language Processing with Disaster Tweets".

- A baseline first: TF-IDF and logistic regression.
- Then a pretrained transformer (for example DistilBERT) fine-tuned with Hugging Face.
- Concepts: tokens, embeddings, attention, fine-tuning against training from scratch.
- Deploying it stretches phase 1: a bigger model, CPU or GPU, latency, batching requests.
- **Leftovers folded in here:**
  - **Learning curves:** would more tweets help the TF-IDF model?
  - **Stacking:** a learned combination of TF-IDF and the transformer.

**Part 2, an LLM application:**

- **Prompting** and calling an LLM API; cost and latency.
- **RAG** (retrieval-augmented generation): split documents into chunks, embed them, find the closest chunks to a question, and give them to the LLM as context.
- **Vector databases:** they store the embeddings and find the nearest neighbours fast, using approximate indexes such as HNSW. Learn with FAISS, then build with pgvector (Postgres). Qdrant, Chroma and Pinecone are worth knowing by name. The distance ideas are the same as in clustering.
- **Evaluation:** a test set of questions; check whether the right chunks were retrieved and whether the answers are faithful to them.

### Phase 3: Time series ⬜

Moved after transformers on 2026-11-14 (phase 2 doesn't depend on it). It still comes before the capstone and interview prep: forecasting is common in ML jobs and in system design questions.

**Problem:** Kaggle "Store Sales – Time Series Forecasting".

- **Validation over time:** train on the past, validate on the future, with expanding windows. Shuffled folds would leak the future into training.
- **Baselines first:** last value, the same day last week, a moving average.
- **Features:** lags, rolling means, day of week and month, holidays, promotions; each must use only data available at prediction time.
- **Models:** gradient boosting on lag features (the usual winner); a short look at classical methods (ARIMA, exponential smoothing).

### Phase 4: Capstone ⬜

One end-to-end project on data I pick. It's the centrepiece of the portfolio.

- **Data pipeline:** ingest the data and validate it, repeatably.
- **Training:** tracked in MLflow, with the validation habits above.
- **Deployment:** the phase 1 setup (API, Docker, CI, cloud).
- **Monitoring:** data drift and prediction drift, using the prediction logs.
- **Retraining:** scheduled or triggered by drift; a new version goes to the registry and replaces the old one only if it scores better.
- **Write-up:** the problem, the decisions, the results and the lessons.
- Could use phase 2 (search or recommendations with embeddings) if the data suits it.

### Phase 5: Interview prep ⬜

- **ML system design:** a framework (requirements, data, features, model, serving, monitoring) practised on common cases: recommendations, fraud detection, search ranking, demand forecasting.
- **Statistics and ML theory:** the bias-variance trade-off, regularization, metrics and when each fits, A/B tests and significance, how each model family works.
- **SQL:** joins, aggregations, window functions, query-writing exercises.
- **Python coding:** data-structure problems (easy and medium), pandas.
- **Telling the story:** each portfolio project in two minutes; backend experience framed as production ML strength.

## Alongside the phases

| Track | What | When |
|---|---|---|
| Leftovers | **Stacking** (House Prices blended by hand); **thresholds and calibration** (Give Me Some Credit: turning a probability into a decision, and probabilities you can trust); **learning curves** (would more data help?). Optional: the rest of House Prices' columns; dropout and voting augmented CNNs in Digit Recognizer. | Stacking and learning curves in phase 2; thresholds and calibration in phase 4 |
| SQL | Regular practice. It comes back in time series features, pgvector and the capstone's data pipeline. | Throughout |
| Cloud ML platform | The registry is on Databricks (managed MLflow 3, Unity Catalog), the container goes to Azure. Azure ML's MLflow follows MLflow 2 (no aliases), so it lost the registry. Know AWS SageMaker and GCP Vertex AI by name. | From phase 1's cloud step, more in phase 4 |
| Portfolio | This GitHub repo: a README per project, the notes and lessons, a live demo link from phase 1, the capstone write-up, the Kaggle profile. | Updated as each phase finishes |
