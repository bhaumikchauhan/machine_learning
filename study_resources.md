# Study resources

The courses, videos and documentation behind the work in this repository, in the order they were used. Only what has already been studied is listed; the plan ahead is in [roadmap.md](roadmap.md).

## 1. Foundations

**Machine Learning Specialization** by Andrew Ng (DeepLearning.AI and Stanford), on Coursera: https://www.coursera.org/specializations/machine-learning-introduction

It has three courses:

1. Supervised Machine Learning: Regression and Classification
2. Advanced Learning Algorithms
3. Unsupervised Learning, Recommenders, Reinforcement Learning

## 2. Practice on Kaggle

| Problem | Link | Notes in this repo |
|---|---|---|
| Titanic | https://www.kaggle.com/competitions/titanic | [titanic/](titanic/feature_notes.md) |
| House Prices | https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques | [house-prices/](house-prices/feature_notes.md) |
| Give Me Some Credit | https://www.kaggle.com/competitions/GiveMeSomeCredit | [give-me-some-credit/](give-me-some-credit/feature_notes.md) |
| Credit Card Dataset for Clustering | https://www.kaggle.com/datasets/arjunbhasin2013/ccdata | [credit-card-clustering/](credit-card-clustering/feature_notes.md) |
| Digit Recognizer | https://www.kaggle.com/competitions/digit-recognizer | [digit-recognizer/](digit-recognizer/feature_notes.md) |

## 3. Hierarchical clustering

| Resource | Source | Link |
|---|---|---|
| StatQuest: Hierarchical Clustering (one video) | YouTube, StatQuest with Josh Starmer | https://www.youtube.com/watch?v=7xHsRkOdVwo |
| Hierarchical Clustering (playlist; it starts with Mahesh Huddar's "Agglomerative Hierarchical Clustering Algorithm"). It explained the topic more clearly than the StatQuest video. | YouTube | https://www.youtube.com/watch?v=i6XIu-H2gOc&list=PLpmY-bsmBBXpgHSx0yJD0o6oUOjqEHB1m |

## 4. Convolutional neural networks

| Resource | Source | Link |
|---|---|---|
| Convolutional Neural Network (playlist; the first 7 videos) | YouTube, Learn With Jay | https://www.youtube.com/watch?v=E5Z7FQp7AQQ&list=PLuhqtP7jdD8CD6rOWy20INGM44kULvrHu |

## 5. Transformers and LLMs

The resources for Phase 2 (GPT from scratch + HuggingFace), in the order studied.

| Resource | Source | Link |
|---|---|---|
| Neural networks series: the chapters on GPT and attention | YouTube, 3Blue1Brown | https://www.youtube.com/watch?v=wjZofJX0v4M |
| The Illustrated Transformer | Blog, Jay Alammar | https://jalammar.github.io/illustrated-transformer |
| Stanford CME295 Transformers & LLMs, Lecture 1 – Transformer (Autumn 2025) | YouTube, Stanford | https://www.youtube.com/watch?v=Ub3GoFaUcds |
| Let's build GPT: from scratch, in code, spelled out (~2 hours) | YouTube, Andrej Karpathy | https://www.youtube.com/watch?v=kCc8FmEb1nY |
| Word Embedding and Word2Vec, Clearly Explained!!! | YouTube, StatQuest with Josh Starmer | https://www.youtube.com/watch?v=viZrOnJclY0 |
| Word Embedding in PyTorch + Lightning | YouTube, StatQuest with Josh Starmer | https://www.youtube.com/watch?v=Qf06XDYXCXI |
| HuggingFace LLM Course, chapters 1–3 | huggingface.co/learn | https://huggingface.co/learn/llm-course |

Build notes in [gpt-from-scratch/](gpt-from-scratch/learning_notes.md). Concepts in [concepts.md](concepts.md) under "Language models and Transformers".

## 6. Building and deploying a model

The credit risk API ([credit-api/](credit-api/README.md)) serves the Give Me Some Credit model. It covers the API, tests, Docker, prediction logging, MLflow, CI and a cloud deployment.

**Course:** FastAPI Foundations by Reindert-Jan Ekker, on Pluralsight (https://www.pluralsight.com). A condensed version of the course project is in [fast-api/](fast-api/README.md).

**Topics, studied through videos and the official documentation:**

| Topic | Official documentation |
|---|---|
| FastAPI | https://fastapi.tiangolo.com |
| pytest | https://docs.pytest.org |
| Docker | https://docs.docker.com |
| MLflow: experiment tracking, model registry, aliases | https://mlflow.org/docs/latest |
| Databricks Free Edition (managed MLflow, Unity Catalog) | https://docs.databricks.com |
| GitHub Actions: workflows, secrets, environments, OIDC | https://docs.github.com/en/actions |
| Azure Container Registry and Azure Container Apps | https://learn.microsoft.com/azure/container-apps |
