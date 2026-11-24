# Disaster Tweets notes

Decisions and reasoning for Kaggle's "Natural Language Processing with Disaster Tweets": classify each tweet as about a real disaster (1) or not (0). The first NLP competition: a TF-IDF and logistic-regression baseline first, then a fine-tuned transformer on the same folds. Scored by **F1 on the disaster class**. The steps are in [plan.md](plan.md).

Status: started 2026-11-24.

**Who writes what** (the same split as before):
- **The user writes:** the Pipeline, the CV, the keep rule, and anything that could leak.
- **Claude writes:** the data analysis, the plots and the setup.

Data: `kaggle/input/nlp-getting-started/` (not in git). `train.csv`: 7,613 tweets with `id`, `keyword`, `location`, `text`, `target`; 57% not a disaster, 43% disaster. `keyword`: 221 values, 0.8% missing; `location`: 3,341 free-text values, 33% missing; `text`: 7 to 157 characters, median 107, none missing. 69 texts appear more than once (179 rows), 18 of them with conflicting labels. `test.csv`: 3,263 tweets without `target`.

**The bar:** "always say disaster" scores F1 0.60 (precision 0.43, recall 1.0). "Always say not disaster" scores 0 (it flags nothing, so recall is 0), despite the higher accuracy, 57% against 43%. The metric decides which trivial model looks better.

Setup: this folder's venv, built with uv (instructions at the top of [requirements.txt](requirements.txt)): the Hugging Face demo's pins (torch 2.15.0 with CUDA 13.2, transformers, datasets, accelerate, evaluate, pandas, scikit-learn) plus matplotlib, so the transformer step needs nothing new. GPU: NVIDIA GeForce RTX 4070 SUPER (12 GB).
