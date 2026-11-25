# Disaster Tweets: the plan

Kaggle's "Natural Language Processing with Disaster Tweets": classify each tweet as about a real disaster (1) or not (0). Scored by **F1 on the disaster class**. This plan covers the baseline, TF-IDF and logistic regression; the fine-tuned transformer gets its own plan afterwards, on the same folds.

- **Data:** `kaggle/input/nlp-getting-started/` (not in git). `train.csv`: 7,613 tweets with `id`, `keyword`, `location`, `text`, `target`; `test.csv`: 3,263 without `target`. 57% not a disaster, 43% disaster.
- **The bar:** "always say disaster" scores F1 0.60 (precision 0.43, recall 1.0); "always say not disaster" scores 0, despite 57% accuracy. A model has to beat 0.60.
- **Why a baseline first:** it fits in seconds, so CV, error analysis and the submission format are worked out cheaply; its per-word weights show what the data rewards; and it's the number that tells whether the transformer is worth its cost.
- **Who writes what** (as before): **the user** writes the Pipeline, the CV, the keep rule and anything that could leak; **Claude** writes the data analysis, the plots and the setup. The user reviews Claude's code.
- **Layout:** `disaster_tweets.py` (the user's: pipeline, CV, submission), `data_analysis.py` (Claude's), `feature_notes.md` (decisions and why, as in the earlier projects).
- **Commits:** one per step, with its section in `feature_notes.md`; new concepts go into `concepts.md`.
- **Leaderboard leak:** the full labelled dataset is public, so the top of the leaderboard is at 1.0 (as with House Prices). Compare with your own CV and with honest scores (roughly 0.78–0.80 for TF-IDF, 0.83–0.84 for fine-tuned BERT-type models).

## Step 0: setup

1. A venv in `disaster-tweets/`, built with uv, with a `requirements.txt`: the same pins as `hugging-face-demo/` (torch with CUDA, transformers, datasets, accelerate, evaluate, pandas, scikit-learn), so the transformer step needs nothing new. Add `matplotlib` for the plots.
2. `feature_notes.md` with the header (problem, metric, data, who writes what).

**Check:** the venv's Python imports `sklearn`, `pandas` and `torch`, and `torch.cuda.is_available()` is True.

## Step 1: look at the data (Claude, `data_analysis.py`)

1. **Text:** length in characters and words; how many tweets have URLs, `@mentions`, `#hashtags`, HTML entities (`&amp;`); examples of each class.
2. **`keyword`:** 221 values, 0.8% missing, spaces written as `%20`. The disaster rate per keyword: which keywords are nearly always 1, nearly always 0, or mixed (the metaphor problem: "ablaze", "fire").
3. **`location`:** 3,341 values for 7,613 tweets, 33% missing, free text ("USA", "New York", "Earth"). Likely too messy for the baseline; check whether missing vs present differs in disaster rate.
4. **Duplicates:** 69 texts appear more than once (179 rows), 18 of them with **conflicting labels**. Decide what to do with them in step 2.
5. **Train vs test:** the same keyword list, and similar lengths.

**Check:** the numbers above are reproduced; the user reads the examples and the keyword table before step 2.

## Step 2: validation (user)

1. **Folds:** stratified 5-fold, fixed seed. **Duplicates in the same fold:** a tweet in both the training and the validation part is a memorised answer, so group duplicates (`StratifiedGroupKFold`, groups = the text normalised: URLs removed, lower case, spaces collapsed, as in `data_analysis.py`). Save the fold assignment (a `fold` column per `id`) so the transformer is scored on exactly the same folds.
2. **Conflicting duplicates:** keep them (the noise is real), drop them, or relabel by majority. Write the choice and why in `feature_notes.md`.
3. **Scoring:** F1 on class 1, plus precision, recall and accuracy for reading; train scores too (`return_train_score=True`), for the gap.
4. **Trivial models through the same CV:** `DummyClassifier(strategy="constant", constant=1)` should give F1 ≈ 0.60. This tests the CV code before any real model.

**Check:** the dummy's F1 is 0.60 on every fold; no text appears in both parts of any fold (an assert).

## Step 3: the baseline (user)

1. `Pipeline([("tfidf", TfidfVectorizer()), ("lr", LogisticRegression(max_iter=1000))])`, text only, default settings.
2. Run it through the step 2 CV. Print the mean and sd of F1 over the folds, and the train F1.
3. **Prediction first:** before running, guess the CV F1 and the train-vs-validation gap.
4. **Keep rule, written before any comparison:** from the baseline's fold-to-fold sd, write the rule for keeping a change (for example: mean F1 up by at least *x* and better on at least 4 of 5 folds, compared fold by fold). Changes in step 5 are judged by it.

**Check:** F1 well above 0.60; the vectorizer is fitted inside each fold (it's in the Pipeline), so the vocabulary and IDF never see the validation tweets.

## Step 4: look inside (Claude)

1. **The words:** the 20 largest positive and negative coefficients, with the word for each.
2. **The mistakes:** the most confident false positives and false negatives (out-of-fold predictions, `cross_val_predict` on the same folds).
3. **The cut-off:** F1 against the probability cut-off (0.5 is the default, not necessarily the best for F1). Choosing the cut-off is tuning, so it has to happen inside CV if it's kept.

**Check:** the user reads the words and the mistakes and predicts which step 5 change should help most.

## Step 5: improvements, one at a time (user, judged by the keep rule)

| | Change | Why it might help |
|---|---|---|
| 5a | word pairs: `ngram_range=(1, 2)` | "not safe", "on fire" carry meaning their words don't |
| 5b | `min_df`, `sublinear_tf=True` | drop words seen once; damp repeated words |
| 5c | tune `C` inside the folds (`LogisticRegressionCV` in the pipeline, nested) | the default C=1 may regularise too much or too little for 20,000 columns |
| 5d | add `keyword` (`ColumnTransformer`: one-hot keyword + TF-IDF text) | step 1 shows how much a keyword says alone |
| 5e | cleaning: URLs → one token, `@mentions` → one token, decode `&amp;` | thousands of unique URLs add columns that mean nothing |
| 5f | character n-grams (`analyzer="char_wb"`, 2–5) | robust to misspellings, hashtags and joined words |
| 5g | the cut-off from step 4, chosen inside the folds | F1 rewards a different balance of precision and recall than accuracy |

Each change is compared with the current model fold by fold; the simpler version stays unless the change clears the rule.

## Step 6: learning curve (Claude)

F1 against the number of training tweets (10% to 100%), train and validation (the roadmap's leftover). Still rising at 100% means more tweets would help; flat means the model has reached what bag-of-words can do.

## Step 7: submission

1. Fit the final pipeline on all of train; predict test; write `submission.csv` (`id`, `target`), checked against `sample_submission.csv`'s format.
2. Submit, and compare the Kaggle score with the CV F1.

**Check:** the Kaggle score lies within about the fold-to-fold spread of the CV F1.

## Progress

| Step | Result | Date |
|---|---|---|
| 0 setup | venv with uv (torch hardlinked from `D:\uv-cache`), imports and CUDA pass | 2026-11-24 |
| 1 data | URL and keyword are signals, location isn't; 12.7% of rows are duplicates once URLs are ignored, 66 groups conflicting | 2026-11-24 |
| 2 validation | grouped folds saved to `folds.csv` (disaster rate 42.94–42.98% per fold); dummy F1 0.6011 on both CVs; no local holdout, Kaggle is the final check | 2026-11-24 |
| 3 baseline | TF-IDF + LR: validation F1 0.748 ± 0.012 (grouped), train 0.859; keep rule: all 5 folds won and mean gain ≥ 0.008 | 2026-11-24 |
| 4 look inside | strongest words news-style (`in`, place names) vs personal (`you`, `my`); URLs cut into `http`/`https` pieces; 983 false negatives vs 562 false positives; best cut-off 0.46 only +0.004 | 2026-11-25 |
| 5 improvements | 5a word pairs −0.020, 5b-i `min_df=2` +0.002, 5b-ii `sublinear_tf` −0.002, 5c tuned C (3–10, nested, grouped inner CV) +0.004, 5d keyword +0.007 (4 of 5), 5e cleaning −0.006, 5f char n-grams +0.003, 5g tuned cut-off (0.45–0.47) +0.004: all 7 dropped; the baseline is the final TF-IDF model (0.7476) | 2026-11-25 |
| 6 learning curve | | |
| 7 submission | | |
