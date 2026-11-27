# Disaster Tweets: the transformer plan

Fine-tune a pretrained transformer on the same task, on the **same grouped folds** (`folds.csv`), so every result compares fold by fold with the TF-IDF model. The TF-IDF plan is in [plan.md](plan.md); its decisions and results are in [feature_notes.md](feature_notes.md).

- **The bar:** the final TF-IDF model, grouped CV F1 **0.7476 ± 0.012** (per fold 0.7247, 0.7516, 0.7490, 0.7555, 0.7571), Kaggle 0.79374. Honest scores quoted for fine-tuned BERT-type models are about 0.83–0.84 on Kaggle.
- **Why a transformer should help:** the learning curve was still rising (about +0.03 F1 per doubling of the tweets), and no setting of bag of words got more out of these tweets. A pretrained model brings what it learned from billions of words: the "more data" the curve asks for. And it reads words in context ("my mixtape is fire" against "the forest is on fire"), which bag of words can't.
- **What carries over from the Hugging Face demo** ([main.py](../hugging-face-demo/main.py)): `AutoModelForSequenceClassification` adds a new random `Linear(768, 2)` head; `set_seed(seed)` goes **before** `from_pretrained`, or the head starts differently each run; `Trainer` with `compute_metrics`; checkpoints in `output_dir` hold the optimizer state and are for resuming, not for keeping.
- **What's new here:** five fine-tunes per comparison (one per fold) instead of one; a metric (F1) that must be computed per fold the same way as in `cross_validate`; and the epoch count must not be chosen by looking at the validation fold (see step 2).
- **Who writes what** (as before): **the user** writes the fold loop, the training and anything that could leak (epoch choice, seeds, what the model sees); **Claude** writes the analysis, the plots and the setup. The user reviews Claude's code.
- **Layout:** `transformer.py` (the user's: fold loop, training, out-of-fold predictions, submission); out-of-fold probabilities saved per run (`oof_<run>.csv`: `id`, `fold`, `p`) for analysis and a possible blend; model files and checkpoints not in git.
- **Commits:** one per step, with its section in `feature_notes.md` ("Transformer" part); new concepts go into `concepts.md`.
- **Compute:** RTX 4070 SUPER (12 GB), the venv already has torch with CUDA, transformers, datasets, accelerate, evaluate. Tweets are short: median **33** BERT tokens, 95% within 53, the longest 84. So `max_length=96` (or dynamic padding) covers every tweet, and one fine-tune of a base model should take a few minutes.

## Step 1: one fold, end to end (user)

1. Fold 0 only: train on folds 1–4, evaluate on fold 0. `bert-base-uncased`, the course's settings (learning rate 5e-5, batch size 8 to 32, **3 epochs**), seeded.
2. `compute_metrics` returns F1 on class 1 (plus precision, recall, accuracy), from argmax of the logits, so it matches `cross_validate`'s F1 at a 0.5 cut-off.
3. Save fold 0's probabilities (softmax of the logits, column 1) with their `id`.

How to write it:
- **Data:** merge `train.csv` with `folds.csv` on `id`; fold 0 is the validation part, folds 1–4 the training part. Each part becomes a `datasets.Dataset` (`Dataset.from_pandas`) with `text`, `label` and `id`. Tokenize with `truncation=True, max_length=96`; a `DataCollatorWithPadding` pads each batch (as in the demo).
- **Seed:** `set_seed(SEED)` before `from_pretrained(..., num_labels=2)`.
- **`TrainingArguments`:** `num_train_epochs=3`, `learning_rate=5e-5`, `per_device_train_batch_size=16`, `bf16=True` (the 4070 supports it; faster), `report_to="none"`.
  - **`eval_strategy="no"`**, then one `trainer.predict()` on fold 0 after training. Scoring fold 0 after every epoch would show which epoch was best, and step 2's epoch decision would then be shaped by the validation tweets.
  - **`save_strategy="no"`**: no checkpoints (GBs each); the `output_dir` itself goes in `.gitignore`.
- **After training:** `p = softmax(logits)[:, 1]`; save `id`, `fold`, `p` to `oof_bert_fold0.csv`; recompute F1 from `p ≥ 0.5` with `f1_score` and compare it with `compute_metrics`.
- **Assert:** no fold 0 `id` is in the training part.
- Run output goes into `output.txt` (gitignored).

**Prediction first:** fold 0's F1 (TF-IDF scored 0.7247 there, its hardest fold), and the minutes per epoch. **The user's prediction (2026-11-25): F1 0.83, about 1 minute per epoch.**

**Check:** the F1 recomputed from the saved probabilities (`p ≥ 0.5`) with `sklearn.metrics.f1_score` equals what `compute_metrics` printed; the validation tweets of fold 0 never reach the training set (an assert on the ids).

## Step 2: decisions before the full CV (user, written in feature_notes.md before any 5-fold run)

1. **The number of epochs:** fixed in advance (the course's 3), or chosen inside each fold's training part (a grouped split of the training part, as with C and the cut-off). **Not** by watching fold *k*'s validation F1 and keeping the best epoch: that choice is fitted to the tweets that score it (step 4 of the TF-IDF plan, the cut-off, was the same trap).
   **Changed 2026-11-27:** fixed at 3 after fold 0's check curves came out flat (means 0.7824–0.7854 over 1–5 epochs; see feature_notes.md). Steps 3 and 5 use 3 epochs, no check split.
2. **Seeds:** a fine-tune's score varies with the seed (the head's start and the batch order; the demo's unseeded run scored 0.79 accuracy against 0.855 seeded). Decide how many seeds per fold (1 is cheap; 3 shows the spread), and how a comparison uses them (the mean over seeds, per fold).
3. **The keep rule for transformer changes:** the TF-IDF rule (all 5 folds won, mean gain ≥ 0.008), or a different one written now, given that seed noise adds to fold noise.
4. **What the model sees:** the text only, as TF-IDF did. Keyword is a later change (step 5).
5. **The seed-spread run (user):** (as first planned; after the epoch change, only the last part, 3 seeds × 3 epochs on folds 1–4, is rerun) fold 0 only, 3 seeds (42, 43, 44), the step 1 code turned into functions so step 3 is a loop over folds:
   - Split the training part (folds 1–4) with `StratifiedGroupKFold(5, shuffle=True, random_state=SEED)` on the normalised text; the first split gives inner-train (≈ 4,870) and check (≈ 1,220), the same for all seeds.
   - Per seed: one 5-epoch run on inner-train, `eval_strategy="epoch"` on the check part (never fold 0); the check F1 per epoch from `trainer.state.log_history`.
   - Average the 3 check curves; *n* = the best average epoch (a tie → fewer epochs).
   - Per seed: a fresh model on the whole training part for *n* epochs, `eval_strategy="no"`; predict fold 0 once; F1 at `p ≥ 0.5`.
   - Print the curves, *n*, fold 0's F1 per seed with mean and sd, and the time; save `oof_bert_seeds_fold0.csv` (`id`, `fold`, `seed`, `p`).
   - The sd decides step 5's seed count: above 0.003 → 3 seeds.

**Check:** no id shared between inner-train, check and fold 0, no group across inner-train and check; 5 check scores per seed.

## Step 3: the transformer baseline, 5 folds (user)

1. The fold loop over `folds.csv`: for each fold, a fresh seeded model, trained on the other four folds, scored on this one. Out-of-fold probabilities for all 7,613 tweets saved.
2. Print F1 per fold, mean and sd, and the time.
3. Compare with the TF-IDF model **fold by fold** with the keep rule.

**Prediction first:** the mean CV F1, and whether it passes the keep rule against TF-IDF.

**Check:** every tweet gets exactly one out-of-fold probability; the per-fold F1 matches the printed one.

## Step 4: look inside (Claude)

1. **Where the two models disagree:** tweets the transformer gets right and TF-IDF wrong, and the reverse, per class (out-of-fold, same folds). Does the transformer fix the metaphors ("I'm On Fire.") and the news-style false alarms?
2. **The mistakes left:** the most confident false positives and negatives; how many are in the 66 conflicting groups (label noise neither model can fix).
3. **The cut-off:** F1 against the cut-off on the out-of-fold probabilities (reading only, as in the TF-IDF step 4).
4. **Fold 1:** does it still lose to the others?

**Check:** the user reads the disagreements and predicts which step 5 change helps most.

## Step 5: improvements, one at a time (user, judged by the keep rule)

| | Change | Why it might help |
|---|---|---|
| 5a | a model pretrained on tweets (`cardiffnlp/twitter-roberta-base` or `vinai/bertweet-base`) | it has seen hashtags, @mentions, slang and t.co links in pretraining |
| 5b | `keyword` in the input (`"keyword: earthquake. text…"`) | TF-IDF's keyword change was the closest to passing (+0.007, 4 of 5 folds) |
| 5c | a larger model (`roberta-large`, `bert-large`) | more capacity; about 3 times the time, and less stable on small data |
| 5d | the cut-off chosen inside the folds | as TF-IDF's 5g |
| 5e | blend: the average of TF-IDF's and the transformer's probabilities (weight fixed in advance, e.g. 0.5, or chosen inside the folds) | the two models make different mistakes; the deferred "blending / stacking" from House Prices |

Each change is compared with the current transformer fold by fold; the simpler version stays unless the change clears the rule.

## Step 6: submission

1. **The final model:** decide between (a) one model fine-tuned on all 7,613 tweets with the step 2 epoch count, or (b) the average of the five fold models' probabilities on test (no extra training; each model saw 80% of the tweets). Write the choice and why before predicting.
2. Write `submission_transformer.csv`, checked against `sample_submission.csv` (as `create_submission`); submit.
3. **Expected score written first,** using what step 7 of the TF-IDF plan showed: Kaggle came out about 0.046 above grouped CV.

**Check:** the Kaggle score against the expectation, and against TF-IDF's 0.79374.

## Progress

| Step | Result | Date |
|---|---|---|
| 1 one fold | fold 0: BERT F1 0.7816 vs TF-IDF 0.7247 (+0.057); 17 s per epoch; checks pass | 2026-11-26 |
| 2 decisions | epochs chosen inside each fold (one check split, up to 5 epochs, seed curves averaged, retrain on the whole training part); seed spread measured on fold 0 first, 1 seed for the baseline, 3 for step 5 if the seed sd is above 0.003; keep rule unchanged; text only. Epochs then fixed at 3 (fold 0's check curves flat over 1–5). Seed-spread run: fold 0 F1 0.7816 / 0.7847 / 0.7848, sd 0.0018 → 1 seed for step 5; step 1 reproduced | 2026-11-27 |
| 3 baseline, 5 folds | | |
| 4 look inside | | |
| 5 improvements | | |
| 6 submission | | |
