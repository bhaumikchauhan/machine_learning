# Disaster Tweets notes

Decisions and reasoning for Kaggle's "Natural Language Processing with Disaster Tweets": classify each tweet as about a real disaster (1) or not (0). The first NLP competition: a TF-IDF and logistic-regression baseline first, then a fine-tuned transformer on the same folds. Scored by **F1 on the disaster class**. The steps are in [plan.md](plan.md).

Status: started 2026-11-24.

**Who writes what** (the same split as before):
- **The user writes:** the Pipeline, the CV, the keep rule, and anything that could leak.
- **Claude writes:** the data analysis, the plots and the setup.

Data: `kaggle/input/nlp-getting-started/` (not in git). `train.csv`: 7,613 tweets with `id`, `keyword`, `location`, `text`, `target`; 57% not a disaster, 43% disaster. `keyword`: 221 values, 0.8% missing; `location`: 3,341 free-text values, 33% missing; `text`: 7 to 157 characters, median 107, none missing. 69 texts appear more than once (179 rows), 18 of them with conflicting labels. `test.csv`: 3,263 tweets without `target`.

**The bar:** "always say disaster" scores F1 0.60 (precision 0.43, recall 1.0). "Always say not disaster" scores 0 (it flags nothing, so recall is 0), despite the higher accuracy, 57% against 43%. The metric decides which trivial model looks better.

Setup: this folder's venv, built with uv (instructions at the top of [requirements.txt](requirements.txt)): the Hugging Face demo's pins (torch 2.15.0 with CUDA 13.2, transformers, datasets, accelerate, evaluate, pandas, scikit-learn) plus matplotlib, so the transformer step needs nothing new. GPU: NVIDIA GeForce RTX 4070 SUPER (12 GB).

## The data (step 1, 2026-11-24)

From [data_analysis.py](data_analysis.py); plots in `plots/`.

- **Reading the file:** a standard CSV. Values with a comma or a line break are in double quotes, so `pd.read_csv` reads them whole: 1,851 locations contain a comma ("Sao Paulo, Brazil"), 435 tweets span several lines, only 2 texts contain one. Splitting lines on commas would break; `read_csv` gives 7,613 rows, `target` only 0 or 1, unique ids.
- **Text:** a URL is a strong signal (54.7% disaster with one, 30.2% without: news links); `@mentions` point the other way (33.2% against 46.5%: conversations). Disaster tweets are longer (median 115 characters against 101). HTML entities are written as text: `&amp;` 342 times, `&gt;` 88, `&lt;` 27. Some labels look wrong: a Hubert Humphrey quote about a dust storm and a snooker match ("whitewashes the Whirlwind") are labelled disaster.
- **`keyword`:** 221 values, the same in train and test, 9 to 45 tweets each. A few decide alone (`derailment`, `debris`, `wreckage` 100%; `aftershock` 0%), but 6,379 of 7,552 tweets have a keyword between 10% and 90%: `fire` is 29% disaster, `ablaze` 36% (metaphors). The 61 tweets without a keyword are 69% disaster.
- **`location`:** no signal as it is: 43.2% disaster with one, 42.4% without; 2,818 of 3,341 values used once, the same place written many ways ("USA", "United States"). Left out of the baseline.
- **Duplicates:** retweets differ only by their `t.co` link, so matching identical text (69 texts, 179 rows, 18 conflicting) misses most. Ignoring URLs, case and spaces: **300 texts repeated, 964 rows (12.7%), 66 groups with conflicting labels (237 rows)**; one tweet appears 17 times, labelled 1 in 5 of them. **400 test tweets (12%)** also appear in train.

## Validation decisions (step 2, decided 2026-11-24)

- **Group duplicates into the same fold, by the normalised text** (URLs removed, lower case, spaces collapsed), and judge the keep rule on this grouped CV. The folds never change the submitted model (step 7 fits on all of train, so it memorises the 400 shared tweets either way); they change how a model is measured, and so which model is chosen. Ungrouped CV rewards memorising: a change that only memorises better (keeping every rare word, say) would look like a gain. Grouped CV measures tweets never seen, as with real new tweets.
- **Also print the ungrouped score**, as Kaggle has a similar overlap (12.7% duplicated rows in train, 12% of test seen in train): it shows how much memorisation is worth, and is the one to compare with the Kaggle score in step 7.
- **Keep the 66 conflicting groups as they are.** The noise is real: Kaggle's test labels were made the same way, so cleaning train wouldn't make test cleaner. They cap the score (a model can't get both labels of the same tweet right), the same for every model compared.
- **No local holdout.** Kaggle keeps the test labels hidden, so its score is the one honest check at the end, and all 7,613 tweets go to CV. Titanic held out 20% as well, and that holdout agreed with CV (0.838 against 0.841) while Kaggle gave 0.787: a holdout cut from `train.csv` shares its quirks. A local holdout here would also have to be split by group.
- **Choose on CV, submit at milestones.** Keeping a change because its Kaggle score rose fits the choice to the test set (overfitting the leaderboard; 3,263 test tweets, so gains of a few thousandths are chance). Submit the baseline and the final model, with the expected score written first.

## Validation (step 2, 2026-11-24)

`disaster_tweets.py`, written by the user.

- **Folds:** `create_folds` builds `StratifiedGroupKFold(5, shuffle=True, random_state=42)` folds with the normalised text as group (`data_analysis.normalise`) and saves `folds.csv` (`id`, `fold`; in git, so the transformer is scored on the same folds). It runs only if the file doesn't exist, so a later sklearn version can't silently change the folds. Asserts: no group on both sides of any fold, every tweet has a fold.
- **Reading them back:** merged onto train by `id` (so the order of `folds.csv` doesn't matter), then `PredefinedSplit(test_fold=...)` as `cv=`.
- **Fold sizes and rates:** 1,524, 1,523, 1,522, 1,522, 1,522 tweets; disaster rate 42.94–42.98% (plain `StratifiedKFold`, same seed: 42.94–43.01%). Grouping didn't cost balance: about 6,900 groups, most of them single tweets, fill the gaps around the big blocks (the 17-row one included).
- **The dummy** ("always disaster") through both CVs, train and validation: F1 0.6011, precision 0.4297, recall 1.0, accuracy 0.4297, sd at most 0.0002. As worked out by hand (2 × 0.4297 / 1.4297), so the folds, scoring and printing are right.

## The baseline (step 3, 2026-11-24)

`build_pipeline()`: `TfidfVectorizer()` + `LogisticRegression(max_iter=1000)`, default settings, on `text` only. The vectorizer is in the Pipeline, so each fold learns its vocabulary and IDF from its training part only.

- **Guessed before running:** validation F1 0.70, train F1 0.73 (gap 0.03).

| | Grouped folds | Plain folds |
|---|---|---|
| Validation F1 | **0.7476 ± 0.0118** | 0.7510 ± 0.0085 |
| Train F1 | 0.8587 | 0.8590 |
| Validation precision / recall | 0.803 / 0.700 | 0.808 / 0.701 |
| Validation accuracy | 0.797 | 0.800 |

- **Per fold (grouped, validation F1):** 0.7247, 0.7516, 0.7490, 0.7555, 0.7571. Fold 0 is the hard one, 0.025 below the rest.
- **0.15 above "always disaster"** (0.60). The guess was 0.05 low.
- **The gap is 0.11, not 0.03:** with about 20,000 columns for 7,613 tweets, the model could separate its own training tweets by their rare words; the penalty (`C=1`) keeps those weights small, which is why train F1 stops at 0.86, not 1.0. Tuning C is step 5c.
- **Precision 0.80, recall 0.70:** the model misses disasters more often than it raises false alarms. The cut-off (step 4 and 5g) trades one for the other.
- **Grouped vs plain:** only 0.003 apart, much less than the 12.7% duplicate rate suggests. At `C=1` rare words get small weights, so a copy in the training part helps little. Watch it in step 5: changes that let the model memorise more (a larger C, more n-grams) could widen it.

**Keep rule, written before any change was tried:** keep a change only if it beats the current model's validation F1 on **all 5 grouped folds** (a tie counts as a loss) **and** the mean gain is **at least 0.008**. Why:
- Judged on the fold-by-fold differences. Fold 0's difficulty hits both models alike, so it cancels in the difference; it stays in (its tweets are real, and probably the hard ones that changes should help).
- With 5 folds, a change with no effect wins all 5 by luck 1 time in 32 (at least 4: 6 in 32, too loose).
- 0.008, not a bar near the fold sd (0.012): that sd is mostly fold difficulty, and a fold-by-fold difference varies much less; a higher bar would reject small, consistent gains, as in Titanic, where the group-fate feature won 46 of 50 folds and was kept only by overriding the rule after the result.
- One condition, so a large gain that loses one fold is rejected: a change has to help everywhere.

## Looking inside the baseline (step 4, 2026-11-25)

From [look_inside.py](look_inside.py) (Claude's; reading only, nothing changes the model); out-of-fold predictions on the grouped folds; plot `plots/cutoff.png`.

- **The words** (fitted on all of train, 21,637 columns): the strongest disaster word is `in` (3.96, in 1,789 tweets: news style, "fire in California"), then `hiroshima` (the 70th anniversary), `fires`, `california`, `http`, `fire`, `train`, `storm`, `suicide`, `bombing`, `killed`, `earthquake`, `derailment`. Towards "not": personal words (`you`, `my`, `your`, `me`, `he`, `she`, `love`, `song`, `nowplaying`, `youtube`, `ebay`) and keywords mostly used as metaphors (`body`, `bags`, `panic`, `traumatised`, `blew`, `upheaval`).
- **URLs are cut into pieces:** `http` leans disaster (2.75, in 3,603 tweets), `https` the other way (−1.41); every link also adds `t`, `co` and a random code. The http/https split probably reflects the app that posted, not the meaning (step 5e).
- **The mistakes:** 562 false positives, 983 false negatives: the model misses disasters about twice as often as it raises false alarms. The most confident false negatives are mostly odd labels ("My back is so sunburned :(", keyword `hurricane`, labelled disaster); the most confident false positives are news-style or metaphors ("I'm On Fire.", a Calgary weather-and-traffic bot, "Governor weighs parole for California school bus hijacker"). 8.7% of false positives are in a conflicting group, against 3.1% of all tweets.
- **The cut-off:** F1 is flat near its top: best 0.46 (F1 0.7514) against 0.7476 at 0.5, +0.004, chosen on the validation tweets themselves (optimistic); each fold's best differs (0.46 to 0.55). Precision and recall cross at about 0.45.
- **Fold 0** is not hard for a reason measured here: the same disaster rate (43.0%), repeats (12.8%), fewer conflicting tweets than folds 3 and 4; accuracy 0.780 against 0.797–0.808. It drew more hard tweets.
- **The user's prediction:** word pairs (5a) should beat the current model; a size was hard to guess. Changes then taken one at a time, each with a pass/fail prediction first.

## Improvements (step 5, from 2026-11-25)

Each change against the current model on the grouped folds, through `keep_rule` (all 5 folds won, mean gain ≥ 0.008). The current model is still the baseline (0.7476).

| Change | Validation F1 | Train F1 | Mean gain | Folds won | Predicted | Result |
|---|---|---|---|---|---|---|
| 5a word pairs, `ngram_range=(1, 2)` | 0.7280 | 0.8777 | −0.020 | 1 of 5 | pass | **dropped** |
| 5b-i `min_df=2` | 0.7496 | 0.8400 | +0.002 | 3 of 5 | pass | **dropped** |
| 5b-ii `sublinear_tf=True` | 0.7461 | 0.8591 | −0.002 | 2 of 5 | fail | **dropped** |
| 5b-i + ii together | 0.7493 | 0.8415 | +0.002 | 4 of 5 | (none) | **dropped** |
| 5c `C` tuned inside the folds | 0.7516 | 0.9479 | +0.004 | 3 of 5 | pass, C below 1 | **dropped**; C 3–10 |
| 5d + `keyword` (one-hot, `ColumnTransformer`) | 0.7545 | 0.8549 | +0.007 | 4 of 5 | pass | **dropped** |
| 5e cleaning (URL, @mention → one token; `&amp;`) | 0.7416 | 0.8519 | −0.006 | 1 of 5 | fail | **dropped** |
| 5f character n-grams (`char_wb`, 2–5) | 0.7509 | 0.8635 | +0.003 | 3 of 5 | F1 up near +0.008, fails on folds | **dropped** |
| 5g cut-off tuned inside the folds | 0.7519 | 0.8753 | +0.004 | 4 of 5 | fail | **dropped**; cut-off 0.45–0.47 |

- **5a: word pairs overfit.** Columns 21,637 → 90,133, of which 72,299 (80%) appear in one tweet only. Training gives such pairs weights because they make the training tweets right, so train F1 rose (0.878) and validation fell, mostly through false alarms (precision 0.803 → 0.761, recall unchanged). And each row is scaled to length 1, so with twice the entries the useful words get smaller values (mean single-word entry 0.239 → 0.148): their weights must grow, and `C=1` charges more. Dropped *as tested* (pairs alone at C=1): it could be retried after `min_df` or a tuned C, since a decision depends on the other settings.
- **5b-i: `min_df=2`** removes about 15,000 single-tweet words, a third of the columns left. Train F1 fell (0.859 → 0.840, the gap 0.11 → 0.09: less to memorise), validation barely moved, folds both ways. The rare words weren't helping or hurting new tweets: `C=1` already kept their weights small. Fewer columns for the same score is tempting, but the rule keeps a change only if it passes, and the rule isn't changed after the result.
- **5b-ii: `sublinear_tf`** (count → 1 + log(count)) changes only entries with a count above 1: 5.3% of them (43% of tweets repeat a word, often URL pieces or "the"). Nothing to gain.
- **Together:** the same as `min_df` alone (0.7493 against 0.7496), so `sublinear_tf` adds nothing on top either.
- **5c: tuning `C`.** `LogisticRegressionCV(Cs=[0.01 … 100], scoring="f1")` in the pipeline, with its **inner** CV grouped too (`StratifiedGroupKFold`, groups passed down by metadata routing: `sklearn.set_config(enable_metadata_routing=True)` and `cross_validate(..., params={"groups": groups})`). An ungrouped inner CV would reward memorising the copies and pick too high a C. Chosen per fold: 3, 10, 3, 10, 3, inside the list. Train F1 0.859 → 0.948, validation +0.004: between C = 1 and 10 validation F1 is a plateau. What a larger C adds is mostly weights on words found in one training tweet, which almost never appear in a validation tweet, so they push train up and leave validation alone. No link between the C chosen and the fold's result (C=10 gave both the worst fold and a winning one). The prediction "C below 1" reasoned from shrinking the gap; the inner CV maximises validation F1, not the gap.
- **The same folds split the same way:** `min_df=2` and tuned C both lose on folds 0 and 1 and win on 2–4.
- **5d: `keyword`** (`ColumnTransformer`: TF-IDF on `text`, given as a string so the vectorizer gets one column; `OneHotEncoder(handle_unknown="ignore")` on `["keyword"]`, a list, as it needs a table; the blank keyword is its own category). The closest yet, and it missed on both conditions (+0.0069, fold 1 −0.009): dropped by the rule, not overridden. It adds one lean per keyword (shared by 9–45 tweets, so train F1 fell slightly, 0.855). Accuracy didn't move at all (0.7971 both): it swapped 85 true negatives for 85 true positives (2,850 → 3,020 tweets predicted disaster), and F1, which ignores true negatives, rose. That's what a lower cut-off does, so part of its gain overlaps 5g. In fold 1 it broke mostly tweets labelled 0 with disaster-sounding keywords ("…thank god for airplane mode", keyword `airplane accident`; a band called Buildings on Fire).
- **Fold 1 lost with `min_df`, tuned C, keyword and char n-grams.** Looked at to understand it, not to tune for it (designing a change around one validation fold would fit that fold). Each change flips about 150 tweets per fold in each direction; the net is a small difference of two large counts, so it varies by about ±√150 ≈ ±12 tweets per fold by chance. Fold 1's −21 (keyword) is at the unlucky end of that, not a special kind of tweet.
- **5e: cleaning** with `TfidfVectorizer(preprocessor=clean)` (`clean` written by Claude at the user's request; a custom preprocessor replaces the built-in lowercasing, so `clean` lowercases). Columns 21,637 → 14,826. Recall fell (0.700 → 0.675), precision rose: merging all links into one token lost a signal. `http://` tweets are 57.2% disaster, `https://` 31.6%, no link 30.2%: only `http` links carry the news signal, so step 4's "artefact" was information. The grouped-vs-plain gap grew (0.003 → 0.008): retweets that differed only by their `t.co` code became identical, so plain folds could match them exactly.
- **5f: character n-grams** (`analyzer="char_wb"`, 2–5): 240,743 columns (11 × the words) for +0.003; recall up (0.715), precision down.
- **5g: the cut-off** with `TunedThresholdClassifierCV(pipeline, scoring="f1", cv=StratifiedGroupKFold(...))`, groups routed to its inner CV as in 5c. Each outer fold chose 0.447–0.473, consistent with step 4's 0.46; precision and recall balanced (0.757 / 0.748, from 0.803 / 0.700), +0.004 on 4 of 5 folds (fold 1 −0.012).

**Step 5 result: 7 changes, none passed; the final TF-IDF model is the baseline** (TF-IDF defaults + `LogisticRegression(C=1)`, 0.7476 on the grouped folds). Several changes gained +0.003 to +0.007 with folds both ways: bag of words looks near what it can do here, about 0.75. The rule held when 5d came close.

**How `TunedThresholdClassifierCV` chose (5g, explained afterwards):** inside each outer training part, 5 grouped inner splits each give an F1-against-cut-off curve (100 cut-offs between the lowest and highest probability); the curves are averaged and the peak is `best_threshold_`; then the pipeline is refitted on the whole training part. For outer fold 0 the inner splits alone would pick 0.39 to 0.505; their average peaks at 0.47, on a flat top (0.40–0.52 within about 0.01), hence the small gain. Plot: `plots/tuned_threshold_fold0.png`.

**Lesson for the next problem, not applied here:** with 5 folds and changes that flip about 150 tweets per fold, "all 5 folds won and +0.008" can only see fairly large gains; repeated CV (10 × 5 grouped folds, as in Titanic) would measure +0.004-sized gains more precisely, at 10 times the run time. Decide the number of folds with the keep rule, before the first comparison.

## Learning curve (step 6, 2026-11-25)

From [learning_curve.py](learning_curve.py) (Claude's): the final model trained on 10% … 100% of each grouped fold's training part (a random subset), scored on that fold's validation tweets. Plot: `plots/learning_curve.png`.

| Training tweets per fold | Train F1 | Validation F1 | Gap |
|---|---|---|---|
| 608 (10%) | 0.880 | 0.612 | 0.27 |
| 1,217 | 0.876 | 0.663 | 0.21 |
| 3,044 (50%) | 0.866 | 0.720 | 0.15 |
| 4,871 | 0.862 | 0.740 | 0.12 |
| 6,089 (100%) | 0.859 | 0.747 | 0.11 |

- **Predicted: flat at 100%. Wrong: still rising**, more slowly with each step: the last doubling (3,044 → 6,089) added +0.027, on all 5 folds (+0.002 to +0.041); over the upper half, about +0.03 per doubling.
- The gap shrinks mostly because validation rises; train falls slightly (fewer chance patterns to memorise per tweet).
- **With step 5:** with these tweets, no change of settings got more out of bag of words; more tweets would still help (perhaps 0.77 at 12,000, 0.80 at 24,000, if the trend holds). More labelled tweets aren't available, but a pretrained transformer brings knowledge from billions of words: the "more data" this curve asks for.

## Submission (step 7, 2026-11-25)

- **The model:** the final TF-IDF model (TF-IDF defaults + `LogisticRegression(C=1)`, cut-off 0.5) fitted on all 7,613 training tweets. `create_submission()` in `disaster_tweets.py`, written by Claude; it checks the columns, the 3,263 rows and the ids against `sample_submission.csv`. 36.8% of test tweets predicted disaster (train: 43.0% are).
- **Expected, written before uploading:** 0.74, "near the validation score".
- **Kaggle: 0.79374**, +0.046 above the grouped CV (0.7476) and +0.042 above the plain CV (0.7510): outside the check in plan.md (within the fold-to-fold spread, ±0.012). In line with the honest TF-IDF scores quoted in plan.md (0.78–0.80).
- **Why higher than CV** (the user named A and C from options):
  - **More training tweets:** 6,089 per fold → 7,613 is 0.32 of a doubling; at about +0.03 per doubling (step 6), about +0.010.
  - **The 400 test tweets also in train:** about +0.003 to +0.01 (plain CV, with the same kind of overlap, was +0.003 above grouped). The model calls 53% of them disaster against 34.5% of unseen test tweets: many are retweeted news.
  - **Chance:** about ±0.008 (F1 on 3,263 tweets; per-fold sd 0.012 on 1,522).
  - **Left unexplained: about +0.025.** The model treats test like train (mean |p − 0.5| 0.230 against 0.219 out-of-fold; 20% near 0.5 in both; 36.8% against 37.4% predicted disaster), so the difference is more likely in the labels (perhaps less noise than train's 66 conflicting groups). Checking would need the leaked test labels, which would make Kaggle another validation set: not done.
- **For the method:** here CV was pessimistic (Titanic: optimistic). A shift like that affects every model alike, so the keep rule, which compares models with each other, still holds: CV ranks changes; it doesn't predict the leaderboard number to within a few points.

# Transformer

The plan is in [plan_transformer.md](plan_transformer.md): fine-tune a pretrained transformer on the same grouped folds (`folds.csv`), compared fold by fold with the TF-IDF model (0.7476).

## One fold, end to end (step 1, 2026-11-26)

`disaster_tweets_transformer.py`, written by the user: `bert-base-uncased`, trained on folds 1–4 (6,089 tweets), scored on fold 0 (1,524). Lr 5e-5 (linear decay to 0 over the 1,143 steps: the Trainer's default schedule), batch 16, 3 epochs, bf16, `max_length=96`, seed 42 set before `from_pretrained`. No evaluation or checkpoints during training (`eval_strategy="no"`, `save_strategy="no"`), so nothing about the validation tweets is seen before the epoch decision in step 2. Probabilities saved to `oof_bert_fold0.csv` (`id`, `fold`, `p`).

- **Predicted:** F1 0.83, about 1 minute per epoch.
- **Fold 0: F1 0.7816** (precision 0.799, recall 0.765, accuracy 0.816), against TF-IDF's 0.7247 on the same fold: **+0.057**, more than any TF-IDF change gained. The prediction was too high: 0.83 is the Kaggle-level figure, and here Kaggle came out about 0.046 above grouped CV.
- **Time:** about 17 s per epoch (52 s for 3), 2.6 s to predict fold 0: about 3.5 times faster than predicted (bf16, short tweets).
- **Checks:** F1 recomputed from the saved probabilities (`p ≥ 0.5`, `f1_score`) equals `compute_metrics`; no fold 0 id in the training part; a second run gave identical losses (seeded).
- **Datasets from pandas:** `preserve_index=False` (after filtering by fold the index isn't 0, 1, 2…, and would become a `__index_level_0__` column); the label column renamed `target` → `label` (the Trainer looks for `label`); only `text`, `label`, `id` converted, `text` and `id` removed after tokenizing.

## Decisions before the full CV (step 2, decided 2026-11-26)

Written before any 5-fold run. Words used below: for fold k, the **training part** is the other four folds (≈ 6,090 tweets) and the **validation part** is fold k (≈ 1,523). The **check part** is a grouped split of the training part (≈ 1,220), leaving the **inner-train part** (≈ 4,870).

**1. The number of epochs: chosen inside each fold's training part (option B).** Per fold k:

| | Run | Trains on | Epochs | Scored on |
|---|---|---|---|---|
| a | one run per seed, check F1 after each epoch | inner-train, ≈ 4,870 | 5 | check, ≈ 1,220, after every epoch |
| b | average the seeds' check curves; the best average epoch is *n* | | | |
| c | a fresh model per seed (`set_seed` + `from_pretrained`) | the whole training part, ≈ 6,090 | *n* | validation (fold k), once |

- **Not fixed at 3 (option A):** 3 comes from the course's recipe, on another data set. A fixed count can't leak, but if it's wrong it only makes BERT score lower than it could.
- **Never by watching fold k's F1 after each epoch:** the count would be fitted to the tweets that score it, the step 4 trap. There, each fold's best cut-off on its own validation tweets gave 0.7560, against 0.7519 when chosen inside the folds (5g): the extra +0.004 was noise counted as gain.
- **Each fold chooses its own *n*:** choosing once inside fold 0's training part would use tweets from folds 1–4, which then score folds 1–4.
- **Why (a) is one run and not 5:** a 5-epoch run passes through epochs 1–4, so scoring after each epoch gives all five counts at once. **Approximately:** the learning rate decays linearly to 0 over the whole run, so after epoch 3 of a 5-epoch run 40% of it is left, while (c)'s 3-epoch run ends at 0. Accepted (the user's choice, 2026-11-26), as is usual with early stopping: it can't leak, at most *n* is slightly off. Not chosen: a constant learning rate in (a) and (c) (a different recipe from step 1's 0.7816), or one run per count (15 epochs per seed instead of 5).
- **Why retrain in (c) instead of keeping (a)'s model:** (a)'s model never learned from the 1,220 check tweets. (c) learns from about 25% more tweets; at about +0.03 F1 per doubling (step 6), that's roughly +0.01. *n* stays honest either way: it was chosen on the check part, and fold k is never used. The same pattern as 5g's refit.
- **Why the curves are averaged:** seeds differ by a few thousandths on 1,220 check tweets, which is noise. A seed's own peak doesn't carry over to its run in (c) anyway: different tweets mean different batches, so a different path. Averaging is steadier, as `TunedThresholdClassifierCV` averages its inner curves.
- **Cost, one seed:** (a) ≈ 68 s per fold (14 s per epoch on 80% of the tweets), (c) up to 85 s, so about 13 minutes for 5 folds, worst case.
- **Changed after the fold 0 run (2026-11-27): fixed at 3 epochs** (the user's choice). Fold 0's check curves (inner-train 4,871, check 1,218; `check_f1_fold0.csv`) were flat:

  | Check F1 | 1 | 2 | 3 | 4 | 5 |
  |---|---|---|---|---|---|
  | seed 42 | 0.7729 | 0.7876 | 0.7871 | 0.7838 | 0.7866 |
  | seed 43 | 0.7953 | 0.7852 | 0.7869 | 0.7786 | 0.7762 |
  | seed 44 | 0.7847 | 0.7796 | 0.7820 | 0.7863 | 0.7845 |
  | **mean** | 0.7843 | 0.7841 | **0.7854** | 0.7829 | 0.7824 |

  The means span 0.003 over 1–5 epochs, less than the seeds differ at any one epoch, so the count barely matters, and option B's per-fold choice (about 4 extra minutes per fold) buys almost nothing. *n* = 3 here anyway. **Known, accepted leak:** the check tweets belong to folds 1–4, so "3" for scoring folds 1–4 was informed by some of their own tweets. It's one number from a flat curve, so it carries almost no information about which tweets are easy; chosen to save time, not because 3 scored best.

**2. Seeds.** The seed sets the new head's starting weights (the ≈ 110 million BERT weights start pretrained, the same for every seed), the batch order and dropout. Different paths settle at different scores, not only at different epochs: on a few thousand examples, ±0.01–0.02, and now and then a seed that barely learns.
- **First, measure the spread:** fold 0, 3 seeds (≈ 8 minutes).
- **The baseline (step 3): 1 seed.** BERT against TF-IDF was +0.057 on fold 0, far above any seed spread.
- **Step 5 changes: 3 seeds if fold 0's seed sd is above 0.003** (≈ 38 minutes per change, worst case), otherwise 1. Why 0.003: a fold's difference carries the luck of two runs, √2 × the seed sd, so at 0.003 it's about ±0.004, half the bar. The user first chose 0.006 ("near the bar"); at that sd the difference's luck (≈ 0.0085) equals the bar, and a real +0.008 change would win all 5 folds only about 38% of the time (89% at 0.003). An example with a true gain of +0.010 and the same luck at both sizes: at sd 0.008, 1 seed lost folds 0 and 4 (luck −0.014 and −0.012) and was rejected, 3 seeds kept it; at sd 0.003, 1 seed kept it. An sd from 3 seeds is rough (it could be half or double), so the switch is set low.
- **With several seeds, a fold's score is the mean F1 over its seeds.**

**3. The keep rule: unchanged** (chosen by Claude at the user's request): all 5 folds won (a tie counts as a loss) and a mean gain of at least 0.008, fold by fold on the same grouped folds. It's the same rule as for TF-IDF, so "kept" means the same thing in both plans, and BERT against TF-IDF is judged on equal terms. Winning 5 of 5 by luck happens about 1 time in 32. Seed noise is handled by decision 2 (3 seeds shrink it by about √3), not by moving the bar.

**4. What the model sees: the text only**, as for TF-IDF. Keyword is a later change (step 5b).

### The seed spread (2026-11-27)

Fold 0, 3 epochs on folds 1–4 (6,089 tweets), one fresh model per seed, fold 0 (1,524) predicted once; `oof_bert_seeds_fold0.csv` (`id`, `fold`, `seed`, `p`). About 55 s per seed. The script now logs to the console and to `logs/run_<date>_<time>.log` (gitignored), one file per run.

| Seed | Fold 0 F1 | Precision | Recall | Accuracy |
|---|---|---|---|---|
| 42 | 0.7816 | 0.799 | 0.765 | 0.816 |
| 43 | 0.7847 | 0.805 | 0.765 | 0.820 |
| 44 | 0.7848 | 0.782 | 0.788 | 0.814 |
| **mean / sd** | **0.7837 / 0.0018** | | | |

- **sd 0.0018, below the 0.003 switch: 1 seed for step 5.** (An sd from 3 seeds is rough; the margin isn't large.)
- **Reproduction check passed:** seed 42 gave 0.7816, exactly step 1; seeds 42 and 43 also matched the earlier crashed run to every digit.
- **Every seed beats TF-IDF on fold 0** (0.7247) by 0.057–0.060.
- **A tie, found by an assert:** seed 44 had one tweet with equal logits (bf16 keeps about 3 significant digits). `argmax` called it 0, `p ≥ 0.5` called it 1, so `compute_metrics` and the F1 from the saved probabilities differed by exactly one true positive (0.7839 against 0.7848). Now one rule everywhere: `p ≥ 0.5` from a float64 softmax, in `compute_metrics` too.

## The transformer baseline, 5 folds (step 3, 2026-11-27)

The fold loop, written by the user: for each fold k, a fresh `bert-base-uncased` (seed 42, 3 epochs, the step 1 recipe) trained on the other four folds and predicting fold k once. Out-of-fold probabilities for all 7,613 tweets are in `oof_bert_all_fold.csv` (`id`, `fold`, `p`). About 46 s per fold, 3.8 minutes in all.

- **Predicted:** a mean of 0.79, and it passes the keep rule ("fold 0, TF-IDF's hardest, was already won by 0.057").

| Fold | TF-IDF | BERT | Diff | BERT precision | BERT recall |
|---|---|---|---|---|---|
| 0 | 0.7247 | 0.7816 | +0.0569 | 0.799 | 0.765 |
| 1 | 0.7516 | 0.7964 | +0.0448 | 0.786 | 0.807 |
| 2 | 0.7490 | 0.8061 | +0.0571 | 0.802 | 0.810 |
| 3 | 0.7555 | 0.7997 | +0.0442 | 0.836 | 0.766 |
| 4 | 0.7571 | 0.7920 | +0.0349 | 0.801 | 0.783 |
| **mean ± sd** | **0.7476 ± 0.012** | **0.7951 ± 0.008** | **+0.0476** | | |

- **Kept:** all 5 folds won, and the mean gain of 0.048 is six times the 0.008 bar. The smallest gain (fold 4, +0.035) is still larger than any TF-IDF change gained. BERT is now the model to beat in step 5.
- **The prediction was right on both parts:** 0.7951 against 0.79, and it passes. Fold 0 is BERT's hardest fold too (0.7816, its lowest), so the reasoning held. Why fold 0 is hard for both models is a question for step 4.
- **The folds vary more than the seeds:** the sd over folds is 0.008, the sd over seeds on fold 0 was 0.0018. Most of the spread is which tweets land in a fold, not training luck.
- **sd with ddof=0** (`np.std`), as for TF-IDF's ± 0.012; `pandas.std()` (ddof=1) would give 0.0092.
- **Checks:** 7,613 rows with unique ids, the same set as `train.csv`; each fold's F1 recomputed from the saved `p`, matched to the labels by `id` (not by row order), equals the F1 logged in the loop; no id in both the training part and fold k. Fold 0 reproduced step 1's 0.7816, and a second run gave every fold's F1 to every digit.
- **A rough Kaggle expectation:** TF-IDF scored about 0.046 above its grouped CV on Kaggle, so about 0.84 here, at the top of the 0.83–0.84 quoted for BERT-type models. It's written down properly in step 6.

## Looking inside the transformer (step 4, 2026-11-27)

From [look_inside_transformer.py](look_inside_transformer.py) (Claude's; reading only). BERT's out-of-fold probabilities are from `oof_bert_all_fold.csv`; TF-IDF's are recomputed on the same folds. Both reproduce their logged per-fold F1. Plot: `plots/cutoff_bert.png`.

**1. Where the two models disagree.**

| Label | Both right | BERT only | TF-IDF only | Both wrong |
|---|---|---|---|---|
| 0, not disaster (4,342) | 3,437 | 279 | 343 | 283 |
| 1, disaster (3,271) | 2,164 | 408 | 124 | 575 |

- **BERT's gain is recall.** It misses 699 disasters against TF-IDF's 983 (−284), and raises 626 false alarms against 562 (+64). Recall rose from 0.699 to 0.786 while precision stayed at 0.80.
- **The two models make different mistakes.** They agree on 84.8% of the predictions, and their probabilities correlate at 0.76. BERT fixes 687 tweets and breaks 467. Of TF-IDF's 1,545 mistakes, BERT shares 858 (56%). This is what a blend needs (5e).
- **The misses BERT fixes** are mostly disaster words in chatty tweets: @mentions and "you", the words that pull TF-IDF towards "not" ("Did anyone else see that fireball falling to earth?", "that storm's been given you guys trouble though"). Some of the fixed tweets have doubtful labels ("…little derailment is possible so I applaud you :)", labelled 1).
- **The false alarms BERT fixes** are word artefacts: a wind-report bot, "Heroes of the Storm", "HAIL STATE", links (`http` leaned disaster in TF-IDF).
- **The metaphors and the news-style false alarms are not fixed.** Of TF-IDF's 20 most confident false alarms, BERT also calls 17 a disaster, "I'm On Fire." among them (BERT 0.991). Most are news about real events labelled 0 (evacuation of an abandoned aircraft, a church arson). These are disputed labels, not something BERT failed to read.
- **BERT also makes new metaphor mistakes** in chatty tweets ("I'll be dam if I get any of that drought bud" 0.992, "thunderstorms… make my soul explode" 0.975). Reading in context isn't enough when the disaster word is strong.

**2. The mistakes left:** 1,325 (626 false positives, 699 false negatives), against TF-IDF's 1,545.
- **The conflicting groups force at least 83 mistakes** on any model: 237 tweets, and in each group the copies with the minority label are lost. 96 of BERT's mistakes are in these groups.
- **The most confident mistakes are mostly label noise.** The false positives at p 0.999 are news about disasters labelled 0 ("No reported flooding po in the area", "Cinema evacuated following false fire alarm"). The false negatives at p 0.001–0.003 are tweets labelled 1 that aren't disasters ("Check out my Lava lamp dude", "THUNDER BUDDYS!!!!", "Fall off a cliff please").

**BERT is very confident, more than it should be.** This was a new reading, added to the plan's four.

| Probability band | BERT share | BERT disaster rate | TF-IDF share | TF-IDF disaster rate |
|---|---|---|---|---|
| < 0.05 | 42.4% | 12.6% | 0.4% | 6.9% |
| 0.4–0.6 | 2.0% | 36.5% | 20.9% | 44.5% |
| ≥ 0.95 | 34.6% | 87.9% | 0.7% | 98.1% |

- 77% of BERT's probabilities are below 0.05 or above 0.95. Of those at 0.95 or more, only 87.9% are disasters, and 12.6% of those under 0.05 are. BERT is **overconfident**: its fine-tuning loss pushes the probabilities to the ends, and a few of these labels are wrong anyway.
- TF-IDF is the other way round: **underconfident**. Its 0.8–0.95 band is 97.7% disasters, because the C=1 penalty keeps its weights small.
- **For a blend (5e):** a plain average of probabilities lets BERT's extremes decide. BERT 0.99 and TF-IDF 0.14 average to 0.565, so BERT's call wins. TF-IDF changes the result mostly where BERT is unsure, about 5% of the tweets. Averaging the logits or the ranks would weigh the two more evenly; that's a choice for step 5e.

**3. The cut-off is flat.** F1 stays within 0.005 of its 0.5 value from cut-off 0.30 to 0.92, because few tweets lie between them. The best over all the tweets is 0.74 (F1 0.7976 against 0.7952, +0.002), chosen on the tweets that score it. Each fold's own best is all over the place (0.44 to 0.83). **So 5d (the cut-off chosen inside the folds) can't come near the 0.008 bar.** Precision and recall cross at about 0.40.

**4. Fold 0 is hard for both models, for the same reason.** (The plan says "Fold 1", a slip: fold 0 is the hard one.)

| Fold | TF-IDF accuracy | BERT accuracy | Both wrong | TF-IDF only wrong | BERT only wrong | Conflicting |
|---|---|---|---|---|---|---|
| 0 | 78.0% | 81.6% | **12.1%** | 9.8% | 6.2% | 2.8% |
| 1 | 79.8% | 82.3% | 11.2% | 9.1% | 6.6% | 2.4% |
| 2 | 79.7% | 83.2% | 10.4% | 9.9% | 6.4% | 2.4% |
| 3 | 80.2% | 83.5% | 11.6% | 8.1% | 4.9% | 4.2% |
| 4 | 80.8% | 82.3% | 11.0% | 8.1% | 6.6% | 3.9% |

- Fold 0 has the most tweets that **both** models get wrong (12.1%), but few conflicting ones (2.8%). It drew more tweets that are hard for any model, or labelled against their wording, rather than more copies with clashing labels.
- BERT narrows the gap: fold 0 is 0.017 below the mean of the other folds for BERT, against 0.029 for TF-IDF.
- **The user's prediction for step 5:** 5a helps most (a model pretrained on tweets has seen more of them). The blend (5e) won't help, because the metaphors and news-style false alarms stay false alarms in both models. 5e was nearly chosen first, on the grounds that BERT is overconfident and TF-IDF underconfident; that says how to blend, not whether it helps.
