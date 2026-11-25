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
