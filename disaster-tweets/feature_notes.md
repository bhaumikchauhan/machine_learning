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
