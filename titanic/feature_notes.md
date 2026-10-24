# Titanic feature notes

Decisions and reasoning behind the feature engineering in [common.py](common.py), [titanic.py](titanic.py) (v1) and [titanic_v2.py](titanic_v2.py) (v2). Model: logistic regression (a linear model, `f(x) = w·x + b`), so every input must be numeric and each feature gets one weight.

Status: finished. Every column has a decision. Logistic regression's settings were tested and left at the defaults. A decision tree and a random forest were tested, with and without raw Age, and neither beat logistic regression, so it stays the model. The final model is v2: the 11 features plus a group-fate feature (what happened to the women and children on the same ticket), kept by a decision that overrides the gain part of the keep rule (see "Ticket fate" and "Outcome and when to stop").

Final scores (v2, `LogisticRegression` defaults): cross-validation 10 × 5 folds on the 80% working set: accuracy 0.841 (±0.032), precision 0.824, recall 0.748. Test set (scored once): 0.838 / 0.823 / 0.739. Kaggle: 0.78708.

Features: `Sex_encoded`, `is_male_adult`, `Embarked_C/Q/S`, `Pclass_1/2/3`, `is_single`, `is_of_small_family`, `is_of_large_family` (the 11 of v1, 0.827 in cross-validation), plus `wc_mates_unknown`, `wc_mates_all_died`, `wc_mate_survived` (built inside each fold).

## How the code is organised

Files:

- `common.py`: everything shared by both versions. Constants, data loading, the feature functions, `feature_engineering`, the split, `fitting_columns`, `build_model`, `repeated_folds()` (the 50 folds, fixed seed), `cross_validate_model`, `apply_keep_rule(base, with_new)` and `compare_with_columns`. Every feature here uses inputs only, never `Survived`, so it is built once before the split.
- `titanic.py` (v1): the 11-feature model. Only what is specific to it: `compare_pclass_interaction`, `evaluate_on_test`, `create_submission`, and the entry point (`uv run python titanic.py`).
- `titanic_v2.py` (v2, final): the 11 features plus a fate feature (see "Ticket fate"). Fate features are built from other passengers' `Survived`, so they can't live in `feature_engineering`.
  - `ticket_mates(X, X_known, y_known)` counts the known ticket-mates and how many survived, never counting a row's own outcome. `add_ticket_fate` (first version, all ticket-mates) and `add_group_fate` (refined, kept) build the three columns from it. `TICKET_FATE` and `GROUP_FATE` pair each function with its columns.
  - `cross_validate_with_fate(X, y, fate=None)` rebuilds the chosen fate feature inside each of the 50 folds; `compare_fate(X, y, fate)` runs with and without it and applies the keep rule. The entry point (`uv run python titanic_v2.py`) runs `compare_fate` with `GROUP_FATE`.
  - `fit_with_fate(X_known, y_known, X_new)` fits on the known rows and predicts the new ones: known rows get their fate from each other, new rows from the known rows only. It prints how many women and boys in the new rows have a known woman/child ticket-mate. `evaluate_on_test` and `create_submission` (written to `submissions_v2.csv`) both use it.
- The split was checked: v1, and v2 with the feature switched off, give exactly the scores from before the split (0.827 / 0.804 / 0.729) on identical folds.

Functions in `common.py`:

- `retrieve_X_And_y()` loads `train.csv` and returns `X` (DataFrame, no `Survived`) and `y` (one-column DataFrame, `Survived`). Column names are kept.
- `feature_engineering(X)` is the one place that adds or updates columns. It changes `X` in place, removes nothing, and calls, in this order: `extract_title`, `add_is_male_adult` (which first calls `fill_missing_age`), `encode_sex`, `one_hot_encode_embarked`, `one_hot_encode_pclass`, `add_pclass_interaction`, `add_family_buckets`, `add_shared_ticket`, `add_has_cabin`, `add_fare_bands_in_class`.
- Features that were tested and dropped are still built, but switched out of the fit, so they can be retested: `USE_PCLASS_INTERACTION` (`PCLASS_INTERACTION_COLUMNS`), `USE_SHARED_TICKET` (`SHARED_TICKET_COLUMNS`), `USE_CABIN` (`CABIN_COLUMNS`) and `USE_FARE_BANDS` (`FARE_BAND_COLUMNS`). All four are off.
- `SMALL_FAMILY_MAX = 3` is the family cut point, and `FAMILY_COLUMNS` lists the three family bucket columns.
- `COLUMNS_NOT_FOR_FITTING` lists the columns to leave out when fitting. Columns are only excluded at fit time, so the originals stay available for later steps and for analysis. It contains PassengerId, Name, Age, Title, Sex, Embarked, Pclass, SibSp, Parch, `family_count`, Ticket, `ticket_group_size`, Fare and Cabin. `UNDECIDED_COLUMNS` is empty now; any new column without a decision goes there. `fitting_columns(X)` returns everything not in the list.
- Constants: `ADULT_AGE = 15` (adult means age greater than 15), `RANDOM_SEED = 42`, `TEST_SIZE = 0.2`, `N_FOLDS = 5`, `N_REPEATS = 10`.
- Evaluation: `split_train_test(X, y)`, `build_model()`, `repeated_folds()`, `cross_validate_model(X, y, columns=None)`, `apply_keep_rule(base, with_new)` and `compare_with_columns(X, y, new_columns)`. `evaluate_on_test(...)` is in `titanic.py`. See "Evaluation".
- Analysis helpers: `title_survival_summary(X, y)`, `age_survival_summary(X, y, by=None, bin_width=5)`, `pclass_survival_summary(X, y, by=None)` and `column_survival_summary(X, y, column, by=None)` (any column).
- `error_analysis.py` and `tree_analysis.py` are separate scripts that import `common.py` and save graphs to `plots/`. They don't change the logistic regression pipeline. See "Error analysis" and "Decision tree and random forest". `error_analysis.py` also writes `female_3rd_class.csv` (see "Ticket fate").

## Encoding: why each choice

| Situation | Encoding | Reason |
|---|---|---|
| Two categories (Sex) | One 0/1 column | With `f(x) = w·x + b` the value 0 gives `b` and the value 1 gives `w + b`, so a single weight is enough to separate two groups. |
| Several unordered categories (Embarked: C, Q, S) | One 0/1 column per category (one-hot) | Coding them 0, 1, 2 forces a straight line through the values, so the model assumes an order and equal distances that don't exist. The result would also depend on which number went to which port. Separate columns give each port its own weight. |
| Ordered categories | Single column 0, 1, 2 | Only when the order is real. Not used so far. |

For Embarked, all three columns are kept (not two). The data supports a real port effect: C 55% survival, Q 39%, S 34%.

## Columns

### Name: not used, but Title is extracted

Names look like `Braund, Mr. Owen Harris`: the title sits between the first comma and the first period after it. `extract_title` slices that and trims it into a new `Title` column. 17 distinct titles; Mr 517, Miss 182, Mrs 125, Master 40, and 13 rare ones with 7 or fewer passengers each.

- `Name` is excluded: free text with a unique value per passenger, nothing a model can learn from.
- `Title` is excluded from fitting, for now. Mr, Mrs and Miss mostly restate `Sex`, and Master (a young boy) is covered by the age feature. The rare "special" titles (Dr, Rev, Major, Col, Capt, Don, Sir, Lady, Countess, Jonkheer) cover about 23 passengers with mixed outcomes (0% for Rev and Capt, 100% for Lady and Countess), which is too little data for a reliable weight. The column stays in `X` because it is used to fill missing ages. If the special-title idea is revisited, test it on held-out data, not training accuracy.
- French titles (Mlle, Mme) and Ms would be grouped with Miss and Mrs, not kept as separate groups, because the groups are tiny.

### Sex: encoded, original excluded

- `Sex_encoded`: male 0, female 1. The original `Sex` text column is kept in `X` but excluded from fitting.

### Age: filled and turned into a flag, raw age excluded

- 177 of 891 ages are missing. Of those, 172 belong to Mr, Miss and Mrs; Master has 4 and Dr 1.
- Missing ages are filled with the mean age for the passenger's title. This is enough because age is only used to decide whether a male passenger is an adult. Mr (mean 32.4) and Dr (42.0) come out adult, Master (4.6) comes out a child, and Miss and Mrs are female, so their filled age doesn't change the flag. Three known Mr are under 15, so a few boys may be misclassified; that is a small risk.
- Binning age into child/young/adult/old was dropped. Survival by age on its own is nearly flat after about 15 (35% to 46%), and the only real effect is children under about 15 (65% under 5).
- The effect depends on sex. Adult women survive at 65% to 90% and adult men at about 10% to 25%, but boys under 10 survive at 44% to 65%. One `is_child` weight can't express that, so the feature is the interaction `is_male_adult` (male and age > 15, as 0/1).
- With `Sex_encoded` and `is_male_adult` the model has three groups: females (girls and women together), boys, and adult men. Girls and women are merged on purpose, because girls' survival is high at every age.
- Raw `Age` is excluded. After the flag exists it adds nothing a linear weight could use, since the curve is not a straight line.
- Pclass interacts with age too (third-class children do much worse than second-class ones). It is handled as a separate feature later.

#### Raw Age for tree models: tested and dropped

Why it was considered:

- The reason for excluding raw Age (a straight-line weight per year can't follow the age curve) applies to logistic regression only. A tree splits at thresholds such as `Age <= 6.5`, and can use different thresholds in different branches, so a curved effect is no problem.
- Age is the one column that varies inside the passenger types. The 11 features describe only 43 types and cap accuracy at 0.838 (see "Error analysis"). Age could, in principle, separate survivors from non-survivors within a type.

The missing-age problem:

- For `is_male_adult`, the title-mean fill only has to put a passenger on the correct side of 15, which it does. A tree uses the exact value.
- 119 of the 517 Mr (23%) have a missing age and all got exactly 32.4. In the 32-year bin, 15 men have a real age and 119 are fills (`plots/mr_ages.png`). With two close thresholds (for example `Age > 32.3` and `Age <= 32.5`), a tree can build a leaf that is essentially "Mr whose age was missing". Every threshold near 32 also moves 119 men at once, so where the tree places its real age splits for men depends on the fill value.
- Does a missing age carry signal? Mr with a missing age survived at 12% against 17% with a known age, but most of the missing ones are in 3rd class (90 of 119). By class (known against missing): 1st 37% against 25% (n=20 missing), 2nd 7% against 22% (n=9), 3rd 13% against 8% (n=90). The 1st- and 2nd-class groups are too small to read, and the 3rd-class gap (5 points) is within its noise band (about ±6 points). Every rate is far below 50%, so a "missing age" leaf would predict "died", as for adult men anyway (`plots/mr_missing_age_survival.png`). So the spike teaches nothing useful, and its real harm is to the placement of real age thresholds.

Options considered for the tree's Age column (all use features only, never `Survived`):

- A. Finer group mean (Title + Pclass): smaller spikes, but still spikes.
- B. Random draw from the group's real ages: no spike, but made-up ages that vary with the seed.
- C. Keep the fill and add a flag `age_missing` (0/1): the tree can separate the filled passengers with one clean split.
- D. Predict age from other columns (e.g. `KNNImputer`): more spread, more machinery, still a guess.
- E. Sentinel value (−1 for missing): moves the spike to the edge, away from real ages. Fine for trees (they only compare against thresholds), nonsense for logistic regression.
- F. Leave it as NaN: scikit-learn 1.9.1 trees learn at each split which side missing values go.
- Dropping the rows was ruled out: 177 of 891 passengers (20%).

In every option, `is_male_adult` and its title-mean fill stay as they are; only a separate Age column for the tree changes. C, E and F were tested (columns `age_missing`, `age_sentinel` and `age_with_nan`, added in `tree_analysis.py`, not in `feature_engineering`).

Results (best validation accuracy over `min_samples_leaf`, 50 folds; details in "Decision tree and random forest"):

| Model | No Age | C: fill + flag | E: sentinel −1 | F: NaN |
|---|---|---|---|---|
| Decision tree | 0.824 (leaf 2) | 0.825 (leaf 20) | 0.818 (leaf 20) | 0.818 (leaf 20) |
| Random forest | 0.820 (leaf 20) | 0.823 (leaf 10) | not tested | not tested |

- With Age, the tree overfits clearly for the first time: with leaves of 2, training accuracy is 0.891 and validation 0.793. Age gives it real numbers to memorize, but the detail doesn't carry over to new passengers.
- E and F give practically the same curve: both just send all missing ages to one side of a split. C is 0.007 higher, which is within noise.
- Keep rule, tree: C at its best against the tree without Age at the same leaf size gives +0.004 (needs +0.024). Fails.
- Keep rule, forest: best with Age against best without Age gives +0.003, better in 52% of folds. Fails. The forest was the fair second chance for Age, because averaging many trees removes exactly the kind of overfitting Age caused in the single tree. It did remove it (validation with leaves of 2 rose from 0.793 to 0.809), but the peak didn't rise.
- Decision: raw Age is out, for trees as well. What matters about age (child or adult, at 15) is already in `is_male_adult`.

### Embarked: filled and one-hot encoded, original excluded

- 2 values are missing; they are filled with the most common port (S).
- `Embarked_C`, `Embarked_Q`, `Embarked_S` are added. The original `Embarked` text column is kept but excluded from fitting.

### Pclass: one-hot encoded, original excluded

- Survival by class: 1st 63.0% (n=216), 2nd 47.3% (n=184), 3rd 24.2% (n=491). The steps are unequal (16 points from 1st to 2nd, 23 from 2nd to 3rd), so a single numeric weight would force a straight line. One-hot (`Pclass_1`, `Pclass_2`, `Pclass_3`) gives each class its own weight. The original `Pclass` is kept but excluded from fitting.
- The class effect depends on the group (survival by class, 1st/2nd/3rd): women 97%/92%/50%, adult men 35%/8%/11%, boys 100% (n=3)/100% (n=9)/34% (n=32). Women lose mainly in 3rd class and men gain mainly in 1st.
- Logistic regression adds weights on the log-odds scale, so with only separate `Sex_encoded`, `is_male_adult` and `Pclass_*` weights the class effect is the same number for every group. In log-odds, 1st versus 3rd class is about +3.4 for women and about +1.5 for men, and 2nd versus 3rd is about +2.5 for women and about -0.4 for men. The additive model has to compromise.
- Interaction experiment. Groups: women and adult men (boys left out: cells of 3, 9 and 32 passengers are too small to learn from). Small set of two columns aimed at the largest deviations: `female_P3` (woman in 3rd class) and `adult_male_P1` (adult man in 1st class), built by `add_pclass_interaction`. Rule set beforehand: keep it if the mean accuracy gains at least 0.03 and it wins at least 4 of 5 folds (`MIN_ACCURACY_GAIN`, `MIN_FOLD_WINS`). `compare_pclass_interaction` runs both versions on the same folds.
- Result: mean accuracy 0.791 → 0.815 (+0.024). Per-fold change +0.049, +0.014, +0.028, +0.028, 0.000, so 4 folds won and 1 tie. Precision rose 0.730 → 0.856 and recall fell 0.736 → 0.638: the model predicts "survived" less often but is right more often when it does.
- First decision: keep the interaction, overriding the rule, since the change was positive or zero in every fold and with this small a sample a 0.03 bar is hard to reach. The threshold was then lowered to 0.024 for later decisions.
- Revised decision: interaction off (`USE_PCLASS_INTERACTION = False`), in favour of the family buckets. See "SibSp and Parch" for how this came about. The columns are still built, so the switch can turn it back on.
- `Fare` moves with class (median 60.3, 14.2 and 8.0 for 1st, 2nd and 3rd), so the two overlap. Keep that in mind when deciding on `Fare`.

### SibSp and Parch: combined into three family buckets, originals excluded

SibSp is the number of siblings or spouses aboard and Parch the number of parents or children aboard.

- Survival by value: 0 → about 34% for both (608 and 678 passengers), 1 → 53.6% and 55.1%, then falling for larger values (SibSp 4: 16.7%; SibSp 5 and 8: 0%). Above 2, most values have fewer than 20 passengers.
- Raw numbers are not used: survival goes up and then down, which one weight can't follow, and the large values are too thin to learn from.
- Large families are mostly in 3rd class (54 of 61 with SibSp or Parch above 2), but the 3rd-class weight doesn't absorb them: within 3rd class, large families survived at 11.1% against 25.9% for the rest. So large families need their own bucket.
- Passengers travelling alone (537, 30.4%) and those with 1–2 family members (293, 57.3%) differ more than large vs small does, so "small" was split into single and small family. Small family beats single in every class (73.8/62.3/38.9% against 53.2/34.6/21.3% for 1st/2nd/3rd), with at least 77 passengers in each cell.
- SibSp and Parch are treated as one idea, because their survival rates line up closely. Buckets are defined by the sum `family_count = SibSp + Parch`: survival is 30.4% at 0, 55.3% at 1, 57.8% at 2, 72.4% at 3 (n=29), then 20.0% at 4 and lower beyond. So: single = 0, small family = 1 to 3, large family = 4 or more (537, 292 and 62 passengers).
- Encoded as three 0/1 columns (`is_single`, `is_of_small_family`, `is_of_large_family`), built by `add_family_buckets`. Exactly one is 1 in each row. SibSp, Parch and `family_count` are excluded from fitting.

How the decision was reached:

1. With 5-fold cross-validation, adding the family buckets to the 10 features (including the class interaction) gave 0.815 → 0.827 (+0.013), 4 of 5 folds won, one fold worse (−0.007) and most of the gain from one fold (+0.042). That fails the rule (gain below 0.024).
2. Rather than override the rule a second time, the comparison was rerun with repeated cross-validation (10 × 5 = 50 folds) to average out lucky folds. Family buckets against the 10 features: +0.009, 32 of 50 folds won (64%), 4 ties, 14 lost. Still fails, and the gain shrank, as expected if one fold had been carrying it.
3. All four combinations on the same 50 folds:

| Features | Accuracy | Precision | Recall |
|---|---|---|---|
| Neither (8 features) | 0.795 | 0.743 | 0.724 |
| Class interaction only | 0.820 | 0.865 | 0.635 |
| Family buckets only | 0.827 | 0.804 | 0.729 |
| Both | 0.828 | 0.815 | 0.720 |

4. Each helps on its own (interaction +0.025, family buckets +0.032 over neither), but together they add almost nothing more: with the family buckets in, the interaction adds +0.001 (15 of 50 folds won, 30 ties). They overlap because they cover many of the same passengers: the 3rd-class women singled out by the interaction include many large families, and large families pull down 3rd-class survival.
5. Decision: keep the family buckets only. They give the larger gain on their own, adding the interaction to them brings nothing, and the simpler model is easier to trust. Precision and recall also stay closer to balanced than with the interaction alone.

### PassengerId: excluded

A row number with no meaning for survival. Excluded without testing.

### Ticket: shared-ticket feature tested and dropped

- The ticket text has no pattern, but tickets are not unique: 681 distinct tickets for 891 passengers, so 344 passengers share a ticket. Survival by passengers per ticket: 1 → 29.8% (547), 2 → 57.4% (188), 3 → 69.8% (63), 4 → 50.0% (44), then 0% to 24% for 5 to 7. The same up-then-down shape as family size, because a shared ticket usually means travelling together.
- Checked the passengers a family feature can't see: 75 passengers with no family aboard but on a shared ticket survived at 50.7%, against 27.1% for the 462 on their own ticket. Most of that gap is composition (41% women against 21%, 45% 1st class against 16%). Within sex the gap shrinks (women 93.5% against 73.7%, men 20.5% against 15.0%), and by class it is inconsistent (higher in 1st and 3rd, lower in 2nd with n=16).
- Tested `is_shared_ticket` (1 if more than one passenger on the ticket, 344 passengers), built by `add_shared_ticket`: accuracy 0.827 → 0.825 (−0.002), 2 of 50 folds won, 37 ties, 11 lost. Fails the keep rule. The family buckets already carry this information. `Ticket` and `ticket_group_size` are excluded.

### Cabin: has-cabin feature tested and dropped

- 687 of 891 cabins are missing (77%). Survival with a cabin 66.7% (204), without 30.0% (687).
- The gap overlaps with class: 176 of the 204 with a cabin are in 1st class. Within class there is still a gap (1st 66.5% against 47.5%, 2nd 81.2% against 44.0%, 3rd 50.0% against 23.6%), but the 2nd- and 3rd-class cabin groups have only 16 and 12 passengers.
- Tested `has_cabin` (0/1), built by `add_has_cabin`: accuracy 0.827 → 0.824 (−0.003), 9 of 50 folds won, 19 ties, 22 lost. Fails the keep rule; `Pclass_1` already carries most of it. `Cabin` is excluded.

### Fare: within-class fare bands tested and dropped

- 0 to 512.3, median 14.5, 248 distinct values, 15 passengers paid 0. Raw numbers are not used.
- Median fare by class 60.3, 14.2 and 8.0, so fare largely repeats class. Across five equal-sized bands, survival rises from about 22% to about 65% for both raw fare and fare per person, which is mostly class.
- On a shared ticket the fare is the total for the ticket, so raw fare also reflects group size. Fare per person = fare divided by passengers on the ticket.
- Within each class (split into thirds by fare): raw fare 1st 44/67/77%, 2nd 35/57/59%, 3rd 23/18/32%; fare per person 1st 59/60/70%, 2nd 45/44/54%, 3rd 25/27/21%. Most of the within-class effect of raw fare disappears per person, which suggests it comes from group size, already covered by the family buckets.
- Fare per person was not tested: there was nothing it could plausibly add beyond class and family.
- Tested raw fare in thirds within class, as three 0/1 columns shared across classes (`fare_low_in_class`, `fare_mid_in_class`, `fare_high_in_class`; 323, 294 and 274 passengers), built by `add_fare_bands_in_class`: accuracy 0.827 → 0.824 (−0.004), 6 of 50 folds won, 27 ties, 17 lost. Fails the keep rule. A version with one column per class and band (9 columns) was not tested. `Fare` is excluded.

### Ticket fate (v2): group fate on a shared ticket, refined version kept

How the idea came about:

- `error_analysis.py` exports all 116 3rd-class women of the working set to `female_3rd_class.csv` (`export_cell`): out-of-fold prediction, `correct`, `probability`, `Title`, `family_count`, `ticket_group_size` and the untouched `train.csv` columns (Age with blanks). 74 right, 42 wrong: 39 died but were predicted to survive, 3 survived but were predicted to die.
- The model predicts "survived" for every 3rd-class woman with 0 to 3 relatives and "died" for every one with 4 or more. It sees only the family bucket, and the single and small-family weights are almost equal, so port is the only other thing that moves her probability (0.62 to 0.77, never below 0.5).
- Actual survival by `family_count` (`plots/female_3rd_by_family_count.png`): 0 → 32 of 51 (63%), 1 → 8 of 20 (40%), 2 → 11 of 18 (61%), 3 → 4 of 5, 4 or more → 3 of 22 (14%). The dip at 1 is within noise (with 20 women the band is about 18% to 62%).
- `SibSp` counts spouses as well as siblings. Using the title as a stand-in for a spouse (Mrs with `SibSp` 1 and no `Parch`): 2 of 6 survived against 32 of 51 travelling alone. So the data gives no support to "a husband helps his wife survive", but 6 is far too few to build a feature on.
- A plausible reason was that people travelling together didn't want to be separated. That idea is not limited to wives: it applies to any group in any class, so it rests on many more passengers. Hypothesis: if one member of a group survived, most of the group did.

The data check (working set, groups = same ticket, outcome of the other members; `plots/ticket_group_fate.png`):

| | Alone on ticket | All others died | At least one other survived |
|---|---|---|---|
| All passengers | 32% (476) | 16% (104) | 80% (132) |
| Women | 77% (124) | 31% (51) | 99% (78) |
| Adult men | 14% (344) | 0% (37) | 43% (44) |
| Boys | 88% (8) | 6% (16) | 90% (10) |
| 3rd-class women | 61% (64) | 11% (37) | 100% (15) |
| 1st-class adult men | 34% (62) | 0% (6) | 50% (22) |
| 3rd-class adult men | 11% (218) | 0% (19) | 50% (14) |

- The groups are large enough to trust (for women, 78 and 51 against a gap of almost 70 points).
- It splits 52 of the 116 3rd-class women into two clear groups. For adult men a 50% rate is still a coin flip, so the gain there is smaller.
- The earlier `is_shared_ticket` failed because it only said *whether* a passenger was in a group, which the family buckets already said. This says *what happened to the group*, which no other column can.
- This table was computed over the whole working set without folds, so it is optimistic.

Which outcomes may be used (label leakage):

- This is the first feature built from `Survived`. Example: in one fold, passengers A (validation), B (training) and C (validation) share a ticket. For A's feature, only B's outcome may be used. A's own outcome is the answer itself, and C is in validation, so unknown at prediction time, like a Kaggle test passenger.
- For training rows the same rule: the other training passengers on the ticket, never the passenger's own outcome. Otherwise the model learns that the column holds its answer.
- So the feature is rebuilt inside every fold from that fold's training rows, never in `feature_engineering`. For Kaggle's `test.csv`, ticket-mates come from the labeled `train.csv` rows.

Encoding and implementation:

- Three situations, one-hot like Embarked, Pclass and the family buckets: `ticket_mates_unknown` (no known ticket-mate), `ticket_mates_all_died` (every known ticket-mate died) and `ticket_mate_survived` (at least one known ticket-mate survived). Ticket-mates are all other passengers on the ticket, whatever their sex or age.
- `add_ticket_fate(X, X_known, y_known)` in `titanic_v2.py` counts the known passengers per ticket and subtracts a row's own outcome when that row is itself among the known ones (matched by index).

Result (same 50 folds, keep rule):

| | Accuracy | Precision | Recall |
|---|---|---|---|
| 11 features | 0.827 | 0.804 | 0.729 |
| + ticket fate | 0.837 | 0.815 | 0.746 |

- Gain +0.010, better in 35 of 50 folds (70%), 3 ties, 12 lost. Fails the keep rule (needs +0.024 and 80%).
- The best score so far, and above the 0.838 ceiling of the 11 features on their own (that ceiling applies to those features only).
- Much smaller than the data check suggested: inside a fold about a fifth of the ticket-mates are in validation and count as unknown, and many passengers have no ticket-mate in the labeled data at all.

Why the first version was too blunt (working set, no folds; `plots/women_by_ticket_mates.png`):

- For women, "at least one ticket-mate survived" works as expected: 65 of 66 survived when a woman or child on the ticket survived, 12 of 12 when only men were on the ticket and one survived.
- "All ticket-mates died" mixed two opposite situations. All woman/child ticket-mates died: 0 of 27 women survived (all 27 in 3rd class). Only men on the ticket, all died: 16 of 24 survived (67%), mostly a wife whose husband died, as "women and children first" predicts. Together they gave the diluted 31%.
- So the men's outcomes say little about a woman, and the other women's and children's outcomes say almost everything.
- A proposal of `is_woman_one_mate_survived` was checked against the 42 errors in 3rd-class women: it could fix at most 2, because the model already predicts "survived" for almost every 3rd-class woman. 39 of the 42 errors are the opposite case (died, predicted to survive). By situation: a woman/child ticket-mate survived 2 wrong of 13, all woman/child ticket-mates died 8 of 27, only men ticket-mates who all died 6 of 10, alone on the ticket 26 of 64. The 26 alone can't be helped by any ticket-mate feature.

The refinement (two decisions):

- Only women and children count as ticket-mates. By what happened to the women and children on the ticket (none known / all died / one survived): women 77% (160) / 0% (27) / 98% (66), boys 89% (9) / 0% (15) / 90% (10), adult men 15% (373) / 0% (14) / 37% (38). Boys behave like women; men mostly died even when their wife and children survived (1st class 11 of 22).
- Only women and boys get the feature; adult men are 0 in all three columns. A linear model gives each column one weight shared by everyone, so the strong positive weight learned from women and boys would otherwise push men whose wife survived towards "survived", wrongly about 63% of the time.
- Encoding: three one-hot columns, `wc_mates_unknown`, `wc_mates_all_died` and `wc_mate_survived` (two were considered, leaving "unknown" at 0 like the men; three were chosen). Built by `add_group_fate`, inside each fold, with the same leakage rules.

Result of the refinement (same 50 folds):

| | Accuracy | Precision | Recall |
|---|---|---|---|
| 11 features | 0.827 | 0.804 | 0.729 |
| + first version (all ticket-mates) | 0.837 | 0.815 | 0.746 |
| + refined (women/children, for women and boys) | 0.841 | 0.824 | 0.748 |

- Against the 11 features: gain +0.014, better in 46 of 50 folds (92%), 3 ties, 1 lost. The gain test fails (needs +0.024), the consistency test passes easily (needs 80%).
- Against the first version: +0.004, 21 won, 12 ties, 17 lost. Most of the gain was already in the first version; the refinement makes it more consistent rather than larger.

Decision: kept, overriding the gain threshold. Reasons:

- Winning 46 of 50 folds is very unlikely by chance (a feature with no effect would win about half), so the effect is real, just small. Filtering out noise is what the threshold was for, and the fold consistency already does that.
- The 0.024 threshold was set early, for features that affect many passengers. A targeted feature can fix only a few errors (about 2 per validation fold of 142 passengers here), so it probably can't reach +0.024 however good it is.
- It adds complexity (it is built from other passengers' labels), but this is a learning project, not a production system, and the aim is to learn how to create features.
- Decided in advance: this was the last feature to try, and the result would be accepted either way.

Results of the final model (v2):

- Test set (scored once): 0.838 / 0.823 / 0.739 against v1's 0.827 / 0.797 / 0.739. +0.011 is 2 more passengers right out of 179, in line with cross-validation, so there is no sign of leakage (leakage would make the test score fall well below the CV score). Precision rose and recall stayed: the feature mostly removes false "survived" predictions. 42% of the women and boys in the test set (30 of 71) have a known woman/child ticket-mate.
- Kaggle: 0.78708 against 0.77751. 41% of the women and boys in Kaggle's `test.csv` (about 72 of 175) have a known woman/child ticket-mate in `train.csv`. Only 8 of 418 predictions changed from v1 (6 survived → died, 2 died → survived). The score covers all 418 passengers (0.77751 × 418 = 325 and 0.78708 × 418 = 329 exactly), so v2 gets 4 more right; with 8 changes, that means 6 fixed and 2 broken.

## Evaluation

- The purpose of fitting at this stage is to measure how well a set of features works, not to produce the final model. Each change is compared with the baseline score.
- Kaggle's `test.csv` has no `Survived` column, so all three sets come from `train.csv`. `test.csv` stays untouched for the final submission.
- Split: 20% of `train.csv` is the test set (stratified by `Survived`, seed 42), set aside until the very end. The remaining 80% (about 713 rows) is the working set.
- Cross-validation (option A): 5 stratified, shuffled folds on the working set. Each fold takes a turn as the validation set while the other four train. The mean score is the estimate for a feature set. With 891 rows, a single fixed validation block would be noisy, and 10 folds would make each validation fold too small (about 71 rows). The test set is scored once at the end, with `evaluate_on_test`.
- Repeated cross-validation: the 5-fold split is repeated 10 times with a different shuffle (`N_REPEATS = 10`), giving 50 folds. Added after a 5-fold result turned out to be carried mostly by one fold. The seed is fixed, so two runs being compared use the same 50 folds and can be compared fold by fold.
- Metrics: accuracy, precision and recall are reported. Accuracy decides which model wins.
- Fold-to-fold accuracy varies by about ±0.03, so a change of one or two points in the mean of a single 5-fold run may not be real.
- Keep rule for a new feature: the mean accuracy gains at least 0.024 and the new version is better in at least 80% of folds (`MIN_ACCURACY_GAIN`, `MIN_FOLD_WIN_SHARE`). The threshold was first 0.03 and was lowered to 0.024 after the interaction experiment; the share was 4 of 5 folds before repeated cross-validation. Ties don't count as wins. `apply_keep_rule` applies it; `compare_with_columns` (v1) and `compare_fate` (v2) call it.
- Lesson from the group-fate decision: a fixed gain threshold suits features that affect many passengers. For a targeted feature that can only fix a few errors, consistency across folds is the better test. The group-fate feature was kept on that basis (+0.014, 92% of folds).
- `RUN_PCLASS_INTERACTION_COMPARISON` switches the entry point between the interaction comparison and a single cross-validation run. It is off.
- Training accuracy is not used for decisions, because the model is scored on rows it learned from.
- `y` is converted to a flat array at fitting time (`y['Survived']`), so there is no `DataConversionWarning`.
- Convergence: `LogisticRegression` uses the L-BFGS solver, which sets its own step size. It stops after 100 iterations at most and warns (`ConvergenceWarning`) if it hasn't converged. The baseline used 18 iterations and gave no warning. `cross_validate_model` prints the iterations per fold. Current features are all 0 or 1; a column on a large scale such as `Fare` may need scaling, and a convergence warning should never be ignored.

### Test set results

| Model | Features | Test accuracy | Test precision | Test recall | CV accuracy (50 folds) |
|---|---|---|---|---|---|
| `LogisticRegression` defaults (v1) | 11 | 0.827 | 0.797 | 0.739 | 0.827 |
| `LogisticRegression` defaults (v2, final) | 11 + group fate | 0.838 | 0.823 | 0.739 | 0.841 |

Trained on the 712-row working set, scored on the 179-row test set (69 survivors), each once. v1 was the baseline for the model stage. For v2, the test rows' group fate comes from the working-set passengers only.

Kaggle submissions:

| Model | Features | Trained on | Kaggle public score |
|---|---|---|---|
| `LogisticRegression` defaults (v1) | 11 | all 891 labeled rows | 0.77751 (325 of 418) |
| `LogisticRegression` defaults (v2, final) | 11 + group fate | all 891 labeled rows | 0.78708 (329 of 418) |

- Built by `create_submission()` in `titanic.py`, written to `submissions.csv` (418 rows, 39.5% predicted to survive). `train.csv` and `test.csv` are combined, `feature_engineering` runs once on all 1,309 rows, then the rows are split apart again.
- After feature engineering no Kaggle test passenger is missing an age. One title appears only in `test.csv` ("Dona", 1 passenger); her age is known, so the fallback TODO wasn't needed.
- About 5 points below the local estimate (0.827). Kaggle's own sex-only `gender_submission.csv` scores about 0.766, so the model is about 1 point above that baseline on the leaderboard.
- v2 is built by `create_submission()` in `titanic_v2.py`, written to `submissions_v2.csv` (418 rows, 38.5% predicted to survive). Feature engineering is the same as v1; the group fate of the Kaggle test rows comes from the 891 labeled rows only. v2 stays about 5 points below its local estimates too (0.841 / 0.838), so the gap is between this data and Kaggle's test passengers, not a v2 problem.
- The leaderboard scores all 418 test passengers (each score × 418 is a whole number), so one passenger is about 0.24 points. An earlier statement that it scores only about half of them (209) was wrong.

### Fill values and the split

- `feature_engineering` runs on all labeled rows before the split. The fill values (mean age per title, most common port) use only features, never `Survived`, so this does not leak labels. The effect on scores is small, and it affects every feature set that is compared in the same way.
- Caveat: the test set's features therefore feed the fill values, so it isn't completely untouched. This stops being acceptable if a later feature is computed from `Survived` (for example the survival rate per title); such a feature needs the split first. The ticket-fate feature in v2 is the first such feature, and it is rebuilt inside every fold (see "Ticket fate").
- For Kaggle's `test.csv`: combine it with the training rows, run `feature_engineering` once on the combined data, then separate the rows again (keep an identifier). This also keeps the one-hot columns identical in both.

## Model

### Logistic regression: settings tested, none help

- Main settings: `C` (inverse of the regularization strength λ, so `C = 1 / λ`; default 1.0), `penalty` (`'l2'` default; L1 can set weights to 0), `class_weight` (`None` default; `'balanced'` counts errors on survivors more, trading precision for recall), `solver` (`'lbfgs'` default; not every solver supports every penalty), `max_iter` and `tol` (stopping), `fit_intercept` (keep the bias). The decision threshold (0.5) is not a model setting but also trades precision for recall.
- Underfitting, not overfitting: training accuracy 0.830, test 0.827; over the 50 folds, training folds 0.829 and validation folds 0.827. No gap, but only about 83% right even on the training rows, so the limit is what a linear model can express with these 11 features (a bias problem). Many passengers have identical feature values and different outcomes.
- Regularization fights overfitting, so it can only help here by being weakened (decrease λ, which means increase `C`). `C` from 1 to 10 in steps of 1: validation accuracy 0.827 → 0.828 (+0.001, 8 of 50 folds won) and flat from 2 upward; training accuracy stays 0.830. The weights grow (largest 2.62 → 3.30) but the predictions barely change. Fails the keep rule; `C` stays at the default.
- Not tested, by reasoning: `C` below 1 (stronger regularization) and the L1 penalty both make the model simpler or drop features, which can only add to the bias of a model that already underfits.
- Conclusion: logistic regression's settings won't improve this model. The remaining ways to reduce bias are better features (the likely ones were tested and added nothing) or a more flexible model type that learns interactions and non-linear patterns by itself. Decision trees are next.

### Error analysis

`error_analysis.py` (run with `uv run python error_analysis.py`) predicts each of the 712 working-set passengers with a model that didn't train on that passenger (5-fold, out-of-fold), and saves graphs to `plots/`. The test set is not used.

- 123 wrong of 712 (0.827): 72 survivors predicted to die, 51 who died predicted to survive.
- Where (`errors_by_group_and_class.png`): 106 of the 123 errors are in three cells: 3rd-class women (42 wrong of 116), 1st-class adult men (34 of 90) and 3rd-class adult men (30 of 251). The first two are the near-coin-flip cells the class interaction targeted. Women in 1st and 2nd class and all boys are almost always right.
- How sure (`probability_by_outcome.png`): most passengers sit far from 0.5 and are predicted right. Some survivors sit near 0.07: 3rd-class men who survived, identical in features to the many who died.
- Weights (`weights.png`, bias +0.74): `is_male_adult` −2.62, `is_of_large_family` −1.31, `Pclass_3` −0.92, `Embarked_S` −0.30, `Pclass_2` −0.04, `Embarked_C` +0.07, `Embarked_Q` +0.23, `Sex_encoded` +0.38, `is_single` +0.60, `is_of_small_family` +0.71, `Pclass_1` +0.97. `Sex_encoded` is small because the reference group is boys, not men. `is_single` and `is_of_small_family` are almost equal: once sex and class are accounted for, the family effect is mostly "large family or not"; the single vs small gap in the raw tables came from single passengers being mostly adult men.
- The ceiling (`passenger_types.png`): every passenger is one of at most 3 × 3 × 3 × 3 = 81 combinations of the 11 features (group, class, port, family); 43 occur in the working set. Passengers of the same type have identical inputs, so every model gives them the same prediction; the best possible is each type's majority outcome. The minorities add up to 115 passengers who are wrong for any model, so the maximum accuracy with these features is 597 / 712 = 0.838 (measured on the same rows, so slightly optimistic). The model is at 0.827, 8 errors above the minimum. Example: adult men, 3rd class, from S, single: 167 passengers, 21 survived and 146 died, so the best prediction is "died" and the 21 survivors are always wrong.
- Consequence: a different model type given the same 11 features can't gain more than about 1 point. To do better, a model needs information that separates passengers within a type (for example raw columns such as Age or Fare, which a decision tree can split on directly). Raw Age was tested this way and didn't help (see "Raw Age for tree models").

### Decision tree and random forest

`tree_analysis.py` (run with `uv run python tree_analysis.py`) draws validation curves on the working set, with the same 50 folds as everywhere else; the test set is not used. `plot_min_samples_leaf_curve` was replaced by `plot_age_variant_curves` (single tree) and `plot_forest_curves` (forest); the entry point currently runs the forest.

Settings and how `min_samples_leaf` was chosen:

- Only `min_samples_leaf` (smallest number of training passengers allowed in a leaf) is varied. Everything else is scikit-learn's default: Gini, no maximum depth; for the forest, 100 trees, bootstrap rows and √(number of features) candidate columns per split. Seed 42.
- Using the smallest passenger type as the leaf size was considered and rejected: the smallest types have 1 passenger, and the leaves change as soon as a continuous column such as Age is added.
- The question is how small a group can be before its survival rate is mostly noise. The observed rate in a group of n varies around the true rate p by about ±2 × √(p(1 − p) / n), which covers about 95% of samples. At p = 0.5: 5 passengers ±45 points, 10 ±32, 20 ±22, 40 ±16 (`plots/sampling_noise.png`). Halving the uncertainty takes 4 times as many passengers. For tiny groups, or a rate of exactly 0% or 100%, the formula fails (3 of 3 gives ±0); then compute directly: 3 of 3 survivors is still 12.5% likely at a true rate of 50%, so plausible true rates run from about 53% to 100%, against 92% to 99% for 91 of 94 (`plots/small_vs_large_group.png`).
- For a leaf, that means a small leaf can learn the wrong rule by chance. A leaf whose true rate is 30% shows a survivor majority with 5 passengers 16% of the time, with 10 passengers 5%, with 20 passengers 2%, and with 40 almost never (`plots/wrong_rule_by_leaf_size.png`).
- Values tried: 2, 5, 10, 20, 40, 80. They were first proposed as 10, 20, 40 and widened, because a best value can only be trusted with worse values on both sides, and the curve has to reach both the overfitting and the underfitting end. Doubling steps, since uncertainty shrinks with the square root. Each training fold has about 570 passengers, so 80 allows at most 7 leaves.

Single tree, 11 features (`plots/tree_curve_11_features.png`):

| min_samples_leaf | 2 | 5 | 10 | 20 | 40 | 80 |
|---|---|---|---|---|---|---|
| Training | 0.839 | 0.838 | 0.836 | 0.833 | 0.822 | 0.802 |
| Validation | 0.824 | 0.823 | 0.823 | 0.821 | 0.819 | 0.795 |

- Flat from 2 to 20, underfitting at 80. There is almost no overfitting even with leaves of 2: training accuracy stops at 0.839, the 0.838 ceiling, because a tree can't separate passengers whose 11 values are identical. It has learned everything the columns hold.
- Best 0.824, below logistic regression (0.827).

Single tree, 11 features + Age, three ways to handle missing ages (`plots/tree_curve_age_variants.png`), validation accuracy:

| min_samples_leaf | 2 | 5 | 10 | 20 | 40 | 80 |
|---|---|---|---|---|---|---|
| C: fill + flag | 0.793 | 0.812 | 0.812 | 0.825 | 0.819 | 0.795 |
| E: sentinel −1 | 0.789 | 0.809 | 0.814 | 0.818 | 0.816 | 0.795 |
| F: NaN | 0.790 | 0.809 | 0.814 | 0.818 | 0.816 | 0.795 |

Training accuracy with leaves of 2 is about 0.89 in all three: classic overfitting at the small-leaf end, best at 20.

Random forest (`plots/forest_curves.png`), validation accuracy:

| min_samples_leaf | 2 | 5 | 10 | 20 | 40 | 80 |
|---|---|---|---|---|---|---|
| 11 features | 0.819 | 0.818 | 0.819 | 0.820 | 0.808 | 0.793 |
| 11 features + Age (C) | 0.809 | 0.821 | 0.823 | 0.823 | 0.803 | 0.792 |

- With Age, the forest fixed the single tree's overfitting at small leaves (0.793 → 0.809 at 2, 0.812 → 0.823 at 10), but its best (0.823) is no higher than the single tree's (0.825). Averaging removes noise; it can't add information the columns don't have.
- With the 11 features, the forest is slightly worse than a single tree (0.820 against 0.824): there was no overfitting to remove, and each tree's randomness (bootstrap rows, about 3 of 11 columns per split) only costs accuracy.
- Keep rule against logistic regression, same 50 folds: forest without Age (leaf 20) −0.007, better in 16% of folds; with Age (leaf 10) −0.004, better in 26%. Both fail. (Choosing the best leaf size on the same folds makes these slightly optimistic for the forest, and it still loses.)

Conclusion:

- Logistic regression, a decision tree and a random forest all land between 0.82 and 0.83 on validation. When three quite different models stop at the same level, the limit is the information in the features, not the model. Logistic regression stays.
- Gradient boosting was set aside for the same reason. Whether trees need one-hot or plain numbers was dropped as a question, since no tree model was kept.
- The next gain has to come from a new feature that separates passengers within the three cells where 106 of 123 errors are: 3rd-class women, 1st-class adult men and 3rd-class adult men.

## Outcome and when to stop

| | CV (50 folds) | Local test | Kaggle |
|---|---|---|---|
| Sex only (Kaggle's baseline) | 0.789 | 0.777 | ~0.766 |
| v1: 11 features | 0.827 | 0.827 | 0.77751 |
| v2: + group fate (final) | 0.841 | 0.838 | 0.78708 |

The improvement from v1 to v2 is consistent across all three measures.

Signals used to decide the work was done:

- Baselines: v2 is clearly above the sex-only baseline and within the range honest Titanic models usually reach on Kaggle (about 0.77 to 0.80). Scores far above about 0.83 usually come from looking up the real outcomes, which are public.
- Chance: much of the remaining error is "same inputs, different outcome" (the 11 features capped accuracy at 0.838, and 26 of the 42 errors in 3rd-class women travel alone, beyond any ticket-mate feature). No column records lifeboat or location luck.
- Different models agree: logistic regression, a tree and a forest all stopped at 0.82 to 0.83, so the limit was the information, not the model.
- Gains near the noise level: fold-to-fold accuracy varies by about ±0.03, and late candidates gained 0.004 to 0.014. Smaller gains can't be told apart from luck.
- Validation wear: every feature tried and kept on the same 712 passengers makes the cross-validation score a little more optimistic (multiple comparisons). After many attempts, it is better to lock in.
- Outside a learning project, a target and a cost would decide: whether another point is worth the effort, the complexity and the risk.

## Open items

None are planned; these are left for anyone who continues.

- A title with no known age leaves `Age` empty. Decide the fallback.
- Ticket group size and within-class fare bands are computed from the rows passed in; for Kaggle's `test.csv`, compute them over train and test combined (comments are in the code).
- The overlap of the family buckets with Sex, the age groups and Pclass was not checked separately; the four-way comparison with the class interaction was judged enough.
- Tree models (or gradient boosting) could be revisited with the group-fate feature, which gives them something new to split on.
- Ticket-mates come from the shared ticket only. Families split across tickets (same surname, different ticket) are not linked; that could extend the coverage beyond about 41% of women and boys.
- `female_3rd_class.csv` and the old `errors_female_3rd_class.csv` are analysis exports, not inputs. The old file is redundant and can be deleted.
