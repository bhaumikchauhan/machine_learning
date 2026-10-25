# ML concepts

Concepts that came up while working on the problems in this repo, each in a few lines: what it is, and where it came up. Mostly the ones that were new, or that I had wrong at first. The per-problem `feature_notes.md` files record the decisions column by column; this file collects the ideas behind them.

Where: **T** = [Titanic](titanic/feature_notes.md), **HP** = [House Prices](house-prices/feature_notes.md).

## Encoding

**Labels stored as numbers.** A column of numbers can still be categories: MSSubClass 20, 60, 120 are dwelling types, and 60 isn't "three times" 20. Using it as a number (scaled or not) forces one straight line through the codes. Treat it as a category. *HP: MSSubClass.*

**One-hot encoding.** One 0/1 column per category, so each category gets its own weight. Coding categories as 0, 1, 2 instead invents an order and equal distances. *T: Embarked, Pclass. HP: MSZoning.*

**Reference category.** With one-hot, one value can be left without a column: its rows have all columns at 0, so they get no weight from this column, and each other weight means "difference from the reference". Usually the most common value. For a near-constant column, a single 0/1 column for the rare value keeps the weight away from everyone else. `OneHotEncoder(drop=...)`. *HP: Street, Alley, LotShape, LandContour, LotConfig.*

**Ordered categories.** A single 0, 1, 2, 3 column with one weight forces *equal steps* between neighbouring values. Use it only when the effect really moves in equal steps; having an order isn't enough. *HP: LotShape (steps +$43k, +$32k, −$6k).*

**Merging rare categories.** A weight learned from very few rows mostly learns those rows. Merge a rare value with its most similar neighbour, after checking they really are similar. *HP: IR2 + IR3, FR2 + FR3.*

**Target encoding.** Replace each category with the mean of the label for that category (one column instead of many). Because it's built from the label, it leaks unless done carefully (next three entries). `TargetEncoder` in scikit-learn does both fixes. *HP: MSSubClass.*

**Leakage inside the training data.** If a row's own label is part of its feature value, the model learns that the column holds part of the answer, gives it too much weight, and that weight fails on test rows, whose labels are in no mean. Worst for small groups: in a group of 4, a row's own price is 25% of its value. *HP: MSSubClass. T: group fate.*

**Leave-one-out doesn't fix it; it flips it.** Leaving only the row's own label out gives (group total − own label) / (n − 1): inside a group, the cheapest row gets the highest value. The leak is reversed, not removed, and a tree can recover each label exactly. Fix: **out-of-fold** values, computed for each part of the training data from the other parts only. *HP: MSSubClass.*

**Blending (smoothing) small groups.** value = (n × group mean + m × overall mean) / (n + m). Small groups lean on the overall mean, large groups keep their own. A category never seen in training (n = 0) gets the overall mean, not 0. Leakage and unreliable small groups are separate problems with separate fixes. *HP: MSSubClass, code 150.*

## Missing values

**"Missing" can mean "none".** Some columns use NA for "the house has none" (no alley, no pool, no garage). pandas reads the text "NA" as a missing value by default, so it looks unknown. Replace it with an explicit value such as "None". *HP: Alley.*

**Missing can carry information, or not.** Check whether rows with a missing value differ from the rest: a missing Cabin went with lower class, a missing LotFrontage didn't go with a different price. If it carries information, a 0/1 "missing" flag can capture it. *T: Cabin, Age. HP: LotFrontage.*

**Filling from related columns.** Fill a missing value from columns that predict it: age from title, frontage from lot size and lot shape, zone from neighbourhood. Check the relationship holds for the rows being filled (cul-de-sac lots are narrower for their size). *T: Age. HP: MSZoning, LotFrontage.*

**Leaking labels vs leaking features.** A fill computed from features only (mean age per title) can use all rows, even test rows: no label is involved. A value computed from the label (survival rate, mean price) must use training rows only. *T: fill before the split.*

**Options for missing values in trees.** Fill + flag, sentinel value (−1), or leave NaN (scikit-learn trees can route missing values at each split). A single fill value creates a spike that a tree can split around. *T: Age for trees.*

## Transforming and scaling

**Log transform.** For a lopsided (skewed) column, log turns equal *ratios* into equal steps (5k → 10k is the same step as 50k → 100k). It spreads out the bulk of the data and stops a few huge values from driving a straight-line fit. It changes the shape; scaling doesn't. *HP: LotArea.*

**Scaling.** Changes a column's units, never its shape. Standardization (`StandardScaler`): (x − mean) / sd, the usual choice for linear models. Min-max (`MinMaxScaler`): to 0–1, sensitive to one extreme value. "Normalization" can mean either; scikit-learn's `Normalizer` is different again (each *row* to length 1). Order: transform, then scale. *HP: LotArea.*

**Why scale.** A penalty (Ridge, Lasso) charges every weight's size equally. A column in thousands needs a tiny weight and a 0/1 column a big one for the same effect, so the penalty would hit them unequally. Plain least squares, with no penalty, gives the same predictions with or without scaling: change the units and the best weight changes to match (weight 10 per sq ft = weight 10,000 per thousand sq ft). *HP: LotArea.*

## Statistics

**Uncertainty of a rate.** An observed rate p from n rows varies around the true rate by about ±2 × √(p(1 − p) / n) (95% of samples). Halving it takes 4 times the rows. It fails for tiny n or p = 0 or 1 (3 of 3 gives ±0); then compute directly: 3 of 3 survivors is still 12.5% likely at a true rate of 50%. *T: min_samples_leaf.*

**Uncertainty of a mean.** ±2 × (spread) / √n. Code 20 (536 houses): ±$6.8k. Code 40 (4 houses): ±$76k. Six gravel-road houses: about ±$50k. *HP: MSSubClass, Street.*

**Other causes (confounding).** A gap between groups may come from what the group members have in common, not from the column itself: irregular lots sell for more because they're newer and bigger, cabins went with 1st class. Another column may already carry it. *T: Cabin, Ticket. HP: Street, LotShape, LandContour, LotConfig.*

**Compare like with like.** "98% of big lots are RL" (any neighbourhood) and "78% of IDOTRR is RM" (any lot) answer different questions, and neither describes a big lot in IDOTRR. Look for rows that match *both* conditions. *HP: MSZoning fill.*

**A single feature can have a big effect.** Living area or overall quality can move prices by hundreds of thousands. What makes a gap unreliable is few rows or other causes, not the size of the gap. *HP: Street.*

## Linear models

**One weight per column, shared by everyone.** A linear model adds weight × value for each column, with the same weight for every row. If an effect differs between groups, it needs its own column (an interaction), or the weight learned from one group is applied to another. *T: is_male_adult, group fate for women and boys only.*

**Additive on the log-odds scale.** Logistic regression adds weights in log-odds, so without interactions the class effect is the same for men and women. *T: Pclass interaction.*

**Least squares and outliers.** Minimizing *squared* error lets one far-off row pull the line: an error of $463k squared outweighs hundreds of ordinary rows, and a row at the far end of the x-axis tilts the line most. Two houses of 1,460 moved the price-per-sq-ft slope by 7%. *HP: Ids 524, 1299.*

**Overlapping (correlated) columns.** They don't cause underfitting. The model can't tell which column deserves the credit, so it splits the weight between them, sometimes unstably. Predictions usually stay fine; the weights become hard to read. Ridge spreads the credit. Overlap is a keep-or-drop question, not an encoding question. *HP: LotShape.*

**Near-constant columns.** A column that is one value for 99.6% of rows can only change the predictions of the few rare rows. A single weight learned from 6 rows can still overfit them; coding the common value as 0 keeps that away from everyone else. If no test row has the rare value, the column can't help at all: drop it. *HP: Street, Utilities.*

## Regularization, bias and variance

**Regularization direction.** λ is the penalty strength. scikit-learn's `LogisticRegression` uses `C = 1 / λ`: a *larger* C means *weaker* regularization. Ridge and Lasso use `alpha`, which is λ itself. *T: logistic regression settings.*

**Underfitting (bias) vs overfitting (variance).** Underfitting: too simple for the pattern, so training and validation are both low and close together. Overfitting: learns the training rows' quirks, so training is high and validation clearly lower. Diagnose by comparing them. Regularization fights overfitting; for an underfitting model it can only help by being weakened. *T: logistic regression (0.830 training, 0.827 validation).*

**Feature ceiling.** When rows with identical inputs have different outcomes, no model can get them all right. Titanic's 11 features made 43 passenger types and capped accuracy at 0.838. When different model types stop at the same score, the limit is the information in the features. *T: error analysis.*

## Trees

**Decision tree.** Splits on thresholds (`Age <= 6.5`), with different thresholds in different branches, so curved effects and interactions are no problem. `min_samples_leaf` limits how small a leaf can be: a small leaf can learn the wrong rule by chance. *T.*

**Random forest.** Averages many trees, each trained on a bootstrap sample of rows, with a random subset of columns per split. It reduces variance (overfitting) but can't add information the columns don't have. *T.*

**Validation curve.** Score against one setting (such as `min_samples_leaf`). Try values on both sides of the best, reaching both the overfitting and the underfitting end, in doubling steps. *T: values 2, 5, 10, 20, 40, 80.*

## Evaluation

**Cross-validation.** Split the working data into folds; each fold takes a turn as validation while the others train. Repeating it with different shuffles (10 × 5 = 50 folds) averages out lucky folds. A fixed seed means two versions are compared on the same folds. Fold-to-fold scores varied by about ±0.03, so small gains in one run may be luck. *T.*

**Test set once.** The held-out test set is scored once, at the end. Choosing models on it makes it a second validation set and its score optimistic. *T.*

**Keep rule, written first.** Decide the rule for keeping a feature (minimum gain, share of folds won) *before* looking at the result, and write down any change to it without referring to the result. Adjusting the rule after seeing the result is how you fool yourself. For a targeted feature that can only fix a few errors, consistency across folds is the better test than a fixed gain. *T.*

**Outlier removal and CV.** Removing hard rows from the validation folds too makes scores look better just because the hard rows are gone. To measure whether removing them helps, remove them from the training part of each fold only. *HP: Ids 524, 1299.*

**When to stop.** Signals: clearly above a simple baseline; much of the remaining error is chance (same inputs, different outcome); different models agree; gains below the fold noise; and validation wear: every comparison made on the same rows makes the CV score a little more optimistic (multiple comparisons). *T.*
