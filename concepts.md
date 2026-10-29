# ML concepts

Concepts that came up while working on the problems in this repo, each in a few lines: what it is, and where it came up. Mostly the ones that were new, or that I had wrong at first. The per-problem `feature_notes.md` files record the decisions column by column; this file collects the ideas behind them.

Where: **T** = [Titanic](titanic/feature_notes.md), **HP** = [House Prices](house-prices/feature_notes.md).

## The leftover

**What it is.** The part of a house's price that the current model gets wrong: log(actual price) − log(predicted price), ×100 ≈ % above (+) or below (−) the prediction. The prediction comes from a model trained *without* that house (out-of-fold, as in a CV fold), so the model can't have memorized it.
- **One house** (HP, Id 494, brick face exterior): predicted $144,546, sold $155,000. Leftover = log(155,000) − log(144,546) = 0.07, so **+7**: sold about 7% above the prediction.
- **A group** (all 50 brick face houses): each house has its own leftover (+9, +17, −82, −4, +4, 0, +12, +24, …). The group's **average** is +7: on average brick face houses sell about 7% above the prediction. The **±** is how much that average could move with a different 50 houses: 2 × spread ÷ √n = 2 × 15.7 ÷ √50 ≈ ±4.
- **Average about 0:** the model already prices the group right. **Clearly away from 0** (beyond its ±): the model keeps mispricing the group in one direction, and a column for it can fix that.

**Why the average, even though single houses can be badly wrong.** A 0/1 column for a group has one weight, so it moves every house in the group by the same amount. It can't fix the −82 house without moving the +17 house too. Two houses in one group, leftovers +50 and −50:

| Shift the column applies | House A | House B | Total squared error |
|---|---|---|---|
| 0 | +50 | −50 | 2,500 + 2,500 = **5,000** |
| +10 | +40 | −60 | 1,600 + 3,600 = 5,200 |
| −10 | +60 | −40 | 3,600 + 1,600 = 5,200 |

Any shift makes one house better and the other worse by more; the best shift is the group's average. So the **average** says what a column *for that group* can fix. The spread *inside* the group needs a *different* column: one that differs between the houses of the group. Finding those is what the leftover ranking does (`plot_leftover_ranking`, `plots/leftover_ranking.png`).

**The pool example.** At $100 per sq ft, three houses without a pool (1,000–1,500 sq ft) sell for about their prediction (leftover ≈ $0); three with a pool (2,000–3,000 sq ft) sell for about $8k above it. Raw, pool houses cost $135k more, but that's mostly size: in a model that already has size, a pool column's weight is about $8k. A column's value is judged by what's left after the others, not by its raw pattern. *HP: ExterQual, OverallQual, MSSubClass, every column since.*

## Encoding

**Labels stored as numbers.** A column of numbers can still be categories: MSSubClass 20, 60, 120 are dwelling types, and 60 isn't "three times" 20. Using it as a number (scaled or not) forces one straight line through the codes. Treat it as a category. *HP: MSSubClass.*

**One-hot encoding.** One 0/1 column per category, so each category gets its own weight. Coding categories as 0, 1, 2 instead invents an order and equal distances. *T: Embarked, Pclass. HP: MSZoning.*

**Reference category.** With one-hot, one value can be left without a column: its rows have all columns at 0, so they get no weight from this column, and each other weight means "difference from the reference". Usually the most common value. For a near-constant column, a single 0/1 column for the rare value keeps the weight away from everyone else. `OneHotEncoder(drop=...)`. *HP: Street, Alley, LotShape, LandContour, LotConfig.*

**Ordered categories.** A single 0, 1, 2, 3 column with one weight forces *equal steps* between neighbouring values. Use it only when the effect really moves in equal steps; having an order isn't enough. *HP: LotShape (steps +$43k, +$32k, −$6k).*

**Merging rare categories.** A weight learned from very few rows mostly learns those rows. Merge a rare value with its most similar neighbour, after checking they really are similar. *HP: IR2 + IR3, FR2 + FR3, Mod + Sev slope.*

**Target encoding.** Replace each category with the mean of the label for that category (one column instead of many). Because it's built from the label, it leaks unless done carefully (next three entries). `TargetEncoder` in scikit-learn does both fixes. It suits many unordered categories; it gives one weight on the *raw* group means, so the steps between groups keep their raw shape, which can be the wrong shape once other columns are in (GarageCars: raw steps biggest at 2 → 3 cars, because of quality; after quality, biggest at 0 → 1). *HP: MSSubClass, GarageCars.*

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

**Log of the target.** A model minimizes squared error on whatever y it's given. On raw prices, a $50k miss counts the same on a $100k and a $500k house, so expensive houses dominate the fit; on log prices, errors are relative (a 10% miss counts the same everywhere). Fitting log(y) also matches a metric scored on log prices, turns multiplicative effects (quality adds a %, not a fixed $) into additive ones a linear model can learn, reduces skew, and can never predict a negative price. Predict, then exp() back for the submission. *HP: SalePrice.*

**Scaling.** Changes a column's units, never its shape. Standardization (`StandardScaler`): (x − mean) / sd, the usual choice for linear models. Min-max (`MinMaxScaler`): to 0–1, sensitive to one extreme value. "Normalization" can mean either; scikit-learn's `Normalizer` is different again (each *row* to length 1). Order: transform, then scale. *HP: LotArea.*

**Why scale.** A penalty (Ridge, Lasso) charges every weight's size equally. A column in thousands needs a tiny weight and a 0/1 column a big one for the same effect, so the penalty would hit them unequally. Plain least squares, with no penalty, gives the same predictions with or without scaling: change the units and the best weight changes to match (weight 10 per sq ft = weight 10,000 per thousand sq ft). *HP: LotArea.*

## Statistics

**Uncertainty of a rate.** An observed rate p from n rows varies around the true rate by about ±2 × √(p(1 − p) / n) (95% of samples). Halving it takes 4 times the rows. It fails for tiny n or p = 0 or 1 (3 of 3 gives ±0); then compute directly: 3 of 3 survivors is still 12.5% likely at a true rate of 50%. *T: min_samples_leaf.*

**Uncertainty of a mean.** *When to use it:* whenever a decision rests on a group's mean or on the gap between two groups' means: whether a category is reliable enough for its own weight, whether two categories can be merged, whether a gap is real or chance. *What it tells:* if the gap between two groups is smaller than their uncertainty, the data can't tell them apart, so a separate weight would mostly learn noise; merge them (or don't trust the gap). If the gap is several times the uncertainty, it's probably real (though it can still come from other causes). Severe vs moderate slope: gap $8k, Sev uncertainty ±$48k, so merged. Small groups have large uncertainty, which is why they're the ones to merge or blend. *Not to confuse:* the uncertainty is not how far the group is from the overall mean (that's the gap being tested). It's how much the group's *own* mean would move with a different sample of its houses: BrDale is $77k below the overall mean, but its 16 townhouses are so alike ($83k–$125k) that any 16 would average within about ±$7k of $104k. *How:* ±2 × sd / √n, where sd is the spread of single rows around the mean. Averaging n independent rows divides the variance by n, so the sd of the mean (the *standard error*) is sd / √n; the mean of a sample is roughly normally distributed, so about 95% of samples land within 2 standard errors of the true mean. Severe slope: sd $86.7k, n = 13 → $86.7k / 3.61 = $24k → ±$48k. Rough for small n (the 2 should be about 2.2 at n = 13, and the sd itself is uncertain). The rate formula is the same idea: a rate is the mean of 0/1 values, whose sd is √(p(1 − p)). Code 20 (536 houses): ±$6.8k. Code 40 (4 houses): ±$76k. Six gravel-road houses: about ±$50k. *HP: MSSubClass, Street, LandSlope.*

**Other causes (confounding).** A gap between groups may come from what the group members have in common, not from the column itself: irregular lots sell for more because they're newer and bigger, cabins went with 1st class. Another column may already carry it. *T: Cabin, Ticket. HP: Street, LotShape, LandContour, LotConfig.*

**Checking what other columns explain.** "The other columns will handle it" can be checked: fit a simple model without the column (quality, size, age), and look at the mean leftover error (residual) per group. If a group's leftover is still clearly away from 0 (beyond its uncertainty), the other columns don't explain its gap and it needs its own weight. Leftover = actual − what the other columns predict; a group's average leftover is roughly what its own weight would still have to learn. Veenker sold for about 10% more than NAmes after quality, size, age and lot size were counted. Houses near parks looked $33–53k dearer, but their leftover was about 0: they're just bigger and better. Careful when a group sits entirely inside a group of the other column (every duplex is MSSubClass 90): subtracting the MSSubClass group average makes its leftover 0 by construction, which is not evidence. The real test is the part that varies *within* the other column's groups (inside vs end townhouses within the same PUD code). *HP: Neighborhood, Condition1/2, BldgType.* The shape of a column's effect can change too: OverallQual's raw steps were even (+21–35% per level), but the steps left in the model were small and uneven (+22% at the bottom, +7% in the middle, +11–14% at the top), so one-hot beat a single number column. *HP: OverallQual number vs one-hot.* **Simple example:** at $100 per sq ft, three houses without a pool (1,000–1,500 sq ft) sell for about their prediction (leftover ≈ $0), and three with a pool (2,000–3,000 sq ft) sell for about $8k above it. Raw, pool houses cost $135k more, but that's mostly size; in a model that already has size, the pool's weight is about $8k.

**Same evidence, same decision.** Two groups whose gaps are about the same size relative to their uncertainty should be treated the same way, both merged or both kept. Reasons like "it's the same kind of thing as X" or "I'd expect an effect" are expectations, not evidence. Feedr (−0.04 from Norm) and PosN (+0.03), both within their uncertainty. *HP: Condition1/2.*

**Compare like with like.** "98% of big lots are RL" (any neighbourhood) and "78% of IDOTRR is RM" (any lot) answer different questions, and neither describes a big lot in IDOTRR. Look for rows that match *both* conditions. *HP: MSZoning fill.*

**A single feature can have a big effect.** Living area or overall quality can move prices by hundreds of thousands. What makes a gap unreliable is few rows or other causes, not the size of the gap. *HP: Street.*

## Linear models

**One weight per column, shared by everyone.** A linear model adds weight × value for each column, with the same weight for every row. If an effect differs between groups, it needs its own column (an interaction), or the weight learned from one group is applied to another. *T: is_male_adult, group fate for women and boys only.*

**Additive on the log-odds scale.** Logistic regression adds weights in log-odds, so without interactions the class effect is the same for men and women. *T: Pclass interaction.*

**Dates and ages in a linear model.** "Today − year" is the year flipped and shifted: same predictions, only the weight's sign changes. Age at sale (YrSold − YearBuilt) does differ, but a single age weight can't express a market trend: a later sale makes the house "older", which lowers its price. A market effect needs its own column (YrSold), and should be checked first: Ames prices were flat 2006–2010 (yearly leftovers within ±0.02), not +5% a year. *HP: YearBuilt, YrSold.*

**Least squares and outliers.** Minimizing *squared* error lets one far-off row pull the line: an error of $463k squared outweighs hundreds of ordinary rows, and a row at the far end of the x-axis tilts the line most. Two houses of 1,460 moved the price-per-sq-ft slope by 7%. *HP: Ids 524, 1299.*

**Overlapping (correlated) columns.** They don't cause underfitting. The model can't tell which column deserves the credit, so it splits the weight between them, sometimes unstably. Predictions usually stay fine; the weights become hard to read. Ridge spreads the credit. Overlap is a keep-or-drop question, not an encoding question. *HP: LotShape, LandSlope vs LandContour.*

**Near-constant columns.** A column that is one value for 99.6% of rows can only change the predictions of the few rare rows. A single weight learned from 6 rows can still overfit them; coding the common value as 0 keeps that away from everyone else. If no test row has the rare value, the column can't help at all: drop it. *HP: Street, Utilities.*

## Regularization, bias and variance

**Regularization direction.** λ is the penalty strength. scikit-learn's `LogisticRegression` uses `C = 1 / λ`: a *larger* C means *weaker* regularization. Ridge and Lasso use `alpha`, which is λ itself. *T: logistic regression settings.*

**The reference category under a penalty.** Without a penalty, the choice of reference changes only what the weights mean; predictions are identical. With a penalty (Ridge), every one-hot weight is pulled towards 0, and 0 means "priced like the reference", so each group's prediction is pulled towards the reference's level, roughly (n × group mean + λ × reference mean) / (n + λ): the blending formula, with the reference as the anchor and λ as m. Big groups barely move; small groups move a lot, towards whichever group is the reference. Choose a large, typical group, or keep a column for every value (with a penalty there's no "dummy variable trap"), so that the pull goes towards the intercept instead of one particular group. *HP: Neighborhood.*

**How much a small group keeps under the penalty.** For one 0/1 column for a group of n houses whose average leftover is r, Ridge minimizes (squared leftovers of the group after adding w) + α × w², which gives w = n / (n + α) × r: the group keeps n / (n + α) of its correction, the rest is pulled back to the reference (w = 0). At α = 6: 4 houses keep 40%, 14 keep 70%, 100 keep 94%. Scaling a 0/1 column makes its weight small for the same correction, so the penalty barely touches it. *HP: scaling all 0/1 columns made RidgeCV raise α from 6 to 25 and scored slightly worse (0.1120 vs 0.1118): the small groups' corrections need holding back.*

**Underfitting (bias) vs overfitting (variance).** Underfitting: too simple for the pattern, so training and validation are both low and close together. Overfitting: learns the training rows' quirks, so training is high and validation clearly lower. Diagnose by comparing them. Regularization fights overfitting; for an underfitting model it can only help by being weakened. *T: logistic regression (0.830 training, 0.827 validation). HP: the 4-column straight line has RMSE 0.1609 on its own houses and 0.1611 in CV (4 weights, no gap); the baseline has 0.1402 and 0.1453 (about 40 weights, small gap). The gap grows with the number of weights relative to the number of rows.*

**Feature ceiling.** When rows with identical inputs have different outcomes, no model can get them all right. Titanic's 11 features made 43 passenger types and capped accuracy at 0.838. When different model types stop at the same score, the limit is the information in the features. *T: error analysis. HP error analysis: abnormal sales ("trade, foreclosure, short sale" in one value) are right on average once they have a column (leftover −0.3), but spread twice as much as normal sales (sd 19 against 9); the data can't tell a foreclosure sold at half price from a trade at market price, so those errors stay.*

## Trees

**Decision tree.** Splits on thresholds (`Age <= 6.5`), with different thresholds in different branches, so curved effects and interactions are no problem. `min_samples_leaf` limits how small a leaf can be: a small leaf can learn the wrong rule by chance. *T.*

**Regression tree.** A leaf predicts the **average** target of its training rows (one number for a group: the average gives the smallest squared error). A split is chosen by trying every column and threshold and keeping the one with the smallest total squared error, each row measured against its own side's average: the sides should be as alike inside as possible (`criterion='squared_error'`, the regression counterpart of Gini purity). Example, 4 houses (800 sq ft $100k, 1,000 $120k, 2,000 $250k, 2,200 $270k): no split 22,900; size ≤ 900 13,267; **size ≤ 1,500 400** (sides average $110k and $260k); size ≤ 2,100 13,267. *HP: before the tree model.*

**Feature engineering for trees.** A split only uses the order of a column, so anything that keeps the order changes nothing: no log of input columns, no scaling (and no weights, so no penalty to balance). No one-hot reference levels either (nothing pulls groups toward a reference), and `min_samples_leaf` mostly replaces merging small groups; uneven steps are no problem. **The log of the target still matters:** the tree minimizes squared error, and in dollars a 20% miss on a $500k house ($100k) counts 25 times as much as on a $100k house ($20k); in log price both are about 0.18. *HP: before the tree model.*

**Random forest.** Averages many trees, each trained on a bootstrap sample of rows, with a random subset of columns per split. It reduces variance (overfitting) but can't add information the columns don't have. *T.*

**Gradient boosting.** Trees are built one after another: tree 1 predicts the target; tree 2 is trained on tree 1's **leftover**; tree 3 on what trees 1 and 2 together still get wrong, and so on. The predictions are **added** (tree 1 + tree 2 + …), each correction scaled down by the **learning rate** (for example ×0.1): many small steps generalize better than a few big ones. Example: A $100k, B $150k, C $300k; tree 1 predicts 125, 125, 300; leftovers −25, +25, 0; tree 2 learns exactly those, and the sum is right. Unlike a random forest (independent trees, averaged, mainly fixing overfitting), boosting mainly fixes underfitting, step by step. XGBoost is gradient boosting; weighting the wrongly predicted examples is AdaBoost, an older method. *HP: before the tree model.*

**Validation curve.** Score against one setting (such as `min_samples_leaf`). Try values on both sides of the best, reaching both the overfitting and the underfitting end, in doubling steps. *T: values 2, 5, 10, 20, 40, 80.*

## Evaluation

**Cross-validation.** Split the working data into folds; each fold takes a turn as validation while the others train. Repeating it with different shuffles (10 × 5 = 50 folds) averages out lucky folds. A fixed seed means two versions are compared on the same folds. Fold-to-fold scores varied by about ±0.03, so small gains in one run may be luck. *T.*

**Compare fold by fold (paired).** Like two runners timed on the same 50 routes: hilly routes slow both, so judge by who wins on each route, not by how much each runner's times vary. A fold's score depends mostly on which hard houses landed in it, and that hits both models alike. So compare the two models' scores fold by fold: the differences vary far less than the scores themselves. *HP: fold scores ±0.94 (×100), fold-by-fold difference ±0.12.*

**Repeating columns add little.** A column's gain is what it adds that the model doesn't already know. A column that explains a lot on its own can add almost nothing if another column already says it. *HP: ExterQual ranks top 3 alone, but adds 0.05 (×100) on top of OverallQual. TotalBsmtSF, information the model didn't have, adds 0.78. LotShape adds 0.00: irregular lots are mostly newer and bigger, which Age, LotArea and GrLivArea already say.*

**Test set once.** The held-out test set is scored once, at the end. Choosing models on it makes it a second validation set and its score optimistic. *T.*

**Keep rule, written first.** Decide the rule for keeping a feature (minimum gain, share of folds won) *before* looking at the result, and write down any change to it without referring to the result. Adjusting the rule after seeing the result is how you fool yourself. For a targeted feature that can only fix a few errors, consistency across folds is the better test than a fixed gain. *T.*

**Pipeline inside CV.** Anything learned from data (a scaler's mean, the one-hot levels, target-encoding means, a chosen alpha) must be learned from the training part of each fold only. Put every such step in one `Pipeline` (with a `ColumnTransformer` to send each column to its own step) and pass the pipeline to `cross_validate`, which copies it and refits all of it per fold. Fixed rules that need no data (log, merging levels, filling one known Id) can be done once beforehand. Don't stratify the folds for a continuous target: stratifying needs classes. *HP: baseline.*

**Tuning inside CV (nested).** `RidgeCV` tries a list of alphas on the training part it's given (with a fast leave-one-out shortcut) and keeps the best. Inside a pipeline inside `cross_validate`, each outer fold picks its own alpha without seeing its validation part, so the outer score stays honest. Picking alpha on the outer CV score itself and reporting that same score would be optimistic. Check that the chosen alphas aren't at the edge of the list; if they are, widen it. *HP: baseline, alpha 0.4–0.8 from 0.001–1000.*

**A decision depends on the other columns.** A comparison is made on a particular model; when columns are added or dropped, the result can change. Comparisons made before a big change may need re-running. *HP: OverallQual one-hot beat a number column by 0.05 (×100); with TotalBsmtSF added, the edge was gone (0.00) and the simpler number column was kept.*

**Simpler version as current.** In a comparison, the version with fewer columns is the current one, so close calls go to the simpler model and every extra column has to earn its place, like a new column does. *HP: keep rule.*

**Extrapolation is not overfitting.** Overfitting: the model learned quirks of its training rows, seen as a clear gap between the error on its own rows and in CV. Extrapolation: a row outside the training range; a linear model continues its straight lines where there's no data to say whether they still hold (timing a runner up to 10 km and assuming the same pace for a marathon). A guard: clip each numeric test value to the training range before predicting. *HP: Id 2550, basement 5,095 sq ft vs training max 3,206, predicted $1.1M, $695k clipped; own houses 0.1198 vs CV 0.1236, so no overfitting.* **Trees don't extrapolate:** every prediction is a leaf average of training rows, so beyond the training range a tree stays flat at the values it has seen (no prediction above the most expensive training house). That's built-in clipping, helpful for a house like Id 2550, but a limit when a trend really continues beyond the data (bigger houses still pricier, prices rising each year).

**Compare with the base rate.** In a selection (the worst-predicted houses, say), a value being common is only a pattern if it's more common than in all the data. *HP error analysis: RL is 67% of the 15 worst houses but 79% of all houses, so no pattern; commercial zone (C) is 27% of the worst but 0.7% of all, and abnormal sales 47% against 7%: patterns.*

**Outlier removal and CV.** Removing hard rows from the validation folds too makes scores look better just because the hard rows are gone. To measure whether removing them helps, remove them from the training part of each fold only. *HP: Ids 524, 1299.*

**When to stop.** Signals: clearly above a simple baseline; much of the remaining error is chance (same inputs, different outcome); different models agree; gains below the fold noise; and validation wear: every comparison made on the same rows makes the CV score a little more optimistic (multiple comparisons). *T.*
