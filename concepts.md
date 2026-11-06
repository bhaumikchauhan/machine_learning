# ML concepts

Concepts that came up while working on the problems in this repo, each in a few lines: what it is, and where it came up. Mostly the ones that were new, or that I had wrong at first. The per-problem `feature_notes.md` files record the decisions column by column; this file collects the ideas behind them.

Where: **T** = [Titanic](titanic/feature_notes.md), **HP** = [House Prices](house-prices/feature_notes.md), **GMSC** = Give Me Some Credit (`give-me-some-credit/`), **CC** = [credit card clustering](credit-card-clustering/feature_notes.md), **DR** = [Digit Recognizer](digit-recognizer/feature_notes.md).

## Lessons from Digit Recognizer

Final result: a CNN (two conv blocks, 32 and 64 filters of 3 × 3, max pooling, then 3,136 → 128 → 10), best of 30 epochs on an early-stopping set: 98.84% held-out over 5 seeds, Kaggle 0.98960. Logistic regression 91.31%, dense network 96.55%.
1. **Neighbourhoods matter for images.** A template per digit (logistic) and a dense layer see each pixel separately; filters see local shape, so the CNN cut the dense network's mistakes by more than half (290 → about 100).
2. **One run is one draw.** Network results vary with starting weights and batch order: seed before building the network, make cuDNN deterministic, and compare 5-seed averages.
3. **Size the keep rule from the noise of a difference.** Only images where two models disagree move the difference: sd ≈ √(disagreements) ÷ n. Near 99% that gave +0.3 instead of +0.5; check the disagreement count in each comparison.
4. **Don't move the rule after the result.** Early stopping (+0.38 vs +0.5), voting (+0.23) and augmentation (+0.27) were all probably real and all not kept. A procedure can change for reasons independent of a result (early stopping became the default because a fixed epoch count doesn't carry over to new networks).
5. **Choose on the early-stopping set, judge once on the held-out set.** Epochs, the voting method and the error analysis all used the early-stopping images, so the held-out 8,400 stayed for final decisions.
6. **Look at the mistakes before predicting a fix.** Many of the CNN's mistakes look like the predicted digit to a person too: no model fixes a label that disagrees with the drawing.

## Lessons from credit card clustering

Final result: k-means, k = 6, on all 17 columns (amounts logged, all scaled, MINIMUM_PAYMENTS blank = 0, `n_init=10`). Six named segments: only cash borrowers 25%, frequent instalment buyers 20%, high usage buyers 16%, moderate users 16%, cash and credit users 14%, inactive 10%.
1. **Without a target, state the goal first.** The dataset had no problem statement, so "segments for a marketing strategy" was ours, and it set how the clusters were judged: the elbow, no cluster under 1%, and every cluster getting its own name and action.
2. **The preprocessing defines "similar".** Distances follow the biggest numbers, so scaling is required; a log changes gaps from dollars to ratios. No score says which is right, so the choice is made on purpose and applied to every amount column alike.
3. **Inertia can't pick k.** It always falls. The elbow narrowed k to 6–8 softly; the profiles decided: 7 and 8 only added a tenure cluster and splits that mixed buying with cash.
4. **Profiles, in original units and medians, are the real check.** They caught a misread column (cash advance is borrowing cash, not paying in advance) and showed which splits are actionable.
5. **Test the choices that could matter, and keep the simple one when they don't.** The KNN fill moved 3.3% of customers and left the segments alike; one bad start moved 1,793, so `n_init=10`.
6. **ColumnTransformer entries run side by side, Pipeline steps in order:** fill → log → scale on one column is a Pipeline inside one entry.
7. **A second method shows which segments are solid.** Ward put 25% of customers elsewhere (predicted 5%); the distinct segments agreed (82–89% stayed), the in-between ones didn't (53–61%). A border flag from the k-means distances (nearest within 10% of the second nearest) marked 15% of customers, whom Ward moved 3× as often; most of Ward's moves still came from its different method.
8. **DBSCAN found no gaps, at any setting.** One cluster at eps 2.5 (from the k-distance curve), and no eps from 1.0 to 3.5 gave two big clusters without half the customers as noise. So the segments are useful cuts through a continuum. When a setting can't be scored, check whether the conclusion depends on it, with the overturning rule written first.

## Lessons from Give Me Some Credit

Final model: gradient boosting on the raw columns. CV AUC 0.865; Kaggle private 0.86670 (logistic regression: 0.853 CV, 0.85386 Kaggle).
1. **Pick the score for the problem.** With 6.7% defaulters, accuracy rewards "always no" (93.3%). AUC judges only the ranking; precision, recall and the cut-off come later, as a business choice.
2. **Find the impossible values first.** Codes stored as numbers (96/98 late counts, $1 incomes), a ratio that is really dollars when income is missing, utilization of 50,708. Each became a 0/1 flag plus a fixed fill, checked against the default rate.
3. **Read the setting the search picks.** C at the edge of every grid showed that AUC goes flat under a strong penalty (only the order matters), and that this frozen model couldn't use new columns.
4. **Check what the model already knows before adding a column.** Open credit lines looked strong alone, but the model already ranked those borrowers as risky; real estate loans were missed, but the tiny C blocked them.
5. **Shapes decide the model.** U shapes and steps that level off defeat one straight line per column. Boosting learned them on its own and won without any feature engineering: the opposite of House Prices, where the effects were near-straight.
6. **CV held again:** Kaggle moved by the same amount as CV (+0.012).

## Lessons from House Prices

Final model: Ridge. CV went from 0.1453 to 0.1118; Kaggle 0.12412 (rank 1,215).
1. **Decide before running.** Write the prediction and the keep rule (gain ≥ 0.1 ×100 and ≥ 80% of folds) first, compare each candidate with the version in place on the same 50 folds, and keep the simpler version on a tie. A test chosen after looking at the results is weak.
2. **The leftover guides the columns.** A group's average leftover shows what a column could fix; the spread is what it can't fix. Many of the gains came from cleaning columns (fills, merges, flags), not from adding more.
3. **Feature engineering depends on the model.** Logs, flags and merged groups took Ridge from 0.145 to 0.112, but gave boosting almost nothing (raw 0.1240, engineered 0.1234). Trees find their own thresholds.
4. **Match the model to the shape of the data.** The main effects (size, quality, age) are close to straight lines that add up, so Ridge beat boosting. A blend with a small boosting weight helped a little (0.1100 at 0.3, chosen after the fact).
5. **Some error can't be removed.** The worst 1% of houses carried 26% of the squared error, mostly unusual sales that no column can explain: a feature ceiling.
6. **Trust CV over the leaderboard.** Kaggle stayed about 0.012 above CV for both submissions, so CV ranked the changes correctly. The scores of 0 on the leaderboard are leaks.

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

**Flag plus a fixed fill.** With a 0/1 flag in a linear model, the fill value for the flagged rows barely matters: they all get the same income part, the flag's weight learns the rest, and only the sum counts (fill $5,000: −0.5 + 0.8; fill 0: 0 + 0.3; both +0.3). A fixed number needs no fitting, so nothing to do out of fold; a median does. Codes stored as impossible numbers (96/98 as a count, $1 as an income) are treated the same way. *GMSC: LateCountsCoded, IncomeNotReported, UtilizationInvalid.*

**Capping for a straight line.** A linear model gives a column one slope. Values far out where the effect stops rising stretch that line and bend the weight for the ordinary rows, so cap where the effect stops. A number is a position on the line, not a group. *GMSC: utilization capped at 2, not 10.*

**Filling from related columns.** Fill a missing value from columns that predict it: age from title, frontage from lot size and lot shape, zone from neighbourhood. Check the relationship holds for the rows being filled (cul-de-sac lots are narrower for their size). *T: Age. HP: MSZoning, LotFrontage.*

**Leaking labels vs leaking features.** A fill computed from features only (mean age per title) can use all rows, even test rows: no label is involved. A value computed from the label (survival rate, mean price) must use training rows only. *T: fill before the split.*

**Options for missing values in trees.** Fill + flag, sentinel value (−1), or leave NaN (scikit-learn trees can route missing values at each split). A single fill value creates a spike that a tree can split around. *T: Age for trees.*

## Transforming and scaling

**Log transform.** For a lopsided (skewed) column, log turns equal *ratios* into equal steps (5k → 10k is the same step as 50k → 100k). It spreads out the bulk of the data and stops a few huge values from driving a straight-line fit. It changes the shape; scaling doesn't. *HP: LotArea.*

**Log of the target.** A model minimizes squared error on whatever y it's given. On raw prices, a $50k miss counts the same on a $100k and a $500k house, so expensive houses dominate the fit; on log prices, errors are relative (a 10% miss counts the same everywhere). Fitting log(y) also matches a metric scored on log prices, turns multiplicative effects (quality adds a %, not a fixed $) into additive ones a linear model can learn, reduces skew, and can never predict a negative price. Predict, then exp() back for the submission. *HP: SalePrice.*

**Scaling.** Changes a column's units, never its shape. Standardization (`StandardScaler`): (x − mean) / sd, the usual choice for linear models. Min-max (`MinMaxScaler`): to 0–1, sensitive to one extreme value. "Normalization" can mean either; scikit-learn's `Normalizer` is different again (each *row* to length 1). Order: transform, then scale. *HP: LotArea.*

**Why scale.** A penalty (Ridge, Lasso) charges every weight's size equally. A column in thousands needs a tiny weight and a 0/1 column a big one for the same effect, so the penalty would hit them unequally. Plain least squares, with no penalty, gives the same predictions with or without scaling: change the units and the best weight changes to match (weight 10 per sq ft = weight 10,000 per thousand sq ft). *HP: LotArea.*

**ColumnTransformer runs side by side, Pipeline runs in order.** A ColumnTransformer entry is (name, step, columns): every entry gets the *original* columns and the outputs are placed next to each other. A Pipeline step is (name, step): each step gets the previous step's output. So "fill, then log, then scale" for one column is a Pipeline inside one ColumnTransformer entry, and each column belongs to exactly one entry. Separate entries for fill, log and scale on the same column give three copies, two of them still blank. Columns not named in any entry are dropped. *CC: KMeans "Input X contains NaN"; then 25 columns instead of 17. GMSC: flags dropped until a passthrough entry was added.*

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

**Polynomial features (not tried).** A squared column lets a linear model bend while staying linear in its weights: risk = w₁ × lines + w₂ × lines² + b is a U when w₁ < 0 and w₂ > 0. `PolynomialFeatures` adds them (also products x × y, which are interactions). Catches: x² of a skewed column explodes, so cap or scale first; the curve keeps rising past the data (extrapolation); and many new columns need regularization. Flags, caps and bins are the hand-made alternative; trees learn the shape without any of this. *GMSC: open credit lines and real estate loans were U-shaped; we used flags and caps, then boosting.*

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

**Feature engineering for trees.** A split only uses the order of a column, so anything that keeps the order changes nothing: no log of input columns, no scaling (and no weights, so no penalty to balance). No one-hot reference levels either (nothing pulls groups toward a reference), and `min_samples_leaf` mostly replaces merging small groups; uneven steps are no problem. **The log of the target still matters:** the tree minimizes squared error, and in dollars a 20% miss on a $500k house ($100k) counts 25 times as much as on a $100k house ($20k); in log price both are about 0.18. **What does help trees: columns that combine columns.** A split looks at one column at a time, so a sum, difference or ratio of two columns takes a staircase of splits on both, while one new column does it in a single split. Examples: total area (above ground + basement), age (year sold − year built), debt / income. Fixing wrong values (a garage built in 2207) helps any model. *HP: before the tree model; the engineered columns (mostly logs, flags and merges) gave boosting almost nothing (0.1234 against 0.1240 raw). The in-between option, raw columns plus a few combined ones, wasn't tried.*

**Random forest.** Averages many trees, each trained on a bootstrap sample of rows, with a random subset of columns per split. It reduces variance (overfitting) but can't add information the columns don't have. *T.*

**Gradient boosting.** Trees are built one after another: tree 1 predicts the target; tree 2 is trained on tree 1's **leftover**; tree 3 on what trees 1 and 2 together still get wrong, and so on. The predictions are **added** (tree 1 + tree 2 + …), each correction scaled down by the **learning rate** (for example ×0.1): many small steps generalize better than a few big ones. Example: A $100k, B $150k, C $300k; tree 1 predicts 125, 125, 300; leftovers −25, +25, 0; tree 2 learns exactly those, and the sum is right. Unlike a random forest (independent trees, averaged, mainly fixing overfitting), boosting mainly fixes underfitting, step by step. XGBoost is gradient boosting; weighting the wrongly predicted examples is AdaBoost, an older method. *HP: before the tree model.*
- **The learning rate** is one number for the whole model: every leaf of every tree is scaled by it. Leaves differ in their value (the average leftover of their rows). `l2_regularization` does treat leaves differently: it shrinks each leaf's value by n / (n + λ), the same formula as Ridge's pull on small groups (default 0, off).
- **Number of trees:** the training error keeps falling with every tree (a leaf's average leftover is the value that minimizes its group's squared error, even if single rows get worse). The validation error falls, reaches a lowest point, then rises, as later trees fit noise. **Early stopping** holds back part of the training data and stops adding trees once it stops improving. A smaller learning rate needs more trees but usually reaches a slightly lower point.

**Trees on a straight-line effect.** A tree turns a steady rise into a staircase: each step is one leaf's average, and with `min_samples_leaf` 20 a step needs at least 20 houses. Ridge draws the line with one weight. When most effects are close to straight lines and add up (size, quality, age), and the data is small, the linear model wins; trees win when effects bend, depend on each other, or the columns aren't prepared. Fill values are a smaller issue: a tree can split a filled value (missing MasVnrArea = 0) away from the real ones, but it can't tell them apart from real zeros; left as missing, a tree can send them to their own side. *HP: boosting 0.1234 against Ridge 0.1118 on the Ridge columns.*

**Leaderboard leak.** A score that is far too good usually means the answers were available, not that the model was better. House Prices' test houses come from the public Ames dataset (De Cock, 2011), which includes their sale prices, so a submission of the real prices scores 0. Honest scores on this competition are around 0.11 to 0.12; compare with those, and with your own CV. *HP: entries at 0 on the leaderboard.*

**Blending.** Average two models' predictions: (1 − w) × model 1 + w × model 2. Where the two are wrong in different directions, the errors partly cancel. Example: actual 12.0; Ridge says 12.1, boosting 11.95; the 70/30 blend says 12.055, closer than either. A weaker model still helps with a small weight; with too much weight it pulls the blend toward its own error. Choosing w on the validation folds flatters it, like any setting chosen there. With several models, the same idea applies with one weight per model. **Stacking** learns the weights: a simple model (often a linear one) is fitted on the models' out-of-fold predictions, which is the honest way to choose them. *HP: Ridge 0.1118, boosting 0.1240; 50/50 0.1116, 70/30 0.1100.*

**Validation curve.** Score against one setting (such as `min_samples_leaf`). Try values on both sides of the best, reaching both the overfitting and the underfitting end, in doubling steps. *T: values 2, 5, 10, 20, 40, 80.*

## Evaluation

**Accuracy with uneven classes; precision and recall.** With 6.7% defaulters, always saying "no default" scores 93.3% accuracy and catches nobody. Accuracy counts both mistakes as equal, but a missed defaulter (the loan is lost) usually costs a bank far more than a wrongly declined good borrower (the interest is lost). **Recall** is the share of real defaulters the model catches; **precision** is the share of flagged borrowers who really default. Example, 1,000 borrowers with 67 defaulters: model X flags 10, all real (accuracy 94.3%, recall 15%, precision 100%); model Y flags 150, of whom 50 are real (accuracy 88.3%, recall 75%, precision 33%). The bank prefers Y despite its lower accuracy. *GMSC: before the first model.*
- **Confusion matrix:** TP (flagged, defaults), FP (flagged, doesn't), FN (not flagged, defaults), TN (not flagged, doesn't). Precision = TP / (TP + FP), recall = TP / (TP + FN), accuracy = (TP + TN) / all.
- **The cut-off:** a classifier outputs a probability, and the cut-off turns it into yes/no. X and Y can be the same model: a high cut-off flags few borrowers (precision up, recall down, like X); a low cut-off flags many (recall up, precision down, like Y). Choosing the cut-off is a business decision about what each mistake costs.

**AUC.** The chance that a random positive gets a higher probability than a random negative; it uses only the order of the probabilities, so it needs no cut-off. Example: defaulters at 0.9 and 0.3, good borrowers at 0.4, 0.2 and 0.1: 5 of the 6 pairs are in the right order, so AUC = 0.83. 1.0 means every positive ranks above every negative; 0.5 is a coin flip; "always no" gives everyone the same probability and scores 0.5, despite 93.3% accuracy. *GMSC: the Kaggle score.*
- **AUC ignores the scale:** halving every score (4, 2, 1 → 2, 1, 0.5) keeps the order, so AUC stays the same. A very strong penalty that shrinks all weights together therefore leaves AUC flat, and tuning C by AUC can end at the edge of any grid. *GMSC: inner AUC 0.8223 for every C from 1e-09 to 1e-05.*

**A chosen setting at the edge of the grid.** The best value may lie beyond it: extend the range until the best is inside, or until the score is flat (then any value there is equivalent). *GMSC: C at the smallest value in every fold.*

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

## Clustering

**Distances need scaling.** k-means groups customers by distance, summed over all columns, so the column with the biggest numbers decides. Purchases $1,000 against $1,100 and frequency 0.1 against 0.9: distance √(100² + 0.8²) ≈ 100, all from dollars, though the frequency gap is the largest possible. After StandardScaler (gaps in sd: 0.05 and 2.0) frequency decides. *CC.*

**Outliers grab clusters.** k-means minimises the sum of *squared* distances, so one customer 22 sd from the rest adds 484 against about 1 for a typical one. Giving them their own centre cuts the total more than splitting hundreds of ordinary customers, so a cluster gets spent on a handful of extremes. This is the objective doing it on purpose, not a local minimum (a bad start, which `n_init` handles). *CC: PURCHASES, biggest customer 22.5 sd raw, 2.0 sd after log(1 + x).*

**The scaling defines "similar".** After a log, gaps are ratios: $0–$20, $20–$420 and $420–$8,800 are each about 1 sd apart (raw: 0.01, 0.19, 3.92 sd). Raw says small dollar gaps don't matter; log says doubling matters the same at any level. With no target, no score says which is right; pick one, apply it to every amount column alike (or the unlogged ones dominate), and judge the clusters it gives. *CC.*

**Inertia always falls with k.** Inertia (the sum of squared distances to each point's centre) is k-means' own score; at k = number of points it is 0. So the lowest inertia isn't the goal. The elbow (where the curve flattens), cluster sizes (no tiny clusters) and whether the segments are usable downstream pick k. *CC.*

**Cluster profiles.** To judge clusters, describe each in the original units (dollars, frequencies), not in sd: its size and the median of key columns (the median, because one extreme customer pulls a mean far from the typical member). A k is usable when every cluster gets its own name and its own action; when a higher k only splits a cluster on a small difference ($900 vs $1,100 balance), it's too high. *CC: k = 6 gave six named segments; 7 and 8 added nothing usable.*

**Clustering mixes every column.** One distance over all columns makes each cluster a mix of behaviours (buying, cash, payments). If the user of the segments thinks in separate questions ("frequent buyer?", "borrows cash?"), segmenting each dimension on its own and crossing them is an alternative, at the cost of many small cells. *CC.*

**No target, so state a goal.** A dataset without a problem statement gives no way to say which clusters are right. Write down what the segments are for before judging them. *CC: "segments for a marketing strategy".*

**PCA.** New axes, each a weighted mix of the columns, ordered by how much of the spread they capture; there are as many as columns, and you keep the first few. Run it on scaled columns, or the biggest-numbered column becomes axis 1. `explained_variance_ratio_` gives each axis's share of the spread, and `components_` its weight on every column, which is how you name an axis. Used here only to draw clusters in 2-D: overlap in the picture doesn't mean overlap in all columns, since the rest of the spread isn't shown. *CC: 56% in 2 axes; axis 1 = buy (+) or borrow cash (−), axis 2 = balance and payments.*

**k-means local minimum and `n_init`.** k-means moves the centres step by step from where they start, and stops when no step helps; a bad start can stop at a poorer arrangement. With `init='k-means++'` scikit-learn starts only once by default; `n_init=10` runs 10 starts and keeps the lowest inertia. Check: fit with a few seeds and compare inertia. *CC: 5 seeds gave 70,861 ± 1, seed 3 gave 73,882 and moved 1,793 customers; best of 10 gave 70,860.*

**Hierarchical (agglomerative) clustering.** Start with every point as its own cluster, then repeatedly merge the two closest clusters until one is left; no k, no centres, no random start. The merge history is the dendrogram, and k is chosen afterwards by cutting it at a height. The linkage defines the distance between two clusters: single = closest pair of points, complete = farthest pair, average = mean over all pairs, Ward = increase in inertia. Example on a line, A=1, B=2, C=6, D=7: {A,B} to {C,D} is 4 with single linkage (B–C) and 6 with complete (A–D). *Studied after CC.*

**Single linkage chains.** A chain of points each close to the next (1, 2, 3, … 20) merges into one cluster early under single linkage, since only the nearest pair counts; a few "bridge" points can join two real groups. Complete linkage resists, because its farthest-pair distance grows with the chain.

**Border customers.** `KMeans.transform` gives each point's distance to every centre; when the nearest is barely closer than the second (ratio nearest ÷ second near 1, at most 1 by definition), the assignment is a close call. Use a relative rule, since loose clusters have larger distances. *CC: ratio > 0.9 flagged 15%; Ward moved 57% of them against 19% of the rest.*

**Comparing two clusterings.** Cluster numbers are arbitrary, so match them first (each cluster to the one it shares most points with: `scipy.optimize.linear_sum_assignment` on the crosstab), then count who changed. Compare with the change from a different seed alone, the "noise". *CC: KNN fill 3.3% vs seeds 0.1–1.6%; Ward 25%.*

**DBSCAN.** Two settings: `eps` (the neighbourhood radius) and `min_samples` (points needed within `eps`, the point itself included). A **core** point has at least `min_samples` neighbours; a **border** point isn't core but lies within `eps` of a core point; everything else is **noise**. Clusters grow through core points: two core points within `eps` of each other are in the *same* cluster, so a cluster is a chain of overlapping neighbourhoods, not "every pair within `eps`". There is no k and no centres. In scikit-learn, `labels_` is −1 for noise, `core_sample_indices_` lists the core points (border = not core and not −1), and each point gets one label: a border point reachable from two clusters goes to whichever reached it first. *Studied after CC.*

**One DBSCAN cluster means no gaps, not "all close".** DBSCAN separates groups only where density drops. One cluster means a path of dense neighbours links any two points, however far apart; segments found by k-means in such data are cuts through a continuum, useful but with borders that are a choice. Its noise marks far-out points, not the points between segments. *CC: eps 2.5 and min_samples 34 gave 1 cluster and 330 noise (62% from the inactive segment); k-means border customers were noise 3× as often, but most noise sat twice as far from any centre.*

## Neural networks

**PyTorch and TensorFlow do the same job.** Both define layers, compute gradients automatically and train on a GPU. Keras hides the training loop in `fit`; in PyTorch you write it: predict, loss, `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`. Deep learning = neural networks with many layers.

**Gradient, then step.** `loss.backward()` computes the gradient: for each weight, how much the loss changes if that weight rises a little (backpropagation does it through all layers at once). `optimizer.step()` moves each weight against its gradient by learning rate × gradient. Example: x = 2, y = 10, w = 3: prediction 6, loss 16, gradient 2 × (6 − 10) × 2 = −16; at learning rate 0.1, w becomes 4.6 and the loss 0.64.

**Learning rate too big: overshoot, then blow up.** At learning rate 0.5 the same step jumps past the minimum to w = 11 (loss 144), then to w = −13 (loss 1,296): each jump lands farther away, so the loss grows instead of shrinking. Too big can diverge, not just slow down.

**Scale inputs for gradient descent.** The gradient of a weight contains its input, so inputs of 255 give steps 255 times bigger than inputs near 1, and a learning rate fit for one diverges on the other. For images, divide all pixels by 255: StandardScaler per pixel turns a rare stray mark in a nearly always blank pixel into a huge input. *DR: pixel14, inked in 1 of 42,000 images, would become 205 sd; a centre stroke 1.1 sd.*

**ReLU keeps layers from collapsing.** ReLU = max(0, x), applied after each hidden layer. Without it, two linear layers multiply out to one linear layer (back to logistic regression); with it, each hidden unit can switch off, so the next layer combines partial patterns. The output isn't ReLU'd: it's 10 raw scores (logits), one per digit. `CrossEntropyLoss` applies softmax itself; `argmax` of the scores is the predicted digit. *DR: 784 → 128 → 10, 91.31% → 96.60%.*

**Mini-batches and epochs.** One update = predict a batch (128 images), average its loss, `backward`, `step`. One epoch = one pass over all training images, here 26,880 ÷ 128 = 210 updates. Batch size trades noise for count: batch 1 gives 26,880 noisy updates per epoch, the whole set gives 1 exact but tiny update (10 epochs = 10 steps, too few). The loss is averaged per batch, so the epoch loss multiplies by `len(batch)` before dividing by the image count, or a short last batch (33,600 = 262 × 128 + 64) counts as much as a full one.

**Where the gradients live.** `loss` and the optimizer never talk directly. The forward pass records every operation back to the weights; `loss.backward()` walks that record and writes each gradient into the weight's own `.grad`; the optimizer holds the same weight tensors (`network.parameters()`) and reads `.grad` in `step()`. `backward` *adds* to `.grad`, so `zero_grad()` clears it each batch, otherwise every update follows the sum of all past batches' gradients.

**Early stopping set.** Training accuracy keeps rising with epochs (100% by epoch 27) while unseen accuracy flattens, then may fall. Choosing the best epoch on the held-out set makes its score the best of 30 looks (validation wear), so the epoch is chosen on a set split off the training part and the held-out set judges once. *DR: 33,600 → 26,880 to train, 6,720 to stop.*

**Seed the network's starting weights.** Two runs of a network differ by their random starting weights and batch order. `torch.manual_seed` must run *before* the network is built, or only the batch order is fixed. Seeded, the same seed gives identical runs; for a comparison, average several seeds, since one run is one draw. *DR: built before seeding, three runs gave 96.74–97.04%; across seeds 0–4, sd 0.11–0.14 points.*

**Don't change the rule after the result.** The keep rule is written first so the result can't shape it. Moving it after seeing a near miss is choosing whatever passes: fitted to the held-out set, and false confidence. A rule can be revised, for the next comparison. *DR: early stopping +0.38 against +0.50 (5 seeds, gain several times the noise): not kept.*

**Convolution filters and ReLU.** A filter (for example 3 × 3 weights) slides over the image; each position gives a weighted sum of that patch, so a convolution is linear like a dense layer. ReLU after it makes each output mean "how much of my pattern is here" (0 when absent or reversed); a mirror-image pattern is caught by another filter of the same layer (`nn.Conv2d(1, 32, 3)` learns 32), which training finds on its own if it helps the loss.

**Receptive field.** Each layer sees a patch of the layer below, so stacked layers see wider areas of the image: two 3 × 3 convolutions reach 5 × 5 pixels (in 1-D: b4 uses a3–a5, which use p2–p6). Without ReLU the two would equal one 5 × 5 filter; with ReLU the second combines what the first found next to each other (strokes → corners → digit parts).

**Cross-entropy = log loss for many classes.** Softmax turns the 10 scores into probabilities adding up to 1; the loss is −log(probability of the true digit): 0.99 → 0.01, 0.70 → 0.36, 0.10 → 2.30. Binary log loss −[y·log(p) + (1 − y)·log(1 − p)] is the same with 2 classes: y and 1 − y just pick the true class's probability (p if y = 1, 1 − p if y = 0). The loss sees only the scores and the label, nothing about images. Its gradient on each score is probability − (1 for the true digit, else 0): true 4 at 0.70 gets −0.30 (pushed up), 9 at 0.25 gets +0.25 (pushed down); `backward` carries that push back to every weight, filters included.

**Noise of a difference between two models.** A single score on n images has sd √(p(1 − p) ÷ n) (each image a yes/no outcome; *DR: 0.12 points at 98.84% on 8,400*). Comparing two models on the same images, only the images where they disagree change the difference, so its sd ≈ √(disagreements) ÷ n. Independent noises add as variances: √(a² + b²). A bar of 1 sd is cleared by a no-better model 16% of the time, 2 sd 2.3%, 3 sd 0.1%. *DR: two CNN seeds disagree on ~69 images, so ≈ 0.10; with training noise 0.11; keep rule +0.3.*

**Averaging several networks (voting).** Train the same network with different seeds and combine: each votes for its digit (the most votes wins), or their probabilities are averaged. It fixes the mistakes specific to one seed (another network gets them right), not the shared ones (hard or mislabelled images). Votes ignore confidence (three hesitant networks outvote two certain ones); averaged probabilities don't; which is better is settled on data, not examples. Costs 5× the training. *DR: two CNN seeds share ~60% of their mistakes; votes and probabilities differed on 6 of 6,720 images (tie, votes); 5 voting CNNs 99.07% vs 98.84%, +0.23, not kept.*

**Data augmentation.** Train on randomly changed copies of the images, changed only in ways a writer could produce (shift a few pixels, rotate ±10°, zoom ±10%), a new draw for every image in every batch; never flips or big rotations (6 ↔ 9). Exact pixels can't be memorised, so the network learns the shape. Training images only: the early-stopping and held-out images stay real. On a batch, `torchvision`'s `RandomAffine` gives all images one shared draw; one `affine_grid` matrix per image plus `grid_sample` gives each its own. *DR: 99.11% vs 98.84%, +0.27, not kept; it got a different set of images right (≈ 94 disagreements vs 69 between seeds).*

**Dropout (not tried).** During training, `nn.Dropout(p)` sets each unit's output to 0 with probability p (often 0.5), a new random choice per image per batch, and multiplies the survivors by 1 ÷ (1 − p) so the total stays about the same. No unit can rely on a particular other one, so fragile combinations that memorise single images can't form; it's like many thinned networks sharing weights, used together at the end (a cheap relative of voting). Usually in the dense part, where most weights are (*DR: about 400,000 of the CNN's 420,000 in `Linear(3136, 128)`*). `network.train()` turns it on, `network.eval()` off; left on during evaluation, every score would come from a different crippled half of the network, lower and random.

**Impossible mistakes.** Some images look like a different digit than their label, even to a person; no model can fix them, so they set a ceiling. Look at the mistakes (on the early-stopping set, not the held-out set) before predicting what a change can fix. *DR: a "3" that looks like a 9, two "9"s that look like 4s, among the CNN's most confident mistakes.*
