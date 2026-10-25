# House Prices feature notes

Decisions and reasoning behind the features in [house_prices.py](house_prices.py). Task: predict `SalePrice` for 1,459 test houses from 79 columns. 1,460 training houses. Kaggle scores the RMSE between log(predicted price) and log(actual price).

Status: in progress. Going through the columns one at a time, in the order of `data_description.txt`. No model or baseline yet.

Graphs are made by [data_analysis.py](data_analysis.py) (`uv run python data_analysis.py`) and saved to `plots/`.

## Columns

### Id: not used

A row number, with no information about the house. Goes in the list of columns not used for fitting.

### MSSubClass: target encoding (mean price per code), out-of-fold and blended

16 codes for the type of dwelling (15 in `train.csv`), for example 20 = 1-story 1946 & newer, 60 = 2-story 1946 & newer, 120 = 1-story PUD.

**Not a number, even though it's stored as one.** The codes are labels: 60 isn't "three times" 20.
- Mean living area per code jumps up and down along the code number (correlation 0.07). It follows the number of stories in the description, and those are scattered across the codes: 2-story is 60, 70 and 160. See `plots/mssubclass_living_area.png`.
- Mean sale price does the same (correlation −0.08): 20 is $185k, 30 is $96k, 60 is $240k and 160 is $139k. See `plots/mssubclass_sale_price.png`.
- So normalizing it as a number is out: one weight would force a straight line through the codes.

**Options considered:**
1. One 0/1 column per code (one-hot, as for `Embarked` on Titanic). 15 columns for 1,460 houses (about 100 per column) is not too many on its own. The risk is the rare codes: 40 has 4 houses, 180 has 10, 45 has 12 and 75 has 16, so their weights would mostly learn those few houses. Regularization or merging rare codes into "other" would reduce that. An unknown code (150) gets 0 in every column, so no adjustment.
2. Replace each code with the mean sale price of its group (target encoding). One column. **Chosen for now.** The user's own check of option 1 is still open (see open items).

**Leakage, and the fix.** The value is built from `SalePrice`, the label, like group fate on Titanic.
- If a training house's own price is part of its group mean, part of the answer is inside the feature. For code 40 (4 houses) that's 25% of the value; for code 20 (536 houses) under 0.2%. The model then trusts the column more than it should, and that trust doesn't carry over to test houses, whose prices are in no mean.
- Leaving only the house's own price out (leave-one-out) doesn't fix it; it flips it. The group total is fixed, so the value becomes (total − own price) / (n − 1): the cheapest house gets the highest value. Code 40: house 590 ($79.5k) gets $181.7k, house 1271 ($260k) gets $121.5k. A tree could recover each price exactly from code + value.
- Fix: **out-of-fold means**. Split the training data into parts and compute each part's values from the other parts only, so a house's own price never touches its value. Test houses get means from all training houses.

**Small groups, and the fix.** A mean from a few houses is unreliable. Uncertainty of a mean: ±2 × (price spread) / √n.
- Code 20: 536 houses, spread $78.8k, so ±$6.8k. Reliable.
- Code 40: 4 houses, spread $75.7k, so ±$76k: the true mean could be anywhere from about $80k to $232k.
- Fix: **blend** the group mean with the overall mean ($180,921), trusting the group more the more houses it has:
  value = (n × group mean + m × overall mean) / (n + m)
  With m = 10: code 40 gets $174k (mostly the overall mean), code 20 gets $185.1k (almost its own mean).
- Leakage and unreliable small groups are two separate problems with two separate fixes. The overall mean is preferred for small groups because it's reliable, not because it avoids leakage: it includes the house's own price too, just at 1/1,460.

**Implementation:** scikit-learn's `TargetEncoder` does both (out-of-fold means inside `fit_transform`, and blending, with m chosen from the data).

**Code 150** (1 house in `test.csv`, none in `train.csv`): n = 0, so the formula gives the overall mean.

### MSZoning: one-hot

General zoning of the sale. 8 zones in `data_description.txt`, 5 in the data (the data writes Commercial as `C (all)`). See `plots/mszoning_counts.png`.
- `train.csv`: RL 1,151 (79%), RM 218, FV 65, RH 16, C 10. A, I and RP have no houses in either file.
- `test.csv` has the same 5 zones in a similar mix, and **4 houses with no value** (`train.csv` has none).

**Why one-hot:** only 5 values, so 5 columns. It's built from the column alone, never from `SalePrice`, so it can't leak, and it needs no out-of-fold step.
- The small zones (FV 4.5%, RH 1.1%, C 0.7%) are not a reason against target encoding. They are just as small under one-hot: C's weight would rest on 10 houses. Target encoding with blending handles small groups *better*, by pulling them towards the overall mean. MSSubClass has even smaller groups (40: 4 houses) and got target encoding. With one-hot, regularization is what keeps small-group weights in check.

**Filling the 4 missing test values.** MSZoning is an input, not the label, so train and test houses are used together. See `plots/mszoning_missing.png` and `plots/big_lots_by_neighborhood.png`.
- Location decides the zone most of the time: giving each known house the most common zone of houses with the same Neighborhood, Condition1 and Condition2 is right for 93% of them (Neighborhood alone 92%, always RL 79%). Somewhat optimistic, because many location groups have only 1–2 houses.
- **Id 2905** (Mitchel, Artery/Norm): the 2 matching houses are both RL, and Mitchel overall is RL 104, RM 9. Filled with **RL**.
- **Ids 1916, 2217, 2251** (IDOTRR, Norm/Norm): matching houses are RM 57, C 16, and IDOTRR has no RL house (RM 68, C 22). But their lots (21.8k, 14.6k, 56.6k sq ft) are bigger than any known IDOTRR lot (max 18k), and lots over 14k are 98% RL across all neighbourhoods.
  - The 98% (big lot, any neighbourhood) and the 78% (IDOTRR, any lot) answer different questions. The closer match is big lots (over 14k) in the 7 neighbourhoods that are mostly not RL: 9 RL out of 15 (OldTown RL 3 / RM 5, Somerst RL 5, BrkSide RL 1, IDOTRR C 1), so about 60% RL, from very few houses.
  - Options were RL, RM, or no zone (all zone columns 0, like MSSubClass code 150). Filled with **RM**, IDOTRR's most common zone (68 of 90 known houses; the rest C).
  - Reason: IDOTRR has 0 RL houses out of 90 known, so RL would create a neighbourhood–zone combination never seen in the data. See `plots/zones_by_neighborhood.png`: 4 neighbourhoods have no RL at all (Blueste, BrDale, MeadowV, IDOTRR), and C is almost only in IDOTRR (22 of 25).
  - Small minority zones do occur elsewhere (14 of 25 neighbourhoods have more than one zone, for example NridgHt RL 165 / RM 1), so the strong part of the reason is the 0 of 90, not that minorities are rare.
  - Affects 3 of 1,459 test predictions.

### LotFrontage: two candidates, decided by cross-validation

Feet of street connected to the lot. 21 to 313, median 69, half between 59 and 80. Correlation with SalePrice 0.35: a positive but loose trend. See `plots/lot_frontage.png`.

**Missing values:** 259 of 1,460 in `train.csv` (18%), 227 of 1,459 in `test.csv` (16%).
- They mean **"not recorded"**, not "no frontage": every lot touches a street, there are no 0 values, and houses with no value have ordinary lots (median 10,624 sq ft, against 9,262) and prices (median $172k, against $160k; means $181.6k and $180.8k).
- Missing much more often on cul-de-sac lots: 49% (87 of 176), against 13% of inside lots. A cul-de-sac lot sits on the curve at the end of the street, so its frontage is a short curved edge that is harder to measure.
- **No missing flag:** with and without a value sold for about the same on average, so a flag on its own adds little.

**Candidates (compare with CV once a baseline exists):**
1. **Fill** with (typical ratio for its LotConfig) × √LotArea. Frontage grows with lot size, but the ratio frontage / √LotArea depends on the lot's shape: cul-de-sac 0.44, inside 0.72, corner 0.81. Filling from LotArea alone would make the cul-de-sac lots far too wide. Built from input columns only, so no leakage; train and test can be used together.
2. **Drop** the column. LotFrontage and LotArea are strongly related (0.75 on a log scale), so LotArea may already carry most of it.

### LotArea: log(LotArea)

Lot size in square feet. No missing values. See `plots/lot_area.png` (without the two removed outliers).
- Very lopsided: median 9,475, half between 7,500 and 11,600, 10 lots over 50,000, maximum 215,245. Skew 12.6. Test lots stay below 56,600.
- On the raw scale, a handful of huge lots drive the straight-line fit (it reaches $600k+ at 200k sq ft, where houses actually sold for $230k–$375k). Correlation with SalePrice 0.27.
- log(LotArea) is nearly symmetric (skew −0.2), and the line follows the bulk of the houses. Correlation 0.39.
- The log turns equal ratios into equal steps: 5k → 10k is the same step as 50k → 100k. That matches how lot size should matter: doubling a small lot means more than adding 5k to a huge one.
- **Decision:** use log(LotArea), standardized. Raw LotArea is not used.

### Street: one 0/1 column, gravel = 1

Type of road access. Almost constant: 1,452 paved, 6 gravel (0.4%) in `train.csv` (without the two outliers); 6 gravel in `test.csv`. See `plots/street_alley.png`.
- Gravel houses: median $114k, against $163k paved. From 6 houses, so uncertain by about ±$50k, and possibly caused by what the gravel houses have in common (age, size, neighbourhood) rather than by the gravel itself. Other columns may already capture that.
- **Decision:** keep, as a single 0/1 column with gravel = 1. Paved houses get 0, so the weight never touches their predictions: whatever the model learns, including overfitting to the 6 gravel houses' quirks, stays with the gravel houses (6 in training, 6 in test). A single weight can still overfit, so the protection is the 0 for paved houses, plus regularization keeping a weight learned from 6 houses small. Two one-hot columns would put a weight on the paved houses too.

## Scaling

Scaling changes the units of a column, never its shape. The terms are used loosely:
- **Standardization** (`StandardScaler`): (x − mean) / standard deviation, so the column has mean 0 and spread 1. The usual choice for linear models.
- **Min-max scaling** (`MinMaxScaler`): (x − min) / (max − min), so the column runs from 0 to 1. One extreme value squeezes everything else together.
- "Normalization" can mean either. scikit-learn's `Normalizer` is something else again: it scales each *row* to length 1, which is not what we want here.
- A log (or another transform) is not scaling: it changes the shape. Order: transform first, then scale.

**Why scale at all:** a regularized linear model (Ridge, Lasso) penalizes the size of every weight equally. A column in square feet (thousands) needs a tiny weight and a column in 0/1 needs a big one for the same effect, so the penalty would hit them unequally. Scaling puts every column on the same footing. Plain least squares, without a penalty, gives the same predictions with or without scaling.

## Outliers

### Ids 524 and 1299: removed from the training data

The two largest houses in `train.csv` sold far below the trend: Id 524 (4,676 sq ft living area, $184,750) and Id 1299 (5,642 sq ft, 63,887 sq ft lot, 313 ft frontage, $160,000). Both are in Edwards with `SaleCondition = Partial` ("Home was not completed when last assessed"). See `plots/living_area_outliers.png`.
- A straight-line fit of SalePrice on GrLivArea gives $115 per sq ft without them and $107 with them: 0.14% of the houses move the slope by 7%. A linear model minimizes squared errors, Id 1299 is $463k below the line, and it sits at the far end of the x-axis, so the line tilts towards it. A flatter line under-prices every big house.
- The dataset's author (De Cock) recommends removing houses over 4,000 sq ft; apart from these two, those are the $745k–$755k houses, which fit the trend.
- **Cost:** test Id 2550 looks like them (5,095 sq ft, 39k lot, Edwards, Partial). Trained without them, the model will price it like a normal big house. If it really sold for about $185k and is predicted at about $600k, the log error is ln(600/185) = 1.18. Over 1,459 test houses that adds about 1.18² / 1,459 ≈ 0.001 to the mean squared log error: an RMSE of 0.130 would become about 0.134.
- **Decision:** remove both from training. Every other house's prediction gets better from a slope that isn't flattened, at the cost of one probably bad prediction.
- **For cross-validation:** removing them from the validation folds too would make the scores look better simply because two hard houses are gone. To measure the effect honestly, remove them only from the training part of each fold. Once removed for good, all scores are on the 1,458 houses, and Kaggle's score still includes Id 2550.

## Open items

- Keep rule for this problem: to be written before the first comparison.
- Compare option 1 (one-hot) against option 2 (target encoding) for MSSubClass with cross-validation, once a baseline exists.
- Compare filling LotFrontage (LotConfig ratio × √LotArea) against dropping it, with cross-validation.
- Columns that may overlap with MSSubClass: `HouseStyle`, `YearBuilt`, `BldgType`. Check when we reach them.
- Group similar columns (garage, basement, pool) instead of going one by one.
