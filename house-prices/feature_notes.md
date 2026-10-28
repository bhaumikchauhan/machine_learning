# House Prices feature notes

Decisions and reasoning behind the features in [house_prices.py](house_prices.py). Task: predict `SalePrice` for 1,459 test houses from 79 columns. 1,460 training houses. Kaggle scores the RMSE between log(predicted price) and log(actual price).

Status: in progress. Columns Id to HouseStyle were gone through one at a time, in the order of `data_description.txt`. That was too slow for 79 columns, so the approach changed (2026-10-26): a small baseline first (see [Baseline](#baseline)), then the remaining columns in groups, with a quick check per column and CV to decide.

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

**CV comparison (2026-10-28), with TotalBsmtSF in the model:** one-hot (20 as the reference, 14 columns) vs target encoding. No prediction written.
- Why it might matter: target encoding's one weight came out at 0.066, so it adds at most ±4 (×100) per code, while four codes have leftovers outside their uncertainty after the other columns (180: +11 ± 5, 85: +6 ± 3, 90: −7 ± 5, 160: −5 ± 3). Blending and out-of-fold barely change the encoded values, except code 40 (4 houses).
- **Result:** one-hot 0.1367 vs target encoding 0.1375: gain 0.08 (×100), wins 72% of folds. Fails both parts of the keep rule (needs 0.1 and 80%), if narrowly. **Target encoding stays.**
- In some folds code 40 (4 houses) is missing from the training part; one-hot then prices it like code 20 (scikit-learn's "unknown categories" warning).

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

**CV result (2026-10-28):** one-hot (RL as the reference, 4 columns) vs not added: RMSE 0.1229 vs 0.1247, gain **0.18** (×100) but wins only **72%** of folds. Fails part 2. **Not added.**
- **Where the gain comes from:** 82% of it from the 10 commercial-zone houses (C), 18% from RM, almost nothing from the rest. Each fold holds about 2 of the 10, so folds without them barely gain. This is the case part 2 of the rule exists for: a gain carried by a handful of houses.
- A single "commercial zone" 0/1 column would target those houses, but that idea comes from looking at this result, so a test of it would be weak (picked to fit what was just seen). Not tried.

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

### Alley: two 0/1 columns (gravel, paved); no alley is the reference

Type of alley access. **An empty cell is not missing data:** `data_description.txt` says `NA = No alley access`, but pandas reads the text "NA" as a missing value by default. Many later columns (pool, garage, basement, fence) work the same way.
- `train.csv`: no alley 1,367 (94%, median $165k), gravel 50 (median $120k), paved 41 (median $172k). `test.csv`: 1,352 / 70 / 37. See `plots/street_alley.png`.
- **Decision:** two 0/1 columns, `alley_gravel` and `alley_paved`. No alley gets no column of its own: it is the reference, with both columns 0, so its houses get no Alley weight, as for paved Street. Each weight then means "difference from a house with no alley".
- Implementation: turn the empty cells into an explicit "None" value first, so they aren't treated as missing. With scikit-learn's `OneHotEncoder`, the reference is set with `drop=`.

### LotShape: dropped (CV: adds nothing)

General shape of the lot, with a natural order: Reg (regular), IR1 (slightly irregular), IR2 (moderately irregular), IR3 (irregular). See `plots/lot_shape.png`.
- `train.csv`: Reg 925 (median $146k), IR1 483 ($189k), IR2 41 ($221k), IR3 9 ($215k). `test.csv`: 934 / 484 / 35 / 6.
- **Irregular lots sell for more**, most likely because they are newer and bigger. Median year built: 1967 / 1984 / 1995 / 1998; median lot: 8,800 / 10,624 / 13,682 / 16,226 sq ft; median living area: 1,365 / 1,588 / 1,668 / 1,675 sq ft. Newer subdivisions have curving streets and culs-de-sac (77 of the 176 cul-de-sac lots are IR1, 1 is Reg). So LotShape may mostly stand in for what YearBuilt, LotArea and GrLivArea already say.
- **Ordered or one-hot:** an ordered column (0–3) with one weight forces equal steps. The steps here are uneven (+$43k, +$32k, −$6k; the last on 9 houses).
- **Overlapping columns don't cause underfitting** (a model too simple for the pattern). They make the model unsure which column deserves the credit, so it splits the weight between them, sometimes unstably. Predictions usually stay fine; the weights become hard to read. Ridge spreads the credit. Overlap is a keep-or-drop question, not an encoding question.
- **Decision:** one-hot with Reg as the reference: `lot_shape_ir1`, and `lot_shape_ir2_ir3` (IR2 and IR3 merged, 50 houses), so no weight is learned from 9 houses.
- **To compare in CV:** ordered (0–3), this one-hot version, and dropping LotShape.
- **CV result (2026-10-28, with TotalBsmtSF in the model; simpler version as current):** dropped 0.1375; ordered 0.1374 (gain 0.00, wins 64%); one-hot 0.1375 (gain 0.00, wins 54%). Neither passes. **Decision: dropped.** As suspected above, LotShape mostly repeats what Age, LotArea and GrLivArea already say.

### LandContour: one-hot, Lvl as reference

Flatness of the lot: Lvl (level), Bnk (banked: quick rise from the street to the building), HLS (hillside: slope from side to side), Low (depression). No natural order: three different kinds of not-flat. See `plots/land_contour.png`.
- `train.csv`: Lvl 1,311 (90%, median $163k), Bnk 61 ($136k), HLS 50 ($222k), Low 36 ($190k). `test.csv`: 1,311 / 54 / 70 / 24.
- Part of each gap is probably age or lot size: median year built 1974 / 1937 / 1996 / 1971, median lot 9,400 / 8,400 / 11,099 / 16,177 sq ft. Banked lots are mostly in old neighbourhoods (OldTown 12, SWISU 11, Crawfor 8), hillside lots in new ones (Crawfor 12, Timber 10, Gilbert 8), and 13 of the 36 depression lots are in ClearCr.
- **Decision:** one-hot with Lvl, the most common value, as the reference: `land_contour_bnk`, `land_contour_hls`, `land_contour_low`. No merging: the smallest group has 36 houses.

### Utilities: dropped

Type of utilities. `train.csv`: AllPub 1,459, NoSeWa 1 (Id 945, Timber, built 1958, $137,500). `test.csv`: AllPub 1,457, and 2 with no value (Ids 1916, 1946). NoSewr and ELO don't appear.
- **Decision:** drop. No test house is anything but AllPub, so the weight could never change a test prediction; in training it would only fit Id 945's price, which is noise. Dropping also means the 2 missing test values need no filling.

### LotConfig: one-hot, Inside as reference, FR2 and FR3 merged

Position of the lot: Inside, Corner, CulDSac (cul-de-sac), FR2 / FR3 (frontage on 2 / 3 sides). No natural order. See `plots/lot_config.png`.
- `train.csv`: Inside 1,051 (72%, median $160k), Corner 262 ($161k), CulDSac 94 ($199k), FR2 47 ($165k), FR3 4 ($195k). `test.csv`: 1,081 / 248 / 82 / 38 / 10.
- Inside, Corner and FR2 sell for about the same. Cul-de-sac stands out (+$39k), and its houses are newer and bigger (median year built 1985 against 1972, lot 12,655 against 9,100 sq ft, living area 1,693 against 1,452): the same pattern as LotShape, and most cul-de-sac lots are IR1, so the two columns overlap.
- **FR3** (see `plots/fr3_houses.png`): the 4 training houses are 532 (1920, single-family, $128k), 1266 (1999, townhouse, $184k), 334 (2004, townhouse, $207k) and 1360 (2004, single-family, OverallQual 9, $315k). Their median rests on 4 spread-out houses (uncertain by roughly ±$80k). Train and test together, FR2 and FR3 are almost identical: median year built 1994 / 1996, lot 9,600 / 9,445 sq ft, living area 1,422 / 1,407 sq ft. The 10 test FR3 houses range from 1910 to 2006.
- **Decision:** one-hot with Inside as the reference: `lot_config_corner`, `lot_config_culdsac`, `lot_config_fr2_fr3` (FR2 and FR3 merged, 51 training houses). Main reason: FR3's gap rests on 4 houses and FR3 looks like FR2 on everything checked. Part of those 4 houses' prices is explained by other columns (one is OverallQual 9, two are new townhouses), but that's an expectation to see in the weights, not a fact yet.

### LandSlope: one 0/1 column, Mod and Sev merged; CV to compare

Slope of the property: Gtl (gentle), Mod (moderate), Sev (severe). See `plots/land_slope.png`.
- `train.csv`: Gtl 1,380 (95%, mean $180k ± $4k), Mod 65 ($197k ± $23k), Sev 13 ($204k ± $48k; median lot 28,698 sq ft). `test.csv`: 1,396 / 60 / 3.
- **Mod and Sev can't be told apart:** the gap between their means ($8k) is far smaller than the uncertainty of the Sev mean (±$48k). A Sev weight would be learned from 13 houses, for 3 test houses. Merged: 78 houses, $198k ± $21k.
- **Overlap with LandContour** (train + test): Gtl is 96% Lvl; Mod is 78% non-level (Lvl 28, Bnk 26, HLS 39, Low 32); Sev is 88% non-level (Lvl 2, Bnk 1, HLS 2, Low 11). The new information is mostly in the 30 lots that are sloped but marked level; elsewhere the column splits credit with the LandContour columns.
- **Decision:** `land_slope_sloped` (Mod or Sev = 1, Gtl = 0, the reference).
- **To compare in CV:** with and without `land_slope_sloped`.

### Neighborhood: one-hot, NAmes as reference, Blueste priced like the reference

Physical location within Ames: 25 neighbourhoods, none missing, all of them in both train and test. See `plots/neighborhoods.png`.
- Median price from $88k (MeadowV) to $315k (NridgHt), a factor of 3.6. Log-price spread within a neighbourhood is typically 0.2–0.3, against 0.40 overall: the neighbourhood explains a good part of the price.
- Part of each gap is age and quality: NridgHt median year built 2006, OverallQual 8; IDOTRR 1922, OldTown 1920. Part is location: Crawfor is old (1938) and still $201k. The weights will be the location effect left over after YearBuilt, OverallQual and the rest.
- Small neighbourhoods (train / test; overall mean $181k):

  | Neighbourhood | Train | Test | Mean | Gap | Uncertainty | Gap vs uncertainty |
  |---|---|---|---|---|---|---|
  | Blueste | 2 | 8 | $138k | −$43k | ±$27k | 2 houses: unreliable |
  | NPkVill | 9 | 14 | $143k | −$38k | ±$6k | real |
  | Veenker | 11 | 13 | $239k | +$58k | ±$44k | doubtful |
  | BrDale | 16 | 14 | $104k | −$77k | ±$7k | real |
  | MeadowV | 17 | 20 | $99k | −$82k | ±$11k | real |
  | Blmngtn | 17 | 11 | $195k | +$14k | ±$15k | could be chance |

- **Decision:** one-hot, each neighbourhood its own column, because the gaps are large and uneven, so each needs its own weight. About 22 columns for 1,458 houses is affordable. Blmngtn keeps its column: an uncertain gap gives a small weight, and the penalty shrinks it.
- **Reference: NAmes** (the most common, 225 houses, mean $146k). With a penalty, each weight is pulled towards 0, i.e. each neighbourhood's prediction towards the reference's price level, small groups most (roughly the blending formula with the reference as the anchor). NAmes sits on the cheap side, so expensive small groups get pulled down.
- **To compare in CV:** NAmes against CollgCr (150 houses, mean $198k, close to the overall mean) as the reference.
- **Leftover check** (does "the other columns will handle it" hold?): a simple fit of log price on OverallQual, log GrLivArea, YearBuilt and log LotArea, without Neighborhood, then the mean leftover (actual − predicted log price) per neighbourhood: NAmes +0.02 ± 0.02, Blueste +0.07 ± 0.04 (2 houses), Veenker +0.12 ± 0.07, Blmngtn +0.05 ± 0.05, Crawfor +0.16 ± 0.04.
- **Blueste: no column** (priced like NAmes). 2 houses, and its leftover can't be told apart from NAmes.
- **Veenker: its own column.** After quality, size, age and lot size, it still sold for about 10% more than NAmes, beyond its uncertainty; without a column its 13 test houses would be under-priced by about that. The penalty decides how far to pull its 11 noisy houses towards NAmes. (Score impact either way is small: about +0.0003 RMSE.)
- **Columns:** one per neighbourhood except NAmes and Blueste: 23.

### Condition1 and Condition2: three 0/1 columns from both columns together

Proximity to conditions: Artery (arterial street), Feedr (feeder street), RRNn / RRAn (within 200 ft of / next to the north-south railroad), RRNe / RRAe (the same for the east-west railroad), PosN / PosA (near / next to a park, greenbelt etc.), Norm (nothing). Condition2 is only filled in when a house has a second condition. See `plots/conditions.png`.
- Train + test: only 30 houses have a second condition, 13 of them repeating Condition1's value (PosN / PosN etc.). No house has Condition1 = Norm with something in Condition2, and the order between the two follows no rule (railroad + feeder street is 8 × RRAn / Feedr, 1 × Feedr / RRAn). So the two columns are a *set* of up to two conditions.
- **Leftover check** (the simple fit of log price on OverallQual, log GrLivArea, YearBuilt, log LotArea, without the conditions), train houses with the condition in either column:

  | Condition | Train | Test | Mean | Leftover |
  |---|---|---|---|---|
  | Norm | 1,260 | 1,251 | $184k | +0.01 ± 0.01 |
  | Artery | 48 | 47 | $135k | −0.09 ± 0.05 |
  | Feedr | 85 | 87 | $141k | −0.03 ± 0.04 |
  | RRNn | 7 | 4 | $179k | +0.02 ± 0.06 |
  | RRAn | 27 | 24 | $183k | −0.06 ± 0.06 |
  | RRNe | 2 | 4 | $191k | −0.07 ± 0.10 |
  | RRAe | 12 | 17 | $143k | −0.14 ± 0.10 |
  | PosN | 18 | 20 | $217k | +0.04 ± 0.06 |
  | PosA | 9 | 12 | $237k | +0.01 ± 0.14 |

  Park-side houses look $33–53k dearer, but the leftover is about 0: they're bigger and better, and the park adds little that can be measured. Busy roads and railways cost something. Measured against each house's neighbourhood average, the leftovers barely change, so this isn't Neighborhood again.
- **Decision:**
  - **Reference (no column):** Norm, Feedr, RRNn, PosN, PosA, whose leftovers can't be told apart from Norm. Feedr and PosN are the same case (difference from Norm −0.04 and +0.03, both ranges reaching Norm), so they're treated the same way.
  - `condition_artery`, `condition_rr_ns_adjacent` (RRAn), `condition_rr_ew` (RRAe + RRNe: the same east-west railroad; RRNe has only 2 training houses).
  - Each column is 1 if the condition is in Condition1 **or** Condition2, so the order between them doesn't matter.
- **To compare in CV:** Feedr and PosN merged into the reference against their own columns.
- **CV result (2026-10-28):** 3 columns vs not added: 0.1186 vs 0.1193, gain 0.07 (×100), wins 80%: fails part 1. 5 columns (+ `ConditionFeeder`, `ConditionParkNear`) vs not added: 0.1180, gain 0.13, wins 84%: passes. (First wrongly compared against the failed 3-column version, 0.06 and 76%; corrected by the "version actually in place" rule before the corrected numbers were computed.) **Decision: the 5 columns added**, not scaled. The current model scores **0.1180 ± 0.0080**.

### BldgType: dropped (MSSubClass already says it)

Type of dwelling: 1Fam (single-family), 2fmCon (two-family conversion), Duplex, Twnhs (townhouse inside unit), TwnhsE (townhouse end unit). The data spells them `2fmCon`, `Duplex`, `Twnhs`; `data_description.txt` says `2FmCon`, `Duplx`, `TwnhsI`. See `plots/bldg_type.png` and `plots/bldg_type_vs_mssubclass.png`.
- `train.csv`: 1Fam 1,218 (mean $186k), 2fmCon 31 ($128k), Duplex 52 ($134k), Twnhs 43 ($136k), TwnhsE 114 ($182k). `test.csv`: 1,205 / 31 / 57 / 53 / 113. None missing.
- **Almost a copy of MSSubClass** (train + test): Duplex is exactly code 90 (109 houses); 60 of the 62 2fmCon are code 190; 322 of the 323 townhouses are the PUD codes 120–180; codes 20–85 are single-family apart from 3 houses.
- **The only new information** is inside vs end unit within a PUD code (code 160: 71 inside, 56 end). After quality, size, age, lot size and the MSSubClass group, their leftovers are the same (−0.003 and −0.001). The raw $46k Twnhs/TwnhsE gap comes from the codes: end units are mostly code 120 (1-story, median built 2003), inside units mostly code 160. (For Duplex and 2fmCon, a leftover of 0 after MSSubClass is automatic, since they sit inside one code; it's the mapping itself that shows they add nothing.)
- **Decision:** drop. Holds whether MSSubClass ends up target-encoded or one-hot: both carry each code's price level.

### HouseStyle: dropped (MSSubClass mostly says it)

Style of dwelling: 1Story, 1.5Fin / 1.5Unf (second level finished / unfinished), 2Story, 2.5Fin / 2.5Unf, SFoyer (split foyer), SLvl (split level). See `plots/house_style.png` and `plots/house_style_vs_mssubclass.png`.
- `train.csv`: 1Story 726 (mean $176k), 1.5Fin 154 ($143k), 1.5Unf 14 ($110k), 2Story 443 ($210k), 2.5Fin 8 ($220k), 2.5Unf 11 ($157k), SFoyer 37 ($135k), SLvl 65 ($167k). `test.csv`: 745 / 160 / 5 / 427 / 0 / 13 / 46 / 63. None missing.
- After quality, size, age and lot size, the raw gaps mostly disappear (leftovers −0.08 to +0.06, mostly within their uncertainty): two-story houses are bigger and newer.
- **Overlap:** most codes fix the style (code 20: 1Story for 1,077 of 1,079; 60: 2Story for 572 of 575; 80 always SLvl; 85 always SFoyer). Codes 90 (duplex) and 190 (two-family conversion) describe the kind of building, and their houses come in every style: there HouseStyle is new information.
- **Within-code check:** the 62 training houses whose style isn't their code's usual one have a leftover after MSSubClass of +0.015 ± 0.041; subgroups are small with mixed signs (code 90 SFoyer +0.10 on 9 houses, code 90 1.5Fin −0.09 on 4, code 190 1Story +0.09 on 6). Noise, not a pattern.
- 2.5Fin has no test houses. "How much upstairs" comes back more precisely with 2ndFlrSF.
- **Decision:** drop.

### OverallQual: one number column, levels 1–3 merged (since 2026-10-28; first one-hot)

Overall material and finish, 1 (very poor) to 10 (very excellent). None missing. On its own it explains 67% of the log-price spread, the most of any column (`plots/column_overview.png`).
- Raw steps in mean log price are fairly even from 4 to 10 (+0.19 to +0.30 per level, about 25%). After size, age and lot size (a fit without OverallQual) they're uneven: 4: +0.21, 5–7: +0.06, +0.04, +0.02, 8–9: +0.14, +0.18, 10: +0.06.
- Levels 1, 2, 3 have 2, 3 and 20 training houses and nearly the same leftover (−0.29, −0.28, −0.32): merged.
- **Decision:** one-hot over 1–3, 4, 5, 6, 7, 8, 9, 10, with 5 (397 houses, the middle) as the reference: 7 columns.
- **Risk:** the penalty pulls each weight towards the reference, hardest on small groups, and here the small groups are at the far ends (10: 16 houses, 9: 43), so the most expensive houses get pulled towards the middle. A number column keeps the trend for them but forces equal steps. **To compare in CV:** number vs one-hot. It's the strongest column, so this comparison matters more than most.
- **Prediction, written before running (2026-10-27):** the number column (levels 1 and 2 set to 3, keeping the 1–3 merge) beats one-hot by 0.2 (×100), because the raw steps from 4 to 10 are fairly even (+21% to +35% per level).
- **Result:** number 0.1457 vs one-hot 0.1453: the number column is worse by 0.05 (×100) and wins only 28% of folds. Fails the keep rule; **one-hot stays**. The prediction was wrong in direction.
- **Why the prediction was wrong:** it was based on the raw steps (mean log price per level), which ignore the other columns. The model's weights measure something else: the step **for two houses that are equal in every other column** (same size, neighbourhood, age, exterior, garage, ...).

  | OverallQual | Raw step | Step left in the model | Model weight (log price vs level 5) |
  |---|---|---|---|
  | 1–3 → 4 | +25% to +67% | +22% | −0.28 → −0.08 |
  | 4 → 5 | +25% | +8% | −0.08 → 0 |
  | 5 → 6 | +21% | +7% | 0 → +0.06 |
  | 6 → 7 | +29% | +7% | +0.06 → +0.13 |
  | 7 → 8 | +32% | +11% | +0.13 → +0.24 |
  | 8 → 9 | +35% | +14% | +0.24 → +0.37 |
  | 9 → 10 | +27% | +11% | +0.37 → +0.47 |

  (Weights from the baseline fitted on all 1,458 training houses; step % = e^(weight difference) − 1.)
  - **The other columns take most of each raw step.** Better-quality houses are also bigger, newer, in better neighbourhoods, with better exteriors and bigger garages. In the raw means, all of that is counted as "quality". In the model, GrLivArea, Neighborhood, ExterQual, GarageCars and Age each take their own share, so only 7–22% per level is left for quality itself, against 21–35% raw.
  - **What's left is uneven:** a big step out of the lowest group (+22%), small steps in the middle (+7–8% from 4 to 7), and bigger steps again at the top (+11–14% from 7 to 10). The raw steps looked even only because the other columns' shares happened to fill in the uneven parts.
  - **A number column can't follow that.** With one weight, every level gets the same % step: a straight line on the log-price scale. Fitting one slope through these steps (roughly 10% each) overshoots the middle and undershoots both ends, so houses at levels 3–4 and 8–10 are mispriced. One-hot gives each level its own weight and can follow the curve. That's why it won 72% of the folds.
  - **The penalty didn't decide it.** At the alpha RidgeCV picks (about 0.6), even level 10 (16 houses) keeps about 96% of its weight, so the worry that one-hot would pull the top levels towards the middle didn't matter here.
  - **Lesson (same as ExterQual):** judge a column's shape by what's left after the other columns (the leftover check, or the model's weights), not by its raw means.
- **Re-run (2026-10-28), with the simpler version as current** (new keep-rule convention) and on the current model (TotalBsmtSF added, LotShape dropped): number (current) 0.1374 vs one-hot 0.1375: one-hot's gain −0.00 (×100), wins 52% of folds. Fails. **Decision: one number column** (`OverallQualNumber`, 1 and 2 set to 3, scaled). One-hot's earlier edge (0.05) is gone now that basement area is in the model; the explanation above describes the earlier model.

### Age at sale: YrSold − YearRemodAdd

YearRemodAdd is the remodel date, the same as YearBuilt if never remodelled. None missing.
- **YearRemodAdd starts at 1950:** 361 houses show 1950, 339 of them built before 1950 (remodels before 1950 apparently weren't recorded). In train, these 166 houses (median built 1928) average $118k ± $6k; genuine 1950 houses never remodelled $111k ± $11k. The floor does little harm.
- Pre-1950 houses remodelled after 1950 average $148k against $118k for those that weren't: remodelling matters, which YearBuilt can't see.
- **Age at sale, not "today − year":** in a linear model, today − year is the year flipped, so the predictions are the same. Age at sale describes the house when sold.
- **No market trend to model:** prices were flat 2006–2010 (median $164k / $166k / $165k / $162k / $155k; yearly leftovers within ±0.02). A trend would need YrSold as its own column anyway: with age alone, a later sale makes the house "older".
- **Decision:** `age = YrSold − YearRemodAdd`, as a number. YearBuilt overlaps heavily (1,560 of 2,919 houses never remodelled); left for later.

### GrLivArea: log(GrLivArea)

Above-ground living area. None missing. Skew 1.01 raw, −0.07 after log; explains 52.6% (raw) vs 54.4% (log) of the log-price spread. Mean log price per size band rises less for big houses, which is what log area gives: with raw area, every extra 1,000 sq ft multiplies the price by the same factor, whatever the size.
- Test Id 2550 has 5,095 sq ft, beyond every training house once 524 and 1299 are removed (largest 4,476); log area extrapolates more gently.
- **Decision:** log(GrLivArea), then standardized.

### ExterQual: one-hot, TA as reference

Quality of the exterior material: Ex, Gd, TA (typical), Fa. Po doesn't occur. None missing.
- `train.csv`: Fa 14 ($88k), TA 906 ($144k), Gd 488 ($232k), Ex 50 ($375k). `test.csv`: 21 / 892 / 491 / 55.
- Raw steps in mean log price are even (+0.53, +0.47, +0.48), but that's mostly OverallQual rising with it (median quality 4 / 5 / 7 / 9). After quality, size, age and lot size, the leftovers are −0.13 ± 0.13 / −0.01 / +0.01 / +0.12 ± 0.05: uneven steps (+0.12, +0.02, +0.10). The model's weights work on top of OverallQual, so those are the steps that count.
- **Decision:** one-hot with TA as the reference: `exter_qual_fa`, `exter_qual_gd`, `exter_qual_ex`. Fa (14 houses, range just reaching 0) keeps its column, to see how far the penalty shrinks it; whether it helps is a CV question.

### GarageCars: one-hot 0 / 1 / 2 / 3+, 2 as reference

Garage size in cars. Test Id 2577 has no value: GarageType says Detchd but every other garage field is empty, so it's treated as no garage (GarageCars 0; GarageType to "no garage" too when the garage columns are added).
- `train.csv`: 0: 81 ($103k), 1: 369 ($128k), 2: 823 ($184k), 3: 180 ($310k), 4: 5 ($193k). `test.csv`: 76 (+ 2577) / 407 / 770 / 193 / 11, and one 5-car house (Id 1829: 1,184 sq ft detached, OverallQual 5, 1,072 sq ft living area).
- Leftover after quality, size, age and lot size: −0.14 ± 0.05 / −0.02 / +0.01 / +0.07 ± 0.02 / +0.12 ± 0.08. Having a garage at all is the big step (+0.12); raw steps are biggest at 2 → 3 (+0.50), because 3-car houses are mostly quality 8.
- **Not target encoding:** it gives one weight on the raw group means, so it keeps the raw shape of the steps (biggest at 2 → 3), not the shape left after quality (biggest at 0 → 1).
- **4 cars merged with 3:** ranges +0.04 to +0.20 and +0.05 to +0.09 overlap; their raw price gap comes from quality.
- **Decision:** `garage_none`, `garage_1`, `garage_3plus` (3, 4 and 5 cars); 2 is the reference. The 5-car test house goes in 3+.

### TotalBsmtSF: added after the baseline; raw vs log + NoBasement decided by CV

Total basement area (= BsmtFinSF1 + BsmtFinSF2 + BsmtUnfSF). Added first after the baseline, because MSSubClass 180's leftover (+11) looked like basement space the model couldn't see: 7 of its 10 houses are split-foyer, 630 sq ft above ground plus a 630 sq ft basement.
- **Kept separate from GrLivArea**, not summed: a basement sq ft isn't worth the same as an above-ground sq ft, and a sum would force one weight for both.
- Test Id 2121 is missing, with every basement column empty: no basement, so 0 (as with Id 2577's garage).
- No basement (0 sq ft): 37 in train, 42 in test (with 2121). They average about 40% below houses with a basement.
- Skew raw 0.51 (mild), log −0.38; correlation with log price raw 0.65, log 0.61 (houses with a basement only). Largest: train 3,206 (Id 333), test 5,095 (Id 2550).
- **log(0) doesn't exist, and replacing 0 by 1 doesn't fix it:** log 1 = 0 against 5.7 for 300 sq ft and 8.0 for 3,000, so one weight would price "no basement" as a huge drop, set by how prices change between basement sizes. So the log version comes with a `NoBasement` 0/1 column, which takes the no-basement step.
- **Two versions, compared by CV:** (a) raw, scaled; (b) log (0 for no basement), scaled, plus `NoBasement`.
- **Prediction, written before running (2026-10-27):** adding the basement passes the keep rule, with a gain of about 0.1 (×100); log + NoBasement beats raw.
- **Result:** raw: RMSE 0.1375, gain **0.78** (×100) over the baseline, wins 100% of folds. Log + NoBasement: 0.1378, gain 0.74, wins 100%. Log + NoBasement vs raw: −0.03, wins 34%: fails. **Decision: TotalBsmtSF raw, scaled.** The new current model scores **0.1375 ± 0.0097**.
- The prediction got "passes" right but the size 8 times too small (0.78, not 0.1): unlike ExterQual, basement area is information the model didn't have. The version was wrong: the mild skew didn't need a log.
- **MSSubClass 180's leftover didn't change** (+11 before and after; 85: +6 → +7, 90: −7, 160: −5 → −4). So basement space wasn't the explanation for 180; the reason is still unknown.

### OverallCond: four versions against not adding it

Condition of the house, 1–9. None missing. Top of the ranking of unused columns by share of the current model's leftover explained (8.7%; see the open items).

| OverallCond | Houses | Test | Mean price | Median age at sale | Leftover (×100) |
|---|---|---|---|---|---|
| 1 | 1 | 6 | $61k | 59 | −12 |
| 2 | 5 | 5 | $142k | 44 | −16 ± 23 |
| 3 | 25 | 25 | $102k | 52 | −22 ± 9 |
| 4 | 57 | 44 | $120k | 53 | −9 ± 5 |
| 5 | 819 | 824 | $203k | 9 | −1 ± 1 |
| 6 | 252 | 279 | $154k | 37 | +3 ± 2 |
| 7 | 205 | 185 | $158k | 16 | +5 ± 2 |
| 8 | 72 | 72 | $156k | 8 | +2 ± 3 |
| 9 | 22 | 19 | $216k | 5 | +6 ± 6 |

- **Raw prices are misleading:** level 5 has the highest mean, because new houses are mostly rated 5 (median age 9). The leftover shows the real order, with big steps at the bottom and small ones from 5 up.
- **The user's assumptions, written before running (2026-10-28):** a number column won't fit (steps unequal); the small levels (1, 2, 3, 9) carry signal and would lose it if merged; Ridge keeps their weights from overfitting.
- **Review of those assumptions:** levels 1 and 2 can't be told apart from 3 (−12 and −16 ± 23 vs −22 ± 9), nor 9 from 7 and 8; level 1 has 1 house in train but 6 in test. At alpha ≈ 3 a level keeps about n / (n + 3) of its weight: level 1 25%, level 2 62%, level 9 88%. Target encoding stores the raw means, whose shape is wrong here (level 5 highest), as with GarageCars.
- **Versions:** number (scaled); target encoding of the levels; one-hot with 1–3 merged (5 as the reference, 6 columns); one-hot unmerged (5 as the reference, 8 columns). Each against not adding it, then the more complex versions against the simpler ones.
- **Result:**

  | Version | RMSE | Comparison | Gain (×100) | Folds won | Passes? |
  |---|---|---|---|---|---|
  | Not added | 0.1374 | | | | |
  | Number | 0.1312 | vs not added | 0.62 | 98% | yes |
  | Target encoding | 0.1372 | vs not added | 0.02 | 70% | no |
  | One-hot, 1–3 merged | 0.1302 | vs not added | 0.72 | 96% | yes |
  | | | vs number | 0.10 | 76% | no |
  | One-hot, unmerged | 0.1307 | vs not added | 0.67 | 96% | yes |
  | | | vs one-hot merged | −0.05 | 24% | no |

  **Decision: one number column** (`OverallCondNumber`, scaled). One-hot with 1–3 merged was slightly better, but missed the 80% (76%), so the simpler version stays. The current model scores **0.1312 ± 0.0091**.
- **Assumptions checked:** "a number column won't fit": wrong, it captures almost all of the gain. "Merging the small levels loses their signal": wrong, merging 1–3 beat unmerged. Target encoding added almost nothing (0.02), as expected from the raw shape.

### Finished basement area: BsmtFinSF1 + BsmtFinSF2, added

How much of the basement is finished. The current model (with TotalBsmtSF) treats every basement sq ft alike; its leftover (×100) by finished share: no basement +2 ± 4 (37 houses), all unfinished −5 ± 1 (430), 1–50% −1 ± 2 (258), 51–99% +3 ± 1 (652), 100% +5 ± 3 (81).
- Test Id 2121 (no basement) has no areas: 0. 167 houses have a second finished part (BsmtFinSF2).
- Finished area in sq ft is the same information as BsmtUnfSF once TotalBsmtSF is in (finished = total − unfinished), so only one of them.
- **Versions** (one column each, scaled): finished area `BsmtFinishedSF`; finished share `BsmtFinishedShare` (0 if no basement).
- **Result (2026-10-28):** area 0.1247 (gain 0.65 ×100, wins 98%); share 0.1249 (gain 0.62, wins 100%). Both pass; same number of columns, so the lower RMSE: **finished area added.** The current model scores **0.1247 ± 0.0091**.

### Functional: one 0/1 column, any deduction, added

Home functionality: Typ (typical) down to Sal (salvage only). 1,358 of 1,458 training houses are Typ; leftover (×100) by level: Typ 0 ± 1, Min1 −5 ± 4 (31 houses), Min2 −2 ± 5 (34), Mod −9 ± 12 (15), Maj1 −7 ± 7 (14), Maj2 −25 ± 7 (5), Sev −53 (1); no Sal.
- Test Ids 2217 and 2474 (both IDOTRR) are missing: filled with Typ, as data_description.txt says ("Assume typical unless deductions are warranted"). 2474 has OverallCond 1, so it may well have deductions; this affects 2 test predictions only, not the CV score.
- **Version (user's choice):** `FunctionalDeduction`, 1 for any level below Typ (100 houses in train, 100 in test), not scaled.
- **Result (2026-10-28):** RMSE 0.1236 vs 0.1247, gain 0.12 (×100), wins exactly 80% of folds (40 of 50). Passes, at the edge of both parts. **Added.** The current model scores **0.1236 ± 0.0089**.

### SaleCondition: two 0/1 columns (Partial, abnormal)

Condition of sale. None missing. Leftover of the current model (×100): Normal 0 ± 1 (1,198 houses), Partial +4 ± 3 (123; 120 of them SaleType New), Abnorml −7 ± 4 (101; trade, foreclosure, short sale), Family −7 ± 10 (20), Alloca −1 ± 15 (12), AdjLand 0 ± 14 (4).
- **Merging (user's rule):** values whose leftover can't be told apart from Normal's are merged into Normal, the reference: Alloca, AdjLand. Partial and Abnorml get a 0/1 column each.
- **Family** can't be told apart from Normal (0) or from Abnorml (−7, the same mean), and is, like Abnorml, not an open-market sale. Two versions, both tested (same number of columns, so the lower RMSE if both pass): Family with Normal, Family with Abnorml. **The user's expectation, written before running:** Family belongs with Abnorml.
- Caution for the test set: Partial gets a small premium in the model, because the two Partial houses that sold far below the trend (524, 1299) were removed; adding it pushes Id 2550's prediction up.
- **Result (2026-10-28):** Family with Normal 0.1215 (gain 0.21 ×100, wins 86%); Family with Abnorml 0.1211 (gain 0.24, wins 88%). Both pass; the lower RMSE is kept: **`SalePartial` and `SaleAbnormalOrFamily` added** (not scaled). The user's expectation was right. The current model scores **0.1211 ± 0.0086**.

### CentralAir: one 0/1 column, no central air, added

None missing. No central air: 95 houses in train, 101 in test, median age 56, leftover −7 ± 4 (×100); central air: 1,363, +1 ± 1.
- **Version:** `NoCentralAir`, 1 for no central air, not scaled (the only sensible version for a yes/no column).
- **Result (2026-10-28):** RMSE 0.1193 vs 0.1211, gain 0.19 (×100), wins 96% of folds. **Added.** The current model scores **0.1193 ± 0.0079**.

### Exterior1st: not added

Exterior material, 15 values in train. Leftover of the current model (×100), clearly away from 0 only for wood siding (Wd Sdng, 206 houses, −3 ± 2) and brick face (BrkFace, 50, +7 ± 4); vinyl siding (VinylSd, 515) 0 ± 1; the rest within their ±. Five rare materials (BrkComm, Stone, AsphShn, CBlock, ImStucc) have 1–2 houses each; BrkComm's −18 is ± 73. Test Id 2152 is missing (Exterior2nd too). Exterior2nd matches Exterior1st for 85% of houses: left for later.
- **Version (user's choice, merging rule as for SaleCondition):** `ExteriorWoodSiding` and `ExteriorBrickFace`; everything else, and Id 2152, priced like VinylSd. The rare materials were first given a column for BrkComm's −18, then merged: 2 houses with ± 73 aren't evidence.
- **Result (2026-10-28):** RMSE 0.1184 vs 0.1193, gain **0.08** (×100), wins 82% of folds. Fails part 1 (needs 0.1). **Not added.** Variants picked after this result (brick face alone, for example) would be weak tests; not tried.

### The remaining columns, in groups (plan fixed 2026-10-28, before testing any group)

Single columns were reaching the limit of the keep rule (Exterior1st 0.08, the 3 condition columns 0.07), so the remaining ~40 columns are tested in fixed groups, each group as one candidate against the current model:

| Group | Columns |
|---|---|
| Garage | GarageType, GarageFinish, GarageArea, GarageQual, GarageCond, GarageYrBlt |
| Basement rest | BsmtQual, BsmtCond, BsmtExposure, BsmtFullBath, BsmtHalfBath |
| Rooms | FullBath, HalfBath, BedroomAbvGr, KitchenAbvGr, KitchenQual, TotRmsAbvGrd, Fireplaces, FireplaceQu |
| Outside | WoodDeckSF, OpenPorchSF, EnclosedPorch, 3SsnPorch, ScreenPorch, PoolArea, Fence |
| Lot (mostly decided) | LotFrontage, Street, Alley, LandContour, LotConfig, LandSlope |
| Other | Foundation, Heating, HeatingQC, Electrical, RoofStyle, MasVnrArea, PavedDrive, YearBuilt, SaleType, MoSold |

**Default encodings within a group:** areas and counts as numbers (scaled); quality scales (Ex, Gd, TA, Fa, Po) as one number (5 to 1, scaled); categories as 0/1 columns, with the user's merging rule (values whose leftover can't be told apart from the reference's are merged into it). Decisions a default doesn't cover are made per group, before its run.

#### Garage group: added

Leftover of the current model (×100) per value, reference = the largest value:
- **GarageType** (reference Attchd, 869): only BuiltIn is clearly off (+3 ± 2, 87 houses); Detchd 0 ± 1, Basment 0 ± 9, CarPort −10 ± 11, 2Types −10 ± 9, no garage 0 ± 4. → `GarageBuiltIn`.
- **GarageFinish** (reference Unf, 605): RFn 0 ± 1, Fin +1 ± 1 against Unf −1 ± 1; the ranges touch. → no column.
- **GarageQual, GarageCond:** quality scale as one number (Ex 5 … Po 1, no garage 0): `GarageQualNumber`, `GarageCondNumber`. Mostly TA.
- **GarageArea** as a number (correlation 0.89 with GarageCars, already in the model). **GarageYrBlt** as a number, `GarageYear`; equal to YearBuilt for 75% of houses.
- **Fills, decided before the run:** no garage (81 train, 76 test): quality, condition and area 0, year = YearBuilt (GarageCars' "0" column gives them their own step). Test Id 2593: GarageYrBlt 2207 → 2007 (typo). Test Id 2127 (detached, 360 sq ft, 1 car, but no finish, quality, condition or year): quality and condition TA, year = YearBuilt. Test Id 2577: no garage, as before (area 0).
- **Result (2026-10-28):** the 5 columns as one candidate: RMSE 0.1168 vs 0.1180, gain 0.12 (×100), wins 88% of folds. **Added.** The current model scores **0.1168 ± 0.0083**.

#### Basement group: added

Leftover of the current model (×100) per value, reference = the largest value:
- **BsmtQual** (basement height): Ex +3 ± 2 (119 houses) clearly off; TA −1 ± 1, Gd 0 ± 1, Fa +1 ± 4. → one number, `BsmtQualNumber` (Ex 5 … Fa 2, no basement 0).
- **BsmtCond:** mostly TA (1,309); Fa −5 ± 5, Gd −1 ± 2. → `BsmtCondNumber`, same scale.
- **BsmtExposure** (reference No, 953): only Gd is clearly off (+4 ± 2, 132 houses); Av +1 ± 2, Mn 0 ± 2. → `BsmtExposureGood`.
- **BsmtFullBath, BsmtHalfBath** as numbers; leftovers all about 0.
- **Fills, decided before the run (most common value):** test 2121, 2189 (no basement): bathrooms 0. Test 2218, 2219: BsmtQual TA. Test 2041, 2186, 2525: BsmtCond TA. Train 949, test 1488, 2349: BsmtExposure No.
- **Result (2026-10-28):** the 5 columns as one candidate: RMSE 0.1157 vs 0.1168, gain 0.12 (×100), wins 84% of folds. **Added.** The current model scores **0.1157 ± 0.0084**.

#### Rooms group: added

Leftover of the current model (×100), clear signals only: KitchenQual Ex +4 ± 3 (98 houses), 2 kitchens −5 ± 4 (65, mostly duplexes), 2 fireplaces +3 ± 2 (115). Everything else about 0 within its ±; TotRmsAbvGrd correlates 0.82 with log living area.
- **Encodings (defaults):** FullBath, HalfBath, BedroomAbvGr, KitchenAbvGr, TotRmsAbvGrd, Fireplaces as numbers; KitchenQual and FireplaceQu as quality numbers (`KitchenQualNumber`, `FireplaceQuNumber`, no fireplace 0). All scaled.
- **Fill:** test Id 1556 (a kitchen, no KitchenQual): TA. Test has 4 houses with 4 full baths (training max 3): clipped at submission.
- **Result (2026-10-28):** the 8 columns as one candidate: RMSE 0.1138 vs 0.1157, gain 0.18 (×100), wins exactly 80% of folds. **Added.** The current model scores **0.1138 ± 0.0086**.

#### Outside group: not added

Leftover of the current model (×100), houses with vs without each: WoodDeckSF (697 houses) 0 ± 1 vs 0 ± 1; OpenPorchSF (802) 0 vs 0; EnclosedPorch (208) 0 vs 0; 3SsnPorch (24) +2 ± 5; ScreenPorch (116) +2 ± 2; PoolArea (6) +11 ± 14. Fence: every type within its ± of no fence, so no columns by the merging rule. None missing.
- **Encodings (defaults):** the 6 areas as numbers, scaled.
- **Result (2026-10-28):** RMSE 0.1138 vs 0.1138, gain 0.00 (×100), wins 66% of folds. **Not added.** As the leftovers suggested, size, quality and neighbourhood already cover what decks and porches say.

#### Lot group: not added

The encodings decided column by column at the start (Street, Alley, LandContour, LotConfig, LandSlope sections): 10 0/1 columns. LotFrontage filled with (median frontage / √LotArea for the LotConfig, train and test together) × √LotArea, as a number.
- **Versions:** A = the 10 columns vs not added; B = A + filled LotFrontage, against the version in place.
- **Result (2026-10-28):** A: RMSE 0.1138, gain 0.02 (×100), wins 54%. A failed, so B against not added: gain 0.01, wins 52%. **Neither added.** These lot details mostly repeat what LotArea, Neighborhood and age already say; this settles the LotFrontage (fill vs drop) and LandSlope (keep vs drop) items: dropped.

#### Other group: added (3 of its 10 columns)

Leftover of the current model (×100), reference = the largest value:
- **No columns** by the merging rule: Foundation (PConc +1 ± 1, CBlock 0 ± 1, BrkTil −3 ± 3, rest within their ±), Electrical, RoofStyle, PavedDrive (N −2 ± 4, P −4 ± 5). Heating (only Wall, 4 houses, +8 ± 6, and OthW, 2, −13 ± 3) and SaleType (only Con, 2 houses, +12 ± 3; New ≈ SalePartial, already in): too few houses, as with BrkComm in Exterior1st.
- **MoSold left out:** every month within ±2, and a month isn't a scale (12 → 1 is one month apart, not 11).
- **Numbers:** `HeatingQCNumber` (Ex +1 ± 1, Gd and TA −1 ± 1), MasVnrArea (missing for 8 train, 15 test houses, whose veneer type is missing too: 0), YearBuilt (houses built before 1900: −5 to −6; correlation −0.59 with Age).
- **Result (2026-10-28):** the 3 columns as one candidate: RMSE 0.1118 vs 0.1138, gain 0.21 (×100), wins 96% of folds. **Added.** The current model scores **0.1118 ± 0.0089**.

## Baseline

A small first model, to get a CV score and its fold-to-fold noise before any comparison. Ridge on log(SalePrice), Ids 524 and 1299 removed from training.

| Column | Encoding |
|---|---|
| MSSubClass | target encoding, out-of-fold and blended (`TargetEncoder`) |
| Neighborhood | one-hot, NAmes as reference, Blueste with no column (23 columns) |
| LotArea | log, standardized |
| LotShape | one-hot, Reg as reference, IR2 + IR3 merged (2 columns) |
| OverallQual | one-hot, 1–3 merged, 5 as reference (7 columns) |
| Age at sale | YrSold − YearRemodAdd, standardized |
| GrLivArea | log, standardized |
| ExterQual | one-hot, TA as reference (3 columns) |
| GarageCars | one-hot 0 / 1 / 2 / 3+, 2 as reference (3 columns); Id 2577 → 0 |

None of these has missing values apart from GarageCars for Id 2577.

**Expected CV RMSE, written before running:** 0.13. (First guess 0.30; revised because a plain least-squares fit on OverallQual, log GrLivArea, YearBuilt and log LotArea already leaves a spread of 0.161, and the baseline has those plus more. Predicting the mean for every house gives about 0.40.)

**Result (2026-10-27):** CV RMSE **0.1453 ± 0.0094** (mean ± standard deviation over 50 folds: 5-fold, repeated 10 times, seed 42, on all 1,458 houses; no hold-out split, since Kaggle's test set is the final test). Worse than the 0.13 expected.
- Everything learned from the data (scaler, one-hot levels, target means, alpha) is inside one `Pipeline`, refitted on each training part.
- Alpha is chosen by `RidgeCV` inside each training part, from 61 values between 0.001 and 1000: it picked 0.40 to 0.79 (median 0.63), well inside the range. That is a weak penalty: a group of n houses keeps about n / (n + alpha) of its weight, so even Fa (14 houses) keeps 96%.
- Unknown one-hot levels (missing from a training part, or new in test) are encoded as all zeros, i.e. priced like the reference. MSSubClass 150 appears in test only, so `TargetEncoder` gives it the overall mean.
- **Why not 0.13:** the 4-column straight line scores 0.1609 on its own houses and 0.1611 in CV, so the 0.161 was a fair starting point. The baseline scores 0.1402 on its own houses and 0.1453 in CV: a small gap, so little overfitting. The extra columns and the more flexible form bought 0.016, about half the 0.03 assumed. The limit is what these 9 columns know, not the fit.
- Test Ids 2296 and 2550 have negative ages (−1, −2: remodelled after the sale was recorded). Harmless for a linear model; left as they are.

## Keep rule

Written 2026-10-27, before the first comparison. A change (new column, other encoding) is kept only if, on the same 50 folds as the model it's compared with:
1. **the mean RMSE improves by at least 0.001** (0.1 on the ×100 scale), and
2. **it wins on at least 80% of the folds** (40 of 50).

- **Compare fold by fold.** The folds' own spread (±0.94 ×100) is mostly which hard houses land in a fold, which hits both models alike. The fold-by-fold difference varies far less (±0.12 in a test without ExterQual).
- **Why 0.1:** ExterQual mostly repeats OverallQual and adds 0.05; a column with new information should clear at least twice that. Smaller consistent gains add up, and part 2 guards against luck.
- **The simpler version is always the current one** (added 2026-10-28, before the LotShape run): fewer columns first (dropped, then one number column, then one-hot), so every extra column has to earn its place, as a new column does. Reason: keep the model simple; close calls go to the simpler version. This applies to earlier comparisons too: OverallQual was run the other way round (one-hot as current), so it gets re-run with the number column as current.
- **Same number of columns** (added 2026-10-28, before the finished-basement run): each version is tested against not adding it; if more than one passes, the one with the lower mean RMSE is kept. Reason: neither is simpler, so there's no reason to give one the benefit of the doubt.
- **Each candidate is compared with the version actually in place** (written 2026-10-28, as done for LotShape): if a simpler version fails, it never becomes current, so the next, bigger version is compared with what is still in place, not with the failed one. Made explicit after the Conditions run, where the 5-column version had wrongly been set up against the failed 3-column one; chosen before computing the corrected comparison.
- Any change to this rule gets written here, with a reason that doesn't refer to a result.

## Submissions

### Submission 1 (2026-10-28): `submission.csv`

The current model (CV RMSE 0.1236 ± 0.0089), fitted on all 1,458 training houses, predictions turned back with exp().
- **Clipped to the training range:** each numeric test value is capped at the training minimum and maximum before predicting, so the straight lines stop at the edge of the data. Changes 3 houses: Id 2550 $1,095,537 → $694,923 (basement 5,095 → 3,206, finished 4,010 → 2,306, living area 5,095 → 4,476 sq ft, age −2 → 0), Id 2377 $380,697 → $374,150 (finished 2,461 → 2,306), Id 2296 $283,586 → $283,046 (age −1 → 0).
- Not overfitting: the model scores 0.1198 on its own houses against 0.1236 in CV. It's extrapolation: beyond the training range, a linear model just continues its lines.
- **Expected Kaggle score, written before uploading:** about 0.128 if Id 2550 sold for about $185k like the two removed houses (ln(695 / 185) = 1.32, adding 1.32² / 1,459 ≈ 0.0012 to the mean squared error), about 0.124 if it didn't.
- **Kaggle score: 0.13669**, worse than both expectations. Possible causes, not separable from one score: Id 2550 (+0.004 to +0.009 if it sold low and is among the scored houses), noise (folds of 292 houses varied by ±0.009), validation wear from about 15 comparisons on the same folds (+0.001 to +0.002). CV stays the guide for choosing; one Kaggle score is too noisy to choose by, and choosing by it would make the test set a second validation set.

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

- CV list below, one comparison at a time. Done: MSSubClass (target encoding stays), LotShape (dropped). OverallQual re-run (number column replaces one-hot). Neighborhood reference skipped (2026-10-28): at alpha ≈ 3 a neighbourhood of n houses keeps about n / (n + 3) of its weight (77% at 10 houses, over 90% for most), so the reference barely matters, and each comparison wears the CV score a little. Next: add more columns.
- Whether to scale the 0/1 columns: in the baseline they aren't scaled (only LotArea, Age and GrLivArea are). At the alpha RidgeCV picks (about 0.6) the pull on small groups is small either way; revisit if alpha grows as columns are added.
- Columns that may overlap with MSSubClass: `YearBuilt`. Check when we reach it. (`BldgType`, `HouseStyle`: dropped.)
- Group similar columns (garage, basement, pool) instead of going one by one.
