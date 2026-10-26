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

### Alley: two 0/1 columns (gravel, paved); no alley is the reference

Type of alley access. **An empty cell is not missing data:** `data_description.txt` says `NA = No alley access`, but pandas reads the text "NA" as a missing value by default. Many later columns (pool, garage, basement, fence) work the same way.
- `train.csv`: no alley 1,367 (94%, median $165k), gravel 50 (median $120k), paved 41 (median $172k). `test.csv`: 1,352 / 70 / 37. See `plots/street_alley.png`.
- **Decision:** two 0/1 columns, `alley_gravel` and `alley_paved`. No alley gets no column of its own: it is the reference, with both columns 0, so its houses get no Alley weight, as for paved Street. Each weight then means "difference from a house with no alley".
- Implementation: turn the empty cells into an explicit "None" value first, so they aren't treated as missing. With scikit-learn's `OneHotEncoder`, the reference is set with `drop=`.

### LotShape: one-hot, Reg as reference, IR2 and IR3 merged; CV to compare

General shape of the lot, with a natural order: Reg (regular), IR1 (slightly irregular), IR2 (moderately irregular), IR3 (irregular). See `plots/lot_shape.png`.
- `train.csv`: Reg 925 (median $146k), IR1 483 ($189k), IR2 41 ($221k), IR3 9 ($215k). `test.csv`: 934 / 484 / 35 / 6.
- **Irregular lots sell for more**, most likely because they are newer and bigger. Median year built: 1967 / 1984 / 1995 / 1998; median lot: 8,800 / 10,624 / 13,682 / 16,226 sq ft; median living area: 1,365 / 1,588 / 1,668 / 1,675 sq ft. Newer subdivisions have curving streets and culs-de-sac (77 of the 176 cul-de-sac lots are IR1, 1 is Reg). So LotShape may mostly stand in for what YearBuilt, LotArea and GrLivArea already say.
- **Ordered or one-hot:** an ordered column (0–3) with one weight forces equal steps. The steps here are uneven (+$43k, +$32k, −$6k; the last on 9 houses).
- **Overlapping columns don't cause underfitting** (a model too simple for the pattern). They make the model unsure which column deserves the credit, so it splits the weight between them, sometimes unstably. Predictions usually stay fine; the weights become hard to read. Ridge spreads the credit. Overlap is a keep-or-drop question, not an encoding question.
- **Decision:** one-hot with Reg as the reference: `lot_shape_ir1`, and `lot_shape_ir2_ir3` (IR2 and IR3 merged, 50 houses), so no weight is learned from 9 houses.
- **To compare in CV:** ordered (0–3), this one-hot version, and dropping LotShape.

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

### BldgType: dropped (MSSubClass already says it)

Type of dwelling: 1Fam (single-family), 2fmCon (two-family conversion), Duplex, Twnhs (townhouse inside unit), TwnhsE (townhouse end unit). The data spells them `2fmCon`, `Duplex`, `Twnhs`; `data_description.txt` says `2FmCon`, `Duplx`, `TwnhsI`. See `plots/bldg_type.png` and `plots/bldg_type_vs_mssubclass.png`.
- `train.csv`: 1Fam 1,218 (mean $186k), 2fmCon 31 ($128k), Duplex 52 ($134k), Twnhs 43 ($136k), TwnhsE 114 ($182k). `test.csv`: 1,205 / 31 / 57 / 53 / 113. None missing.
- **Almost a copy of MSSubClass** (train + test): Duplex is exactly code 90 (109 houses); 60 of the 62 2fmCon are code 190; 322 of the 323 townhouses are the PUD codes 120–180; codes 20–85 are single-family apart from 3 houses.
- **The only new information** is inside vs end unit within a PUD code (code 160: 71 inside, 56 end). After quality, size, age, lot size and the MSSubClass group, their leftovers are the same (−0.003 and −0.001). The raw $46k Twnhs/TwnhsE gap comes from the codes: end units are mostly code 120 (1-story, median built 2003), inside units mostly code 160. (For Duplex and 2fmCon, a leftover of 0 after MSSubClass is automatic, since they sit inside one code; it's the mapping itself that shows they add nothing.)
- **Decision:** drop. Holds whether MSSubClass ends up target-encoded or one-hot: both carry each code's price level.

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
- Compare LotShape ordered (0–3), one-hot (IR1, IR2+IR3) and dropped, with cross-validation.
- Compare keeping `land_slope_sloped` against dropping it (overlap with LandContour), with cross-validation.
- Compare NAmes against CollgCr as the Neighborhood reference, with cross-validation.
- Compare Feedr and PosN merged into the condition reference against their own columns, with cross-validation.
- Whether to scale the 0/1 columns: decide when writing the pipeline (it changes how hard the penalty pulls on small groups).
- Columns that may overlap with MSSubClass: `HouseStyle`, `YearBuilt`. Check when we reach them. (`BldgType`: dropped.)
- Group similar columns (garage, basement, pool) instead of going one by one.
