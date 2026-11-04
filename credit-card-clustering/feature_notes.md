# Credit card clustering notes

Decisions and reasoning for grouping credit card customers into segments. The task is unsupervised: there is no target column, so there is no score to cross-validate against, and judging the groups is part of the work.

Status: finished (2026-11-03). Final result: k-means, k = 6, version 1 (MINIMUM_PAYMENTS blank = 0), `n_init=10`; six named segments (see "Decision: `n_init=10`" at the end for the current cluster numbers). Lessons are at the top of `concepts.md`.

**Who writes what** (the same split as Give Me Some Credit):
- **The user writes:** the Pipeline and anything fitted on data (scaler, PCA, k-means).
- **Claude writes:** plots and analysis scripts.

Data: `kaggle/input/ccdata/CC GENERAL.csv` (Kaggle dataset `arjunbhasin2013/ccdata`, not in git). 8,950 customers, 6 months of card use. The columns are `CUST_ID` plus 17 numbers: amounts in dollars (balance, purchases, cash advance, payments, credit limit), frequencies from 0 to 1, transaction counts, and tenure in months. Missing values: MINIMUM_PAYMENTS (313), CREDIT_LIMIT (1).

## Scaling (decided 2026-11-02)

k-means uses distances, so unscaled dollars decide everything: for purchases $1,000 vs $1,100 and frequency 0.1 vs 0.9, the distance is about 100 and nearly all of it comes from dollars. After StandardScaler (gaps in sd: 0.05 and 2.0), frequency decides.

Amounts are very skewed. PURCHASES: median $361, 75% under $1,110, maximum $49,040 (22.5 sd from the mean; 135 customers beyond 3 sd). Plain scaling would let k-means spend a cluster on a handful of big spenders, because squared distance makes a 22 sd customer weigh 484 against about 1 for a typical one. After log(1 + x), the biggest customer is 2.0 sd away and no one is beyond 3 sd (`plots/purchases_raw_against_log.png`). 2,044 customers (23%) bought nothing and form a spike at 0.

The log changes what "similar" means: $0–$20, $20–$420 and $420–$8,800 are each about 1 sd apart (raw: 0.01, 0.19, 3.92 sd). With no target, no score says whether raw or log is right.

**Decision:** log(1 + x) on every skewed amount column alike, then StandardScaler. Mixing logged and raw amount columns would let the raw ones' outliers dominate again. The clusters it gives are judged by the rules below.

## How the clusters are judged

There is no target, and inertia (the sum of squared distances to the centre) always falls as k rises: it is 0 at k = 8,950. So k is picked by:
- **the elbow:** the k after which inertia stops falling much;
- **cluster sizes:** a k with several clusters under 1% of customers is not good;
- **downstream use:** the segments are for marketing strategy, so each must be describable and different enough to act on.

**Range for k (fixed before seeing the curve):** 5 to 10. A marketing team can't run separate strategies for many more segments, and fewer than 5 would be too coarse to act on. The elbow is read only inside this range.

## Missing values

**MINIMUM_PAYMENTS (313 missing).** The column behaves like the minimum *due*, not the minimum paid: for 2,365 of the 8,637 customers who have it, it is larger than PAYMENTS (the total paid). Of the 313 blanks, 240 have PAYMENTS = 0 (none of the other customers do), 74 have a $0 balance, 188 a balance of $50 or less, and 51 over $1,000.

35 customers have balance, payments and minimum due all 0, yet 33 of them bought something (median $110) and 2 took a cash advance. That doesn't add up (refunds, payments recorded elsewhere, or bad records; the description doesn't say). They are 0.4% of customers, too few to form or move a cluster, so this isn't pursued.

**Decision:** test two fills.
- **Version 1 (now):** 0, meaning nothing due.
- **Version 2 (later):** the median MINIMUM_PAYMENTS of customers with a similar balance.

Without a target, the two are compared with the judging rules above, and by how many customers change cluster between them.

**CREDIT_LIMIT (1 missing).** An ordinary customer (balance $18, one $187 cash advance). **Decision:** fill with the median credit limit ($3,000), fitted inside the Pipeline so new customers get the same fill.

## Columns: log or not (decided 2026-11-02)

- **log(1 + x), then scaled:** MINIMUM_PAYMENTS, ONEOFF_PURCHASES, INSTALLMENTS_PURCHASES, PURCHASES, CASH_ADVANCE, CASH_ADVANCE_TRX, PAYMENTS, PURCHASES_TRX, BALANCE, CREDIT_LIMIT. Most extreme customer 7 to 32 sd from the mean when only scaled.
- **Scaled only:** PRC_FULL_PAYMENT, PURCHASES_FREQUENCY, ONEOFF_PURCHASES_FREQUENCY, PURCHASES_INSTALLMENTS_FREQUENCY, BALANCE_FREQUENCY, TENURE. Bounded, most extreme customer under 3 sd (BALANCE_FREQUENCY and TENURE have their tail on the low side, at most 0.5 sd above).
- **CASH_ADVANCE_FREQUENCY: clipped at 1, scaled, not logged.** 8 customers are above 1 (max 1.5), impossible for a frequency. The log barely bends a 0–1 scale (log(1 + 1) = 0.69); its customers beyond 3 sd come from 52% sitting at 0, not from a long tail.

## First k-means run (version 1, k = 5)

Pipeline: ColumnTransformer (MINIMUM_PAYMENTS: fill 0, log, scale; CREDIT_LIMIT: fill median, log, scale; the other 8 amount and count columns: log, scale; CASH_ADVANCE_FREQUENCY: clip at 1, scale; the other 6 bounded columns: scale), then KMeans.

**Prediction (before running):** the biggest of the 5 clusters holds about 40% of customers.

**Result:** the preprocessing gives 17 columns, no blanks, each with mean 0 and sd 1; the most extreme value is 5.1 sd (32 sd before the log). Cluster shares: 27.8%, 22.4%, 21.2%, 15.2%, 13.4%. The biggest cluster holds 28%, not the predicted 40%: k-means split the customers fairly evenly. No cluster is under 1%.

Two earlier attempts went wrong, and are worth remembering: a ColumnTransformer runs its entries side by side on the original columns, not one after another. Entries for "fill", "log" and "scale" on the same column gave three copies, two still with blanks (KMeans: "Input X contains NaN"). Later, overlapping "log" and "scale" entries gave 25 columns instead of 17. Chained steps go in a Pipeline inside one entry, and each column belongs to exactly one entry.

## Choosing k (version 1)

**Prediction (before running):** the inertia curve flattens at k = 7.

**Result** (`plots/inertia_by_k.png`, k = 2 to 15 for context, read only in 5 to 10):

| k | Inertia | Drop from k − 1 | Smallest cluster | Largest cluster |
|---|---|---|---|---|
| 5 | 77,397 | 10.2% | 13.4% | 27.8% |
| 6 | 70,862 | 8.4% | 10.1% | 24.7% |
| 7 | 67,394 | 4.9% | 4.7% | 22.5% |
| 8 | 65,100 | 3.4% | 6.4% | 22.0% |
| 9 | 61,205 | 6.0% | 2.8% | 21.9% |
| 10 | 59,299 | 3.1% | 2.8% | 21.8% |

The drops halve from 6 to 7 (8.4% to 4.9%), so the curve bends around 7, as predicted, but softly: there is no sharp elbow, and the drops after 7 are uneven (9 drops more than 8). Every k in the range passes the size check (smallest cluster above 1%). The elbow narrows k to about 6 to 8; it doesn't pick one.

**Profiles** (`cluster_profiles(k)` in `data_analysis.py`: size and median of key columns per cluster, in original units; medians because one $49,040 spender would pull a mean far from the typical member). CASH_ADVANCE is cash withdrawn on the card (a loan with a fee and interest from day one), not a payment in advance.

k = 6, named by the user:

| Cluster | Share | Median balance | Purchases | Cash advance | Purchase frequency | Paid in full | Credit limit | Name |
|---|---|---|---|---|---|---|---|---|
| 1 | 25% | $1,599 | $0 | $1,364 | 0.00 | 0% | $3,000 | only cash borrowers |
| 5 | 19% | $97 | $469 (all instalments) | $0 | 0.92 | 18% | $2,000 | frequent buyers, low balance and limit |
| 2 | 16% | $759 | $2,321 | $0 | 1.00 | 8% | $6,500 | high usage buyers |
| 3 | 16% | $812 | $425 (all one-off) | $0 | 0.25 | 0% | $2,500 | moderate users |
| 0 | 14% | $2,641 | $833 | $1,945 | 0.75 | 0% | $5,000 | cash and credit users |
| 4 | 10% | $12 | $114 | $0 | 0.17 | 0% | $2,700 | inactive |

k = 7 keeps the six and adds a 5% cluster of newer customers (median tenure 7 months, against 12 everywhere else) who mainly borrow cash. Not a separate strategy: onboarding offers can use TENURE directly. k = 8 drops that cluster and splits two of the six: high usage buyers into those who carry a balance (12%, 0% paid in full) and those who pay off (6%, 88%); and cash and credit users into frequent (10%, purchase frequency 0.90) and occasional buyers (8%, 0.25). The user judged that mixing buying frequency with cash isn't useful: marketing would treat "frequent vs occasional buyer" and "borrows cash or not" as separate questions.

**Goal, stated by us:** the Kaggle dataset has no problem statement, so the goal is ours: segments for a marketing strategy, from k-means on all columns (this problem is for learning unsupervised methods).

**Decision: k = 6 for version 1.** All six clusters have distinct names and actions; 7 and 8 add nothing usable.

**Alternative, not taken:** segment each dimension on its own (buying frequency, cash advance, paying in full) by rules or by k-means on that dimension's columns, and cross the results. Each answer stays separate, but the cells multiply (3 × 2 × 2 = 12) and some become tiny. That is rule-based segmentation, not clustering on all columns.

## PCA picture (version 1, k = 6)

PCA on the 17 scaled columns, only to draw the clusters in 2-D; k-means still clusters on all 17.

**Prediction (before running):** the first 2 axes capture 44% of the spread.

**Result** (`plots/pca_clusters.png`): the first 2 axes capture 33.7% + 22.4% = **56.1%** of the spread, more than the predicted 44%. The cluster numbers match the k = 6 profiles. Only cash borrowers (cluster 1) stand clearly apart on the left; the others touch or overlap. Overlap in 2-D doesn't mean overlap in 17 columns: 44% of the spread isn't in the picture.

Reading the axes (`pca.components_`, each axis's weight on every scaled column):
- **Axis 1, buy or borrow cash:** purchase columns +0.21 to +0.38, cash advance columns −0.24 to −0.29, the rest near 0. Only cash borrowers sit far left, high usage buyers far right.
- **Axis 2, how much balance the customer carries and moves:** BALANCE +0.43, MINIMUM_PAYMENTS +0.40, PAYMENTS +0.35, BALANCE_FREQUENCY +0.34. Inactive customers ($12 balance) sit at the bottom, cash and credit users ($2,641 balance) at the top.

## Version 2: MINIMUM_PAYMENTS filled from similar customers

**Decision:** `KNNImputer`, with neighbours found on BALANCE, CREDIT_LIMIT and PAYMENTS. It measures distance, so the same scaling lesson applies: the neighbour columns are logged and scaled before imputing. It fills with the *mean* of the neighbours' values, not the median planned earlier; on the log scale that mean is far less pulled by one big customer.

Implemented by Claude (on request) as `build_pipeline(version=2)`: one entry for MINIMUM_PAYMENTS, BALANCE, CREDIT_LIMIT and PAYMENTS with log, then scale, then `KNNImputer(n_neighbors=5)` (impute last, because the neighbours are found by distance). The single CREDIT_LIMIT blank is filled by its neighbours too.

**Result** (`compare_versions()` in `data_analysis.py`; cluster numbers matched by the most shared customers):

| | Customers who change cluster |
|---|---|
| All | 297 of 8,950 (3.3%) |
| The 313 with a blank MINIMUM_PAYMENTS | 60 |
| The other 8,637 | 237 (the blanks move the centres a little, and customers near a boundary flip) |
| For comparison: version 1 with a different KMeans seed (0, 1, 2, 4) | 8 to 143 |

The version 2 fills are small: median $35, quartiles $14 to $154 (version 1: $0). Cluster shares barely move (the largest change: inactive 10.1% to 11.0%). One seed (3) changed 1,793 customers: a different solution, not noise (see below).

**Decision: keep version 1** (blank = 0). Version 2 moves 3.3% of customers and leaves the six segments and their shares about the same, so the simpler fill stays.

## Local minimum: the seed that changed 1,793 customers

KMeans with `init='k-means++'` runs once by default (`n_init='auto'` means 1 start). Inertia at k = 6, version 1:

| Start | Inertia |
|---|---|
| seeds 42, 0, 1, 2, 4 | 70,860 to 70,862 |
| seed 3 | **73,882** (4% worse: a different, poorer solution) |
| `n_init=10`, seed 42 (best of 10 starts) | 70,860 |

Seed 3 got stuck in a local minimum: from its starting centres, every small move made things worse, though a better arrangement existed. Our seed 42 is at the good solution (best of 10 gives the same inertia), so the profiles stand.

**Decision: `n_init=10` added to KMeans** (both versions). Inertia 70,860; 108 customers (1.2%) change cluster against the single start, the six profiles stay within a few dollars, but the cluster **numbers** change (`plots/pca_clusters.png` regenerated):

| Name | Number before (single start) | Number now (n_init=10) | Share |
|---|---|---|---|
| only cash borrowers | 1 | 0 | 25% |
| frequent buyers, low balance and limit | 5 | 4 | 20% |
| high usage buyers | 2 | 2 | 16% |
| moderate users | 3 | 1 | 16% |
| cash and credit users | 0 | 3 | 14% |
| inactive | 4 | 5 | 10% |

## After closing: hierarchical clustering (Ward) against k-means

Same preprocessing (version 1), `AgglomerativeClustering(n_clusters=6, linkage='ward')` in place of KMeans. Ward merges the pair of clusters whose merge raises inertia least, the quantity k-means minimises, but it never undoes a merge, so the clusters should be similar, not identical. Judged by: customers who change cluster against k-means (after matching numbers), whether the six profiles keep their names, and no cluster under 1%.

**Prediction (before running):** 5% of customers change cluster.

**Result** (`compare_ward_with_kmeans()` and `plot_dendrogram()` in `data_analysis.py`, `plots/ward_dendrogram.png`): **25.1%** of customers (2,242) change cluster, five times the predicted 5%.

| Segment (k-means name) | k-means share | Ward share | Stayed in the same segment |
|---|---|---|---|
| only cash borrowers | 25% | 25% | 1,938 of 2,218 (87%) |
| high usage buyers | 16% | 16% | 1,137 of 1,445 (79%) |
| cash and credit users | 14% | 21% | 1,101 of 1,241 (89%) |
| inactive | 10% | 14% | 732 of 893 (82%) |
| frequent buyers, low balance and limit | 20% | 14% | 1,071 of 1,765 (61%) |
| moderate users | 16% | 11% | 729 of 1,388 (53%) |

The six profiles keep their names (the medians move, e.g. Ward's cash and credit users have a $2,046 balance against $2,610), and the smallest cluster is 11%. The clear segments (cash borrowers, cash and credit) agree; the in-between ones (moderate users, frequent buyers) lose 39–47% of their customers to neighbours. In the dendrogram, the cut for 6 sits among merges at about 98, 99 and 102, so 5, 6 and 7 clusters are almost equally far apart; the big gaps are higher up (2 or 3 clusters).

**Spread within each segment** (the user's check; mean squared distance to the segment's centre, on the 17 logged and scaled columns the clustering used):

| Segment | k-means | Ward |
|---|---|---|
| only cash borrowers | 6.3 | 8.6 |
| moderate users | 7.0 | 5.1 |
| high usage buyers | 7.2 | 7.5 |
| cash and credit users | 9.2 | 9.2 |
| frequent buyers, low balance and limit | 7.1 | 9.3 |
| inactive | 14.2 | 14.1 |
| **Total inertia** | **70,860** | **80,619** (+14%) |

k-means is tighter overall, but that is the quantity it minimises, so this alone doesn't make it "right". Ward's moderate users are tighter (5.1) because Ward made the segment smaller (11% against 16%) and handed its looser members to neighbours. Measured on raw dollars, the spread would again be ruled by the big spenders, so the clustering's own space is the one to measure in.

Where k-means' 1,388 moderate users went under Ward: 729 stayed, 213 inactive, 213 cash and credit users, 125 high usage buyers, 67 frequent buyers, 41 only cash borrowers. The border customers lean three ways (going quiet, borrowing cash, spending more), not mainly towards the buyer segments.

**Conclusion (the user's message for marketing):** about half of the moderate users are clearly moderate; the rest are border customers whose segment is uncertain. The clear segments (only cash borrowers, cash and credit users, inactive, high usage buyers) agree across both methods, so k-means stays the result.

**Border flag (rule fixed before looking):** from `pipeline.transform(X)`, each customer's distance to the 6 k-means centres. Border when nearest ÷ second nearest > 0.9 (the two nearest centres within about 10%). Relative, not a fixed gap, because the inactive segment's spread (14) is twice the others' and its distances run larger. **Expectation:** the flagged customers should roughly be the ones Ward moved.

**Result** (`border_flag()` in `data_analysis.py`): 1,360 border customers (15.2%). Ward moved 57% of them against 19% of the others, so the flag does find the uncertain customers. But only 35% of Ward's 2,242 moves are flagged: most of Ward's disagreement is not about close calls but about clusters of a different shape (its early merges can't be undone). Frequent buyers show it most: 15% border, but 39% moved.

| Segment | Border share | Ward moved |
|---|---|---|
| moderate users | 22% | 47% |
| inactive | 21% | 18% |
| cash and credit users | 18% | 11% |
| high usage buyers | 16% | 21% |
| frequent buyers, low balance and limit | 15% | 39% |
| only cash borrowers | 6% | 13% |

The expectation held only partly: the flag marks the close calls, which Ward moves three times as often, but Ward's other moves come from its different method, not from the customers being uncertain.

**Ward comparison closed (2026-11-04).** k-means stays the result; the border flag (15% of customers) is the per-customer uncertainty to hand marketing alongside the segment.

## After closing: DBSCAN

Same 17 scaled columns (version 1). DBSCAN finds dense regions by chaining core points (at least `min_samples` neighbours within `eps`); points outside every dense region are noise (−1). Question: does it mark the k-means border customers as noise, and does it find the same segments?

**min_samples = 34** (2 × 17 columns, the rule of thumb; fixed before running). min_samples is the neighbours one point needs within eps to be core, not the minimum cluster size; clusters grow far bigger by chaining. 500 was considered and rejected: 5.6% of all customers inside each radius needs a large eps, which chains neighbouring segments into one blob and swallows the border customers.

**eps rule (fixed before looking):** from the k-distance curve, each customer's distance to their 34th nearest neighbour (themselves included, as DBSCAN counts), sorted. eps = the value where the curve bends sharply upwards: below it, customers sit in dense regions; above it, they are the sparse ones.

**eps = 2.5** (the user's choice; `plots/dbscan_k_distance.png`). About 12% of customers have their 34th neighbour farther than 2.5, so about 88% are core points; 2.5 sits where the curve starts to bend (around customer 7,500 to 8,500). 1.2 was considered and rejected: it is read off the curve at customer 2,100, so 77% of customers would not be core.

**Prediction (before running):** fewer clusters than k-means' six, because the segments touch (PCA picture, Ward's weak borders) and with 88% of customers as core points, chains of cores will run across the touching zones and merge segments into big clusters.

**Result** (user's `build_dbscan_pipeline()`, `compare_dbscan()` in `data_analysis.py`): **1 cluster**; 7,938 core, 682 border, 330 noise (3.7%). The prediction (fewer than six) held, in the extreme: chains of core points connect every segment.

The noise is the far-out customers, not the between-segment ones: median distance to the nearest k-means centre 5.0 against 2.5 for the rest. k-means border customers are noise 3 times as often (8.4% against 2.8%), but 205 of the 330 come from the inactive segment (23% of it; the segment with the widest spread, 14). Noise customers have a shorter median tenure (10 months against 12), small balances ($196) and some cash advance ($369).

| k-means segment | Customers | Noise | Share |
|---|---|---|---|
| inactive | 893 | 205 | 23.0% |
| cash and credit users | 1,241 | 52 | 4.2% |
| high usage buyers | 1,445 | 23 | 1.6% |
| moderate users | 1,388 | 19 | 1.4% |
| only cash borrowers | 2,218 | 19 | 0.9% |
| frequent buyers, low balance and limit | 1,765 | 12 | 0.7% |

**Conclusion (the user's reading):** one cluster means there is no low-density gap anywhere: between any two customers, even far-apart ones (cash borrowers at −4 and high usage buyers at +4 on PCA axis 1), runs a path of dense neighbours. So the six k-means segments are cuts through one continuous cloud: useful for marketing (distinct profiles and actions), but their borders are a choice, not something the data shows. That fits Ward putting 25% of customers elsewhere. DBSCAN's noise (3.7%) is a different thing from the border flag: it marks the far-out customers, mostly inactive ones.

**Is eps = 2.5 right? Sensitivity check (criterion fixed before running):** there's no score for eps, so the check is whether the conclusion depends on it. DBSCAN is run for eps 1.0 to 3.5 (min_samples 34) and for min_samples 17 and 68 at eps 2.5. The "no gaps" conclusion is overturned if any setting gives **at least 2 clusters of 5% of customers or more, with under 30% noise**.

**Result** (`dbscan_sensitivity()` in `data_analysis.py`):

| eps | min_samples | Clusters | Clusters ≥ 5% | Largest | Second | Noise |
|---|---|---|---|---|---|---|
| 1.00 | 34 | 4 | 2 | 14.1% | 6.7% | 77.1% |
| 1.25 | 34 | 5 | 2 | 25.9% | 16.6% | 52.4% |
| 1.50 | 34 | 4 | 1 | 69.4% | 0.6% | 29.1% |
| 1.75 | 34 | 1 | 1 | 84.2% | | 15.8% |
| 2.00 | 34 | 1 | 1 | 90.0% | | 10.0% |
| 2.25 | 34 | 2 | 1 | 93.2% | 0.5% | 6.3% |
| 2.50 | 34 | 1 | 1 | 96.3% | | 3.7% |
| 3.00 | 34 | 2 | 1 | 98.7% | 0.3% | 1.0% |
| 3.50 | 34 | 1 | 1 | 99.9% | | 0.1% |
| 2.50 | 17 | 1 | 1 | 98.5% | | 1.5% |
| 2.50 | 68 | 1 | 1 | 94.0% | | 6.0% |

No setting meets the criterion: two big clusters appear only at eps 1.0 and 1.25, where half or more of the customers are noise. So the "no gaps" conclusion doesn't depend on eps = 2.5.

At eps 1.25, the two dense cores are 1,487 only cash borrowers and a buyers' core (1,077 frequent buyers, 614 high usage buyers, 562 moderate users); all 1,241 cash and credit users are noise or tiny clusters. At eps 1.5 the two cores join, and 491 cash and credit users are in the joined cluster. The densest structure is two poles, buying and borrowing cash (PCA axis 1), linked by customers who do both.
