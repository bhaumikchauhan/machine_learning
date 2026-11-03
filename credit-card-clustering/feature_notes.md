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
