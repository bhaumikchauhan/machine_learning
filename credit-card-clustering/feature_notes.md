# Credit card clustering notes

Decisions and reasoning for grouping credit card customers into segments. The task is unsupervised: there is no target column, so there is no score to cross-validate against, and judging the groups is part of the work.

Status: in progress (started 2026-10-31).

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
