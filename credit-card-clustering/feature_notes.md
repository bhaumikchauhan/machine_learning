# Credit card clustering notes

Decisions and reasoning for grouping credit card customers into segments. The task is unsupervised: there is no target column, so there is no score to cross-validate against, and judging the groups is part of the work.

Status: in progress (started 2026-10-31).

**Who writes what** (the same split as Give Me Some Credit):
- **The user writes:** the Pipeline and anything fitted on data (scaler, PCA, k-means).
- **Claude writes:** plots and analysis scripts.

Data: `kaggle/input/ccdata/CC GENERAL.csv` (Kaggle dataset `arjunbhasin2013/ccdata`, not in git). 8,950 customers, 6 months of card use. The columns are `CUST_ID` plus 17 numbers: amounts in dollars (balance, purchases, cash advance, payments, credit limit), frequencies from 0 to 1, transaction counts, and tenure in months. Missing values: MINIMUM_PAYMENTS (313), CREDIT_LIMIT (1).
