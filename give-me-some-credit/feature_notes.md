# Give Me Some Credit feature notes

Decisions and reasoning for the Give Me Some Credit model. The task is to predict, for 101,503 test borrowers, the probability of a serious late payment (90+ days) within 2 years (`SeriousDlqin2yrs`). There are 150,000 training borrowers, 6.7% of whom are positive. Kaggle scores the AUC.

Status: in progress.

**Who writes what:**
- **The user writes:** the Pipeline, the CV loop and the leakage-sensitive parts (anything fitted on data).
- **Claude writes:** plots and analysis scripts.
- **The user reviews** Claude's code.

Data: `kaggle/input/GiveMeSomeCredit/` (not in git). `cs-training.csv`, `cs-test.csv`, `sampleEntry.csv` (Id, Probability), `Data Dictionary.xls`.

## Validation and keep rule

Decided before the first model (2026-10-29):
- **Score:** AUC. It uses only the order of the probabilities, so no cut-off is needed. Accuracy is ruled out: "always no" scores 93.3%.
- **Folds:** `RepeatedStratifiedKFold`, 5 folds × 10 repeats, fixed seed. Stratified, so every fold keeps the 6.7% share of positives; plain folds could vary by chance. The fixed seed puts the same borrowers in the same folds, so versions can be compared fold by fold.
- **Keep rule:** the candidate must win at least 80% of folds and gain at least a minimum mean AUC. **The minimum gain is still to be set**: after the first model shows the fold-to-fold spread, and before any comparison.
- Conventions carried over from House Prices: compare each candidate with the version in place; the simpler version is current; write the prediction before running.

## Column overview (2026-10-29)

All 10 columns are numeric. Values from `cs-training.csv` (150,000 borrowers):

| Column | Missing | Median | 99th percentile | Max |
|---|---|---|---|---|
| RevolvingUtilizationOfUnsecuredLines | 0 | 0.15 | 1.09 | 50,708 |
| age | 0 | 52 | 87 | 109 |
| NumberOfTime30-59DaysPastDueNotWorse | 0 | 0 | 4 | 98 |
| DebtRatio | 0 | 0.37 | 4,979 | 329,664 |
| MonthlyIncome | 29,731 | 5,400 | 25,000 | 3,008,750 |
| NumberOfOpenCreditLinesAndLoans | 0 | 8 | 24 | 58 |
| NumberOfTimes90DaysLate | 0 | 0 | 3 | 98 |
| NumberRealEstateLoansOrLines | 0 | 1 | 4 | 54 |
| NumberOfTime60-89DaysPastDueNotWorse | 0 | 0 | 2 | 98 |
| NumberOfDependents | 3,924 | 0 | 4 | 20 |

**Past late payments predict future ones:** the target is 90+ days late in the *next* 2 years, while NumberOfTimes90DaysLate counts the past.

| NumberOfTimes90DaysLate | Borrowers | Default rate |
|---|---|---|
| 0 | 141,662 | 4.6% |
| 1 to 17 | 8,069 | 41.2% |
| 96 or 98 | 269 | 54.6% |

- The user predicted the "1 to 17" group would be lower than average, reasoning that the bank still gave them credit. It is 6 times higher: being given credit doesn't mean being low risk, since banks also lend to risky borrowers at higher interest.

**Values that look wrong, to look at:**
- **96 and 98 in the three late-payment counts:** 5 and 264 borrowers, the same in all three columns. The next highest values are 11 to 17. Most likely codes ("unknown" or "not reported"), not counts; not documented in `Data Dictionary.xls`. The user's check: if they meant days late, every such borrower would be positive, and not all are.
  - **Decided (2026-10-29): one 0/1 column `LateCountsCoded`** (1 for the 269 borrowers), and their three late-payment counts set to 0. The model learns their risk (54.6%) from the 0/1 column. The flag doesn't use the target, so nothing has to be fitted out of fold.
  - **Considered first:** fill each column with the count whose default rate is closest to 54.6%, chosen out of fold because it uses the target.
    - NumberOfTimes90DaysLate: 3 (57.7%; 2 is 49.9%).
    - 60–89 days: 3 (56.6%; 2 is 50.2%).
    - 30–59 days: no count reaches 54.6% (6 or more is 50.2%), so the rule would pick the top group and double it from 239 to 508 borrowers, most of them coded. The rule breaks down because these borrowers aren't like any count.
  - **Shape of the effect:** it rises steeply, then levels off. For 90+ days: 4.6% at 0, 33.7% at 1, 49.9% at 2, 57.7% at 3, about 60 to 70% from 4 on. Groups above 4 are small, so their rates are noisy.
- **DebtRatio when MonthlyIncome is missing:** median 1,159 against 0.30 when income is present.
  - **Why:** when income is missing or 0, DebtRatio holds the monthly debt payment in dollars (divided by 1). It is a whole number for 100% of those 31,365 borrowers, against 2.3% when income is above 0.
  - **Decided (2026-10-29): a `Debt` column (monthly debt payments in dollars) in place of DebtRatio.** Debt = DebtRatio when income is missing or 0; otherwise Debt = MonthlyIncome × DebtRatio. (For $1 incomes, the product is the debt as well.)
- **Very small incomes:** 605 borrowers have exactly $1; 38 have $2 to $99.
  - Their default rate (about 2.7%) is like that of 0 (4.0%) and missing (5.6%), not like real low earners ($1,000 to $1,999: 10.8%, the riskiest group). So they look like "not reported", not tiny incomes. $100 to $499 (416 borrowers, 5.3%) is unclear, so the cut-off is $100.
  - **Decided (2026-10-29): a 0/1 column `IncomeNotReported`** for income missing, 0 or under $100 (32,008 borrowers). 
  - **Their MonthlyIncome, decided (2026-10-30): a fixed 0.** Logistic regression needs a number. With the flag in the model the value barely matters: all flagged borrowers get the same income part, and the flag's weight learns the rest (only the sum counts). A fixed number needs no fitting, so nothing to do out of fold. Boosting could take the blanks as they are.
- **RevolvingUtilization** (the share of the credit limit used) goes up to 50,708, while the 99th percentile is 1.09.
  - **Above 1 is possible:** interest or fees can push a balance over the limit. Default rate by utilization: 0 to 0.5: 2.7% (108,712 borrowers); 0.5 to 1: 15.3%; 1 to 1.5: 39.7% (2,721); 1.5 to 2: 44.5% (229); 2 to 10: 28.5% (130); 10 to 100: 33.3% (18); 100 or more: **4.9%** (223).
  - **Decided (2026-10-30):**
    - **100 or more is invalid:** risk that far over the limit should be very high, but it is below average. Perhaps a balance not divided by the limit. These 223 borrowers get a 0/1 column `UtilizationInvalid`, and their utilization is set to a fixed 0 (the same reasoning as for income).
    - **Cap 2 to 100 at 2.** Logistic regression gives utilization one straight line; values out to 10 or 100, where risk no longer rises, would stretch the line and bend the weight for the ordinary 0 to 1 borrowers.
    - The user first suggested a cap at 10, thinking of the values as groups; the straight-line point changed that. A fixed rule, nothing fitted.
  - Even from 0 to 2 the rise isn't straight (steep up to about 1.5, then flatter); to look at after the first model.
- **age 0:** 1 borrower; the next youngest is 21, and the test set's youngest is 21. **Decided (2026-10-30): set to a fixed 52**, the mean age (52.3), rounded. Written as a fixed number, so nothing is fitted; for one borrower, the mean's dependence on the fold doesn't matter.
- **MonthlyIncome missing:** 20%. The default rate is 5.6% when it is missing and 6.9% when it is present.
