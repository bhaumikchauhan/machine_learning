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
- **DebtRatio when MonthlyIncome is missing:** median 1,159 against 0.30 when income is present.
- **RevolvingUtilization** (the share of the credit limit used) goes up to 50,708, while the 99th percentile is 1.09.
- **age 0:** 1 borrower.
- **MonthlyIncome missing:** 20%. The default rate is 5.6% when it is missing and 6.9% when it is present.
