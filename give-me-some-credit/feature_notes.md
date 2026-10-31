# Give Me Some Credit feature notes

Decisions and reasoning for the Give Me Some Credit model. The task is to predict, for 101,503 test borrowers, the probability of a serious late payment (90+ days) within 2 years (`SeriousDlqin2yrs`). There are 150,000 training borrowers, 6.7% of whom are positive. Kaggle scores the AUC.

Status: finished (2026-10-31). Final model: gradient boosting on the raw columns (CV AUC 0.865, Kaggle private 0.86670).

**Who writes what:**
- **The user writes:** the Pipeline, the CV loop and the leakage-sensitive parts (anything fitted on data).
- **Claude writes:** plots and analysis scripts.
- **The user reviews** Claude's code.

Data: `kaggle/input/GiveMeSomeCredit/` (not in git). `cs-training.csv`, `cs-test.csv`, `sampleEntry.csv` (Id, Probability), `Data Dictionary.xls`.

## Validation and keep rule

Decided before the first model (2026-10-29):
- **Score:** AUC. It uses only the order of the probabilities, so no cut-off is needed. Accuracy is ruled out: "always no" scores 93.3%.
- **Folds:** `RepeatedStratifiedKFold`, 5 folds × 10 repeats, fixed seed. Stratified, so every fold keeps the 6.7% share of positives; plain folds could vary by chance. The fixed seed puts the same borrowers in the same folds, so versions can be compared fold by fold.
- **Keep rule:** the candidate must win at least 80% of folds and gain **at least 0.001 mean AUC**. The gain was set on 2026-10-30, after the baseline showed a fold spread of ±0.005 and before any comparison. Don't change it because of results.
- Conventions carried over from House Prices: compare each candidate with the version in place; the simpler version is current; write the prediction before running.

## Baseline (2026-10-30)

Logistic regression on a few columns, chosen by the user as the ones that tell most about the borrower's finances:
- NumberOfTimes90DaysLate and LateCountsCoded
- RevolvingUtilizationOfUnsecuredLines and UtilizationInvalid
- DebtToIncome, MonthlyIncome and IncomeNotReported

**MonthlyIncome clipped at $15,000** (decided 2026-10-30). For reported incomes, risk falls steadily from $1,000 to about $15,000 (10.6% at $1,000 to $2,000, 4.2% at $10,000 to $15,000), then stops falling (4.4% at $15,000 to $25,000, 5.3% above). Not fitted to the line: $100 to $1,000 (2,518 borrowers, 6.5%) is lower than the next band.

**Result (2026-10-30, `give_me_some_credit.py`, written by the user):** `LogisticRegressionCV(scoring='roc_auc')`, numbers scaled, flags passed through. **AUC 0.822 ± 0.005** (mean and sd over the 50 folds).
- Bugs found in review: the flags were dropped (a ColumnTransformer drops unnamed columns): 0.818 → 0.820. C was chosen by accuracy (the default): → 0.821.
- **The chosen C sits at the bottom edge of every grid tried.** Inner AUC is flat at 0.8223 for C = 1e-09 to 1e-05, against 0.8211 at 1e-04 and about 0.8195 from 1e-03 up.
  - Below about 1e-05 the penalty only shrinks all weights together, in the same proportions. The order of the borrowers is then frozen, and AUC uses only the order. For AUC, any C at or below 1e-05 is the same model.
  - The probabilities from such a model are all close to the 6.7% base rate. That doesn't matter for Kaggle's AUC, but it would for a cut-off or for the probabilities themselves (calibration, later).

## Additions

### 30–59 and 60–89 day late counts

- **Plan, written before running (2026-10-30):** add both columns as one step (scaled, with their 96/98 codes already set to 0), and compare with the baseline on the same 50 folds using the keep rule.
- **Prediction (user):** mean AUC gain of 0.02 (from 0.822 to about 0.842).
- **Result:** gain **0.0307**, winning **100%** of folds. **Kept: AUC 0.822 → 0.853.** The gain was larger than predicted: the earlier late payments carry information the 90-day count doesn't (5,731 borrowers have a 60–89 day late payment, against 8,069 with a 90-day one, and 16,033 have a 30–59 day one).

### NumberOfOpenCreditLinesAndLoans

- **Shape:** a lopsided U. 0 lines: 25.6% (1,888 borrowers); 1: 13.2%; 2: 9.1%; 3: 7.5%; 4 to 5: 6.4%; 6 to 8: 5.3% (the lowest); 9 to 12: 5.9%; 13 to 16: 6.5%; 17 to 20: 6.9%; 21 or more: about 7%.
- **Plan, written before running (2026-10-30):** add it raw (scaled) and compare with the current model (AUC 0.853) using the keep rule.
- **Prediction (user):** a gain below 0.005. One straight line can't follow the U.
- **Result:** AUC 0.851, a gain of **−0.0020**, winning **0%** of folds. **Not added.** Worse in every fold, not just no better.
  - At the chosen C (1e-05 in most folds), the weights are tiny, and each weight follows mostly that column's own straight-line link to default; the model hardly weighs the columns against each other.
  - The raw column's line is shallow (falling up to 6–8 lines, rising after), so it adds a small, wrong push for most borrowers.
  - Candidates for the shape: the steep part only (few lines).
- **Second try, written before running (2026-10-30):** a capped count, min(lines, 4), scaled; the raw column stays out.
  - The line then covers only the steep part (25.6% at 0 down to 7.5% at 3), the same idea as the utilization cap.
  - Chosen over a 0/1 flag for 0 to 3 lines, which would treat 0 lines (25.6%) like 3 (7.5%).
  - This is a second candidate for the same column, chosen after seeing the first fail, so it is a weaker test.
- **Prediction (user):** a gain of 0.02.
- **Result:** AUC 0.852, a gain of **−0.0007**, winning **16%** of folds. **Not added.**
- **Why: the current model already ranks these borrowers as risky.** Average risk rank (percentile, from 5-fold out-of-fold predictions at C = 1e-05) by open lines:
  - 0 lines: **0.85**, higher than the average defaulter's 0.83
  - 1 line: 0.64
  - 2 lines: 0.53
  - 3 lines: 0.49
  - 4 or more: 0.49

  Other columns already carry the information, so a column can show a strong default rate on its own and still add nothing (as in House Prices, where decisions depended on the other columns). The leftover check, how far the model already goes for a group, would have shown this before running.

### NumberRealEstateLoansOrLines

- **Check before predicting (2026-10-31):** default rate by count, against the current model's average risk rank (5-fold out-of-fold, C = 1e-05):

  | Count | Borrowers | Default rate | Risk rank |
  |---|---|---|---|
  | 0 | 56,188 | 8.3% | 0.49 |
  | 1 | 52,338 | 5.3% | 0.49 |
  | 2 | 31,522 | 5.6% | 0.52 |
  | 3 | 6,300 | 6.7% | 0.52 |
  | 4 | 2,170 | 8.9% | 0.55 |
  | 5 or more | 1,482 | 15.2% | 0.57 |

  - A U shape. Unlike open lines, the model misses it: 0 loans (probably no mortgage) defaults at 8.3% against 5.3% for 1 loan, yet both are ranked alike.
- **Plan, written before running:**
  - A 0/1 column `NoRealEstateLoans` (passthrough).
  - Plus the count capped at 5 (scaled), for the rising end: from 1 loan up, the rate rises 5.3%, 5.6%, 6.7%, 8.9%, 15.2%.
  - Compared as one step with the current model (AUC 0.853).
- **Prediction (user):** a negative gain.
- **Result:** AUC 0.852, a gain of **−0.0007**, winning **8%** of folds. **Not added**, as predicted. Code by Claude, at the user's request (`real_estate_loans_feature_engineering`).
  - The model misses the pattern (the risk-rank check), yet the columns don't help. That points at the model, not the columns: at C = 1e-05 each weight mostly follows its own column's straight-line link, so the model can't use a column whose value shows only next to the others. A U-shaped count split into a flag and a rising part is that kind of column. A hypothesis, not tested.
- **Diagnostic, written before running:** the same two versions at a fixed C = 1 (weak penalty, columns fully combined), on the same 50 folds. Not a keep-rule decision. If the columns win at C = 1, the tiny C was blocking them; if not, the columns are the problem. **Prediction (user):** they win.
- **Diagnostic result** (`diagnostics.py`, by Claude): at C = 1, current 0.8484, with the real estate columns 0.8515: a gain of **+0.0031**, winning **100%** of folds. **The tiny C was blocking them.**
  - But both are below the current model at C = 1e-05 (0.853): the fully fitted line ranks worse than the tiny-C one. With the real estate columns, the gap narrows from 0.0046 to 0.0015.
Left for later: age, NumberOfDependents, NumberOfOpenCreditLinesAndLoans, NumberRealEstateLoansOrLines, and the 30–59 and 60–89 day counts.

## Gradient boosting

- **Plan, written before running (2026-10-31):**
  - `HistGradientBoostingClassifier` with its defaults. Early stopping is on automatically above 10,000 rows: 10% of each training fold is held back, so the number of trees is chosen inside the fold.
  - All 10 raw columns: blanks and 96/98 left as they are.
  - Compared with the current logistic regression (0.853) on the same 50 folds, using the keep rule.
  - The user writes the model code.
- **Prediction (user):** close to the logistic regression, but a small negative gain.
- **Result:** gain **+0.0116** (AUC about 0.865 against 0.853), winning **100%** of folds. **Kept: boosting on the raw columns is the current model.**
  - Code by the user (`cross_validate_tree_model`). Review: a Regressor was used first, which has no probabilities for AUC; fixed to the Classifier.
  - The opposite of House Prices: here the columns bend (U shapes, steps that level off), which the trees learn on their own, and with 150,000 rows there is enough data for them. None of our feature engineering was needed.

## Submissions

### Submission 1 (2026-10-30): `submission.csv`

- **The model:** the current model (CV AUC 0.853 ± 0.004, baseline plus the 30–59 and 60–89 day counts), fitted on all 150,000 training borrowers. `create_submission` in `give_me_some_credit.py`, written by Claude.
- **The file:** 101,503 rows, ids match `sampleEntry.csv`. Probabilities from 0.055 to 0.863, median 0.060: bunched near the 6.7% base rate because of the tiny C. That doesn't affect AUC.
- **Expected Kaggle score, written before uploading:** about 0.853. Kaggle's AUC is computed on part of the test set, so a difference of about ±0.005 would be noise.
- **Kaggle score: public 0.85030, private 0.85386** (submitted with the Kaggle command-line tool). Both are within the expected noise: private is 0.001 above CV, public 0.003 below. The competition has ended, so both parts of the test set are scored; the private one decided the final ranking.

### Submission 2 (2026-10-31): `submission_2.csv`

- **The model:** boosting on the raw columns (CV AUC about 0.865), fitted on all 150,000 training borrowers. `create_boosting_submission`, written by Claude.
- **Expected Kaggle score, written before uploading:** about 0.865, and about 0.012 above submission 1 on both the public and the private part.
- **Kaggle score: public 0.86161, private 0.86670**: +0.0113 and +0.0128 over submission 1, as CV said (+0.0116). The probabilities now spread out (median 0.021, against 0.060 for the tiny-C logistic regression).

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
  - **First decided (2026-10-29): a `Debt` column** (monthly debt payments in dollars) in place of DebtRatio. Debt = DebtRatio when income is missing or 0; otherwise Debt = MonthlyIncome × DebtRatio. (For $1 incomes, the product is the debt as well.)
  - **Replaced for the baseline (2026-10-30) by `DebtToIncome`.** The user's reason for Debt and income was the share of income that goes to debt, but a linear model adds columns and can't divide them ($500 / $1,000 and $5,000 / $10,000 are both 50%, yet they score differently). That share is DebtRatio itself, for reported incomes.
    - DebtToIncome = DebtRatio when income is $100 or more; a fixed 0 when `IncomeNotReported` is 1.
    - **Capped at 2:** risk rises up to about 2, then stops (5.9% at 0 to 0.25, 11.4% at 0.75 to 1, 13.7% at 1 to 2, 10.5% at 2 to 5, 8.6% at 5 to 170). 99% are below 2.02. This is the same straight-line reasoning as for utilization.
    - Debt stays a candidate for later.
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
- **NumberOfDependents missing:** 3,924 borrowers, all with missing income, so already flagged by `IncomeNotReported`. They default at 4.6%, against 5.8% for the other missing-income borrowers. **Decided (2026-10-30): fill with a fixed 0, no flag of its own.** 0 is also the most common value (86,902 borrowers). Unlike income, the fill does matter a little here, because the flag covers 32,008 borrowers and not just these 3,924; with a 1.2-point difference, a separate flag would add little.
- **MonthlyIncome missing:** 20%. The default rate is 5.6% when it is missing and 6.9% when it is present.
