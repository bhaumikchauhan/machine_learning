import numpy as np
import pandas as pd
from sklearn import pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TEST_SIZE = 0.2
RANDOM_SEED = 42
N_FOLDS = 5
# 5-fold cross-validation is repeated with a different shuffle each time, to average out lucky folds.
N_REPEATS = 10

SCALE_COLUMNS = ['MonthlyIncome','DebtToIncome',
                   'NumberOfTimes90DaysLate','RevolvingUtilizationOfUnsecuredLines',
                   'NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTime60-89DaysPastDueNotWorse']

PASSTHROUGH_COLUMNS = ['IncomeNotReported','LateCountsCoded','UtilizationInvalid']
MIN_AUC_GAIN = 0.001
MIN_FOLD_WIN_SHARE = 0.8

def get_train_data_from_csv():
    return pd.read_csv("kaggle\\input\\GiveMeSomeCredit\\cs-training.csv")

def get_kaggle_test_data_from_csv():
    return pd.read_csv("kaggle\\input\\GiveMeSomeCredit\\cs-test.csv")

def retrieve_X_And_y():
    df = get_train_data_from_csv()
    y = df['SeriousDlqin2yrs']
    X = df.drop(columns=['SeriousDlqin2yrs'])
    return X,y

def monthly_income_feature_engineering(X):
    #clip monthly income at $15,000
    X['MonthlyIncome'] = X['MonthlyIncome'].fillna(0)
    X['MonthlyIncome'] = X['MonthlyIncome'].clip(upper=15000)
    #if monthlyincome is less than 100, set it to 0, so that the debt-to-income ratio is not used for those with very low income
    X['MonthlyIncome'] = X['MonthlyIncome'].apply(lambda x: 0 if x < 100 else x)
    return X

def late_count_coded_feature_engineering(X):
    X['LateCountsCoded'] = X['NumberOfTimes90DaysLate'].apply(lambda x: 1 if x == 96 or x == 98 else 0)
    X['NumberOfTimes90DaysLate'] = X['NumberOfTimes90DaysLate'].apply(lambda x: 0 if x == 96 or x == 98 else x)
    X['NumberOfTime30-59DaysPastDueNotWorse'] = X['NumberOfTime30-59DaysPastDueNotWorse'].apply(lambda x: 0 if x == 96 or x == 98 else x)
    X['NumberOfTime60-89DaysPastDueNotWorse'] = X['NumberOfTime60-89DaysPastDueNotWorse'].apply(lambda x: 0 if x == 96 or x == 98 else x)
    return X

def income_not_reported_feature_engineering(X):
    #create a new column that codes whether the monthly income is not reported
    X['IncomeNotReported'] = X['MonthlyIncome'].apply(lambda x: 1 if x == 0 else 0)
    return X

def debt_to_income_feature_engineering(X):
    #DebtToIncome = DebtRatio when income is $100 or more;
    X['DebtToIncome'] = X.apply(lambda row: row['DebtRatio'] if row['IncomeNotReported'] == 0 else 0, axis=1)
    X['DebtToIncome'] = X['DebtToIncome'].clip(upper=2)
    return X

def utilization_invalid_feature_engineering(X):
    #create a new column that codes whether the utilization is invalid
    X['UtilizationInvalid'] = X['RevolvingUtilizationOfUnsecuredLines'].apply(lambda x: 1 if x >=100 else 0)
    return X

def revolving_utilization_feature_engineering(X):
    X['RevolvingUtilizationOfUnsecuredLines'] = X['RevolvingUtilizationOfUnsecuredLines'].apply(lambda x: 0 if x >= 100 else x)
    X['RevolvingUtilizationOfUnsecuredLines'] = X['RevolvingUtilizationOfUnsecuredLines'].apply(lambda x: 2 if x > 2 and x < 100 else x)
    return X

def number_of_open_credit_lines_and_loans_feature_engineering(X):
    #create a new column that caps the number of open credit lines and loans at 4
    X['NumberOfOpenCreditLinesAndLoans'] = X['NumberOfOpenCreditLinesAndLoans'].apply(lambda x: 4 if x > 4 else x)
    return X

def real_estate_loans_feature_engineering(X):
    #0 real estate loans gets its own 0/1 column; the count is capped at 5 for the rising end (1 to 5+)
    X['NoRealEstateLoans'] = X['NumberRealEstateLoansOrLines'].apply(lambda x: 1 if x == 0 else 0)
    X['NumberRealEstateLoansOrLines'] = X['NumberRealEstateLoansOrLines'].clip(upper=5)
    return X

def feature_engineering(X, y):
    #clip monthly income at $15,000
    X = monthly_income_feature_engineering(X)
    X = late_count_coded_feature_engineering(X)
    X = income_not_reported_feature_engineering(X)
    X = debt_to_income_feature_engineering(X)
    X = utilization_invalid_feature_engineering(X)
    X = revolving_utilization_feature_engineering(X)
    return X, y

def split_train_test(X, y):
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y['SeriousDlqin2yrs'], random_state=RANDOM_SEED)


def repeated_folds():
    # Fixed seed, so every caller gets the same 50 folds and results can be compared fold by fold.
    return RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=RANDOM_SEED)

def build_pipeline(scale_columns=None, passthrough_columns=None):
    if scale_columns is None:
        scale_columns = SCALE_COLUMNS
    if passthrough_columns is None:
        passthrough_columns = PASSTHROUGH_COLUMNS

    preprocess = ColumnTransformer([
        ('scale', StandardScaler(), scale_columns),
        ('passthrough', 'passthrough', passthrough_columns)
    ])
    return Pipeline([('preprocess', preprocess), ('model', LogisticRegressionCV(Cs=np.logspace(-5,-1, 6), scoring='roc_auc'))])

def cross_validate_model(X, y, scale_columns=None, passthrough_columns=None):
    if scale_columns is None:
        scale_columns = SCALE_COLUMNS
    if passthrough_columns is None:
        passthrough_columns = PASSTHROUGH_COLUMNS
    cols = scale_columns + passthrough_columns
    print("Fitting on:", cols)
    pipeline = build_pipeline(scale_columns=scale_columns, passthrough_columns=passthrough_columns)
    scores = cross_validate(pipeline, X[cols], y, cv=repeated_folds(),
                            scoring=['roc_auc'], return_estimator=True)
    #print the mean AUC, the standard deviation across the 50 folds, and the chosen C per fold (min, median, max)
    aucs = scores['test_roc_auc']
    print(f"AUC mean: {aucs.mean():.3f} AUC std (+/- {aucs.std():.3f})")
    #scores['estimator'] holds the 50 fitted pipelines, one per fold. in each pipeline, the model step is pipeline.maned_steps['<model_step_name>'] and the chosen value is in C_
    #C_ is an array with a 0/1 target, it has one element, so take C_[0]
    #collect the 50 values into a list and print min, np.median, max
    Cs = [est.named_steps['model'].C_[0] for est in scores['estimator']]
    print(f"C per fold: min {min(Cs)}, median {np.median(Cs)}, max {max(Cs)}")
    return scores

def cross_validate_tree_model(X, y):
    model = HistGradientBoostingClassifier(random_state=RANDOM_SEED)
    pipeline = Pipeline([('model', model)])
    scores = cross_validate(pipeline, X, y, cv=repeated_folds(),
                            scoring=['roc_auc'], return_estimator=True)
    #print the mean AUC, the standard deviation across the 50 folds, and the chosen C per fold (min, median, max)
    aucs = scores['test_roc_auc']
    print(f"AUC mean: {aucs.mean():.3f} AUC std (+/- {aucs.std():.3f})")
    return scores

def keep_rule(current_auc, candidate_auc):
    cur_auc_mean = current_auc.mean()
    candidate_auc_mean = candidate_auc.mean()
    gain = candidate_auc_mean - cur_auc_mean
    win_share = (candidate_auc > current_auc).mean()
    keep = gain >= MIN_AUC_GAIN and win_share >= MIN_FOLD_WIN_SHARE
    print(f"Mean gain: {gain:.4f} (need >= {MIN_AUC_GAIN:.3f}), "
          f"folds won: {win_share:.0%} (need >= {MIN_FOLD_WIN_SHARE:.0%})")
    print("Keep the candidate." if keep else "Keep the current version.")
    return keep

def test_new_features(X, y, current_scores):
    # NumberOfOpenCreditLinesAndLoans: raw and capped at 4 both not added (feature_notes)
    X = real_estate_loans_feature_engineering(X)
    scale_columns = SCALE_COLUMNS + ['NumberRealEstateLoansOrLines']
    passthrough_columns = PASSTHROUGH_COLUMNS + ['NoRealEstateLoans']
    candidate_scores = cross_validate_model(X, y, scale_columns=scale_columns, passthrough_columns=passthrough_columns)
    keep_rule(current_scores['test_roc_auc'], candidate_scores['test_roc_auc'])

def create_submission(path='submission.csv'):
    # Written by Claude. Fits the current model on all 150,000 training borrowers and writes one
    # probability per test borrower. The feature engineering is fixed rules only, so applying it to
    # the test set leaks nothing.
    X, y = retrieve_X_And_y()
    X, y = feature_engineering(X, y)
    cols = SCALE_COLUMNS + PASSTHROUGH_COLUMNS
    pipeline = build_pipeline().fit(X[cols], y)

    test = get_kaggle_test_data_from_csv()
    ids = test.iloc[:, 0]  # the unnamed first column holds the Id
    X_test, _ = feature_engineering(test.drop(columns=['SeriousDlqin2yrs']), None)
    probability = pipeline.predict_proba(X_test[cols])[:, 1]  # column 1 = probability of class 1 (default)
    pd.DataFrame({'Id': ids, 'Probability': probability}).to_csv(path, index=False)
    print(f"Saved {path}: {len(ids)} rows, probability min {probability.min():.4f}, "
          f"median {np.median(probability):.4f}, max {probability.max():.4f}")

def create_boosting_submission(path='submission_2.csv'):
    # Written by Claude. The current model (boosting on the raw columns), fitted on all 150,000 training borrowers.
    X, y = retrieve_X_And_y()
    X = X.drop(columns=X.columns[0])  # the id column
    model = HistGradientBoostingClassifier(random_state=RANDOM_SEED).fit(X, y)

    test = get_kaggle_test_data_from_csv()
    ids = test.iloc[:, 0]  # the first column holds the Id (unnamed in cs-test.csv)
    X_test = test.drop(columns=[test.columns[0], 'SeriousDlqin2yrs'])
    probability = model.predict_proba(X_test)[:, 1]
    pd.DataFrame({'Id': ids, 'Probability': probability}).to_csv(path, index=False)
    print(f"Saved {path}: {len(ids)} rows, probability min {probability.min():.4f}, "
          f"median {np.median(probability):.4f}, max {probability.max():.4f}")




def main():
    X, y = retrieve_X_And_y()
    X, y = feature_engineering(X, y)
    current_scores = cross_validate_model(X, y)

    X, y = retrieve_X_And_y()
    X = X.drop(columns=['id'])
    candidate_scores = cross_validate_tree_model(X, y)
    keep_rule(current_scores['test_roc_auc'], candidate_scores['test_roc_auc'])
    #test_new_features(X, y, current_scores)

if __name__ == '__main__':
    main()