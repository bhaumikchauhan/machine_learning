# Analysis checks for give_me_some_credit.py. Written by Claude; the model code is the user's.

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

import give_me_some_credit as g


def fold_aucs_at_fixed_c(X, y, C, scale_columns, passthrough_columns):
    # Same pipeline and the same 50 folds as cross_validate_model, with C fixed instead of searched
    pipeline = g.build_pipeline(scale_columns=scale_columns, passthrough_columns=passthrough_columns)
    pipeline.set_params(model=LogisticRegression(C=C, max_iter=1000))
    cols = scale_columns + passthrough_columns
    return cross_validate(pipeline, X[cols], y, cv=g.repeated_folds(), scoring='roc_auc')['test_score']


def real_estate_at_fixed_c(C=1.0):
    # Does the tiny C block the real estate columns? Current model against current + the two columns.
    X, y = g.retrieve_X_And_y()
    X, y = g.feature_engineering(X, y)
    X = g.real_estate_loans_feature_engineering(X)
    current = fold_aucs_at_fixed_c(X, y, C, g.SCALE_COLUMNS, g.PASSTHROUGH_COLUMNS)
    candidate = fold_aucs_at_fixed_c(X, y, C, g.SCALE_COLUMNS + ['NumberRealEstateLoansOrLines'],
                                     g.PASSTHROUGH_COLUMNS + ['NoRealEstateLoans'])
    print(f"C = {C}: current {current.mean():.4f}, with real estate columns {candidate.mean():.4f}")
    g.keep_rule(current, candidate)


def column_order_check(n=5):
    # Does the column order matter when predicting? The boosting model as in create_boosting_submission,
    # then the first n Kaggle test borrowers: as trained, with age and MonthlyIncome swapped (DataFrame),
    # and the same swap as a numpy array (no column names).
    X, y = g.retrieve_X_And_y()
    X = X.drop(columns=X.columns[0])
    model = HistGradientBoostingClassifier(random_state=g.RANDOM_SEED).fit(X, y)

    test = g.get_kaggle_test_data_from_csv()
    X_test = test.drop(columns=[test.columns[0], 'SeriousDlqin2yrs']).head(n)
    columns = list(X_test.columns)
    i, j = columns.index('age'), columns.index('MonthlyIncome')
    columns[i], columns[j] = columns[j], columns[i]
    swapped = X_test[columns]

    print("as trained:        ", model.predict_proba(X_test)[:, 1].round(4))
    try:
        print("swapped, DataFrame:", model.predict_proba(swapped)[:, 1].round(4))
    except ValueError as error:
        print("swapped, DataFrame: ValueError:", str(error).splitlines()[0])
    print("swapped, numpy:    ", model.predict_proba(swapped.to_numpy())[:, 1].round(4))


if __name__ == '__main__':
    column_order_check()
