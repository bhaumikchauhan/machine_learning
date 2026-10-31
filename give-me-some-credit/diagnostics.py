# Analysis checks for give_me_some_credit.py. Written by Claude; the model code is the user's.

import numpy as np
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


if __name__ == '__main__':
    real_estate_at_fixed_c()
