# Kaggle: House Prices - Advanced Regression Techniques
# https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
# Predict SalePrice. Scored by RMSE between log(predicted price) and log(actual price).


import pandas as pd
import common

def fold_rmse(X_train, y, **pipeline_columns):
    scores = common.cross_validate_model(X_train, y, **pipeline_columns)
    rmse = -scores['test_neg_root_mean_squared_error']
    print(f"RMSE: {rmse.mean():.4f} (+/- {rmse.std():.4f})\n")
    return rmse

def main():
    # Comparisons done on 2026-10-27 (feature_notes.md): OverallQual one-hot stays;
    # TotalBsmtSF added raw (log + NoBasement lost to raw).
    X, y = common.retrieve_X_And_y()
    X_test = common.get_kaggle_test_data_from_csv()
    X_train, y, X_test = common.feature_engineering(X, y, X_test)

    # MSSubClass one-hot vs target encoding was compared on 2026-10-28: target encoding stays.
    # LotShape (one-hot, ordered, dropped) was compared on 2026-10-28: dropped.
    # OverallQual re-run with the simpler version as current on 2026-10-28: number column replaces one-hot.
    # OverallCond (number, target encoding, one-hot merged / unmerged) was compared on 2026-10-28: number added.
    # Finished basement (area in sq ft vs share) was compared on 2026-10-28: area added.
    fold_rmse(X_train, y)


if __name__ == '__main__':
    main()
