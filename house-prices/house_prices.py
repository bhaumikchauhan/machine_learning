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

    current = fold_rmse(X_train, y)
    # candidate: MSSubClass one-hot with 20 (536 houses, the largest code) as the reference, instead of target encoding
    one_hot_mssubclass = fold_rmse(X_train, y, one_hot_references={**common.ONE_HOT_REFERENCES, 'MSSubClass': '20'},
                                   target_columns=[])
    print("MSSubClass one-hot vs target encoding:")
    common.keep_rule(current, one_hot_mssubclass)


if __name__ == '__main__':
    main()
