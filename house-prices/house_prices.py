# Kaggle: House Prices - Advanced Regression Techniques
# https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
# Predict SalePrice. Scored by RMSE between log(predicted price) and log(actual price).


import numpy as np
import pandas as pd
import common

def fold_rmse(X_train, y, **pipeline_columns):
    scores = common.cross_validate_model(X_train, y, **pipeline_columns)
    rmse = -scores['test_neg_root_mean_squared_error']
    print(f"RMSE: {rmse.mean():.4f} (+/- {rmse.std():.4f})\n")
    return rmse

def create_submission(path='submission.csv'):
    # the current model, fitted on all 1,458 training houses; it predicts log price, so exp() turns it back to $
    X, y = common.retrieve_X_And_y()
    X_test = common.get_kaggle_test_data_from_csv()
    X_train, y, X_test = common.feature_engineering(X, y, X_test)
    pipeline = common.build_pipeline().fit(X_train, y)
    # clip each numeric test value to the training range, so the straight lines stop at the edge of the data
    # (Id 2550 is far beyond it: 5,095 sq ft basement against a training max of 3,206)
    for column in common.NUMERIC_COLUMNS:
        X_test[column] = X_test[column].clip(X_train[column].min(), X_train[column].max())
    prices = np.exp(pipeline.predict(X_test))
    submission = pd.DataFrame({'Id': X_test['Id'], 'SalePrice': prices})
    submission.to_csv(path, index=False)
    print(f"Saved {path}: {len(submission)} rows, missing {submission['SalePrice'].isna().sum()}, "
          f"median ${np.median(prices):,.0f}, min ${prices.min():,.0f}, max ${prices.max():,.0f}")
    print(f"Id 2550 (like the two removed outliers): ${prices[X_test['Id'].values == 2550][0]:,.0f}")
    return submission

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
    # MSZoning one-hot vs not added was compared on 2026-10-28: not added (wins only 72% of folds).
    # Functional (one 0/1 column for any deduction) vs not added was compared on 2026-10-28: added.
    # SaleCondition (Partial + abnormal, Family with Normal or with Abnorml) was compared on 2026-10-28:
    # added, Family with Abnorml.
    # CentralAir (one 0/1 column for no central air) vs not added was compared on 2026-10-28: added.
    # Exterior1st (0/1 columns for wood siding and brick face) vs not added was compared on 2026-10-28:
    # not added (gain 0.08 x100).
    # Conditions (3 columns, or 5 with Feedr and PosN) were compared on 2026-10-28: 5 columns added.
    # Garage group (area, quality, condition, year, built-in) vs not added was compared on 2026-10-28: added.
    # Basement group (quality, condition, bathrooms, good exposure) vs not added was compared on 2026-10-28: added.
    # Rooms group (8 room columns) vs not added was compared on 2026-10-28: added.
    # Outside group (deck, porch and pool areas) vs not added was compared on 2026-10-28: not added (gain 0.00).
    fold_rmse(X_train, y)


if __name__ == '__main__':
    main()
    #create_submission()  # Kaggle submission, trained on all training houses
