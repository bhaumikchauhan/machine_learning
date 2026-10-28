# Error analysis of the current model. Written by Claude; the model code is in common.py.

import numpy as np
import pandas as pd

import common


def out_of_fold_predictions():
    # Every training house is predicted 10 times (once per CV repeat) by a model that didn't see it;
    # the average of the 10 is its out-of-fold prediction, in log price.
    X, y = common.retrieve_X_And_y()
    X_train, y, _ = common.feature_engineering(X, y, common.get_kaggle_test_data_from_csv())
    total = np.zeros(len(y))
    count = np.zeros(len(y))
    for train_index, validation_index in common.repeated_folds().split(X_train):
        model = common.build_pipeline().fit(X_train.iloc[train_index], y.iloc[train_index])
        total[validation_index] += model.predict(X_train.iloc[validation_index])
        count[validation_index] += 1
    return X_train, y, pd.Series(total / count, index=y.index)


def worst_houses(top=15):
    X_train, y, prediction = out_of_fold_predictions()
    raw = common.get_train_data_from_csv().loc[X_train.index]
    leftover = (y - prediction) * 100
    table = pd.DataFrame({
        'Id': raw['Id'], 'actual': np.exp(y).round(-3), 'predicted': np.exp(prediction).round(-3),
        'leftover': leftover.round(0), 'Neighborhood': raw['Neighborhood'], 'MSZoning': raw['MSZoning'],
        'OverallQual': raw['OverallQual'], 'OverallCond': raw['OverallCond'], 'GrLivArea': raw['GrLivArea'],
        'YearBuilt': raw['YearBuilt'], 'SaleCondition': raw['SaleCondition'], 'Functional': raw['Functional'],
    })
    worst = table.loc[leftover.abs().sort_values(ascending=False).index[:top]]
    squared = leftover ** 2
    print(worst.to_string(index=False))
    print(f"These {top} houses ({top / len(y):.1%} of the houses) carry "
          f"{squared[worst.index].sum() / squared.sum():.0%} of the squared error.")
    print(f"RMSE: {np.sqrt((squared / 1e4).mean()):.4f} with them, "
          f"{np.sqrt((squared.drop(worst.index) / 1e4).mean()):.4f} without them.")
    return worst


if __name__ == '__main__':
    worst_houses()
