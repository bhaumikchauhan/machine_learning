# Kaggle: House Prices - Advanced Regression Techniques
# https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
# Predict SalePrice. Scored by RMSE between log(predicted price) and log(actual price).


import pandas as pd
import common

def main():
    use_overall_qual_number = False
    X, y = common.retrieve_X_And_y()
    X_test = common.get_kaggle_test_data_from_csv()
    print(f"using overall_qual_number = {use_overall_qual_number}")
    X_train, y, X_test = common.feature_engineering(X, y, X_test, use_overall_qual_number)
    scores = common.cross_validate_model(X_train, y)
    fold_rmse_cat = -scores['test_neg_root_mean_squared_error']

    use_overall_qual_number = True
    X, y = common.retrieve_X_And_y()
    X_test = common.get_kaggle_test_data_from_csv()
    print(f"using overall_qual_number = {use_overall_qual_number}")
    X_train, y, X_test = common.feature_engineering(X, y, X_test, use_overall_qual_number)
    scores = common.cross_validate_model(X_train, y)
    fold_rmse_num = -scores['test_neg_root_mean_squared_error']
    

    print(f"Comparing the two approaches:")
    print(f"RMSE with overall_qual_number = True: {fold_rmse_num.mean():.4f} (+/- {fold_rmse_num.std():.4f})")
    print(f"RMSE with overall_qual_number = False: {fold_rmse_cat.mean():.4f} (+/- {fold_rmse_cat.std():.4f})")
    # one-hot is the current version, the number column is the candidate
    common.keep_rule(fold_rmse_cat, fold_rmse_num)

if __name__ == '__main__':
    main()
