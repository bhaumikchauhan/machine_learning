# Kaggle: House Prices - Advanced Regression Techniques
# https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
# Predict SalePrice. Scored by RMSE between log(predicted price) and log(actual price).


import pandas as pd
import common

def main():
    X, y = common.retrieve_X_And_y()
    X_test = common.get_kaggle_test_data_from_csv()
    X_train, y, X_test = common.feature_engineering(X, y, X_test)
    common.cross_validate_model(X_train, y)


if __name__ == '__main__':
    main()
