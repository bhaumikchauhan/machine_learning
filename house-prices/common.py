import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, RepeatedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder


RANDOM_SEED = 42
N_FOLDS = 5
# 5-fold cross-validation is repeated with a different shuffle each time, to average out lucky folds.
N_REPEATS = 10
# columns no decision has been taken on yet
UNDECIDED_COLUMNS = []
COLUMNS_FOR_FITTING = ['MSSubClass', 'Neighborhood', 'LotArea', 'LotShape',
                       'OverallQual', 'Age', 'GrLivArea', 'ExterQual', 'GarageCars', 'TotalBsmtSF']
COLUMNS_NOT_FOR_FITTING = ([] + UNDECIDED_COLUMNS)
MIN_FOLD_WIN_SHARE = 0.8
# keep rule (feature_notes.md): mean RMSE gain of at least 0.001 and wins on at least 80% of folds
MIN_RMSE_GAIN = 0.001

NUMERIC_COLUMNS = ['LotArea', 'Age', 'GrLivArea', 'TotalBsmtSF']
# one-hot column -> reference level (the level that gets no column of its own)
ONE_HOT_REFERENCES = {'Neighborhood': 'NAmes', 'LotShape': 'Reg', 'OverallQual': 'Average',
                      'ExterQual': 'TA', 'GarageCars': '2'}
TARGET_ENCODED_COLUMNS = ['MSSubClass']
# RidgeCV tries every alpha on each training part and keeps the best one (61 values from 0.001 to 1000).
ALPHAS = np.logspace(-3, 3, 61)
USE_OVERALL_QUAL_NUMBER = True

def get_train_data_from_csv():
    return pd.read_csv("kaggle\\input\\competitions\\house-prices-advanced-regression-techniques\\train.csv")

def get_kaggle_test_data_from_csv():
    return pd.read_csv("kaggle\\input\\competitions\\house-prices-advanced-regression-techniques\\test.csv")


def get_col_name_idx():
    return {col: idx for idx, col in enumerate(get_train_data_from_csv().columns)}

def retrieve_X_And_y():
    df = get_train_data_from_csv()
    y = df[['SalePrice']]
    X = df.drop(columns=['SalePrice'])
    return X,y

def target_log_transform(y):
    # Apply log transform to SalePrice to reduce skewness.
    return np.log(y['SalePrice'])

def ms_sub_class_to_str(X_train, X_test):
    # Convert MSSubClass to string type for categorical processing
    X_train['MSSubClass'] = X_train['MSSubClass'].astype(str)
    X_test['MSSubClass'] = X_test['MSSubClass'].astype(str)
    return X_train, X_test

def neighbourhood_update_blueste(X_train, X_test):
    # Blueste has only 2 houses and its price can't be told apart from NAmes,
    # so it is priced like the reference (NAmes).
    X_train.loc[X_train['Neighborhood'] == 'Blueste', 'Neighborhood'] = 'NAmes'
    X_test.loc[X_test['Neighborhood'] == 'Blueste', 'Neighborhood'] = 'NAmes'
    return X_train, X_test

def lot_area_log(X_train, X_test):
    # LotArea is highly skewed. Apply log transform to reduce skewness.
    X_train['LotArea'] = np.log(X_train['LotArea'])
    X_test['LotArea'] = np.log(X_test['LotArea'])
    return X_train, X_test


def ids_remove_outlier(X, y):
    # remove outliers from training data
    #find index of rows with outlier Ids
    outlier_ids = [524, 1299]
    outlier_indices = X[X['Id'].isin(outlier_ids)].index
    #drop outliers from X and y
    X = X.drop(outlier_indices)
    y = y.drop(outlier_indices)
    return X, y

def overall_qual_merge(X_train, X_test, use_overall_qual_number):

    if not use_overall_qual_number:
    # Merge OverallQual values 1, 2, 3 into a single category 'Low'
    # and others into descriptive categories for better interpretability.
        X_train['OverallQual'] = X_train['OverallQual'].replace({1: 'Low', 2: 'Low', 3: 'Low',
                             4: "Below Average", 5: "Average", 6: "Above Average",
                             7: "Good", 8: "Very Good", 9: "Excellent", 10: "Very Excellent"})
        X_test['OverallQual'] = X_test['OverallQual'].replace({1: 'Low', 2: 'Low', 3: 'Low',
                                     4: "Below Average", 5: "Average", 6: "Above Average",
                                     7: "Good", 8: "Very Good", 9: "Excellent", 10: "Very Excellent"})
        ONE_HOT_REFERENCES['OverallQual'] = 'Average'
        if 'OverallQual' in NUMERIC_COLUMNS:
            NUMERIC_COLUMNS.remove('OverallQual')
        
    else:
        #override qual 1 and 2 with 3, since they are very rare and don't have enough examples to learn from
        X_train['OverallQual'] = X_train['OverallQual'].replace({1: 3, 2: 3})
        X_test['OverallQual'] = X_test['OverallQual'].replace({1: 3, 2: 3})
        if 'OverallQual' not in NUMERIC_COLUMNS:
            NUMERIC_COLUMNS.append('OverallQual')
        ONE_HOT_REFERENCES.pop('OverallQual', None)
    return X_train, X_test


def age_create(X_train, X_test):
    # Create a new feature 'Age' as the difference between YrSold and YearRemodAdd.
    X_train['Age'] = X_train['YrSold'] - X_train['YearRemodAdd']
    X_test['Age'] = X_test['YrSold'] - X_test['YearRemodAdd']
    return X_train, X_test


def gr_liv_area_log(X_train, X_test):
    # GrLivArea is highly skewed. Apply log transform to reduce skewness.
    X_train['GrLivArea'] = np.log(X_train['GrLivArea'])
    X_test['GrLivArea'] = np.log(X_test['GrLivArea'])
    return X_train, X_test

def garage_cars_modification(X_train, X_test):
    # fill id 2577 with 0
    X_test.loc[X_test['Id'] == 2577, 'GarageCars'] = 0
    X_test['GarageCars'] = X_test['GarageCars'].astype(int)
    X_train['GarageCars'] = X_train['GarageCars'].astype(int)
    #merge garagecars 3,4,5 into 3+
    X_train['GarageCars'] = X_train['GarageCars'].replace({3: '3+', 4: '3+', 5: '3+'})
    X_test['GarageCars'] = X_test['GarageCars'].replace({3: '3+', 4: '3+', 5: '3+'})
    X_train, X_test = garage_cars_to_str(X_train, X_test)
    return X_train, X_test

def garage_cars_to_str(X_train, X_test):
    # Convert GarageCars to string type for categorical processing
    X_train['GarageCars'] = X_train['GarageCars'].astype(str)
    X_test['GarageCars'] = X_test['GarageCars'].astype(str)
    return X_train, X_test

def lot_shape_merge(X_train, X_test):
    # Merge LotShape values 'IR2', 'IR3' into a single category 'IR' since they dont have much examples
    # and other reason mentioned in feature_notes.md.
    X_train['LotShape'] = X_train['LotShape'].replace({'IR2': 'IR', 'IR3': 'IR'})
    X_test['LotShape'] = X_test['LotShape'].replace({'IR2': 'IR', 'IR3': 'IR'})
    return X_train, X_test

def total_bsmt_sf_modification(X_train, X_test):
    # test Id 2121 has every basement column empty: no basement, so 0 sq ft
    X_test.loc[X_test['Id'] == 2121, 'TotalBsmtSF'] = 0
    for X in (X_train, X_test):
        # log for houses with a basement; log(0) doesn't exist, so houses without one get 0 here
        # and the NoBasement column takes their step (its weight absorbs whatever value they get)
        X['TotalBsmtSFLog'] = np.log(X['TotalBsmtSF'].where(X['TotalBsmtSF'] > 0, 1))
        X['NoBasement'] = (X['TotalBsmtSF'] == 0).astype(int)
    return X_train, X_test

def feature_engineering(X_train, y, X_test, use_overall_qual_number=False):
    y = target_log_transform(y)
    X_train, y = ids_remove_outlier(X_train, y)
    X_train, X_test = ms_sub_class_to_str(X_train, X_test)
    X_train, X_test = neighbourhood_update_blueste(X_train, X_test)
    X_train, X_test = lot_area_log(X_train, X_test)
    X_train, X_test = overall_qual_merge(X_train, X_test, use_overall_qual_number)
    X_train, X_test = age_create(X_train, X_test)
    X_train, X_test = gr_liv_area_log(X_train, X_test)
    X_train, X_test = garage_cars_modification(X_train, X_test)
    X_train, X_test = lot_shape_merge(X_train, X_test)
    X_train, X_test = total_bsmt_sf_modification(X_train, X_test)
    return X_train, y, X_test

def build_pipeline(numeric_columns=None, passthrough_columns=(), one_hot_references=None, target_columns=None):
    # Everything learned from the data (scaler means, one-hot levels, target means, alpha) sits in here,
    # so cross_validate refits all of it on each training part and the held-out part stays unseen.
    # numeric_columns: scaled columns; passthrough_columns: 0/1 columns used as they are;
    # one_hot_references: column -> reference level; target_columns: target-encoded columns.
    # Each defaults to the current model.
    if numeric_columns is None:
        numeric_columns = NUMERIC_COLUMNS
    if one_hot_references is None:
        one_hot_references = ONE_HOT_REFERENCES
    if target_columns is None:
        target_columns = TARGET_ENCODED_COLUMNS
    preprocess = ColumnTransformer([
        ('scale', StandardScaler(), numeric_columns),
        ('flags', 'passthrough', list(passthrough_columns)),
        # a level missing from a training part (or new in test) gets all zeros, i.e. priced like the reference
        ('one_hot', OneHotEncoder(drop=list(one_hot_references.values()), handle_unknown='ignore',
                                  sparse_output=False), list(one_hot_references)),
        # out-of-fold and blended mean log price per code; an unseen code (150 in test) gets the overall mean
        # Pylance infers cv: int from the default; sklearn accepts a splitter too (and wants one for the seed)
        ('target', TargetEncoder(cv=KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)),  # pyright: ignore[reportArgumentType]
         list(target_columns)),
    ])
    return Pipeline([('preprocess', preprocess), ('model', RidgeCV(alphas=ALPHAS))])

def repeated_folds():
    # Fixed seed, so every caller gets the same 50 folds and results can be compared fold by fold.
    # Not stratified: stratifying needs classes, and SalePrice is continuous.
    return RepeatedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=RANDOM_SEED)

def cross_validate_model(X, y, **pipeline_columns):
    # pipeline_columns: any of build_pipeline's arguments, to change the columns of the current model
    pipeline = build_pipeline(**pipeline_columns)
    # the ColumnTransformer drops any column it isn't given, so X can hold extra columns
    columns = [c for _, _, cols in pipeline.named_steps['preprocess'].transformers for c in cols]
    print("Fitting on:", columns)
    scores = cross_validate(pipeline, X[columns], y, cv=repeated_folds(),
                            scoring=['neg_root_mean_squared_error'], return_estimator=True)
    alphas = [model.named_steps['model'].alpha_ for model in scores['estimator']]
    print(f"Chosen alpha over {len(alphas)} folds: min {min(alphas):.3g}, "
          f"median {np.median(alphas):.3g}, max {max(alphas):.3g}")
    return scores

def keep_rule(current_rmse, candidate_rmse):
    # Both arrays come from the same 50 folds (fixed seed, same rows in the same order),
    # so position k holds the same houses in both: compare them fold by fold.
    gain = current_rmse.mean() - candidate_rmse.mean()
    win_share = (candidate_rmse < current_rmse).mean()
    keep = gain >= MIN_RMSE_GAIN and win_share >= MIN_FOLD_WIN_SHARE
    print(f"Mean gain: {gain * 100:.2f} x100 (need >= {MIN_RMSE_GAIN * 100:.1f}), "
          f"folds won: {win_share:.0%} (need >= {MIN_FOLD_WIN_SHARE:.0%})")
    print("Keep the candidate." if keep else "Keep the current version.")
    return keep
