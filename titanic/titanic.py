# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os


# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Use the kagglehub client library to attach Kaggle resources like competitions, datasets, and models to your session
# Learn more about kagglehub: https://github.com/Kaggle/kagglehub/blob/main/README.md

from numpy.lib import recfunctions as rfn
# kagglehub.dataset_download('<owner>/<dataset-slug>')

import common
# v1: the 11 features only, all built in common.feature_engineering. Shared code lives in common.py.


# Run the with/without comparison of the class interaction instead of a single cross-validation.
RUN_PCLASS_INTERACTION_COMPARISON = False

def compare_pclass_interaction(X, y):
    return common.compare_with_columns(X, y, common.PCLASS_INTERACTION_COLUMNS)

def evaluate_on_test(X_train, y_train, X_test, y_test):
    # Run once at the very end, after features and model are chosen with cross-validation.
    columns = common.fitting_columns(X_train)
    model = common.build_model().fit(X_train[columns], y_train['Survived'])
    y_pred = model.predict(X_test[columns])
    print(f"Test accuracy: {accuracy_score(y_test['Survived'], y_pred):.3f}")
    print(f"Test precision: {precision_score(y_test['Survived'], y_pred):.3f}")
    print(f"Test recall: {recall_score(y_test['Survived'], y_pred):.3f}")

def create_submission(path='submissions.csv'):
    # Train on all labeled rows and predict Kaggle's test.csv. Feature engineering runs once on
    # train and test combined, so fill values and counts come from both (features only, never Survived).
    train = common.get_train_data_from_csv()
    kaggle_test = common.get_kaggle_test_data_from_csv()
    y = train[['Survived']]
    combined = pd.concat([train.drop(columns=['Survived']), kaggle_test], ignore_index=True)
    common.feature_engineering(combined)
    X_train = combined.iloc[:len(train)]
    X_kaggle = combined.iloc[len(train):]
    columns = common.fitting_columns(X_train)
    missing = X_kaggle[columns].isna().sum()
    if missing.any():
        raise ValueError(f"Missing values in Kaggle test features: {missing[missing > 0].to_dict()}")
    model = common.build_model().fit(X_train[columns], y['Survived'])
    submission = pd.DataFrame({'PassengerId': X_kaggle['PassengerId'],
                               'Survived': model.predict(X_kaggle[columns])})
    submission.to_csv(path, index=False)
    print(f"Saved {path}: {len(submission)} rows, {submission['Survived'].mean():.1%} predicted to survive")
    return submission


if __name__ == "__main__":
    #get current directory
    # for dirname, _, filenames in os.walk('kaggle\\input\\competitions\\titanic'):
    #     for filename in filenames:
    #         print(os.path.join(dirname, filename))

    X, y = common.retrieve_X_And_y()
    common.feature_engineering(X)
    X_train, X_test, y_train, y_test = common.split_train_test(X, y)
    if RUN_PCLASS_INTERACTION_COMPARISON:
        compare_pclass_interaction(X_train, y_train)
    else:
        common.cross_validate_model(X_train, y_train)
    #evaluate_on_test(X_train, y_train, X_test, y_test)  # only once, at the very end
    #create_submission()  # Kaggle submission, trained on all labeled rows

