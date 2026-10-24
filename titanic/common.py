# Shared by titanic.py (v1) and titanic_v2.py: constants, data loading, feature engineering,
# the split, cross-validation and the keep rule.

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split

ADULT_AGE = 15
RANDOM_SEED = 42
TEST_SIZE = 0.2
N_FOLDS = 5
# 5-fold cross-validation is repeated with a different shuffle each time, to average out lucky folds.
N_REPEATS = 10
# columns no decision has been taken on yet
UNDECIDED_COLUMNS = []
COLUMNS_NOT_FOR_FITTING = (['PassengerId', 'Name', 'Age', 'Title', 'Sex', 'Embarked', 'Pclass', 'SibSp', 'Parch',
                            'family_count', 'Ticket', 'ticket_group_size', 'Fare', 'Cabin'] + UNDECIDED_COLUMNS)
# family_count = SibSp + Parch: 0 is single, 1 to SMALL_FAMILY_MAX is a small family, above it a large family.
SMALL_FAMILY_MAX = 3
FAMILY_COLUMNS = ['is_single', 'is_of_small_family', 'is_of_large_family']
SHARED_TICKET_COLUMNS = ['is_shared_ticket']
# Failed the keep rule (gain -0.002, 2 of 50 folds won); built but left out of the fit.
USE_SHARED_TICKET = False
CABIN_COLUMNS = ['has_cabin']
# Failed the keep rule (gain -0.003, 9 of 50 folds won); built but left out of the fit.
USE_CABIN = False
# Raw fare split into thirds within each class: low, mid or high for that passenger's class.
FARE_BAND_COLUMNS = ['fare_low_in_class', 'fare_mid_in_class', 'fare_high_in_class']
# Failed the keep rule (gain -0.004, 6 of 50 folds won); built but left out of the fit.
USE_FARE_BANDS = False
# Class-by-group interaction columns: 3rd-class women and 1st-class adult men.
# Switch off to leave them out of the fit (they are still built by feature_engineering).
USE_PCLASS_INTERACTION = False
PCLASS_INTERACTION_COLUMNS = ['female_P3', 'adult_male_P1']
# New columns are kept only if the mean accuracy gain and the share of folds they win reach these.
MIN_ACCURACY_GAIN = 0.024
MIN_FOLD_WIN_SHARE = 0.8

def get_train_data_from_csv():
    return pd.read_csv("kaggle\\input\\competitions\\titanic\\train.csv")

def get_kaggle_test_data_from_csv():
    return pd.read_csv("kaggle\\input\\competitions\\titanic\\test.csv")


def get_col_name_idx():
    return {col: idx for idx, col in enumerate(get_train_data_from_csv().columns)}



def retrieve_X_And_y():
    df = get_train_data_from_csv()
    y = df[['Survived']]
    X = df.drop(columns=['Survived'])
    return X,y

def extract_title(X):
    def title_from_name(name):
        comma_idx = name.find(',')
        period_idx = name.find('.', comma_idx)
        return name[comma_idx + 1:period_idx].strip()

    X['Title'] = X['Name'].apply(title_from_name)

def title_survival_summary(X, y):
    summary = (
        X[['Title', 'Sex']]
        .join(y)
        .groupby(['Title', 'Sex'])['Survived']
        .agg(count='count', survival_rate='mean')
        .sort_values('count', ascending=False)
    )
    print(summary)
    return summary

def age_survival_summary(X, y, by=None, bin_width=5):
    known = X[['Age'] + ([by] if by else [])].join(y).dropna(subset=['Age'])
    edges = np.arange(0, known['Age'].max() + bin_width, bin_width)
    keys = [pd.cut(known['Age'], edges, right=False)] + ([known[by]] if by else [])
    summary = known.groupby(keys, observed=True)['Survived'].agg(count='count', survival_rate='mean')
    if by:
        summary = summary.unstack(by)
    print(summary.round(3))
    return summary

def pclass_survival_summary(X, y, by=None):
    summary = (
        X[['Pclass'] + ([by] if by else [])]
        .join(y)
        .groupby(['Pclass'] + ([by] if by else []))['Survived']
        .agg(count='count', survival_rate='mean')
    )
    print(summary.round(3))
    return summary

def column_survival_summary(X, y, column, by=None):
    summary = (
        X[[column] + ([by] if by else [])]
        .join(y)
        .groupby([column] + ([by] if by else []))['Survived']
        .agg(count='count', survival_rate='mean')
    )
    print(summary.round(3))
    return summary

def fill_missing_age(X):
    # Fill values use features only (never Survived), so this runs before the split.
    # TODO: a title with no known age leaves Age as NaN. Decide the fallback. For Kaggle's
    # test.csv, combine it with train and run feature_engineering once on the combined rows.
    X['Age'] = X['Age'].fillna(X.groupby('Title')['Age'].transform('mean'))

def add_is_male_adult(X):
    fill_missing_age(X)
    X['is_male_adult'] = ((X['Sex'] == 'male') & (X['Age'] > ADULT_AGE)).astype(int)

def encode_sex(X):
    X['Sex_encoded'] = X['Sex'].map({'male': 0, 'female': 1})

def one_hot_encode_embarked(X):
    # Features only (never Survived), so this runs before the split. For Kaggle's test.csv,
    # combine it with train first (see fill_missing_age).
    X['Embarked'] = X['Embarked'].fillna(X['Embarked'].mode()[0])
    for port in ['C', 'Q', 'S']:
        X[f'Embarked_{port}'] = (X['Embarked'] == port).astype(int)

def one_hot_encode_pclass(X):
    for pclass in [1, 2, 3]:
        X[f'Pclass_{pclass}'] = (X['Pclass'] == pclass).astype(int)

def add_pclass_interaction(X):
    X['female_P3'] = ((X['Sex_encoded'] == 1) & (X['Pclass'] == 3)).astype(int)
    X['adult_male_P1'] = ((X['is_male_adult'] == 1) & (X['Pclass'] == 1)).astype(int)

def add_family_buckets(X):
    X['family_count'] = X['SibSp'] + X['Parch']
    X['is_single'] = (X['family_count'] == 0).astype(int)
    X['is_of_small_family'] = X['family_count'].between(1, SMALL_FAMILY_MAX).astype(int)
    X['is_of_large_family'] = (X['family_count'] > SMALL_FAMILY_MAX).astype(int)

def add_shared_ticket(X):
    # Counted over the rows passed in. For Kaggle's test.csv, count over train and test combined.
    X['ticket_group_size'] = X.groupby('Ticket')['Ticket'].transform('size')
    X['is_shared_ticket'] = (X['ticket_group_size'] > 1).astype(int)

def add_has_cabin(X):
    X['has_cabin'] = X['Cabin'].notna().astype(int)

def add_fare_bands_in_class(X):
    # Cut points come from the rows passed in. For Kaggle's test.csv, compute over train and test combined.
    band = X.groupby('Pclass')['Fare'].transform(lambda fares: pd.qcut(fares, 3, labels=False))
    for idx, name in enumerate(FARE_BAND_COLUMNS):
        X[name] = (band == idx).astype(int)

def feature_engineering(X):
    extract_title(X)
    add_is_male_adult(X)
    encode_sex(X)
    one_hot_encode_embarked(X)
    one_hot_encode_pclass(X)
    add_pclass_interaction(X)
    add_family_buckets(X)
    add_shared_ticket(X)
    add_has_cabin(X)
    add_fare_bands_in_class(X)

def split_train_test(X, y):
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y['Survived'], random_state=RANDOM_SEED)

def fitting_columns(X):
    excluded = (COLUMNS_NOT_FOR_FITTING
                + ([] if USE_PCLASS_INTERACTION else PCLASS_INTERACTION_COLUMNS)
                + ([] if USE_SHARED_TICKET else SHARED_TICKET_COLUMNS)
                + ([] if USE_CABIN else CABIN_COLUMNS)
                + ([] if USE_FARE_BANDS else FARE_BAND_COLUMNS))
    return [col for col in X.columns if col not in excluded]

def build_model():
    return LogisticRegression()

def repeated_folds():
    # Fixed seed, so every caller gets the same 50 folds and results can be compared fold by fold.
    return RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=RANDOM_SEED)

def cross_validate_model(X, y, columns=None):
    columns = columns or fitting_columns(X)
    print("Fitting on:", columns)
    scores = cross_validate(build_model(), X[columns], y['Survived'], cv=repeated_folds(),
                            scoring=['accuracy', 'precision', 'recall'], return_estimator=True)
    iterations = [int(model.n_iter_[0]) for model in scores['estimator']]
    print(f"Iterations over {len(iterations)} folds: min {min(iterations)}, max {max(iterations)}")
    for metric in ['accuracy', 'precision', 'recall']:
        fold_scores = scores[f'test_{metric}']
        print(f"{metric}: {fold_scores.mean():.3f} (+/- {fold_scores.std():.3f})")
    return scores

def apply_keep_rule(base, with_new):
    # base and with_new are per-fold accuracies on the same folds, so they are compared fold by fold.
    gain = with_new.mean() - base.mean()
    change = with_new - base
    wins, ties = int((change > 0).sum()), int((change == 0).sum())
    win_share = wins / len(change)
    keep = gain >= MIN_ACCURACY_GAIN and win_share >= MIN_FOLD_WIN_SHARE
    print(f"Mean accuracy gain: {gain:+.3f} (needs >= {MIN_ACCURACY_GAIN}); "
          f"folds won: {wins}/{len(change)} = {win_share:.0%} (needs >= {MIN_FOLD_WIN_SHARE:.0%}), "
          f"ties: {ties}, lost: {len(change) - wins - ties}")
    print("Keep:", keep)
    return keep

def compare_with_columns(X, y, new_columns):
    base_columns = [col for col in fitting_columns(X) if col not in new_columns]
    print(f"--- without {new_columns}")
    base = cross_validate_model(X, y, base_columns)['test_accuracy']
    print(f"--- with {new_columns}")
    with_new = cross_validate_model(X, y, base_columns + new_columns)['test_accuracy']
    return apply_keep_rule(base, with_new)
