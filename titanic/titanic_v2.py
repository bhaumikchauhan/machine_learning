# v2: the 11 features of v1 plus group fate on a shared ticket. Shared code lives in common.py.

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score

import common

# Both fate features are one-hot and built from other passengers' Survived, so they are computed
# inside each fold from that fold's training rows only, never in feature_engineering.
# First version: all ticket-mates, every passenger. Failed the keep rule (gain +0.010, 35 of 50 folds won).
TICKET_FATE_COLUMNS = ['ticket_mates_unknown', 'ticket_mates_all_died', 'ticket_mate_survived']
# Refined version: only women and children count as ticket-mates, and only women and boys get the
# feature; adult men are 0 in all three columns.
GROUP_FATE_COLUMNS = ['wc_mates_unknown', 'wc_mates_all_died', 'wc_mate_survived']

def ticket_mates(X, X_known, y_known):
    # Ticket-mates are the known (training) passengers on the same ticket. A passenger's own
    # outcome is never counted: rows of X that are also in X_known are subtracted from their ticket.
    known = pd.DataFrame({'Ticket': X_known['Ticket'], 'Survived': y_known['Survived']})
    tickets = known.groupby('Ticket')['Survived']
    mates = X['Ticket'].map(tickets.size()).fillna(0)
    mates_survived = X['Ticket'].map(tickets.sum()).fillna(0)
    own = known['Survived'].reindex(X.index)
    return mates - own.notna(), mates_survived - own.fillna(0)

def add_ticket_fate(X, X_known, y_known):
    mates, mates_survived = ticket_mates(X, X_known, y_known)
    X['ticket_mates_unknown'] = (mates == 0).astype(int)
    X['ticket_mates_all_died'] = ((mates > 0) & (mates_survived == 0)).astype(int)
    X['ticket_mate_survived'] = (mates_survived > 0).astype(int)

def add_group_fate(X, X_known, y_known):
    is_woman_or_child = X_known['is_male_adult'] == 0
    mates, mates_survived = ticket_mates(X, X_known[is_woman_or_child], y_known[is_woman_or_child])
    gets_feature = X['is_male_adult'] == 0
    X['wc_mates_unknown'] = (gets_feature & (mates == 0)).astype(int)
    X['wc_mates_all_died'] = (gets_feature & (mates > 0) & (mates_survived == 0)).astype(int)
    X['wc_mate_survived'] = (gets_feature & (mates_survived > 0)).astype(int)

TICKET_FATE = (add_ticket_fate, TICKET_FATE_COLUMNS)
GROUP_FATE = (add_group_fate, GROUP_FATE_COLUMNS)

def cross_validate_with_fate(X, y, fate=None, columns=None):
    # Same 50 folds as common.cross_validate_model, but the fate feature (if any) is rebuilt in every fold.
    add_fate, fate_columns = fate if fate else (None, [])
    columns = (columns or common.fitting_columns(X)) + fate_columns
    print("Fitting on:", columns)
    scores = {'accuracy': [], 'precision': [], 'recall': []}
    for train_idx, validation_idx in common.repeated_folds().split(X, y['Survived']):
        X_fold_train, X_validation = X.iloc[train_idx].copy(), X.iloc[validation_idx].copy()
        y_fold_train, y_validation = y.iloc[train_idx], y.iloc[validation_idx]
        if add_fate:
            add_fate(X_fold_train, X_fold_train, y_fold_train)
            add_fate(X_validation, X_fold_train, y_fold_train)
        model = common.build_model().fit(X_fold_train[columns], y_fold_train['Survived'])
        predicted = model.predict(X_validation[columns])
        actual = y_validation['Survived']
        scores['accuracy'].append(accuracy_score(actual, predicted))
        scores['precision'].append(precision_score(actual, predicted))
        scores['recall'].append(recall_score(actual, predicted))
    scores = {metric: np.array(values) for metric, values in scores.items()}
    for metric, values in scores.items():
        print(f"{metric}: {values.mean():.3f} (+/- {values.std():.3f})")
    return scores

def compare_fate(X, y, fate):
    print("--- without fate")
    base = cross_validate_with_fate(X, y)['accuracy']
    print(f"--- with {fate[1]}")
    with_new = cross_validate_with_fate(X, y, fate)['accuracy']
    return common.apply_keep_rule(base, with_new)

def fit_with_fate(X_known, y_known, X_new, fate=GROUP_FATE):
    # Known rows get their fate from each other (never their own outcome); new rows from the known rows only.
    add_fate, fate_columns = fate
    columns = common.fitting_columns(X_known) + fate_columns
    X_known, X_new = X_known.copy(), X_new.copy()
    add_fate(X_known, X_known, y_known)
    add_fate(X_new, X_known, y_known)
    model = common.build_model().fit(X_known[columns], y_known['Survived'])
    women_and_boys = X_new['is_male_adult'] == 0
    known_share = 1 - X_new.loc[women_and_boys, fate_columns[0]].mean()
    print(f"Women and boys with a known woman/child ticket-mate: {known_share:.0%} of {women_and_boys.sum()}")
    return model.predict(X_new[columns])

def evaluate_on_test(X_train, y_train, X_test, y_test):
    # Run once at the very end, after features and model are chosen with cross-validation.
    y_pred = fit_with_fate(X_train, y_train, X_test)
    print(f"Test accuracy: {accuracy_score(y_test['Survived'], y_pred):.3f}")
    print(f"Test precision: {precision_score(y_test['Survived'], y_pred):.3f}")
    print(f"Test recall: {recall_score(y_test['Survived'], y_pred):.3f}")

def create_submission(path='submissions_v2.csv'):
    # Feature engineering runs once on train and test combined (features only, as in v1). The fate
    # feature for Kaggle's test rows comes from the 891 labeled rows only.
    train = common.get_train_data_from_csv()
    kaggle_test = common.get_kaggle_test_data_from_csv()
    y = train[['Survived']]
    combined = pd.concat([train.drop(columns=['Survived']), kaggle_test], ignore_index=True)
    common.feature_engineering(combined)
    X_train = combined.iloc[:len(train)]
    X_kaggle = combined.iloc[len(train):]
    y_pred = fit_with_fate(X_train, y, X_kaggle)
    submission = pd.DataFrame({'PassengerId': X_kaggle['PassengerId'], 'Survived': y_pred})
    submission.to_csv(path, index=False)
    print(f"Saved {path}: {len(submission)} rows, {submission['Survived'].mean():.1%} predicted to survive")
    return submission


if __name__ == "__main__":
    X, y = common.retrieve_X_And_y()
    common.feature_engineering(X)
    X_train, X_test, y_train, y_test = common.split_train_test(X, y)
    compare_fate(X_train, y_train, GROUP_FATE)
    #evaluate_on_test(X_train, y_train, X_test, y_test)  # only once, at the very end
    #create_submission()  # Kaggle submission, trained on all labeled rows
