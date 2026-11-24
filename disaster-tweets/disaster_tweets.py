import pandas as pd
import numpy as np
import data_analysis
from sklearn.model_selection import PredefinedSplit, StratifiedGroupKFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier

RANDOM_SEED = 42
N_FOLDS = 5
FOLDS_CSV = data_analysis.HERE / "folds.csv"

def get_train_data_from_csv():
    
    return pd.read_csv(data_analysis.DATA / "train.csv")

def get_X_y(df):
    X = df.drop(columns=['target'])
    y = df['target']
    return X, y

def build_pipeline():
    return Pipeline([("tfidf", TfidfVectorizer()), ("lr", LogisticRegression(max_iter=1000))])

def cross_validate_model(pipeline, X, y, cv):
    return cross_validate(pipeline, X, y, cv=cv,
                  scoring=["f1", "precision", "recall", "accuracy"], return_train_score=True)

def create_folds(X, y):
    groups = X['text'].map(data_analysis.normalise)
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold = np.full(len(X), -1)
    for k, (train_idx, val_idx) in enumerate(splitter.split(X,y, groups)):
        fold[val_idx] = k
        groups_in_train = groups.iloc[train_idx]
        groups_in_val = groups.iloc[val_idx]
        assert len(set(groups_in_val) & set(groups_in_train)) == 0
    
    assert (fold >= 0).all()
    df_fold = pd.DataFrame({"id": X["id"], "fold": fold})
    df_fold.to_csv(FOLDS_CSV, index=False)

def get_dummy_classifier():
    return DummyClassifier(strategy="constant", constant=1)

def print_scores(scores, label):
    print(f"{label}:")
    for score_type in ["f1", "precision", "recall", "accuracy"]:
        test_score = scores['test_' + score_type]
        train_score = scores['train_' + score_type]
        print(f"{score_type} TRAIN mean: {train_score.mean():.4f} std (+/- {train_score.std():.4f})")
        print(f"{score_type} TEST mean: {test_score.mean():.4f} std (+/- {test_score.std():.4f})")
        if score_type == 'f1':
            print("Test F1 score for each fold:")
            for i, score in enumerate(test_score):
                print(f"Fold_{i}: {score}")

def cross_validate_model_and_print_scores(model, X, y, df_fold):
    scores = cross_validate_model(model, X, y, PredefinedSplit(test_fold=df_fold["fold"]))
    print_scores(scores, "PredefinedSplit")
    scores = cross_validate_model(model, X, y, StratifiedKFold(shuffle=True, random_state=RANDOM_SEED, n_splits=N_FOLDS))
    print_scores(scores, "StratifiedKFold")

if __name__ == "__main__":
    df = get_train_data_from_csv()
    X, y = get_X_y(df)
    if not FOLDS_CSV.exists():
        create_folds(X, y)
    df_fold = pd.read_csv(FOLDS_CSV)
    df_fold = df.merge(df_fold, on="id")
    cross_validate_model_and_print_scores(build_pipeline(), X['text'], y, df_fold)