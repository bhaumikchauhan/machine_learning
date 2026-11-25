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
MIN_F1_GAIN = 0.008
MIN_FOLD_WIN_SHARE = 1

def get_train_data_from_csv():
    
    return pd.read_csv(data_analysis.DATA / "train.csv")

def get_X_y(df):
    X = df.drop(columns=['target'])
    y = df['target']
    return X, y

def build_pipeline(tfidfVectorizer):
    return Pipeline([("tfidf", tfidfVectorizer), ("lr", LogisticRegression(max_iter=1000))])

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
        train_mean_std = f"{score_type} TRAIN mean: {train_score.mean():.4f} std (+/- {train_score.std():.4f})"
        test_mean_std = f"{score_type} TEST mean: {test_score.mean():.4f} std (+/- {test_score.std():.4f})"
        print(f"{train_mean_std} / {test_mean_std}")
        if score_type == 'f1':
            print("Test F1 score for each fold:")
            for i, score in enumerate(test_score):
                print(f"Fold_{i}: {score}")

def cross_validate_model_and_print_scores(model, X, y, df_fold):
    pre_defined_split_scores = cross_validate_model(model, X, y, PredefinedSplit(test_fold=df_fold["fold"]))
    print_scores(pre_defined_split_scores, "PredefinedSplit")
    stratified_split_scores = cross_validate_model(model, X, y, StratifiedKFold(shuffle=True, random_state=RANDOM_SEED, n_splits=N_FOLDS))
    print_scores(stratified_split_scores, "StratifiedKFold")
    return pre_defined_split_scores

def keep_rule(current_f1, candidate_f1):
    #all 5 folds won and mean gain ≥ 0.008
    cur_f1_mean = current_f1.mean()
    candidate_f1_mean = candidate_f1.mean()
    gain = candidate_f1_mean - cur_f1_mean
    win_share = (candidate_f1 > current_f1).mean()
    keep = gain >= MIN_F1_GAIN and win_share >= MIN_FOLD_WIN_SHARE
    print(f"Mean gain: {gain:.4f} (need >= {MIN_F1_GAIN:.3f}), "
            f"folds won: {win_share:.0%} (need >= {MIN_FOLD_WIN_SHARE:.0%})")
    for fold, score in enumerate(current_f1):
        print(f"Fold_{fold} diff: {(candidate_f1[fold] - score):.4f}")
    print("Keep the candidate." if keep else "Keep the current version.")
    return keep


if __name__ == "__main__":
    df = get_train_data_from_csv()
    X, y = get_X_y(df)
    if not FOLDS_CSV.exists():
        create_folds(X, y)
    df_fold = pd.read_csv(FOLDS_CSV)
    df_fold = df.merge(df_fold, on="id")
    print("Baseline:")
    base_line_scores = cross_validate_model_and_print_scores(build_pipeline(TfidfVectorizer()), X['text'], y, df_fold)
    print("sublinear_tf=True:")
    #bigram_scores = cross_validate_model_and_print_scores(build_pipeline(TfidfVectorizer(ngram_range=(1,2))), X['text'], y, df_fold)
    min_df_scores = cross_validate_model_and_print_scores(build_pipeline(TfidfVectorizer(sublinear_tf=True)), X['text'], y, df_fold)

    keep_rule(base_line_scores['test_f1'], min_df_scores['test_f1'])