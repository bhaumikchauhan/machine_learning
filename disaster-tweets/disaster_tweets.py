import html
import re
import pandas as pd
import numpy as np
import sklearn
import data_analysis
from sklearn.model_selection import PredefinedSplit, StratifiedGroupKFold, StratifiedKFold, TunedThresholdClassifierCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

RANDOM_SEED = 42
N_FOLDS = 5
FOLDS_CSV = data_analysis.HERE / "folds.csv"
SUBMISSION_CSV = data_analysis.HERE / "submission.csv"
MIN_F1_GAIN = 0.008
MIN_FOLD_WIN_SHARE = 1
CS=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
MAX_ITER = 1000
URL = re.compile(r"https?://\S+")
MENTION = re.compile(r"@\w+")
sklearn.set_config(enable_metadata_routing=True)

def get_train_data_from_csv():
    return pd.read_csv(data_analysis.DATA / "train.csv")

def get_X_y(df):
    X = df.drop(columns=['target'])
    y = df['target']
    return X, y

def clean(text: str) -> str:
    """5e: one token per URL and per @mention, HTML entities decoded.
    Given to TfidfVectorizer as preprocessor=, which replaces its built-in lowercasing, so lower() is done here."""
    text = html.unescape(text)                  # &amp; -> &, &gt; -> >, &lt; -> <
    text = URL.sub(" urltoken ", text)          # every link: one column instead of http, t, co and a random code
    text = MENTION.sub(" mentiontoken ", text)  # every @user: one column instead of one per user name
    return text.lower()


def build_pipeline(tfidfVectorizer, lr=None):
    if lr is None:
        lr = LogisticRegression(max_iter=MAX_ITER)
    return Pipeline([("tfidf", tfidfVectorizer), ("lr", lr)])

def build_pipeline_with_column_transformer():
    preprocess = ColumnTransformer([
            ('text', TfidfVectorizer(), 'text'),
            ('keyword', OneHotEncoder(handle_unknown="ignore"), ['keyword'])
        ])
    return Pipeline([('preprocess', preprocess),('lr', LogisticRegression(max_iter=MAX_ITER))])

def cross_validate_model(pipeline, X, y, cv, params=None):
    return cross_validate(pipeline, X, y, cv=cv, scoring=["f1", "precision", "recall", "accuracy"], return_train_score=True, return_estimator=True, params=params)

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

def print_scores(scores, label, print_c, print_threshold=False):
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

    if print_threshold:
        for fold, score in enumerate(scores['estimator']):
            print(f"Best threshold for Fold_{fold}: {score.best_threshold_}")
    #scores['estimator'][0].best_threshold_

        
    if print_c:
        for fold, score in enumerate(scores['estimator']):
            print(f"C for Fold_{fold}: {score.named_steps['lr'].C_[0]}")
        Cs = [est.named_steps['lr'].C_[0] for est in scores['estimator']]
        print(f"C: min {min(Cs)}, median {np.median(Cs)}, max {max(Cs)}")

def cross_validate_model_and_print_scores(model, X, y, df_fold, params=None):
    pre_defined_split_scores = cross_validate_model(model, X, y, PredefinedSplit(test_fold=df_fold["fold"]), params)
    print_c = isinstance(model, Pipeline) and isinstance(model.named_steps["lr"], LogisticRegressionCV)
    print_threshold = isinstance(model, TunedThresholdClassifierCV)
    print_scores(pre_defined_split_scores, "PredefinedSplit", print_c, print_threshold)
    stratified_split_scores = cross_validate_model(model, X, y, StratifiedKFold(shuffle=True, random_state=RANDOM_SEED, n_splits=N_FOLDS), params)
    print_scores(stratified_split_scores, "StratifiedKFold", print_c)
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

def get_tfidf_base_line():
    return TfidfVectorizer()

def get_tfidf_bigram():
    return TfidfVectorizer(ngram_range=(1,2))

def get_tfidf_min_df_2():
    return TfidfVectorizer(min_df=2)

def get_tfidf_sublinear_tf():
    return TfidfVectorizer(sublinear_tf=True)

def get_tfidf_min_df_2_sublinear_tf():
    return TfidfVectorizer(min_df=2, sublinear_tf=True)

def get_logistic_regression_withcv_grouped():
    return LogisticRegressionCV(Cs=CS, cv=StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED), scoring="f1", max_iter=MAX_ITER)

def get_tfidf_clean():
    return TfidfVectorizer(preprocessor=clean)

def get_tfidf_char_wb():
    return TfidfVectorizer(analyzer='char_wb', ngram_range=(2,5))

def get_tuned_threshold_cv():
    return TunedThresholdClassifierCV(build_pipeline(get_tfidf_base_line()), scoring="f1",
                           cv=StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED))

def create_submission(path=SUBMISSION_CSV):
    """Step 7 (written by Claude): the final model (TF-IDF defaults + LogisticRegression(C=1), cut-off 0.5)
    fitted on all 7,613 training tweets, predicting test.csv. Run with:
        venv/Scripts/python.exe -c "import disaster_tweets; disaster_tweets.create_submission()"
    """
    train = get_train_data_from_csv()
    test = pd.read_csv(data_analysis.DATA / "test.csv")
    sample = pd.read_csv(data_analysis.DATA / "sample_submission.csv")

    model = build_pipeline(get_tfidf_base_line()).fit(train['text'], train['target'])
    submission = pd.DataFrame({"id": test["id"], "target": np.asarray(model.predict(test['text']), dtype=int)})

    assert list(submission.columns) == list(sample.columns), "columns differ from sample_submission.csv"
    assert len(submission) == len(sample) == 3263, "wrong number of rows"
    assert submission["id"].equals(sample["id"]), "ids differ from sample_submission.csv"
    assert submission["target"].isin([0, 1]).all()

    submission.to_csv(path, index=False)
    print(f"{path}: {len(submission)} rows, {submission.target.mean():.1%} predicted disaster "
          f"(train: {train.target.mean():.1%} disaster)")
    return submission

if __name__ == "__main__":
    df = get_train_data_from_csv()
    X, y = get_X_y(df)
    if not FOLDS_CSV.exists():
        create_folds(X, y)
    df_fold = pd.read_csv(FOLDS_CSV)
    df_fold = df.merge(df_fold, on="id")
    print("Baseline:")
    base_line_scores = cross_validate_model_and_print_scores(build_pipeline(get_tfidf_base_line()), X['text'], y, df_fold)
    print("Tuned threshold:")
    #bigram_scores = cross_validate_model_and_print_scores(build_pipeline(TfidfVectorizer(ngram_range=(1,2))), X['text'], y, df_fold)
    groups = X['text'].map(data_analysis.normalise)
    #.
    # candidate_scores = cross_validate_model_and_print_scores(build_pipeline(get_tfidf_base_line(), get_logistic_regression_withcv_grouped()), X['text'], y, df_fold, params={"groups": groups})

    candidate_scores = cross_validate_model_and_print_scores(get_tuned_threshold_cv(), X['text'], y, df_fold, params={"groups": groups})

    keep_rule(base_line_scores['test_f1'], candidate_scores['test_f1'])