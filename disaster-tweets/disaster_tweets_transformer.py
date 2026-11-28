import numpy as np
import contextlib
import gc
import io
import re
import logging
import sys
import time
from datetime import datetime

import torch
from urllib.parse import unquote

import disaster_tweets as dt
import data_analysis
import pandas as pd
import datasets
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import Trainer
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import set_seed
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from transformers import PreTrainedTokenizerBase
from transformers import TrainerCallback

#CHECKPOINT = "cardiffnlp/twitter-roberta-base"
SEED = 42
SEEDS = [42, 43, 44]
# Fixed (step 2, 2026-11-27): fold 0's check curves were flat over 1-5 epochs (means 0.7824-0.7854),
# so the per-fold epoch choice was dropped; the run is in git history and check_f1_fold0.csv.
NUM_EPOCHS = 3
LEARNING_RATE = 5e-5
PER_DEVICE_TRAIN_BATCH_SIZE=16
STEP_1_FOLD_0_F1 = 0.7816   # seed 42, 3 epochs: must be reproduced
TFIDF_FOLD_F1 = np.array([0.7247, 0.7516, 0.7490, 0.7555, 0.7571])
BERT_FOLD_F1 = np.array([0.7816, 0.7964, 0.8061, 0.7997, 0.7920])

LOG_DIR = data_analysis.HERE / "logs"
START = time.perf_counter()
logger = logging.getLogger("transformer")             # console and the run's file
file_logger = logging.getLogger("transformer_file")   # the run's file only (the Trainer already prints to the console)

def setup_logging():
    """A new file per run, logs/run_<date>_<time>.log, so the files sort by start time."""
    LOG_DIR.mkdir(exist_ok=True)
    path = LOG_DIR / f"run_{datetime.now():%Y-%m-%d_%H-%M-%S}.log"
    formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    file = logging.FileHandler(path, encoding="utf-8")
    for handler in (console, file):
        handler.setFormatter(formatter)
    for lg, handlers in ((logger, (console, file)), (file_logger, (file,))):
        lg.setLevel(logging.INFO)
        lg.propagate = False
        for handler in handlers:
            lg.addHandler(handler)
    return path

def elapsed():
    return f"[{(time.perf_counter() - START) / 60:5.1f} min]"

def log(message):
    """A step or a result, with the clock time and the minutes since the script started."""
    logger.info(f"{elapsed()} {message}")

class LogTrainerToFile(TrainerCallback):
    """The Trainer's loss lines and run summary into the run's file too."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        file_logger.info(f"{elapsed()}   trainer: {logs}")

def preprocess_roberta(keyword, text):
    """The tweet as twitter-roberta saw it in pretraining (Cardiff NLP's model card):
    every word starting with @ becomes @user, every word starting with http becomes http.
    A "word" starts after a space, a newline or the start of the text, and runs to the next whitespace."""
    text = re.sub(r"(?<!\S)@\S+", "@user", text)
    return re.sub(r"(?<!\S)http\S*", "http", text)

def preprocess_keyword(keyword, text):
    keyword = "none." if pd.isna(keyword) else f"{unquote(keyword)}."
    return f"Keyword: {keyword} {text}"

def tokenize_function(row, tokenizer: PreTrainedTokenizerBase):
    return tokenizer(row['text'], truncation=True, max_length=96)

def tokenize_function_check(row, tokenizer: PreTrainedTokenizerBase):
    return tokenizer(row['text'])

def probability(logits):
    """P(class 1) per row, softmax in float64. The one rule for a prediction is p >= 0.5, everywhere:
    with bf16 two logits can be equal, and argmax would then say 0 while p >= 0.5 says 1."""
    return torch.softmax(torch.tensor(logits, dtype=torch.float64), dim=-1)[:, 1].numpy()

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    preds = probability(logits) >= 0.5
    return {"f1": f1_score(labels, preds),
            "precision": precision_score(labels, preds),
            "recall": recall_score(labels, preds),
            "accuracy": accuracy_score(labels, preds)}

def make_dataset(df, tokenizer):
    columns_for_transformer = ['text', 'label', 'id']
    df_ds = datasets.Dataset.from_pandas(df[columns_for_transformer], preserve_index=False)

    tokenized_ds = df_ds.map(tokenize_function, batched=True, fn_kwargs={"tokenizer": tokenizer})
    tokenized_ds = tokenized_ds.remove_columns(['text', 'id'])

    return tokenized_ds

def fine_tune(model_name, seed, tokenizer, data_collator, train_ds, num_epochs, eval_ds=None):
    """A fresh model, seeded before its head is drawn. With eval_ds, eval_ds is scored after every epoch."""
    set_seed(seed)
    training_args = TrainingArguments(
            output_dir="bert_runs",
            num_train_epochs=num_epochs,
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
            bf16=True,
            report_to="none",
            eval_strategy="no" if eval_ds is None else "epoch",
            save_strategy="no"
        )
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    trainer = Trainer(
            model,
            training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=data_collator,
            processing_class=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[LogTrainerToFile()]
        )
    trainer.train()
    return trainer

def free_gpu():
    """Call after the trainer's last name is gone: gc breaks the Trainer's reference cycles, then the cache is released."""
    gc.collect()
    torch.cuda.empty_cache()

def predict_p(trainer, ds, labels):
    """Probability of class 1 per row; F1 at p >= 0.5, checked against compute_metrics."""
    predictions = trainer.predict(ds)
    logits = predictions.predictions
    p = probability(logits)
    f1 = f1_score(labels, p >= 0.5)
    assert f1 == predictions.metrics['test_f1']
    ties = int((logits[:, 0] == logits[:, 1]).sum())
    if ties:
        log(f"{ties} tweet(s) with equal logits (p = 0.5, predicted 1)")
    log(predictions.metrics)
    return p, f1

def fold_probabilities(model_name, seed, tokenizer, data_collator, train_ds, num_epochs, val_ds, val_labels):
    """A fresh model on the whole training part for num_epochs; the validation fold predicted once."""
    trainer = fine_tune(model_name, seed, tokenizer, data_collator, train_ds, num_epochs)
    p, f1 = predict_p(trainer, val_ds, val_labels)
    del trainer
    free_gpu()
    return p, f1

def seed_spread(model_name):
    log(f"Seed-spread run: fold 0, seeds {SEEDS}, {NUM_EPOCHS} epochs, {model_name}")

    #merge train and folds on id
    log("Loading train.csv and folds.csv")
    train_all = dt.get_train_data_from_csv()
    folds = pd.read_csv(dt.FOLDS_CSV)
    train_all_folds = train_all.merge(folds, on="id", how="inner")
    assert len(train_all) == len(train_all_folds)

    train_all_folds = train_all_folds.rename(columns={'target' : 'label'})
    train_df = train_all_folds[train_all_folds['fold'] != 0]
    val_df = train_all_folds[train_all_folds['fold'] == 0]

    assert len(set(train_df.id) & set(val_df.id)) == 0

    log(f"Training part {len(train_df)}, fold 0 {len(val_df)}")

    log("Tokenizing the training part and fold 0")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    tokenized_train = make_dataset(train_df, tokenizer)
    tokenized_val = make_dataset(val_df, tokenizer)

    # per seed, a fresh model on the whole training part for NUM_EPOCHS; fold 0 predicted once
    fold_f1 = {}
    oof = []
    for i, seed in enumerate(SEEDS, start=1):
        log(f"Seed {seed} ({i} of {len(SEEDS)}): {NUM_EPOCHS} epochs on the training part, then predict fold 0")
        t = time.perf_counter()
        p, fold_f1[seed] = fold_probabilities(model_name, seed, tokenizer, data_collator, tokenized_train, NUM_EPOCHS, tokenized_val, val_df['label'])
        oof.append(pd.DataFrame({"id": val_df['id'].to_numpy(), "fold": val_df['fold'].to_numpy(), "seed": seed, "p": p}))
        log(f"Seed {seed}: fold 0 F1 {fold_f1[seed]:.4f} ({time.perf_counter() - t:.0f} s)")

    # the seed spread
    log("Saving oof_bert_seeds_fold0.csv and the seed spread")
    oof_df = pd.concat(oof, ignore_index=True)
    assert len(oof_df) == len(SEEDS) * len(val_df)
    oof_df.to_csv(data_analysis.HERE / "oof_bert_seeds_fold0.csv", index=False)

    f1s = pd.Series(fold_f1)
    sd = f1s.std()   # sample sd (ddof=1), over the seeds
    log(f"Fold 0 F1 per seed: {f1s.round(4).to_dict()}")
    log(f"mean {f1s.mean():.4f}, sd {sd:.4f}, min {f1s.min():.4f}, max {f1s.max():.4f}")
    log(f"Step 5 seeds: {'3 (sd above 0.003)' if sd > 0.003 else '1 (sd at most 0.003)'}")
    log(f"Reproduction check: seed 42 {fold_f1[42]:.4f} vs step 1 {STEP_1_FOLD_0_F1}")
    log("Done")

def seed_check(model_name, fold_ids, seeds, oof_name):
    """Step 5 check (2026-11-28): the seed spread on other folds than fold 0, for the baseline recipe.
    Per fold and seed, a fresh model on the other folds; the fold predicted once. Seed 42's runs are in oof_bert_all_fold.csv."""
    log(f"Seed check: folds {fold_ids}, seeds {seeds}, {NUM_EPOCHS} epochs, {model_name}")
    train_all_folds = dt.get_train_data_from_csv().merge(pd.read_csv(dt.FOLDS_CSV), on="id", how="inner")
    train_all_folds = train_all_folds.rename(columns={'target': 'label'})
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    rows, oof = [], []
    for k in fold_ids:
        train_df = train_all_folds[train_all_folds['fold'] != k]
        val_df = train_all_folds[train_all_folds['fold'] == k]
        assert len(set(train_df.id) & set(val_df.id)) == 0
        tokenized_train = make_dataset(train_df, tokenizer)
        tokenized_val = make_dataset(val_df, tokenizer)
        for seed in seeds:
            log(f"Fold {k}, seed {seed}")
            t = time.perf_counter()
            p, f1 = fold_probabilities(model_name, seed, tokenizer, data_collator, tokenized_train, NUM_EPOCHS, tokenized_val, val_df['label'])
            oof.append(pd.DataFrame({"id": val_df['id'].to_numpy(), "fold": k, "seed": seed, "p": p}))
            rows.append({"fold": k, "seed": seed, "f1": f1})
            log(f"Fold {k}, seed {seed}: F1 {f1:.4f} ({time.perf_counter() - t:.0f} s)")
    pd.concat(oof, ignore_index=True).to_csv(data_analysis.HERE / f"{oof_name}.csv", index=False)
    log(f"F1 per fold and seed:\n{pd.DataFrame(rows).pivot(index='fold', columns='seed', values='f1').round(4)}")
    log("Done")

def main(model_name, oof_name, preprocess=None):
    log("Loading train.csv and folds.csv")
    train_all = dt.get_train_data_from_csv()
    test_all = pd.read_csv(data_analysis.DATA / "test.csv")
    all_data = pd.concat([train_all, test_all], ignore_index=True)
    

    folds = pd.read_csv(dt.FOLDS_CSV)
    train_all_folds = train_all.merge(folds, on="id", how="inner")
    assert len(train_all) == len(train_all_folds)

    train_all_folds = train_all_folds.rename(columns={'target' : 'label'})
    if preprocess is not None:
        train_all_folds['text'] = train_all_folds.apply(lambda row: preprocess(row["keyword"], row["text"]), axis=1)
        all_data['text'] = all_data.apply(lambda row: preprocess(row["keyword"], row["text"]), axis=1)


    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    df_ds = datasets.Dataset.from_pandas(all_data[['text']], preserve_index=False)
    all_data_tokenized = df_ds.map(tokenize_function_check, batched=True, fn_kwargs={"tokenizer": tokenizer})
    max_len = max((len(item['input_ids']) for item in all_data_tokenized))
    log(f"max len of token: {max_len}")
    log(f"number of tokenized tweet having length > 96: {sum(1 for item in all_data_tokenized if len(item['input_ids']) > 96)}")
    assert max_len <=96
    
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    fold_f1 = {}
    oof = []
    fold_ids = sorted(folds.fold.unique())
    log(f"Baseline: {len(fold_ids)} folds, seed {SEED}, {NUM_EPOCHS} epochs, {model_name}")
    for k in fold_ids:
        log(f"Fold {k} ({k + 1} of {len(fold_ids)}): {NUM_EPOCHS} epochs on the other folds, then predict fold {k}")
        t = time.perf_counter()
        train_df = train_all_folds[train_all_folds['fold'] != k]
        val_df = train_all_folds[train_all_folds['fold'] == k]
        assert len(set(train_df.id) & set(val_df.id)) == 0
        
        tokenized_train = make_dataset(train_df, tokenizer)
        tokenized_val = make_dataset(val_df, tokenizer)
        
        p, fold_f1[k] = fold_probabilities(model_name, SEED, tokenizer, data_collator, tokenized_train, NUM_EPOCHS, tokenized_val, val_df['label'])
               
        oof.append(pd.DataFrame({"id": val_df['id'].to_numpy(), "fold": val_df['fold'].to_numpy(), "p": p}))
        log(f"Fold {k}: F1 {fold_f1[k]:.4f} ({time.perf_counter() - t:.0f} s)")

    log("Validating output of all trained folds")
    oof_df = pd.concat(oof, ignore_index=True)
    assert len(oof_df) == len(train_all) == 7613
    assert oof_df.id.is_unique
    assert set(oof_df.id) == set(train_all.id)   # every tweet gets exactly one out-of-fold probability

    # Recompute each fold's F1 from the saved p, matched to the labels by id (not by row order)
    oof_labelled = oof_df.merge(train_all_folds[['id', 'label']], on='id', how='inner', validate='one_to_one')
    assert len(oof_labelled) == len(oof_df)
    for k, g in oof_labelled.groupby('fold'):
        assert f1_score(g.label, g.p >= 0.5) == fold_f1[k]
    log("Checks passed: 7,613 tweets, one probability each; per-fold F1 recomputed by id matches")

    oof_df.to_csv(data_analysis.HERE / f"{oof_name}.csv", index=False)

    log("Summary:")
    f1s = pd.Series(fold_f1)
    sd = np.std(f1s.to_numpy())   # ddof=0, as TF-IDF's ± 0.012
    log(f"All folds F1: {f1s.round(4).to_dict()}")
    log(f"mean {f1s.mean():.4f}, sd {sd:.4f}, min {f1s.min():.4f}, max {f1s.max():.4f}")

    # keep_rule prints; its lines are caught and logged so they reach the run's file too
    log("Keep rule: BERT + keyword against BERT, fold by fold")
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        dt.keep_rule(BERT_FOLD_F1, f1s.to_numpy())
    for line in printed.getvalue().splitlines():
        log(line)
    log("Done")


if __name__ == "__main__":
    log_path = setup_logging()
    log(f"Logging to {log_path}")
    try:
        #main("cardiffnlp/twitter-roberta-base", "oof_twitter_roberta", preprocess_roberta)
        main("bert-base-uncased", "oof_bert_keyword", preprocess_keyword)
    except Exception:
        logger.exception(f"{elapsed()} Run failed")   # the traceback goes into the file too
        raise
