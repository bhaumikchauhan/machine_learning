import torch

import disaster_tweets as dt
import data_analysis
import pandas as pd
import datasets
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import Trainer
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import set_seed
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from transformers import PreTrainedTokenizerBase

CHECKPOINT = "bert-base-uncased"
SEED = 42
NUM_TRAIN_EPOCHS = 3
LEARNING_RATE = 5e-5
PER_DEVICE_TRAIN_BATCH_SIZE=16

def tokenize_function(row, tokenizer: PreTrainedTokenizerBase):
    return tokenizer(row['text'], truncation=True, max_length=96)

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    preds = np.argmax(logits, axis=-1)
    return {"f1": f1_score(labels, preds),
            "precision": precision_score(labels, preds),
            "recall": recall_score(labels, preds),
            "accuracy": accuracy_score(labels, preds)}

if __name__ == "__main__":
    #merge train and folds on id
    train_all = dt.get_train_data_from_csv()
    folds = pd.read_csv(dt.FOLDS_CSV)
    train_all_folds = train_all.merge(folds, on="id", how="inner")
    assert len(train_all) == len(train_all_folds)

    train_all_folds = train_all_folds.rename(columns={'target' : 'label'})
    columns_for_transformer = ['text', 'label', 'id']
    
    train_df = train_all_folds[train_all_folds['fold'] != 0]
    val_df = train_all_folds[train_all_folds['fold'] == 0]

    assert len(set(train_df.id) & set(val_df.id)) == 0
    
    train_ds = datasets.Dataset.from_pandas(train_df[columns_for_transformer], preserve_index=False)
    val_ds = datasets.Dataset.from_pandas(val_df[columns_for_transformer], preserve_index=False)



    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    tokenized_train_ds = train_ds.map(tokenize_function, batched=True, fn_kwargs={"tokenizer": tokenizer})
    tokenized_train_ds = tokenized_train_ds.remove_columns(['text', 'id'])
    tokenized_val_ds = val_ds.map(tokenize_function, batched=True, fn_kwargs={"tokenizer": tokenizer})
    tokenized_val_ds = tokenized_val_ds.remove_columns(['text', 'id'])

    set_seed(SEED)

    training_args = TrainingArguments(
            output_dir="bert_runs",
            num_train_epochs=NUM_TRAIN_EPOCHS,
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
            bf16=True,
            report_to="none",
            eval_strategy="no",
            save_strategy="no"
        )

    model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT, num_labels=2)
    trainer = Trainer(
            model,
            training_args,
            train_dataset=tokenized_train_ds,
            eval_dataset=tokenized_val_ds,
            data_collator=data_collator,
            processing_class=tokenizer,
            compute_metrics=compute_metrics
        )

    trainer.train()

    predictions = trainer.predict(tokenized_val_ds)
    logits = predictions.predictions
    metric = predictions.metrics
    print(metric)
    p = torch.softmax(torch.tensor(logits), dim = -1)[:, 1].numpy()

    oof_bert_fold0_df = pd.DataFrame()
    oof_bert_fold0_df['id'] = val_df['id']
    oof_bert_fold0_df['fold'] = val_df['fold']
    oof_bert_fold0_df['p'] = p
    oof_bert_fold0_df.to_csv(data_analysis.HERE / 'oof_bert_fold0.csv', index=False)

    assert f1_score(val_df.label, p >= 0.5) == metric['test_f1']