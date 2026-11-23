import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import Trainer
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import set_seed
import evaluate

SEED = 42

raw_datasets = load_dataset("nyu-mll/glue", "mrpc")
# loaded once: compute_metrics runs after every evaluation
metric = evaluate.load("glue", "mrpc")

def tokenize_function(example, tokenizer):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def main(local, train, seed=SEED):
    if local:
        checkpoint = "bert-mrpc"
    else:
        checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, fn_kwargs={"tokenizer": tokenizer})
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(
        "test-trainer",
        seed=seed,
        eval_strategy="epoch",
        save_strategy="epoch",              # a checkpoint per epoch, so the best one can be reloaded
        metric_for_best_model="accuracy",   # tracks the best epoch; F1 flatters "always say paraphrase" (0.81)
        save_total_limit=2,                 # keep the best and the latest checkpoint only
        # not load_best_model_at_end=True: bert-base-uncased names LayerNorm's weights gamma/beta,
        # from_pretrained renames them but the Trainer's best-model reload doesn't, so LayerNorm
        # would keep the last epoch's values. The best epoch is reloaded below instead.
    )
    # the classifier head is drawn here: seed first, so every run starts from the same head
    # (the Trainer seeds too, but only inside Trainer(...), after the head exists)
    set_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics
    )

    if train:
        trainer.train()
        # reload the best epoch: from_pretrained reads every weight, LayerNorm included,
        # and load_state_dict copies them into the model the Trainer already uses (on the GPU)
        print(f"best checkpoint: {trainer.state.best_model_checkpoint}")
        best = AutoModelForSequenceClassification.from_pretrained(trainer.state.best_model_checkpoint)
        trainer.model.load_state_dict(best.state_dict())
        trainer.save_model("bert-mrpc")

    return trainer, tokenized_datasets


def predict(trainer: Trainer, tokenized_datasets, label):
    predictions = trainer.predict(tokenized_datasets["validation"])
    print(f"===== {label} =====")
    print(predictions.predictions.shape, predictions.label_ids.shape)
    print(predictions.metrics)
    df = pd.DataFrame({"logit_0" : predictions.predictions[:,0],
                       "logit_1": predictions.predictions[:,1],
                       "label": predictions.label_ids})
    print(df.head())
    return df


if __name__ == '__main__':
    # baseline: bert-base-uncased with its random classifier head (the same head the
    # seeded training runs start from), no training
    trainer, tokenized_datasets = main(local=False, train=False)
    predict(trainer, tokenized_datasets, "baseline (untrained head)")

    # two training runs with the same seed: should match to about 3 decimals
    for run in (1, 2):
        trainer, tokenized_datasets = main(local=False, train=True)
        predict(trainer, tokenized_datasets, f"trained, seed {SEED}, run {run}")
