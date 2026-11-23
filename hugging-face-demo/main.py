import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import Trainer
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification

raw_datasets = load_dataset("nyu-mll/glue", "mrpc")

def tokenize_function(example, tokenizer):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

def main(local, train):
    if local:
        checkpoint = "bert-mrpc"
    else:
        checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, fn_kwargs={"tokenizer": tokenizer})
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments("test-trainer")
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    if train:
        trainer.train()
        trainer.save_model("bert-mrpc")

    return trainer, tokenized_datasets

def predict(trainer: Trainer, tokenized_datasets, train):
    predictions = trainer.predict(tokenized_datasets["validation"])
    print(predictions.predictions.shape, predictions.label_ids.shape)
    df = pd.DataFrame({"logit_0" : predictions.predictions[:,0], 
                       "logit_1": predictions.predictions[:,1],
                       "label": predictions.label_ids})
    print(F"train: {train}")
    print(df.head())
    return df

if __name__ == '__main__':
    trainer, tokenized_datasets = main(local=False, train=True)
    df_from_training = predict(trainer, tokenized_datasets, True)
    trainer, tokenized_datasets = main(local=True, train=False)
    df_from_eval = predict(trainer, tokenized_datasets, False)

    pd.testing.assert_frame_equal(df_from_training, df_from_eval, check_exact=False, atol=1e-4, rtol=0)

    