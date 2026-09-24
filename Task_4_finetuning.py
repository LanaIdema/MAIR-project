import time

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, EarlyStoppingCallback, Trainer, TrainingArguments, DataCollatorWithPadding, set_seed)
import torch


SEED = 42
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 512
VAL_SIZE = 0.1

DATASETS = {
    "original": ("./data/processed/original/train.csv", "./data/processed/original/test.csv"),
    "grouped":  ("./data/processed/grouped/train.csv",  "./data/processed/grouped/test.csv"),
}


def load_data(train_path, test_path, tokenizer):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    label_encoder = LabelEncoder()
    df_train['label'] = label_encoder.fit_transform(df_train['label'])
    df_test['label'] = label_encoder.transform(df_test['label'])


    df_train, df_val = train_test_split(
        df_train, test_size=VAL_SIZE, stratify=df_train['label'], random_state=SEED
    )

    train_ds = tokenize(Dataset.from_pandas(df_train, preserve_index=False), tokenizer)
    val_ds = tokenize(Dataset.from_pandas(df_val, preserve_index=False), tokenizer)
    test_ds = tokenize(Dataset.from_pandas(df_test, preserve_index=False), tokenizer)

    return train_ds, val_ds, test_ds, label_encoder


def load_model(num_labels, model_name=MODEL_NAME):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    print(f"Model parameter size: {size_mb:.2f} MB")
    return model


def tokenize(dataset, tokenizer):
    def _tokenize(batch):
        texts = [str(x) for x in batch["text"]]
        return tokenizer(texts, truncation=True, max_length=MAX_LENGTH)

    return dataset.map(_tokenize, batched=True, remove_columns=["text"])


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    pred = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, pred),
        "balanced_accuracy": balanced_accuracy_score(labels, pred),
        "macro_f1": f1_score(labels, pred, average="macro", zero_division=0),
    }


def make_training_args(name):
    return TrainingArguments(
        output_dir=f"./finetuning_results_{name}",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        learning_rate=3e-5,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=10,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=SEED,
    )


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    for name, (train_path, test_path) in DATASETS.items():
        train_ds, val_ds, test_ds, label_encoder = load_data(train_path, test_path, tokenizer)
        set_seed(SEED)
        model = load_model(num_labels=len(label_encoder.classes_))

        trainer = Trainer(
            model=model,
            args=make_training_args(name),
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            data_collator=DataCollatorWithPadding(tokenizer)
        )

        start_time = time.time()
        trainer.train()
        train_time = time.time() - start_time

        res = trainer.evaluate(test_ds, metric_key_prefix="test")
        print(name, "fine-tuned", MODEL_NAME,
              "acc:", round(res['test_accuracy'], 4),
              "balanced acc:", round(res['test_balanced_accuracy'], 4),
              "macro-F1:", round(res['test_macro_f1'], 4),
              "train time (s):", round(train_time, 1))

