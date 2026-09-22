import pandas as pd
import torch
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
from datasets import Dataset
import time


MODEL_NAME = "distilbert-base-uncased"

DATASETS = {
    "original": ("./data/processed/original/train.csv", "./data/processed/original/test.csv"),
    "grouped":  ("./data/processed/grouped/train.csv",  "./data/processed/grouped/test.csv"),
}

def compute_metrics(to_evaluate):
    logits, labels = to_evaluate
    pred = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, pred),
            "balanced_accuracy": balanced_accuracy_score(labels, pred),
             "macro_f1": f1_score(labels, pred, average="macro", zero_devision=0)
             }


if __name__ == "__main__":
      tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
      for name, (train_path, test_path) in DATASETS.items():

           df_train = pd.read_csv(train_path)
           df_test = pd.read_csv(test_path)

           label_encoder = LabelEncoder()
           df_train['label'] = label_encoder.fit_transform(df_train['label'])
           df_test['label'] = label_encoder.transform(df_test['label'])
           num_labels = len(label_encoder.classes_)
           train_ds = Dataset.from_pandas(df_train)
           test_ds = Dataset.from_pandas(df_test)

def tokenize_func(examples):
     # if tokens <64 add zeros, if 64<tokens, truncnate to 64, ensure all items have equal length
     # texts are ensured to be string, when empty: "" is handeled easier
     texts = [str(x) for x in examples["text"]]
     return tokenizer(texts, padding="max_length", truncation=True, max_length=64) 

train_ds = train_ds.map(tokenize_func, batched=True, remove_columns=["text"])
test_ds = test_ds.map(tokenize_func, batched=True, remove_columns=["text"])

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)

if name == "original":
     parameter_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
     print(f"\n Model Parameter size: {parameter_size} MB")
     args = TrainingArguments(
          output_dir = f"./finetuning_results_{name}",
          eval_strategy="epoch",
          learning_rate=0.00005,
          per_device_train_batch_size=8,
          per_device_eval_batch_size=16,
          num_train_epochs=3,
          weight_decay= 0.01,
          report_to="none",
          save_strategy="no"
     )
     trainer = Trainer(
              model=model,
              args=args,
              train_dataset=train_ds,
              eval_dataset=test_ds,
              compute_metrics=compute_metrics
     
             )

start_time = time.time()
trainer.train()
train_time = time.time() - start_time
print(f"time to train model {name}: {train_time} seconds")
res = trainer.evaluate()
print(f"model {name} Accuracy: {res['eval_accuracy']}")
print(f"model {name} Balanced Accuracy: {res['eval_balanced_accuracy']}")
print(f"model {name} macro-F1: {res['eval_macro_f1']}")



        






        