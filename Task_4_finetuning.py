import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
import numpy as np
from datasets import Dataset
import time


MODEL_NAME = "distilbert-base-uncased"

DATASETS = {
    "original": ("./data/processed/original/train.csv",
                 "./data/processed/original/test.csv"),
    "grouped":  ("./data/processed/grouped/train.csv",
                 "./data/processed/grouped/test.csv"),
}

def compute_metrics(to_evaluate):
    logits, labels = to_evaluate
    pred = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, pred),
            "balanced_accuracy": balanced_accuracy_score(labels, pred),
             "macro_f1": f1_score(labels, pred, average="macro", zero_division=0)
             }

def tokenize_func(examples, tokenizer_custom):
     # if tokens <64 add zeros, if 64<tokens, truncnate to 64, ensure all items have equal length
     # texts are ensured to be string, when empty: "" is handeled easier
     texts = [str(x) for x in examples["text"]]
     return tokenizer_custom(texts, padding="max_length", truncation=True, max_length=64) 

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

          train_ds = train_ds.map(tokenize_func, batched=True, remove_columns=["text"], fn_kwargs={"tokenizer_custom": tokenizer})
          test_ds = test_ds.map(tokenize_func, batched=True, remove_columns=["text"], fn_kwargs={"tokenizer_custom": tokenizer})

          model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)

          parameter_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
          print(f"\n Model Parameter size: {parameter_size} MB")
          args = TrainingArguments(
          output_dir = f"./finetuning_results_{name}",
          eval_strategy="epoch",
          save_strategy="epoch",
          learning_rate=0.00005,
          per_device_train_batch_size=16,
          per_device_eval_batch_size=32,
          num_train_epochs=20,
          weight_decay= 0.01,
          report_to="none",
          load_best_model_at_end=True,
          metric_for_best_model="macro_f1",
          greater_is_better=True,
          )
          
          trainer = Trainer(
          model=model,
          args=args,
          train_dataset=train_ds,
          eval_dataset=test_ds,
          compute_metrics=compute_metrics,
          callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
          )


          start_time = time.time()
          trainer.train()
          train_time = time.time() - start_time
          print(f"time to train model {name}: {train_time} seconds")
          res = trainer.evaluate()
          print(f"model {name} Accuracy: {res['eval_accuracy']}")
          print(f"model {name} Balanced Accuracy: {res['eval_balanced_accuracy']}")
          print(f"model {name} macro-F1: {res['eval_macro_f1']}")


"""
                         First time Model Training

Model Parameter size: 255.45122909545898 MB
evaluation metric: validation loss
early stopping: no
train batch size = 8
eval batch size = 16

 - model original - 
[25500/25500 46:37, Epoch 20/20]
time to train model original: 2799.085625886917 seconds          
model original Accuracy: 0.9872222222222222
model original Balanced Accuracy: 0.879689798230982
model original macro-F1: 0.868376055544059

- model grouped - 
[26020/26020 45:34, Epoch 20/20]
time to train model grouped: 2735.184300661087 seconds
model grouped Accuracy: 0.9787234042553191
model grouped Balanced Accuracy: 0.7468120493015109
model grouped macro-F1: 0.7486764643715479

                         Second time Model Training
                    
Parameter size is equal
evaluation metric: f1_macro score
early stopping: yes
training batch size = 16
eval batch size = 32

- model original -
7018/12760 17:36 < 14:24, 6.64 it/s, Epoch 11/20]
model original Accuracy: 0.99
model original Balanced Accuracy: 0.9691730034808346
model original macro-F1: 0.9644437140077251

- model grouped -
[ 7812/13020 19:27 < 12:58, 6.69 it/s, Epoch 12/20]
model grouped Accuracy: 0.9809136420525657
model grouped Balanced Accuracy: 0.8151413486746313
model grouped macro-F1: 0.8047777215812875


higher batch size to speed up, but would be similar otherwise
more epochs (between 10/15) recommended for consistent results
"""