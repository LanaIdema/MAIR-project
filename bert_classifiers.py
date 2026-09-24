import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from pickle import dump


SEED = 42
MODEL_NAME = "distilbert-base-uncased"

DATASETS = {
    "original": ("./data/processed/original/train.csv", "./data/processed/original/test.csv"),
    "grouped":  ("./data/processed/grouped/train.csv",  "./data/processed/grouped/test.csv"),
}

def load_data(train_path, test_path, tokenizer, model, device):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(df_train['label'])
    y_test = label_encoder.transform(df_test['label'])

    X_train = embed(df_train['text'].astype(str), tokenizer, model, device, desc="Embedding train")
    X_test = embed(df_test['text'].astype(str), tokenizer, model, device, desc="Embedding test")

    return X_train, X_test, y_train, y_test, label_encoder


def load_model(model_name = MODEL_NAME):
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    return tokenizer, model, device



def embed(texts, tokenizer, model, device, batch_size=64, desc="Embedding"):
    vectors = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc=desc, unit="batch"):
            batch = tokenizer(
                list(texts[i:i + batch_size]),
                padding=True, truncation=True, max_length=512,
                return_tensors="pt",
            ).to(device)
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1)
            vectors.append(((hidden * mask).sum(1) / mask.sum(1)).cpu())
    return torch.cat(vectors).numpy()



def make_classifiers():
    return [
        SVC(),
        DecisionTreeClassifier(),
        MLPClassifier()
    ]




if __name__ == '__main__':
    tokenizer, model, device = load_model()

    for name, (train_path, test_path) in DATASETS.items():
        X_train, X_test, y_train, y_test, label_encoder = load_data(
            train_path, test_path, tokenizer, model, device
        )

        for clf in make_classifiers():
            pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
            pred = pipe.fit(X_train, y_train).predict(X_test)
            print(name, type(clf).__name__,
                  "acc:", round(accuracy_score(y_test, pred), 4),
                  "balanced acc:", round(balanced_accuracy_score(y_test, pred), 4),
                  "macro-F1:", round(f1_score(y_test, pred, average="macro", zero_division=0), 4))

            with open(f"./models/model_{type(clf).__name__}_{name}.pkl".lower(), "wb") as f:
                dump(pipe, f, protocol=5)


"""
original SVC acc: 0.9836 balanced acc: 0.8288 macro-F1: 0.8647
original DecisionTreeClassifier acc: 0.9561 balanced acc: 0.8097 macro-F1: 0.7699
original MLPClassifier acc: 0.9875 balanced acc: 0.947 macro-F1: 0.9462

grouped SVC acc: 0.9465 balanced acc: 0.6157 macro-F1: 0.616
grouped DecisionTreeClassifier acc: 0.709 balanced acc: 0.3029 macro-F1: 0.2723
grouped MLPClassifier acc: 0.9546 balanced acc: 0.6245 macro-F1: 0.6501

"""


