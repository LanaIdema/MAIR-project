import pandas as pd
import torch
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from transformers import AutoModel, AutoTokenizer

SEED = 42
MODEL_NAME = "distilbert-base-uncased"

DATASETS = {
    "original": ("./data/processed/original/train.csv", "./data/processed/original/test.csv"),
    "grouped":  ("./data/processed/grouped/train.csv",  "./data/processed/grouped/test.csv"),
}


def load_model(model_name = MODEL_NAME):
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    return tokenizer, model, device



def embed(texts, tokenizer, model, device):
    vectors = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):

            batch = tokenizer(list(texts[i:i + 64]), padding=True, return_tensors="pt").to(device)
            
            hidden = model(**batch).last_hidden_state

            mask = batch["attention_mask"].unsqueeze(-1)

            vectors.append(((hidden * mask).sum(1) / mask.sum(1)).cpu())

    return torch.cat(vectors).numpy()



if __name__ == '__main__':
    tokenizer, model, device = load_model()
 
    for name, (train_path, test_path) in DATASETS.items():
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
 
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(df_train['label'])
        y_test = label_encoder.transform(df_test['label'])
 
        X_train = embed(df_train['text'].astype(str), tokenizer, model, device)
        X_test = embed(df_test['text'].astype(str), tokenizer, model, device)
 
        for clf in [SVC(), MLPClassifier(random_state= SEED)]:
            pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
            pred = pipe.fit(X_train, y_train).predict(X_test)
            
            print(name, type(clf).__name__,
                "acc:", round(accuracy_score(y_test, pred), 4),
                "balanced acc:", round(balanced_accuracy_score(y_test, pred), 4),
                "macro-F1:", round(f1_score(y_test, pred, average="macro", zero_division=0), 4))







"""
original SVC acc: 0.9836 balanced acc: 0.8288 macro-F1: 0.8647
original MLPClassifier acc: 0.9861 balanced acc: 0.9402 macro-F1: 0.9338

"""