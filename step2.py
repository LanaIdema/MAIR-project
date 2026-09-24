from collections import Counter

import numpy as np

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from pickle import dump

# Bag of words extractor, found on https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
from sklearn.feature_extraction.text import CountVectorizer

# Since we are using the sklearn pipe pattern, we have to create our own _UNK_-handler
class UnknownHandler(BaseEstimator, TransformerMixin):
    def __init__(self, unknown_treshold=2):
        self.known_vocab = set()
        self.unknown_treshold = unknown_treshold

    def fit(self, X, y=None):
        all_text = " ".join(X)
        # Get the counts of each word
        word_counts = Counter(all_text.split())
        
        # If above treshold
        self.known_vocab = {
            word for word, count in word_counts.items() 
            if count >= self.unknown_treshold #Treshold value
        }
        return self

    def transform(self, X):
        # Replace rare or unseen words with the unk_token
        return [
            " ".join([word if word in self.known_vocab else "_UNK_" for word in doc.split()])
            for doc in X
        ]

def read_data(loc_training_data: str, loc_testing_data: str):
    df_train = pd.read_csv(loc_training_data)
    df_test = pd.read_csv(loc_testing_data)

    # Two todo's for both: encode the labels to categorical features, based on trainingsdata
    label_encoder = LabelEncoder()

    X_train = df_train['text']
    X_test = df_test['text']
    y_train = label_encoder.fit_transform(df_train['label'])
    y_test = label_encoder.transform(df_test['label'])

    return {
        "X_train": X_train,
        "y_train": y_train, 
        "X_test": X_test,
        "y_test": y_test,
        "label_encoder": label_encoder
    }

if __name__ == '__main__':
    # Load the regular data
    regular_data = read_data(
        "./data/processed/original/train.csv", 
        "./data/processed/original/test.csv"
    )
    # Load stratisfied data
    stratisfied_data = read_data(
        "./data/processed/grouped/train.csv", 
        "./data/processed/grouped/test.csv"
    )

    datasets = [
        ("Regular data", regular_data), 
        ("Stratisfied data", stratisfied_data)
    ]
    model_types = [
        ("mlp_classifier", MLPClassifier()),
        ("svm", SVC()),
        ("decision_tree", DecisionTreeClassifier())
    ]

    FILENAME = "output_step_2_unk.md"
    with open(FILENAME, "w", encoding="utf-8") as file:
        for model_type in model_types:
            for dataset_name, dataset in datasets:
                pipe = Pipeline([
                    ('unknown-handler', UnknownHandler()), # Make unknown if token is below certain treshold
                    ('bag-of-words-vectorizer', CountVectorizer()), # Creates a bag of words representation
                    ('normalizer', Normalizer('l2')), # Normalize the bag-of-words vectors to optimize distance calculations
                    model_type # Chain the model in the pipe
                ])

                # Fit the model to the data
                model = pipe.fit(dataset["X_train"], dataset["y_train"])
                # Predict the test set
                pred_y = model.predict(dataset["X_test"])
                # Report
                file.write(f"## {model_type[0]} [{dataset_name}]:\n")
                file.write(f"- {balanced_accuracy_score(dataset["y_test"], pred_y)} balanced_accuracy\n")
                file.write(f"- {f1_score(dataset["y_test"], pred_y, average='macro')} f1-score\n")

                with open(f"./models/model_{model_type[0]}_{dataset_name}.pkl".lower().replace(" ", "_"), "wb") as f:
                    dump(model, f, protocol=5)

    print("Done...")

            