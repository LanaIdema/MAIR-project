import numpy as np

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import SVC

# Bag of words extractor, found on https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
from sklearn.feature_extraction.text import CountVectorizer


if __name__ == '__main__':
    # Load the normal data
    df_train = pd.read_csv("./data/processed/original/train.csv")
    df_test = pd.read_csv("./data/processed/original/test.csv")

    # Load the stratisfied data
    df_strat_train = pd.read_csv("./data/processed/grouped/train.csv")
    df_strat_test = pd.read_csv("./data/processed/grouped/test.csv")

    # Two todo's for both: encode the labels to categorical features, based on trainingsdata
    label_encoder = LabelEncoder()

    X_train = df_train['text']
    X_test = df_test['text']
    y_train = label_encoder.fit_transform(df_train['label'])
    y_test = label_encoder.transform(df_test['label'])

    label_encoder_strat = LabelEncoder()
    
    X_strat_train = df_strat_train['text']
    X_strat_test = df_strat_test['text']
    y_strat_train = label_encoder_strat.fit_transform(df_strat_train['label'])
    y_strat_test = label_encoder_strat.transform(df_strat_test['label'])

    

    pipe = Pipeline([
            ('bag-of-words-vectorizer', CountVectorizer()), # Count Vectorizer ignores unknown words, creates a bag of words representation
            ('normalizer', Normalizer('l2')), # Normalize the bag-of-words vectors to optimize distance calculations
            ('svc', SVC()) # Apply Support Vector Machine, normal settings
    ])

    print(pipe
          .fit(X_train, y_train)
          .score(X_test, y_test)
    )

    print(pipe
            .fit(X_strat_train, y_strat_train)
            .score(X_strat_test, y_strat_test)
    )
