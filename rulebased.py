import pandas as pd
import re
from sklearn.metrics import accuracy_score

from rulebased_rules import RULES


def apply_rulebased(df: pd.DataFrame) -> pd.DataFrame:
    predictions = []

    for text in df["text"]:
        prediction = "null"
        
        for label, rules in RULES.items():
            for rule in rules:
                # The first rule that matches will be the prediction
                # To check whether a keyword matches the text, we need to use regex
                # otherwise the keyword "no" will match with "noise" for example
                if re.search(r"\b" + re.escape(rule) + r"\b", text):
                    prediction = label
                    break

            # when a prediction is made, we do not look further
            if prediction != "null":
                break

        predictions.append(prediction)

    df["prediction"] = predictions                         
    return df


def calculate_accuracy_rulebased(df: pd.DataFrame) -> float:
    # We need to replace the null values with the string " null"
    # to avoid errors when calculating the accuracy
    df["label"] = df["label"].fillna("null")
    true_labels = df["label"]

    predicted_labels = df["prediction"]

    return accuracy_score(true_labels, predicted_labels)


# calculate accuracy on the train set
df_train = pd.read_csv("data/processed/grouped/train.csv")
df_train_predictions = apply_rulebased(df_train)

accuracy = calculate_accuracy_rulebased(df_train_predictions)
print(f"Rule-based accuracy train: {accuracy}")
# Rule-based accuracy train: 0.9189981732525718


# calculate accuracy on the test set
df_test = pd.read_csv("data/processed/grouped/test.csv")
df_test_predictions = apply_rulebased(df_test)

accuracy = calculate_accuracy_rulebased(df_test_predictions)
print(f"Rule-based accuracy test: {accuracy}")
# Rule-based accuracy test: 0.8967459324155194