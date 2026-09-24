import pandas as pd
import re
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

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


def calculate_metrics_rulebased(df: pd.DataFrame) -> float:
    # We need to replace the null values with the string " null"
    # to avoid errors when calculating the accuracy
    df["label"] = df["label"].fillna("null")
    true_labels = df["label"]

    predicted_labels = df["prediction"]

    return accuracy_score(true_labels, predicted_labels), balanced_accuracy_score(true_labels, predicted_labels), f1_score(true_labels, predicted_labels, average="macro", zero_division=0)


# calculate accuracy on the train set
df_train = pd.read_csv("data/processed/grouped/train.csv")
df_train_predictions = apply_rulebased(df_train)

accuracy, balanced_accuracy, macro_F1 = calculate_metrics_rulebased(df_train_predictions)
print(f"Rule-based accuracy train: {accuracy}")
print(f"Rule-based accuracy train: {balanced_accuracy}")
print(f"Rule-based accuracy train: {macro_F1}")
# Rule-based accuracy train: 0.9189981732525718
# Rule-based accuracy train: 0.846409270894487
# Rule-based accuracy train: 0.806821034238376


# calculate accuracy on the test set
df_test = pd.read_csv("data/processed/grouped/test.csv")
df_test_predictions = apply_rulebased(df_test)

accuracy, balanced_accuracy, macro_F1 = calculate_metrics_rulebased(df_test_predictions)
print(f"Rule-based accuracy test: {accuracy}")
print(f"Rule-based accuracy test: {balanced_accuracy}")
print(f"Rule-based accuracy test: {macro_F1}")
# Rule-based accuracy test: 0.8967459324155194
# Rule-based accuracy test: 0.7747955222295051
# Rule-based accuracy test: 0.7265835369351522