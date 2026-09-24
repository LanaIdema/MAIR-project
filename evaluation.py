import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from pickle import dump

from rulebased_rules import RULES

from src.unk_handler import UnknownHandler
from src.rule_based_classifier import RuleBasedClassifier
from src.decision_tree_classifier import TextualDecisionTreeClassifier
from src.support_vector_machine_classifier import TextualSVMClassifier

# Bag of words extractor, found on https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
from sklearn.feature_extraction.text import CountVectorizer



def read_data(loc_training_data: str, loc_testing_data: str):
    df_train = pd.read_csv(loc_training_data, keep_default_na=False)
    df_test = pd.read_csv(loc_testing_data, keep_default_na=False)

    X_train = df_train['text']
    X_test = df_test['text']
    y_train = df_train['label']
    y_test = df_test['label']

    return {
        "X_train": X_train,
        "y_train": y_train, 
        "X_test": X_test,
        "y_test": y_test,
    }

if __name__ == '__main__':
    # Load the regular data
    regular_data = read_data(
        "./data/processed/original/train.csv", 
        "./data/processed/original/test.csv"
    )
    # Load stratisfied data
    grouped_data = read_data(
        "./data/processed/grouped/train.csv", 
        "./data/processed/grouped/test.csv"
    )

    datasets = [
        ("regular_data", regular_data), 
        ("grouped_data", grouped_data)
    ]
    model_types = [
        (("rule_based_classifier", RuleBasedClassifier(rules=RULES)), False), # Step 1
        (("decision_tree", TextualDecisionTreeClassifier()), True), # Step 2
        (("support_vector_machine", TextualSVMClassifier()), True), # Step 2
    ]

    FILENAME = "evaluation.md"
    with open(FILENAME, "w", encoding="utf-8") as file:
        file.write("| Model | Dataset | Accuracy | Balanced Accuracy | Macro F1-score |\n")
        file.write("| --- | --- | --- | --- | --- |\n")
        for model_type, bag_of_words in model_types:
            for dataset_name, dataset in datasets:

                print(f"Training and evaluating {model_type[0]} on {dataset_name} {"(bag-of-words)" if bag_of_words else ""}")
                
                pipe = None
                
                if bag_of_words:
                    pipe = Pipeline([
                        ('unknown-handler', UnknownHandler()), # Make unknown if token is below certain treshold
                        ('bag-of-words-vectorizer', CountVectorizer()), # Creates a bag of words representation
                        ('normalizer', Normalizer('l2')), # Normalize the bag-of-words vectors to optimize distance calculations
                        model_type # Chain the model in the pipe
                    ])
                else:
                    pipe = Pipeline([
                        model_type
                    ])

                # Fit the model to the data
                model = pipe.fit(dataset["X_train"], dataset["y_train"])
                # Predict the test set
                pred_y = model.predict(dataset["X_test"])
                # Transform the y_true of the test set
                y_test = model.named_steps[model_type[0]].label_encoder_.transform(dataset["y_test"])
                # Report
                file.write(f"| {model_type[0]} | {dataset_name} | {accuracy_score(y_test, pred_y)} | {balanced_accuracy_score(y_test, pred_y)} | {f1_score(y_test, pred_y, average='macro')} |\n")

                with open(f"./models/model_{model_type[0]}_{dataset_name}.pkl".lower().replace(" ", "_"), "wb") as f:
                    dump(model, f, protocol=5)
                print("Done...")

    print("Finished...")

            