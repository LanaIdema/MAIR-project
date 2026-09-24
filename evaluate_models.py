
from pickle import load
import sys
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import os
import pandas as pd

def load_dialog_data(path) -> pd.DataFrame:
    # load the data

    rows = []

    with open (path) as f:
        for line in f:

            parts = line.strip().lower().split(maxsplit=1) # split on the first space

            if not parts:
                continue

            label = parts[0]
            text = parts[1]
            rows.append({"label": label, "text": text}) # append back to the rows list

    return pd.DataFrame(rows)

def main(args):

    # Make the data from this path
    path = args[0]
    data = load_dialog_data(path)

    for model in os.listdir("./models"):
        print(model)
        with open(os.path.join("./models", model), "rb") as f:
            model = load(f)

        # Score model on data loaded from path
        try:
            pred_y = model.predict(data['text'])
            # Assuming last step is estimator and estimator has a label encoder
            y = model[-1].label_encoder_.transform(data["label"])
            print(f"Accuracy: {accuracy_score(y, pred_y)}, balanced_accuracy: {balanced_accuracy_score(y, pred_y)}")
        except:
            print("Not implemented yet")
        print("-"*30)

if __name__ == '__main__':
    main(sys.argv[1:]) 