import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
TEST_SIZE = 0.15


def load_dialog_data(path) -> pd.DataFrame:

    rows = []

    with open (path) as f:
        for line in f:

            parts = line.strip().lower().split(maxsplit=1)

            if not parts:
                continue

            label = parts[0]
            text = parts[1]
            rows.append({"label": label, "text": text})

    return pd.DataFrame(rows)



def stratified_split(df: pd.DataFrame, test_size = TEST_SIZE, seed = SEED ):

    train, test = train_test_split(df, test_size=test_size, stratify=df["label"], random_state=seed)

    return train.reset_index(drop = True), test.reset_index(drop=True)



def grouped_split(df: pd.DataFrame, test_size: float = TEST_SIZE, seed: int = SEED):

    uniq = df.drop_duplicates("text")
    _, test_uniq = train_test_split(
        uniq, test_size=test_size, stratify=uniq["label"], random_state=seed)

    in_test = df["text"].isin(set(test_uniq["text"]))
    return (
        df[~in_test].reset_index(drop=True),
        df[in_test].reset_index(drop=True),
    )


if __name__ == "__main__":
    pass

