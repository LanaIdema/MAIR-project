import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

SEED = 42
TEST_SIZE = 0.15

DATA_PATH = "data/dialog_acts.dat"
OUT_DIR = "data/processed"


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



def stratified_split(df: pd.DataFrame, test_size = TEST_SIZE, seed = SEED ):
    # perform stratified split with given test size and seed
    train, test = train_test_split(df, test_size=test_size, stratify=df["label"], random_state=seed)

    return train.reset_index(drop = True), test.reset_index(drop=True)



def grouped_split(df, test_size=TEST_SIZE, seed=SEED):
    # perform grouped stratified split 
    uniq = df.drop_duplicates("text") # get unique utterances
    counts = uniq["label"].value_counts()
    rare = set(counts[counts < 2].index) # classes with only one instance can't be stratified so we leave them out

    splittable = uniq[~uniq["label"].isin(rare)]

    _, test_uniq = train_test_split(
        splittable,
        test_size=test_size,
        stratify=splittable["label"],
        random_state=seed,
    )

    in_test = df["text"].isin(set(test_uniq["text"])) # get ALL the rows (incl. duplicates) that are in test set

    return df[~in_test].reset_index(drop=True), df[in_test].reset_index(drop=True)

def save_splits(train, test, out_dir, variant):
    # save the splits to a given dir as csv

    out = Path(out_dir) / variant
    out.mkdir(parents=True, exist_ok=True)

    train.to_csv(out / "train.csv", index=False)
    test.to_csv(out / "test.csv", index=False)

    print(f"wrote {out}/train.csv ({len(train)}) and test.csv ({len(test)})")


def describe_classes(df, train, test):
    # see the exact class sizes and their percentage of the dataset

    table = pd.DataFrame({
        "full": df["label"].value_counts(),
        "train": train["label"].value_counts(),
        "test": test["label"].value_counts(),
    }).fillna(0).astype(int)

    table["full_%"] = (table["full"] / len(df) * 100).round(2)
    table["train_%"] = (table["train"] / len(train) * 100).round(2)
    table["test_%"] = (table["test"] / len(test) * 100).round(2)

    return table.sort_values("full", ascending=False)


if __name__ == "__main__":
    # perform splitting, print summary, save datasets

    df = load_dialog_data(DATA_PATH)
    print(f"rows={len(df)}  labels={df['label'].nunique()}  "
        f"unique_texts={df['text'].nunique()}")

    for variant, split_fn in [("original", stratified_split),
                              ("grouped", grouped_split)]:
        train, test = split_fn(df)

        print(f"\n=== {variant} ===")
        print(describe_classes(df, train, test).to_string())
        save_splits(train, test, OUT_DIR, variant)

