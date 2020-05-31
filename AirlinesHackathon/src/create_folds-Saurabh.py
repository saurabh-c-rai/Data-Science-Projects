import pandas as pd
from sklearn import model_selection
from os import environ

TRAIN_FILE_PATH = environ.get("TRAIN_FILE_PATH")
TRAINING_DATA = environ.get("TRAINING_DATA")
if __name__ == "__main__":
    df = pd.read_csv(TRAIN_FILE_PATH)
    df["kfold"] = -1

    df = df.sample(frac=1).reset_index(drop=True)

    kf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    X = df.iloc[:, 1:].copy()
    y = df.iloc[:, 0].copy()
    for fold, (train_idx, val_idx) in enumerate(kf.split(X=X, y=y)):
        print(len(train_idx), len(val_idx))
        df.loc[val_idx, "kfold"] = fold

    df.to_csv(TRAINING_DATA, index=False)

