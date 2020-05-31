import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

if __name__ == "__main__":
    df = pd.read_csv('input/train.csv')
    df['Kfold'] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X = df.drop(columns='case_count')
    y = df.loc[:, 'case_count']
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(len(train_idx), len(val_idx))
        df.loc[val_idx, 'Kfold'] = fold

    df.to_csv("input/train_folds.csv", index=False)
