import os

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from . import dispatcher

TRAINING_DATA = os.environ.get("TRAINING_DATA")
FOLD = int(os.environ.get("FOLD"))
MODEL = os.environ.get("MODELS")
FOLD_MAPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3],
}

if __name__ == "__main__":
    df = pd.read_csv(TRAINING_DATA)
    train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))]
    valid_df = df[df.kfold == FOLD]
    print df["kfold"].value_counts()
    print (train_df.shape, valid_df.shape)
    print (train_df["kfold"].value_counts())
    print (valid_df["kfold"].value_counts())

    X = train_df.drop(columns=["case_"])

    clf = dispatcher.MODELS[MODEL]
    clf.fit(X, y)
    y_pred = clf.predict(X_test)
