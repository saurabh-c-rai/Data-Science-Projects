import math
import os
import sys
from os import environ
import pandas as pd
from sklearn import model_selection
import mlflow
from mlflow import log_param, log_artifact

import dispatcher

MODEL = environ.get("MODEL")
TRAIN_PATH = environ.get("TRAIN_FILE_PATH")
TRAINING_FOLD = environ.get("TRAINING_DATA")

if __name__ == "__main__":
    mlflow.set_experiment("Airlines Hackathon")
    with mlflow.start_run(run_name=f"Run_Create_Folds_{MODEL}"):

        df = pd.read_csv(TRAIN_PATH)
        df["kfold"] = -1

        df = df.sample(frac=1).reset_index(drop=True)

        kf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        X = df.iloc[:, 1:].copy()
        y = df.iloc[:, 0].copy()
        for fold, (train_idx, val_idx) in enumerate(kf.split(X=X, y=y)):
            print(len(train_idx), len(val_idx))
            df.loc[val_idx, "kfold"] = fold

        df.to_csv(TRAINING_FOLD, index=False)
        log_param("Stage", "Fold_created")
        log_artifact(local_path=TRAINING_FOLD)
