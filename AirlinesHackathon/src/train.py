#! /usr/local/bin/python3.8
from os import environ
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    f1_score,
    classification_report,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
)
import joblib

import dispatcher

# getting input from environment variables
TRAINING_DATA = environ.get("TRAINING_DATA")
FOLD = int(environ.get("FOLD"))
MODEL = environ.get("MODEL")

FOLD_MAPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3],
}


def metric_data(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    # roc_auc = roc_auc_score(y_test, y_pred, average="weighted", multi_class="ovo")
    print("Accuracy Score of the model = ", accuracy)
    print("Precision Score of the model = ", precision)
    print("Recall Score of the model = ", recall)
    print("F1 Score of the model = ", f1)
    # print("ROC AUC Score of the model = ", roc_auc)
    print("Classification Report is ")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    df = pd.read_csv(TRAINING_DATA)
    train_df = df[df["kfold"].isin(FOLD_MAPPING.get(FOLD))].copy()
    valid_df = df[df["kfold"] == FOLD].copy()

    ytrain = train_df["Severity"]
    yvalid = valid_df["Severity"]

    # print(train_df.columns)
    # print(ytrain.head())

    train_df.drop(columns=["Severity", "Accident_ID", "kfold"], inplace=True)
    valid_df.drop(columns=["Severity", "Accident_ID", "kfold"], inplace=True)

    targetEncoding = {
        "Highly_Fatal_And_Damaging": 0,
        "Significant_Damage_And_Serious_Injuries": 1,
        "Minor_Damage_And_Injuries": 2,
        "Significant_Damage_And_Fatalities": 3,
    }

    ytrain = ytrain.map(targetEncoding)
    yvalid = yvalid.map(targetEncoding)

    clf = dispatcher.MODELS[MODEL]
    clf.fit(train_df, ytrain)
    preds = clf.predict(valid_df)
    metric_data(yvalid, preds)
    joblib.dump(
        clf,
        f"/Users/raisaurabh04/Downloads/GitHub/Data-Science-Projects/AirlinesHackathon/model/{MODEL}_{FOLD}.pkl",
    )
