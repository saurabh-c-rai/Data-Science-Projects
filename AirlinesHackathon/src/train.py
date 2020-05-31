#! D:\PythonEnvironment\ML_Environment\Scripts python3
from os import environ
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    classification_report,
    precision_score,
    recall_score,
    accuracy_score,
)

TRAINING_DATA = environ.get("TRAINING_DATA")
FOLD = int(environ.get("FOLD"))

FOLD_MAPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3],
}

if __name__ == "__main__":
    df = pd.read_csv(TRAINING_DATA)
    train_df = df[df["kfold"].isin(FOLD_MAPPING.get(FOLD))]
    valid_df = df[df["kfold"] == FOLD]

    ytrain = train_df["Severity"]
    yvalid = valid_df["Severity"]
    
    print(train_df.head())

    train_df.drop(columns = ['Severity', "Accident_ID", "kfold"], inplace=True)
    valid_df.drop(columns =['Severity', "Accident_ID", "kfold"], inplace=True)

    targetEncoding = {
        "Highly_Fatal_And_Damaging": 0,
        "Significant_Damage_And_Serious_Injuries": 1,
        "Minor_Damage_And_Injuries": 2,
        "Significant_Damage_And_Fatalities": 3,
    }

    ytrain = ytrain.map(targetEncoding)
    yvalid = yvalid.map(targetEncoding)

    clf = RandomForestClassifier(n_jobs=-1, verbose=2)
    clf.fit(train_df, ytrain)
    preds = clf.predict_proba(valid_df)[:, 1]
    print(preds)
