#! /usr/local/bin/python3.8
from os import environ, path
import numpy as np
import pandas as pd
import joblib

import dispatcher
import mlflow
from mlflow import log_artifacts, log_param

# getting input from environment variables
TESTING_DATA = environ.get("TEST_FILE_PATH")
MODEL = environ.get("MODEL")


def predict():
    df = pd.read_csv(TESTING_DATA)
    test_df = df.drop(columns=["Accident_ID"])
    predictions = None

    targetDecoding = {
        0: "Highly_Fatal_And_Damaging",
        1: "Significant_Damage_And_Serious_Injuries",
        2: "Minor_Damage_And_Injuries",
        3: "Significant_Damage_And_Fatalities",
    }

    for FOLD in range(5):
        clf = joblib.load(path.join("../models", f"{MODEL}_{FOLD}/model.pkl",))

        preds = clf.predict_proba(test_df)
        print(1, preds.shape)
        if FOLD == 0:
            predictions = preds
        else:
            predictions += preds

    predictions = predictions / 5
    pred_class = np.argmax(predictions, axis=1)
    submission_df = pd.DataFrame(columns=["Accident_ID", "Severity"])
    submission_df["Accident_ID"] = df["Accident_ID"]
    submission_df["Severity"] = pred_class
    submission_df["Severity"] = submission_df["Severity"].map(targetDecoding)
    return submission_df


if __name__ == "__main__":
    mlflow.set_experiment("Airlines Hackathon")
    with mlflow.start_run(run_name=f"Run_Predict_{MODEL}"):

        submission = predict()
        print(submission)
        submission.to_csv(
            f"../output/submission_{MODEL}.csv", index=False,
        )
        log_param("Stage", "Prediction Done")
        log_artifacts(f"../output")
