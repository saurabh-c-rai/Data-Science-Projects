#! /usr/local/bin/python3.8
from os import environ, path
import numpy as np
import pandas as pd
import joblib
import mlflow
from mlflow import log_artifacts, log_param

# getting input from environment variables
TESTING_DATA = environ.get("TEST_FILE_PATH")
MODEL = environ.get("MODEL")


def predict():
    df = pd.read_csv(TESTING_DATA)
    test_df = df.drop(columns=["ID"])
    predictions = None

    for FOLD in range(5):
        clf = joblib.load(path.join("../models", f"{MODEL}_{FOLD}/model.pkl",))

        preds = clf.predict_proba(test_df)
        if FOLD == 0:
            predictions = preds
        else:
            predictions += preds

    predictions = predictions / 5
    pred_class = np.argmax(predictions, axis=1)
    submission_df = pd.DataFrame(columns=["ID", "Crop_Damage"])
    submission_df["ID"] = df["ID"]
    submission_df["Crop_Damage"] = pred_class
    return submission_df


if __name__ == "__main__":
    mlflow.set_experiment("Toxic Pesticides Hackathon")
    with mlflow.start_run(run_name=f"Run_Predict_{MODEL}"):

        submission = predict()
        submission.to_csv(
            f"../output/submission_{MODEL}.csv", index=False,
        )
        log_param("Stage", "Prediction Done")
        log_artifacts(f"../output")
