#! /usr/local/bin/python3.8
from os import environ
import pandas as pd
from sklearn.metrics import (
    f1_score,
    classification_report,
    precision_score,
    recall_score,
    accuracy_score,
)
from sklearn.model_selection import RandomizedSearchCV
import dispatcher
import mlflow
from mlflow import log_artifact, log_metric, log_param, log_params, sklearn

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
    log_metric("Accuracy", accuracy)
    log_metric("Precision", precision)
    log_metric("Recall", recall)
    log_metric("F1 Score", f1)

    # roc_auc = roc_auc_score(y_test, y_pred, average="weighted", multi_class="ovo")
    print("Accuracy Score of the model = ", accuracy)
    print("Precision Score of the model = ", precision)
    print("Recall Score of the model = ", recall)
    print("F1 Score of the model = ", f1)
    # print("ROC AUC Score of the model = ", roc_auc)
    print("Classification Report is ")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    mlflow.set_experiment("Toxic Pesticides Hackathon")
    with mlflow.start_run(run_name=f"Run_Train_{MODEL}_{FOLD}"):
        df = pd.read_csv(TRAINING_DATA)
        train_df = df[df["kfold"].isin(FOLD_MAPPING.get(FOLD))].copy()
        valid_df = df[df["kfold"] == FOLD].copy()

        ytrain = train_df["Crop_Damage"]
        yvalid = valid_df["Crop_Damage"]

        train_df.drop(columns=["Crop_Damage", "ID", "kfold"], inplace=True)
        valid_df.drop(columns=["Crop_Damage", "ID", "kfold"], inplace=True)

        # param_grid = {
        #     "preprocessor__numerical_transformation_pipeline__imputer__strategy": [
        #         "mean",
        #         "median",
        #     ],
        #     "classifier__max_depth": [2, 5, 7],
        #     "classifier__n_estimators": [100, 200, 500, 700],
        #     "classifier__class_weight": ["balanced", "balanced_subsample"],
        # }
        clf = dispatcher.MODELS[MODEL]
        # grid_search = RandomizedSearchCV(
        #     clf, param_grid, n_iter=7, scoring="accuracy", verbose=2
        # )
        clf.fit(train_df, ytrain)
        preds = clf.predict(valid_df)
        metric_data(yvalid, preds)
        # joblib.dump(
        #     clf,
        #     f"/Users/raisaurabh04/Downloads/GitHub/Data-Science-Projects/AirlinesHackathon/model/{MODEL}_{FOLD}.pkl",
        # )
        # sklearn.save_model(
        #     clf, f"../models/{MODEL}_{FOLD}",
        # )
        sklearn.log_model(
            sk_model=clf, artifact_path="model", registered_model_name="model-sklearn",
        )
        log_param("FOLD", FOLD)
        log_param("Stage", "Training completed")
        log_artifact(f"../models/{MODEL}_{FOLD}")
