# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


# %%
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import set_config


# %%
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    f1_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, StackingClassifier
from xgboost import XGBClassifier


# %%
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
#%%
import os

cd_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(cd_path)

# %%
data = pd.read_csv("../input/train.csv", index_col="ID")
data.sort_index(axis=0, inplace=True)


# %%
data.shape
data.head()


# %%
X = data.iloc[:, :-1]
y = data.iloc[:, -1]


# %%
y


# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)


# %%
X_train.dtypes


# %%
X_train.isna().sum()
X_test.isna().sum()


# %%
mean_imputer = SimpleImputer(strategy="mean")
median_imputer = SimpleImputer(strategy="median")
mode_imputer = SimpleImputer(strategy="most_frequent")
unknown_imputer = SimpleImputer(strategy="constant", fill_value="Unknown")


# %%
std_scalar = StandardScaler()
onehot_encoder = OneHotEncoder(drop="first", handle_unknown="error", sparse=False)
label_encoder = LabelEncoder()


# %%
XGBClassifier().get_params()


# %%
randomforest = RandomForestClassifier(
    n_jobs=-1,
    random_state=4,
    verbose=4,
    n_estimators=2000,
    class_weight="balanced",
    max_depth=7,
    max_features=0.9,
)
extratrees = ExtraTreesClassifier(
    n_jobs=-1,
    class_weight="balanced",
    random_state=4,
    verbose=4,
    n_estimators=2000,
    max_depth=7,
    max_features=0.9,
)
svc = SVC(C=1, kernel="linear", random_state=4, verbose=1, class_weight="balanced")

xgboost = XGBClassifier(
    objective="multi:softmax",
    num_class=4,
    random_state=4,
    verbosity=1,  # num_parallel_tree=500,
    n_estimators=500,
)


# %%
stack_clf = StackingClassifier(
    estimators=[("rfc", randomforest), ("xgb", xgboost),],
    final_estimator=extratrees,
    n_jobs=-1,
    verbose=1,
)

bag_clf = BaggingClassifier(
    base_estimator=randomforest, n_estimators=5, oob_score=True, n_jobs=-1, verbose=1
)


# %%
X_train.dtypes


# %%
# class CategoricalTransformer( BaseEstimator, TransformerMixin ):
#     #Class constructor method that takes in a list of values as its argument
#     def __init__(self, cat_features):
#         self._cat_features = cat_features

#     #Return self nothing else to do here
#     def fit( self, X, y = None  ):
#         return self

#     #Transformer method we wrote for this transformer
#     def transform(self, X , y = None ):
#        #Depending on constructor argument break dates column into specified units
#        #using the helper functions written above
#        for feature in self._cat_features:
#            if feature == 'Var_1':
#                mode_imputer.fit(X[[feature]])
#                X[feature] = mode_imputer.transform(X[[feature]])
#             else :
#                 unknown_imputer.fo
#        return X.values
# work in progress


# %%
categorical_feature_mode = ["Var_1"]
categorical_feature_unknown = [
    "Gender",
    "Ever_Married",
    "Graduated",
    "Profession",
    "Spending_Score",
]
numerical_features = ["Age", "Work_Experience", "Family_Size"]


# %%
numerical_transformer = Pipeline(
    steps=[("mean_imputer", mean_imputer), ("scaler", std_scalar)], verbose=True
)
categorical_transformer_unknown = Pipeline(
    steps=[("unknown_imputer", unknown_imputer), ("onehot", onehot_encoder)],
    verbose=True,
)

categorical_transformer_mode = Pipeline(
    steps=[("mode_imputer", mode_imputer), ("onehot", onehot_encoder)], verbose=True
)


# %%
#%%
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical_transformation_pipeline",
            numerical_transformer,
            numerical_features,
        ),
        (
            "categorical_transformation_pipeline_unknown",
            categorical_transformer_unknown,
            categorical_feature_unknown,
        ),
        (
            "categorical_transformation_pipeline_mode",
            categorical_transformer_mode,
            categorical_feature_mode,
        ),
    ],
    verbose=True,
)


# %%
randomforest_pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", randomforest)], verbose=True
)
randomforest_pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", randomforest)], verbose=True
)

extratrees_pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", extratrees)], verbose=True
)

svc_pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", svc)], verbose=True
)

xgboost_pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", xgboost)], verbose=True
)


# %%
stacking_pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", stack_clf)], verbose=True
)
bagging_pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", bag_clf)], verbose=True
)


# %%
set_config(display="diagram")


# %%
randomforest_pipeline
extratrees_pipeline
svc_pipeline
xgboost_pipeline
stacking_pipeline
bagging_pipeline


# %%
set_config(display="text")


# %%
label_encoder.fit(y_train)
y_train = label_encoder.transform(y_train)
y_test = label_encoder.transform(y_test)


# %%
randomforest_pipeline.fit(X_train, y_train)
y_pred = randomforest_pipeline.predict(X_test)
print(accuracy_score(y_test, y_pred))

# %%
extratrees_pipeline.fit(X_train, y_train)
y_pred = extratrees_pipeline.predict(X_test)
print(accuracy_score(y_test, y_pred))

# %%
svc_pipeline.fit(X_train, y_train)
y_pred = svc_pipeline.predict(X_test)
print(accuracy_score(y_test, y_pred))

# %%
xgboost_pipeline.fit(X_train, y_train)
y_pred = xgboost_pipeline.predict(X_test)
print(accuracy_score(y_test, y_pred))

# %%
stacking_pipeline.fit(X_train, y_train)
y_pred = stacking_pipeline.predict(X_test)
print(accuracy_score(y_test, y_pred))

# %%
bagging_pipeline.fit(X_train, y_train)
y_pred = bagging_pipeline.predict(X_test)
print(accuracy_score(y_test, y_pred))

#%%
test_data = pd.read_csv("../input/test.csv", index_col="ID")


# %%
test_data["Segmentation"] = randomforest_pipeline.predict(test_data)

#%%
test_data["Segmentation"] = label_encoder.inverse_transform(test_data[["Segmentation"]])
# %%
test_data.reset_index(inplace=True)


# %%
test_data[["ID", "Segmentation"]].to_csv(
    "../output/randomforest_submission_1.csv", index=False
)


# %%

