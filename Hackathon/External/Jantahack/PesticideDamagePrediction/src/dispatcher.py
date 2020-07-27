#%%
import numpy as np
import sklearn
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, StackingClassifier
from xgboost import XGBClassifier

#%%
std_scalar = StandardScaler()
min_max_scalar = MinMaxScaler()
onehot_encoder = OneHotEncoder(drop="first")
imputer = SimpleImputer(strategy="median", missing_values=np.nan)
#%%
randomforest = RandomForestClassifier(
    n_jobs=-1, n_estimators=100, random_state=4, verbose=1,
)
extratrees = ExtraTreesClassifier(
    n_jobs=-1, class_weight="balanced", random_state=4, verbose=1,
)
svc = SVC(C=1, kernel="linear", random_state=4, verbose=1)

xgboost = XGBClassifier(
    objective="multi:softmax", num_class=3, random_state=4, verbosity=1
)

#%%
stack_clf = StackingClassifier(
    estimators=[("rfc", randomforest), ("xgb", xgboost),],
    final_estimator=extratrees,
    n_jobs=-1,
    verbose=1,
)

bag_clf = BaggingClassifier(
    base_estimator=randomforest, n_estimators=5, oob_score=True, n_jobs=-1, verbose=1
)
#%%
numerical_features = [
    "Estimated_Insects_Count",
    "Number_Doses_Week",
    "Number_Weeks_Used",
    "Number_Weeks_Quit",
]

categorical_features = ["Crop_Type", "Soil_Type", "Pesticide_Use_Category", "Season"]
#%%
numerical_transformer = Pipeline(
    steps=[("imputer", imputer), ("scaler", std_scalar)], verbose=True
)
categorical_transformer = Pipeline(steps=[("onehot", onehot_encoder)], verbose=True)

#%%
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical_transformation_pipeline",
            numerical_transformer,
            numerical_features,
        ),
        (
            "categorical_transformation_pipeline",
            categorical_transformer,
            categorical_features,
        ),
    ],
    verbose=True,
)
#%%
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

#%%
stacking_pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", stack_clf)], verbose=True
)
bagging_pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", bag_clf)], verbose=True
)

#%%
MODELS = {
    "randomforest": randomforest_pipeline,
    "extratrees": extratrees_pipeline,
    "svc": svc_pipeline,
    "xgboost": xgboost_pipeline,
    "stack_clf": stacking_pipeline,
    "bagging_clf": bagging_pipeline,
}
# %%
