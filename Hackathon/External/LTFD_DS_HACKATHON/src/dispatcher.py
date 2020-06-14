from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


MODELS = {
    'randomforest' : RandomForestClassifier(n_estimators=200, n_jobs=-1, verbose=2),
    'logisticRegression' : LogisticRegression(C=10)
}