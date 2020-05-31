from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


MODELS = {
    "randomforest" : RandomForestClassifier(n_estimators = 200, n_jobs=-1, verbose=2),
    "extratrees" : ExtraTreesClassifier(n_estimators = 200, n_jobs=-1, verbose=2)
}