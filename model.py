from sklearn.ensemble import RandomForestClassifier

def build_model():
    """Returns a shallow RandomForest classifier with smoothed leaves."""
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=100,
        max_features=None,
        n_jobs=-1,
        random_state=42,
    )
