"""
EDITABLE -- The agent modifies this file.
Define the model pipeline for Polymarket Whale Flow classification.
Starting point: Logistic Regression Baseline.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_model():
    """
    Return a scikit-learn Pipeline. 
    Experiment 2: Increased Forest depth and trees for complexity.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )),
    ])
