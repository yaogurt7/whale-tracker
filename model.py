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
    Experiment 1: Tuned Random Forest Classifier for non-linear interactions.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )),
    ])
