"""
EDITABLE -- The agent modifies this file.
Define the model pipeline for Polymarket Whale Flow classification.
Starting point: Logistic Regression Baseline.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


def build_model():
    """
    Return a scikit-learn Pipeline. 
    Experiment 4: Logistic Regression with Polynomial Interactions.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
        ("model", LogisticRegression(random_state=42, max_iter=1000)),
    ])
