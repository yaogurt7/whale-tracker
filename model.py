"""
EDITABLE -- The agent modifies this file.
Define the model pipeline for Polymarket Whale Flow classification.
Starting point: Logistic Regression Baseline.
"""
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Use headless backend for Windows/Server stability
plt.switch_backend('Agg')

def build_model():
    """
    Returns the core model pipeline. 
    Modify this function to test new architectures.
    Experiment 3: HistGradientBoostingClassifier.
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', HistGradientBoostingClassifier(random_state=42))
    ])
    return pipeline

def check_importance():
    from prepare import load_data
    X_train, y_train, X_val, y_val, feature_names = load_data()
    model = build_model()
    model.fit(X_train, y_train)

    # Extract LogReg coefficients
    importances = model.named_steps['model'].coef_[0]
    feat_df = pl.DataFrame({
        "feature": feature_names,
        "importance": np.abs(importances)
    }).sort("importance", descending=True)

    print("\n--- Feature Importance ---")
    print(feat_df.head(10))

if __name__ == "__main__":
    check_importance()
