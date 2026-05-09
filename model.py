from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

def build_model():
    """Returns a robust-scaled Logistic Regression pipeline."""
    return Pipeline([
        ('scaler', RobustScaler()),
        ('model', LogisticRegression(C=1.0, max_iter=1000, random_state=42))
    ])
