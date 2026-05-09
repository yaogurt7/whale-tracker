from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

def build_model():
    """Returns a polynomial interaction Logistic Regression pipeline."""
    return Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(C=0.1, max_iter=1000, random_state=42))
    ])
