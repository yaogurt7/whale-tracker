from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer

def build_model():
    """Returns a quantile-normalized Logistic Regression pipeline."""
    return Pipeline([
        ('quantile', QuantileTransformer(
            n_quantiles=1000,
            output_distribution='normal',
            random_state=42,
        )),
        ('model', LogisticRegression(C=1.0, max_iter=1000, random_state=42))
    ])
