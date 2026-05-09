from sklearn.ensemble import HistGradientBoostingClassifier

def build_model():
    """Returns a regularized histogram gradient boosting classifier."""
    return HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.03,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        random_state=42,
    )
