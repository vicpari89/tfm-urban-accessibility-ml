"""Logistic regression: the reference model chosen in the thesis.

Docs:
    - API:        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    - User guide: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
"""

from sklearn.linear_model import LogisticRegression

from urban_accessibility_ml.config import SEED

CLASS_WEIGHT = "balanced"
MAX_ITER = 1000

def build_model(seed: int = SEED) -> LogisticRegression:
    """Return a multiclass logistic regression with class weights.

    Input:  seed (int)
    Output: an UNTRAINED LogisticRegression
    """
    return LogisticRegression(class_weight=CLASS_WEIGHT, max_iter=MAX_ITER, random_state=seed)
