"""Logistic regression: the reference model chosen in the thesis.

Docs:
    - API:        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    - User guide: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
"""

from sklearn.linear_model import LogisticRegression

from urban_accessibility_ml.config import SEED

CLASS_WEIGHT = "balanced"
MAX_ITER = 1000
NEED_SAMPLE_WEIGHT = False
SOLVERS = {"l2": "lbfgs", "l1": "saga"}


def build_model(seed: int = SEED, *, C: float = 1.0, penalty: str = "l2") -> LogisticRegression:
    """Return a multiclass logistic regression with class weights.

    Input:  seed, C (regularization strength, inverse: small C = stronger), penalty ("l2" or "l1").
    Output: an UNTRAINED LogisticRegression
    """
    return LogisticRegression(
        C=C,
        penalty=penalty,
        solver=SOLVERS[penalty],
        class_weight=CLASS_WEIGHT,
        max_iter=MAX_ITER,
        random_state=seed,
    )
