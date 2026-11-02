"""Dummy baseline: the floor model, the minimum one. All the other models should improve this one.

Docs:
    - API:        https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    - User guide: https://scikit-learn.org/stable/modules/model_evaluation.html#dummy-estimators
"""

from sklearn.dummy import DummyClassifier

from urban_accessibility_ml.config import SEED

STRATEGY = "most_frequent"

def build_model(seed: int = SEED) -> DummyClassifier:
    """Return a classifier that ignores the inputs and always predicts the most frequent class.

    Input:  seed (int)
    Output: an UNTRAINED DummyClassifier
    """
    return DummyClassifier(strategy=STRATEGY, random_state=seed)
