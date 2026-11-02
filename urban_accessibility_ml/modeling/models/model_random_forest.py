"""Random forest: bagged decision trees.

Docs:
    - API:        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    - User guide: https://scikit-learn.org/stable/modules/ensemble.html#random-forests
"""

from sklearn.ensemble import RandomForestClassifier

from urban_accessibility_ml.config import SEED

N_ESTIMATORS = 300
CLASS_WEIGHT = "balanced"
N_JOBS = -1
NEED_SAMPLE_WEIGHT = False


def build_model(seed: int = SEED) -> RandomForestClassifier:
    """Return a random forest with class weights (untuned hyperparameters).

    Input:  seed (int)
    Output: an UNTRAINED RandomForestClassifier
    """
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS, class_weight=CLASS_WEIGHT, random_state=seed, n_jobs=N_JOBS
    )
