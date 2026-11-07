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


def build_model(
    seed: int = SEED,
    *,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    max_features: str | float = "sqrt",
) -> RandomForestClassifier:
    """Return a random forest with class weights. The defaults are the sklearn ones.

    Input:  seed, max_depth (depth of each tree), min_samples_leaf (tramos per leaf),
        max_features (variables per split).
    Output: an UNTRAINED RandomForestClassifier
    """
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=CLASS_WEIGHT,
        random_state=seed,
        n_jobs=N_JOBS,
    )
