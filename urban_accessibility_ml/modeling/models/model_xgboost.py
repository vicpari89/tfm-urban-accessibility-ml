"""XGBoost: gradient boosted trees.

Docs:
    - API: https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier

TODO: analyze how to use a parameter similar to CLASS_WEIGHT = "balanced"
"""

from xgboost import XGBClassifier

from urban_accessibility_ml.config import SEED

N_ESTIMATORS = 300
N_JOBS = -1


def build_model(seed: int = SEED) -> XGBClassifier:
    """Return an XGBoost classifier (untuned hyperparameters).

    Input:  seed (int)
    Output: an UNTRAINED XGBClassifier
    """
    return XGBClassifier(n_estimators=N_ESTIMATORS, random_state=seed, n_jobs=N_JOBS)
