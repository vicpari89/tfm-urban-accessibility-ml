"""LightGBM: gradient boosted trees, histogram based.

Docs:
    - API: https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html
"""

from lightgbm import LGBMClassifier

from urban_accessibility_ml.config import SEED

N_ESTIMATORS = 300
CLASS_WEIGHT = "balanced"
N_JOBS = -1
VERBOSITY = -1  # silent


def build_model(seed: int = SEED) -> LGBMClassifier:
    """Return a LightGBM classifier with class weights (untuned hyperparameters).

    Input:  seed (int)
    Output: an UNTRAINED LGBMClassifier
    """
    return LGBMClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight=CLASS_WEIGHT,
        random_state=seed,
        n_jobs=N_JOBS,
        verbose=VERBOSITY,
    )
