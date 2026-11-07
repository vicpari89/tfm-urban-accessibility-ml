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
SUBSAMPLE_FREQ = 1  # without it LightGBM ignores subsample (no effect while subsample = 1.0)
NEED_SAMPLE_WEIGHT = False


def build_model(
    seed: int = SEED,
    *,
    learning_rate: float = 0.1,
    num_leaves: int = 31,
    n_estimators: int = N_ESTIMATORS,
    min_child_samples: int = 20,
    subsample: float = 1.0,
    colsample_bytree: float = 1.0,
) -> LGBMClassifier:
    """Return a LightGBM classifier with class weights. The defaults are the LightGBM ones.

    Input:  seed, learning_rate, num_leaves, n_estimators, min_child_samples, subsample, colsample_bytree.
    Output: an UNTRAINED LGBMClassifier
    """
    return LGBMClassifier(
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        n_estimators=n_estimators,
        min_child_samples=min_child_samples,
        subsample=subsample,
        subsample_freq=SUBSAMPLE_FREQ,
        colsample_bytree=colsample_bytree,
        class_weight=CLASS_WEIGHT,
        random_state=seed,
        n_jobs=N_JOBS,
        verbose=VERBOSITY,
    )
