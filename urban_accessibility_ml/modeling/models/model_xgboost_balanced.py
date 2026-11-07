"""XGBoost with balanced classes: gradient boosted trees, trained with balanced sample weights.

Docs:
    - API: https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier

TODO: analyze how to use a parameter similar to CLASS_WEIGHT = "balanced"
"""

from xgboost import XGBClassifier

from urban_accessibility_ml.config import SEED

N_ESTIMATORS = 300
N_JOBS = -1
NEED_SAMPLE_WEIGHT = True


def build_model(
    seed: int = SEED,
    *,
    learning_rate: float | None = None,
    max_depth: int | None = None,
    n_estimators: int = N_ESTIMATORS,
    subsample: float | None = None,
    colsample_bytree: float | None = None,
    min_child_weight: float | None = None,
) -> XGBClassifier:
    """Return an XGBoost classifier. None means the XGBoost default.

    Input:  seed, learning_rate, max_depth, n_estimators, subsample, colsample_bytree, min_child_weight.
    Output: an UNTRAINED XGBClassifier
    """
    return XGBClassifier(
        learning_rate=learning_rate,
        max_depth=max_depth,
        n_estimators=n_estimators,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        random_state=seed,
        n_jobs=N_JOBS,
    )
