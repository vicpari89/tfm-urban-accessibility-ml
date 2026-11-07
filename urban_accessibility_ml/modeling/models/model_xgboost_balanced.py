"""XGBoost with balanced classes: gradient boosted trees, trained with balanced sample weights.

Docs:
    - API: https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier

XGBoost has no class_weight, so BalancedXGBClassifier computes the balanced sample weights inside its
own fit, with the rows it receives. This way it works like class_weight="balanced" in the other
models: also inside the inner folds of the tuning.
"""

from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from urban_accessibility_ml.config import SEED

N_ESTIMATORS = 300
N_JOBS = -1


class BalancedXGBClassifier(XGBClassifier):
    """XGBClassifier that balances the classes with sample weights, using the rows given to fit."""

    def fit(self, X, y, sample_weight=None, **kwargs):
        if sample_weight is None:
            sample_weight = compute_sample_weight("balanced", y)
        return super().fit(X, y, sample_weight=sample_weight, **kwargs)


def build_model(
    seed: int = SEED,
    *,
    learning_rate: float | None = None,
    max_depth: int | None = None,
    n_estimators: int = N_ESTIMATORS,
    subsample: float | None = None,
    colsample_bytree: float | None = None,
    min_child_weight: float | None = None,
) -> BalancedXGBClassifier:
    """Return an XGBoost classifier with balanced classes. None means the XGBoost default.

    Input:  seed, learning_rate, max_depth, n_estimators, subsample, colsample_bytree, min_child_weight.
    Output: an UNTRAINED BalancedXGBClassifier
    """
    return BalancedXGBClassifier(
        learning_rate=learning_rate,
        max_depth=max_depth,
        n_estimators=n_estimators,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        random_state=seed,
        n_jobs=N_JOBS,
    )
